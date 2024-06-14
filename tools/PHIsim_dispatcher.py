import concurrent.futures
import os
import shutil
import subprocess
import traceback
import numpy as np

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from tqdm import tqdm

from tools.PHIsim_sim_params import PHIsim_SimulationParams


""" 
By default, ProcessPoolExecutor uses all CPU cores. I'm reducing this a bit, that way the computer 
remains somewhat responsive during simulation, as a few cores will still be available for other tasks.
This appears to be necessary even, with all cores under load I'm getting errors due to filesystem 
timeouts and access denials.
"""
MAX_NUM_CONCURRENT = max(2, int(os.cpu_count()*0.8))

@dataclass
class PHIsim_Result:
    P_LR_out: np.array 
    F_LR_out: np.array 
    P_RL_out: np.array 
    F_RL_out: np.array
    num_cycles_completed: int


class PHIsim_ConcurrentSetup(ABC):
    """
    Abstract class for all simulation setups.

    The PHIsim_ConcurrentSetup object is passed on to a different process. As a result, the object should be picklable, 
    so subclasses may not contain unpicklable objects, such as lambda function references.

    Subclasses MUST implement the 'initialize_input_files()' method, which sets up the 5 required input files:
        - device input file
        - simulation parameters file
        - signal input file
        - carrier input file
        - photon density input file

    The names of these files MUST match the names provided by the sim_parameters object.

    For the latter 3 files, a default implementation is provided, which simply creates empty files. 
    However, these default functions must still be called from initialize_input_files() by a subclass
    if default initialization is desired.

    A local simulation folder, with the name provided by the variable sim_folder, is created for each simulation.
    This name should be unique among all subprocesses running concurrently. It is suggested to use a meaningful name,
    such as "test_with_1ps_pulse", "test_with_2ps_pulse", ... which encodes the parameters being sweeped.

    Subclasses MAY override or extend the 'process_result()' method, to process the results of a single simulation in the
    running subprocess. You may add additional fields and values to the result, but they must be picklable, as the
    result will be passed back to the main process.
    """
    def __init__(self, sim_parameters: PHIsim_SimulationParams, sim_folder: str, num_PHIsim_cycles: int = 1) -> None:
        super().__init__()
        self.sim_params = sim_parameters
        self.sim_folder = sim_folder
        self.num_cycles = num_PHIsim_cycles

    @abstractmethod
    def initialize_input_files(self):
        pass

    def process_result(self, result : PHIsim_Result):
        pass

    def setup_next_cycle(self, cycles_completed: int):
        """Called between cycles if num_PHIsim_cycles > 1
        Can be overridden by subclasses to do something with the results of each cycle, 
        or change the simulation parameters from cycle to cycle (if needed).
        """
        # PHIsim derives the names of all output files from the simulation parameter 'output_filename'
        # (see open_output_data_files() in PHIsim_v3.c) 
        output_base = self.sim_params.output_filename[:-4] # strip '.txt' the same way PHIsim does

        # use output of previous cycle as input for next cycle
        os.remove(self.sim_params.photond_file)
        os.remove(self.sim_params.carrier_file)
        os.rename(f"{output_base}_opt.txt", self.sim_params.photond_file)
        os.rename(f"{output_base}_car.txt", self.sim_params.carrier_file)	

        # TODO delete signal file?

        # rename intermediate results so we don't lose them
        os.rename(self.sim_params.output_filename, f"{output_base}_{cycles_completed}.txt")
        if self.sim_params.video_N > 0:
            for suffix in ("carriers", "LRf", "LRp", "RLf", "RLp"):	
                os.rename(f"{output_base}_vid_{suffix}.txt", 
                          f"{output_base}_vid_{suffix}_{cycles_completed}.txt")

    ## a few convenience methods that can be called by subclasses,
    ## in case a default/empty photond, carrier or signal input is needed

    def default_initialize_photond_file(self):
        with open(self.sim_params.photond_file, "w") as f:
            f.write("   -1 -1 -1 -1 -1 -1 -1 \n")
            f.write("-1  -1 -1 -1 -1 -1 -1 -1")

    def default_initialize_carrierfile(self):
        with open(self.sim_params.carrier_file, "w") as f:
            f.write("   -1 -1 \n")
            f.write("-1  -1")

    def default_initialize_signal_input_file(self):
        with open(self.sim_params.signal_input_file, "w") as f:
            pass


class PHIsim_Executables:
    """
    Class describing where the PHIsim executables can be found.
    By default, this assumes the executables are in the same folder and named "PHIsim.exe" and "PHIsim_input.exe".
    If that's the case, you can simply instantiate this with only the path argument, e.g.

        executables = PHIsim_Executables("C:/path/to/executables")	
    """

    def __init__(self, PHIsim_executable_path, PHIsim_input_executable_path=None,
                 PHIsim_executable_name="PHIsim.exe", PHIsim_input_executable_name="PHIsim_input.exe"):
        
        self.PHIsim_executable_path = PHIsim_executable_path
        if PHIsim_input_executable_path is None:
            self.PHIsim_input_executable_path = PHIsim_executable_path
        else:
            self.PHIsim_input_executable_path = PHIsim_input_executable_path

        self.PHIsim_executable_name = PHIsim_executable_name
        self.PHIsim_input_executable_name = PHIsim_input_executable_name

    def copy_to(self, path):
        shutil.copy(f"{self.PHIsim_input_executable_path}/{self.PHIsim_input_executable_name}", path)
        shutil.copy(f"{self.PHIsim_executable_path}/{self.PHIsim_executable_name}", path)


def _run_simulation(config : PHIsim_ConcurrentSetup, exec: PHIsim_Executables, path, skip_simulation) -> PHIsim_Result:
    os.chdir(path)

    cycles_completed = 0
    simpars = config.sim_params 

    if not skip_simulation:
        # run PHIsim_input to create devicefile.txt
        os.system(f"{exec.PHIsim_input_executable_name} {simpars.params_file} {simpars.device_file} > PHIsim_input_stdout.txt")

        # run main PHIsim simulation, cycle if needed
        while cycles_completed < config.num_cycles:

            # if this is not the first cycle, use output from previous cycle as input for next cycle
            # (no need to run PHIsim_input again, we re-use the same devicefile)
            if cycles_completed > 0:
                config.setup_next_cycle(cycles_completed)

            # TODO read stdout and report progress to main process?
            # NOTE sometimes the os.system() call fails because it cannot access the files
            #      subprocess.call() appears to be more robust, but at the same time appears to consume more memory
            #      if you run out of memory, it could be worth it to try the os.system() version.
            #os.system(f"{exec.PHIsim_executable_name} {simpars.params_file} devicefile.txt {simpars.carrier_file} {simpars.photond_file} {simpars.signal_input_file}) > PHIsim_stdout.txt")
            subprocess.call([exec.PHIsim_executable_name, simpars.params_file, "devicefile.txt" , simpars.carrier_file, 
                            simpars.photond_file, simpars.signal_input_file], stdout=open("PHIsim_stdout.txt", "w"))
            cycles_completed += 1

    P_LR_out, F_LR_out, P_RL_out, F_RL_out = np.loadtxt(simpars.output_filename, dtype=np.float_, unpack=True, ndmin=2)
    result = PHIsim_Result(P_LR_out, F_LR_out, P_RL_out, F_RL_out, cycles_completed)
    config.process_result(result)

    return result


def PHIsim_run_concurrent(configs : Iterable[PHIsim_ConcurrentSetup], 
                          work_folder: str, 
                          executables: PHIsim_Executables,
                          debug_serial=False,
                          skip_simulation=False
                          ) -> dict[PHIsim_ConcurrentSetup, PHIsim_Result]:
    """
    The main method to run a set of simulations concurrently. Uses a ProcessPoolExecutor to run the simulations in parallel.
    Creates a workspace (i.e., a subfolder) for each simulation, and runs the simulations in parallel. After the simulations
    are done, the results are read and returned. The work_folders are kept after the simulations. This way, for example, 
    you can analyze the results again later, or (for example) create video files from the results. If the simulations live 
    in a version-controlled (e.g. git) folder, you probably want to set up your version-control such that it ignores these 
    workspace sub-folders. If you have sufficient RAM, consider mounting a RAM-disk and using that for the workspaces.

    (Note that I attempted to implement a solution to clean up the subfolders automatically, but I had some problems getting 
    it to work consistently. I believe this is due to the large number of file requests being made overwhelming the OS subsystem.)

    For convenience, "dispatching" and "running" progress is shown. Note that "running" only updates when a subprocess is
    finished, so it may take a while for this one to start updating. Subprocesses will start running as soon as the first one
    is sheduled.

    If you run into trouble, you can pass debug_serial=True to disable parallelization and execute the processes in the same 
    context as the main process. This may help in debugging issues with either the dispatcher or PHIsim.

    If you want to re-run the data processing on existing results without re-running the simulations, set skip_simulation=True.
    """
    os.chdir(work_folder)
    current_folder = os.getcwd()
    results = {}

    progress_scheduling = tqdm(total=len(configs), bar_format=" -> dispatching simulations: [{n}/{total}]")
    progress_running = tqdm(total=len(configs), desc=" -> running simulations", ncols=100)
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_NUM_CONCURRENT) as executor:
        futures = []
        futures_to_config = {}
        subfolders = set()
        for config in configs:
            ## create a workspace (i.e., a subfolder) for each simulation
            if config.sim_folder in subfolders:
                print(f"WARNING: Duplicate results_folder name {config.sim_folder}! Skipping simulation.")
                continue
            subfolders.add(config.sim_folder)

            local_sim_path = current_folder + "/" + config.sim_folder
            if not skip_simulation:
                try:
                    os.mkdir(config.sim_folder)
                except FileExistsError:
                    pass # folder already exists, ignore error (but results in the folder will be overwritten)
            else:
                # check if folder exists
                if not os.path.exists(config.sim_folder):
                    raise RuntimeError(f"Error: skip_simulation is True, but {config.sim_folder} does not exist.")

            # setup the executables and input files 
            # we do this here to avoid overloading the file system with requests from different processes
            # BUT: we don't rebuild the input files if we are skipping the simulation 
            # (otherwise we risk overwriting them with different parameters than the ones used for the simulation)
            os.chdir(config.sim_folder)
            if not skip_simulation:
                executables.copy_to(local_sim_path)
                config.initialize_input_files()
            os.chdir("..")

            if not debug_serial:
                ## spin out a process to run the simulation in the subfolder
                future = executor.submit(_run_simulation, config, executables, local_sim_path, skip_simulation)
                futures.append(future)
                futures_to_config[future] = config
                future.add_done_callback(lambda _: progress_running.update(1))
            else:
                # debug mode - run each simulation in the same process
                results[config] = _run_simulation(config, executables, local_sim_path, skip_simulation)
                os.chdir(work_folder)
                progress_running.update(1)

            progress_scheduling.update(1)

        # block until all processes have finished or terminated
        for future in concurrent.futures.as_completed(futures):
            config = futures_to_config[future]
            try:
                results[config] = future.result()
            except Exception:
                print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                print(f"ERROR: Exception thrown by process in {config.sim_folder}: ")	
                print(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                traceback.print_exc()
                print(f"----------------------------------------------------------------")
                results[config] = None


    return results


