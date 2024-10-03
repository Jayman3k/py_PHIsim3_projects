import numpy as np
import scipy.signal
import scipy.optimize
import os

from itertools import product, chain
from matplotlib import pyplot as plt
from typing import override
from dataclasses import dataclass, replace

from tools.fmt_utils import fmt_eng, fmt_pow10, plot_latex_style

import tools.PHIsim_sim_params as phip
import tools.PHIsim_dispatcher as phid
import tools.PHIsim_signal_input_util as phis 
import tools.PHIsim_signal_processing as phisig
import tools.PHIsim_constants as constants

"""A module to run tests on the PHIsim SOA model, typically we simulate

   IN -> waveguide -> SOA -> waveguide -> OUT

The tests here were written to test the transmission and gain of the SOA,
in particular the modifications we want to add to make the spectral behavior of the
SOA more realistic.
"""

##############################################################################
executables = phid.PHIsim_Executables("C:/PHISim_v3")

work_folder_default = "C:/Users/20244629/Documents/temp"
work_folder_RAMdisk = "R:/Temp"

# prefer RAM-disk folder if it is mounted
if os.path.isdir(work_folder_RAMdisk):
    work_folder = work_folder_RAMdisk
else:
    work_folder = work_folder_default

##############################################################################
## some precalculated filter weights to match gain curves
## note that these are for PHIsim with n_wavelen_segment = 8,
## for different segment length, you will need to recalculate these

VanGasse_Weights = [0.012851639924, 0.041698958521, 0.119001787219, 0.205084278797,
                    0.242726671078, 0.205084278797, 0.119001787219, 0.041698958521,
                    0.012851639924] # match Van Gasse curve with n_wavelen_segment = 8

Judo1A_Weights = [0.035702134069, 0.241065536045, 0.446464659773, 0.241065536045, 0.035702134069] # match Judo 1A curve with n_wavelen_segment = 8

All_Weights = {"VanGasse": VanGasse_Weights, 
               "Judo"    :   Judo1A_Weights}

##############################################################################

@dataclass
class SoaParams:
    soa_current  : float      # Ampere
    soa_segments : int = 100  # number of segments in the waveguide
    wg_segments  : int = 60   # number of waveguide segments before and after SOA (mutiply by 2 for total)
    use_lowpass  : bool = False
    lowpass_w    : float = None   # weight of the lowpass filter, ignored if use_lowpass == False
    use_mzi      : bool = False  # ignored if use_lowpass == True

    def total_segments(self):
        filter_segments = 0
        if self.use_lowpass and self.lowpass_w is not None:
            filter_segments = (len(self.lowpass_w) - 1) // 2

        return self.soa_segments + self.wg_segments*2 + filter_segments \
                + (3 if self.use_mzi else 0)
        

@dataclass
class SignalInputParams:
    pulse_type   : str   = "sech2"   # "sech2", "gaussian" or "chirp"
    pulse_energy : float = 100e-12   # Joule
    pulse_FWHM   : float = 5e-12     # seconds (ignored if "chirp")
    pulse_chirp  : float = 1.0
    pulse_side   : str   = "left"    # "left" or "both" ("right" not implemented yet)
    input_size   : float = 500e-12   # seconds
    first_pulse  : float = 400e-12   # seconds - center of first pulse if "sech2" or "gaussian", start of chirp if "chirp"
    num_pulses   : int   = 1         # (ignored if "chirp")
    distance     : float = 10e-12    # seconds - distance between pulse centers (ignored if repeat == 1) 
    complex_chirp: bool = True       # use complex (phase) chirp, else use real (power) chirp, ignored if not "chirp"
    chirp_min    : float = 1e9       # Hz
    chirp_max    : float = 1e12      # Hz

# when considering a pulse in a signal for processing (i.e. calculate the gain spectrum of a single pulse)
# we clip the pulse to be in the range [CENTER - PULSE_CLIP_FWHM_FACTOR*FWHM, CENTER + PULSE_CLIP_FWHM_FACTOR*FWHM]
PULSE_CLIP_FWHM_FACTOR = 3.5

class TestSOASetup(phid.PHIsim_ConcurrentSetup):

    def __init__(self, sim_params : phip.PHIsim_SimulationParams, soa_params : SoaParams, signal: SignalInputParams,
                 key, work_folder=None):
        
        assert not (soa_params.use_lowpass and soa_params.use_mzi), "use_lowpass and use_mzi are mutually exclusive"
        
        if work_folder is None:
            work_folder = f"transmission_with_single_soa_{fmt_eng(soa_params.soa_current)}A_{soa_params.soa_segments}segm"
        if soa_params.use_lowpass:
            work_folder += "_lowpass"
        if soa_params.use_mzi:
            work_folder += "_mzi"

        super().__init__(sim_params, work_folder)

        self.key = key
        self.soa_params = soa_params
        self.signal_params = signal

    @override
    def initialize_input_files(self):
        if self.soa_params.use_lowpass:
            if self.soa_params.soa_segments == 0:
                self.initialize_device_input_file_with_lowpass_filter_no_soa()
            else:
                self.initialize_device_input_file_with_lowpass_filter()
        elif self.soa_params.use_mzi:
            self.initialize_device_input_file_with_mzi_filter()
        else:
            self.initialize_device_input_file()

        self.initialize_sim_parameter_file()
        self.initialize_signal_input_file()
        ## everything else defaults
        self.default_initialize_carrierfile()
        self.default_initialize_photond_file()

    def initialize_sim_parameter_file(self):
        self.sim_params.write_to_file()
    
    def initialize_device_input_file(self):
        device_input_content = """\
IO_left__  8		            # this always needs to be in the input file
paswg_L__  1   {wg}               # passive waveguide on the left
soa_1____  2   {seg}	  0         # 100 segm = amplifier 800 micron approx (100fs, 20wl segments)
paswg_R__  1   {wg}               # passive waveguide on the right
IO_right_  9		            # this always needs to be in the input file
-1 -1               # end of component list - start of connections list
IO_left__  R0 paswg_L__  L0    #
paswg_L__  R0 soa_1____  L0    #
soa_1____  R0 paswg_R__  L0    #
paswg_R__  R0 IO_right_  L0    #
xxx  -1  xxx -1 	# end of connections list - start of current source list
0     {cur}    # current in soa_1 in amps 0.2 amps = 13kA/cm2
-1 -1	# end of current source list
From this point on the file can contain any text.
""".format(cur = self.soa_params.soa_current, seg = self.soa_params.soa_segments, wg = self.soa_params.wg_segments)

        with open(self.sim_params.device_file, 'w') as f:
            f.write(device_input_content)

    def initialize_device_input_file_with_lowpass_filter(self):
        device_input_content = """\
IO_left__  8		            # this always needs to be in the input file
paswg_L__  1   {wg}               # passive waveguide on the left
soa_1____  2   {seg}	  0         # 100 segm = amplifier 800 micron approx (100fs, 20wl segments)
lowpass_1  14   1  lowpass_filter.txt  # lowpass LR filter 
paswg_R__  1   {wg}             # passive waveguide on the right
IO_right_  9		            # this always needs to be in the input file
-1 -1               # end of component list - start of connections list
IO_left__  R0 paswg_L__  L0    #
paswg_L__  R0 soa_1____  L0    #
soa_1____  R0 lowpass_1  L0    #
lowpass_1  R0 paswg_R__  L0    #
paswg_R__  R0 IO_right_  L0    #
xxx  -1  xxx -1 	# end of connections list - start of current source list
0     {cur}    # current in soa_1 in amps 0.2 amps = 13kA/cm2
-1 -1	# end of current source list
From this point on the file can contain any text.
""".format(cur = self.soa_params.soa_current, seg = self.soa_params.soa_segments, wg = self.soa_params.wg_segments)

        with open(self.sim_params.device_file, 'w') as f:
            f.write(device_input_content)

        assert self.soa_params.lowpass_w is not None, "lowpass filter weights must not be None"	

        with open("lowpass_filter.txt", 'w') as f:
            f.write(f"{len(self.soa_params.lowpass_w)} ")
            for w in self.soa_params.lowpass_w:
                f.write(f"{w} ")
            f.write("\n")

    def initialize_device_input_file_with_lowpass_filter_no_soa(self):
        assert self.soa_params.soa_segments == 0
        ## for comparison: only a filter and no soa
        device_input_content = """\
IO_left__  8		            # this always needs to be in the input file
paswg_L__  1   {wg}               # passive waveguide on the left
lowpass_1  14   1  lowpass_filter.txt  # lowpass LR filter 
paswg_R__  1   {wg}             # passive waveguide on the right
IO_right_  9		            # this always needs to be in the input file
-1 -1               # end of component list - start of connections list
IO_left__  R0 paswg_L__  L0    #
paswg_L__  R0 lowpass_1  L0    #
lowpass_1  R0 paswg_R__  L0    #
paswg_R__  R0 IO_right_  L0    #
xxx  -1  xxx -1 	# end of connections list - start of current source list
0     {cur}    # current in soa_1 in amps 0.2 amps = 13kA/cm2
-1 -1	# end of current source list
From this point on the file can contain any text.
""".format(cur = self.soa_params.soa_current, wg = self.soa_params.wg_segments)

        with open(self.sim_params.device_file, 'w') as f:
            f.write(device_input_content)

        assert self.soa_params.lowpass_w is not None, "lowpass filter weights must not be None"	

        with open("lowpass_filter.txt", 'w') as f:
            f.write(f"{len(self.soa_params.lowpass_w)} ")
            for w in self.soa_params.lowpass_w:
                f.write(f"{w} ")
            f.write("\n")



    def initialize_device_input_file_with_mzi_filter(self):
        device_input_content = """\
IO_left__  8		            # this always needs to be in the input file
paswg_L__  1   {wg}             # passive waveguide on the left
soa_1____  2   {seg}	0       # 100 segm = amplifier 800 micron approx (100fs, 20wl segments)
split_L    101			        # splitter L for MZI filter
mzi_wg     1	1		        #  wg in MZI filter
split_R    102			        # splitter R for MZI filter
paswg_R__  1   {wg}             # passive waveguide on the right
IO_right_  9		            # this always needs to be in the input file
-1 -1               # end of component list - start of connections list
IO_left__  R0 paswg_L__  L0    #
paswg_L__  R0 soa_1____  L0    #
soa_1____  R0 split_L    L0    #
split_L    R0 split_R    L0    #
split_L    R1 mzi_wg     L0    #
mzi_wg     R0 split_R    L1    #
split_R    R0 paswg_R__  L0    #
paswg_R__  R0 IO_right_  L0    #
xxx  -1  xxx -1 	# end of connections list - start of current source list
0     {cur}    # current in soa_1 in amps 0.2 amps = 13kA/cm2
-1 -1	# end of current source list
From this point on the file can contain any text.
""".format(cur = self.soa_params.soa_current, seg = self.soa_params.soa_segments, wg = self.soa_params.wg_segments)

        with open(self.sim_params.device_file, 'w') as f:
            f.write(device_input_content)

    def initialize_signal_input_file(self):
        pulse_power, pulse_phase = self.input_signal()
        phis.write_signal_from_side(self.sim_params.signal_input_file, pulse_power, pulse_phase, self.signal_params.pulse_side)


    def input_signal(self):
        pars = self.signal_params

        sim_timestep = self.sim_params.simulation_time_step()
        input_num_points = int(pars.input_size/sim_timestep)
        assert input_num_points > 0 and input_num_points < constants.MAXNR_OPT_DAT, \
            f"Too many points in the input signal! {input_num_points} exceeds {constants.MAXNR_OPT_DAT} limit."
        

        input_time_array = np.linspace(0, (input_num_points-1)*sim_timestep, input_num_points)
        if pars.pulse_type == "sech2":
            pulse_power, pulse_phase = phis.sech2_pulse_with_total_energy(time_array=input_time_array, 
                                           pulse_energy=pars.pulse_energy, pulse_FWHM=pars.pulse_FWHM, 
                                           pulse_center=pars.first_pulse, pulse_chirp=pars.pulse_chirp)
        elif pars.pulse_type == "gaussian":	
            pulse_power, pulse_phase = phis.gaussian_pulse(time_array=input_time_array, 
                                           pulse_energy=pars.pulse_energy, pulse_FWHM=pars.pulse_FWHM, 
                                           pulse_center=pars.first_pulse, pulse_chirp=pars.pulse_chirp)
        elif pars.pulse_type == "chirp":
            pulse_power, pulse_phase = phis.chirped_signal(time_array=input_time_array, 
                                            peak_power=1, #TODO
                                            start_time=pars.first_pulse,
                                            start_f=pars.chirp_min, end_f=pars.chirp_max,
                                            use_complex=pars.complex_chirp
                                            )
        else:
            raise NotImplementedError(f"unknown pulse type {pars.pulse_type}")
        
        if pars.num_pulses > 1 and pars.pulse_type != "chirp":
            assert pars.distance > pars.pulse_FWHM, f"distance {pars.distance} must be larger than pulse FWHM {pars.pulse_FWHM}"
            pulse_power, pulse_phase = phis.repeat_pulse(input_time_array, pulse_power, pulse_phase, pars.num_pulses, pars.distance)

        return pulse_power, pulse_phase
    
    def reduced_input_signal(self):
        """Input signal with only the input pulse, i.e., all "empty space" clipped off"""
        assert self.signal_params.num_pulses == 1, "can't reduce input signal when it contains repeated pulses"
        pulse_power, pulse_phase = self.input_signal()
        pulse_center_idx = int(self.signal_params.first_pulse/self.signal_params.input_size*len(pulse_power))
        pulse_delta_idx = int(self.signal_params.pulse_FWHM * PULSE_CLIP_FWHM_FACTOR / self.sim_params.simulation_time_step())
        pulse_slice = slice(pulse_center_idx - pulse_delta_idx, pulse_center_idx + pulse_delta_idx)
        return pulse_power[pulse_slice], pulse_phase[pulse_slice]


def plot_output_power_for_various_soa_currents_with_single_pulse():
    """Plot the power output for various laser powers."""
    sim_params = phip.PHIsim_params_InGaAsP_ridge.copy(
        n_wavelen_segment = 5,
        nr_cycles         = 20000,

        # the following almost completely turns off the gaussian noise in the SOA
        # (a value of zero causes numerical issues in the simulation code)
        spontaneous_emission_coupling_amp = 1e-12, 
        # turn off kerr effect of SOA
        #n2_index_amp = 0.0,

        # video_N           = 10,
        # video_start       = 3000
    )

    soa_current = np.linspace(0.01, 0.2, 8)
    soa_segments = 100
    soa_len_text = f"{fmt_eng(sim_params.simulation_segment_length(soa_segments))}m"
    signal = SignalInputParams("sech2", 0.001e-12, 0.3e-12, 0.0, "left")

    configs = [TestSOASetup(sim_params, SoaParams(soa_current=soa_cur, soa_segments=soa_segments), signal, key=i) 
               for i, soa_cur in enumerate(soa_current)] 
    results = phid.PHIsim_run_concurrent(configs, work_folder, executables)

    # plot output power and spectrum
    fig_signal, axs_signal = plt.subplots(4, 2, constrained_layout=True)
    fig_signal.suptitle(f"Single SOA [{soa_len_text}] Output optical pulse")	
    fig_spectrum, axs_spectrum = plt.subplots(4, 2, constrained_layout=True)
    fig_spectrum.suptitle(f"Single SOA [{soa_len_text}] Output spectrum")

    time_out = None 
    for (cfg, result) in results.items():
        assert isinstance(cfg, TestSOASetup)

        data = result.P_LR_out
        if time_out is None: # only need to do this once, all output should have same length
            time_out = np.linspace(0, sim_params.simulation_time_step() * (len(data) - 1), len(data))

        # find the peak in the output and reduce the output to only the output pulse
        peak_idx = np.argmax(data)
        delta_idx = int(phisig.fwhm(range(0, len(data)), data) * PULSE_CLIP_FWHM_FACTOR)
        time_slice = slice(peak_idx-delta_idx, peak_idx+delta_idx)

        row = cfg.key // 2
        col = cfg.key % 2
        ax = axs_signal[row, col]
        ax.set_title(f"SOA current = {fmt_eng(cfg.soa_params.soa_current)}A")

        if (col == 0):
            ax.set_ylabel("Power (W)")
        if (row == 3):
            ax.set_xlabel("time (ps)")

        ax.plot(time_out[time_slice]*1e12, data[time_slice])
        # instantiate a second axes that shares the same x-axis for drawing phase
        axp = ax.twinx() 
        axp.set_ylabel("Phase (rad)", color="r")
        axp.plot(time_out[time_slice]*1e12, result.F_LR_out[time_slice], 'r')
        axp.tick_params(axis='y', labelcolor='r')
        
        # plot spectrum in separate window
        ax2 = axs_spectrum[row, col]
        ax2.set_title(f"soa current = {fmt_eng(cfg.soa_params.soa_current)}A")

        input_P, input_F = cfg.reduced_input_signal()
        freq, psd_LR = phisig.calculate_spectrum_input(input_P, input_F, sim_params.simulation_time_step())
        freq2, psd_out = phisig.calculate_spectrum_input(result.P_LR_out[time_slice], result.F_LR_out[time_slice], sim_params.simulation_time_step())
             
        freq, psd_LR = phisig.fftshift_if_needed(freq, psd_LR)
        freq2, psd_out = phisig.fftshift_if_needed(freq2, psd_out)
        freq_gain, gain = phisig.calculate_spectral_gain(freq, psd_LR, freq2, psd_out)

        ax2.set_ylabel('signal or gain (dB)')
        ax2.set_title(f'SOA current = {fmt_eng(cfg.soa_params.soa_current)}A')
        ax2.set_xlabel('Frequency (GHz)')
        ax2.plot(freq/1e9,      10*np.log10(psd_LR),   label='input')	
        ax2.plot(freq2/1e9,     10*np.log10(psd_out),  label='output')	
        ax2.plot(freq_gain/1e9, 10*np.log10(gain),'r', label='gain')
        if row == 0 and col == 0:
            ax2.legend()


def plot_soa_gain_for_various_soa_currents_with_single_pulse():
    sim_params = phip.PHIsim_params_InGaAsP_ridge.copy(
        n_wavelen_segment = 5,
        nr_cycles         = 20000,

        # the following almost completely turns off the gaussian noise in the SOA
        # (a value of zero causes numerical issues in the simulation code)
        spontaneous_emission_coupling_amp = 1e-12, 

        # turn off waveguide loss
        other_loss_pwg = 0.001,

        # video_N           = 10,
        # video_start       = 3000
    )

    soa_current = np.linspace(0.00, 0.10, 24)
    soa_len = 500e-6 

    soa_segments = sim_params.length_to_num_segments(soa_len)
    soa_len = soa_segments * sim_params.simulation_segment_length() # update length to fix any rounding errors

    soa_len_text = f"{fmt_eng(soa_len)}m".replace('μ', '\\textmu ')	
    signal = SignalInputParams("sech2", 1e-15, 4e-12, 0.0, "left") # very small power 

    configs = [TestSOASetup(sim_params, SoaParams(soa_current=soa_cur, soa_segments=soa_segments), signal, key=i) 
               for i, soa_cur in enumerate(soa_current)] 
    results = phid.PHIsim_run_concurrent(configs, work_folder, executables, skip_simulation=False)

    gain = np.zeros(len(soa_current))

    time_out = None 
    for (cfg, result) in results.items():
        assert isinstance(cfg, TestSOASetup)

        data = result.P_LR_out
        if time_out is None: # only need to do this once, all output should have same length
            time_out = np.linspace(0, sim_params.simulation_time_step() * (len(data) - 1), len(data))

        # find the peak in the output and reduce the output to only the output pulse
        peak_idx = np.argmax(data)
        delta_idx = int(phisig.fwhm(range(0, len(data)), data) * PULSE_CLIP_FWHM_FACTOR)
        time_slice = slice(peak_idx-delta_idx, peak_idx+delta_idx)

        energy_out = np.sum(data[time_slice]) * sim_params.simulation_time_step()

        gain[cfg.key] = energy_out / cfg.signal_params.pulse_energy
    
    gain_per_m = np.log(gain) / soa_len

    plt.figure()
    #plt.suptitle(f"small signal SOA gain (test with L$_{{SOA}}$={soa_len_text})")
    plt.plot(soa_current * 1e3, gain_per_m)
    plt.xlabel("I$_{SOA}$ (mA)")
    plt.ylabel("Gain (1/m)")



def plot_output_power_for_various_soa_currents_with_pulse_train():
    """Plot the power output for various laser powers."""
    sim_params = phip.PHIsim_params_InGaAsP_ridge.copy(
        n_wavelen_segment = 5,
        nr_cycles         = 20000,

        ## we turn off a few effects of the SOA to focus on the linear frequency-dependent gain

        # the following almost completely turns off the gaussian noise in the SOA
        # (a value of zero causes numerical issues in the simulation code)
        spontaneous_emission_coupling_amp = 1e-20, 
        # turn off gain saturation
        epsilon1_amp = 0.0,
        epsilon2_amp = 0.0,
        two_photon_absorption_amp = 0.0,
        # turn off kerr effect of SOA
        n2_index_amp = 0.0,

        # video_N           = 10,
        # video_start       = 3000
    )

    soa_current = np.linspace(0.01, 0.2, 8)
    soa_segments = 100
    soa_len_text = f"{fmt_eng(sim_params.simulation_segment_length(soa_segments))}m"
    signal = SignalInputParams("sech2", 0.01e-12, 1e-12, 0.0, "left", first_pulse=200e-12, input_size=500e-12, num_pulses=int(300/20), distance=20e-12)

    configs = [TestSOASetup(sim_params, SoaParams(soa_current=soa_cur, soa_segments=soa_segments), signal, key=i) 
               for i, soa_cur in enumerate(soa_current)] 
    results = phid.PHIsim_run_concurrent(configs, work_folder, executables)

    # plot output power and spectrum
    fig_signal, axs_signal = plt.subplots(4, 2, constrained_layout=True)
    fig_signal.suptitle(f"Single SOA [{soa_len_text}] Output optical pulse")	
    fig_spectrum, axs_spectrum = plt.subplots(4, 2, constrained_layout=True)
    fig_spectrum.suptitle(f"Single SOA [{soa_len_text}] Output spectrum")

    time_out = None 
    for (cfg, result) in results.items():
        assert isinstance(cfg, TestSOASetup)

        data = result.P_LR_out
        if time_out is None: # only need to do this once, all output should have same length
            time_out = np.linspace(0, sim_params.simulation_time_step() * (len(data) - 1), len(data))

        time_slice = slice(int(len(data)*2/5), -1)

        row = cfg.key // 2
        col = cfg.key % 2
        ax = axs_signal[row, col]
        ax.set_title(f"SOA current = {fmt_eng(cfg.soa_params.soa_current)}A")

        if (col == 0):
            ax.set_ylabel("Power (W)")
        if (row == 3):
            ax.set_xlabel("time (ps)")

        ax.plot(time_out[time_slice]*1e12, data[time_slice])
        # instantiate a second axes that shares the same x-axis for drawing phase
        axp = ax.twinx() 
        axp.set_ylabel("Phase (rad)", color="r")
        axp.plot(time_out[time_slice]*1e12, result.F_LR_out[time_slice], 'r')
        axp.tick_params(axis='y', labelcolor='r')
        
        # plot spectrum in separate window
        ax2 = axs_spectrum[row, col]
        ax2.set_title(f"soa current = {fmt_eng(cfg.soa_params.soa_current)}A")

        input_P, input_F = cfg.input_signal()
        freq, psd_LR = phisig.calculate_spectrum_input(input_P, input_F, sim_params.simulation_time_step())
        freq2, psd_out = phisig.calculate_spectrum_input(result.P_LR_out[time_slice], result.F_LR_out[time_slice], sim_params.simulation_time_step())
             
        freq, psd_LR = phisig.fftshift_if_needed(freq, psd_LR)
        freq2, psd_out = phisig.fftshift_if_needed(freq2, psd_out)
        freq_gain, gain = phisig.calculate_spectral_gain(freq, psd_LR, freq2, psd_out)

        ax2.set_ylabel('signal or gain (dB)')
        ax2.set_title(f'SOA current = {fmt_eng(cfg.soa_params.soa_current)}A')
        ax2.set_xlabel('Frequency (GHz)')
        ax2.plot(freq/1e9,      10*np.log10(psd_LR),   label='input')	
        ax2.plot(freq2/1e9,     10*np.log10(psd_out),  label='output')	
        ax2.plot(freq_gain/1e9, 10*np.log10(gain),'r', label='gain')
        if row == 0 and col == 0:
            ax2.legend()


def plot_output_power_for_various_soa_currents_with_chirped_input():
    """Plot the power output for various laser powers."""
    sim_params = phip.PHIsim_params_InGaAsP_ridge.copy(
        n_wavelen_segment = 5,
        #nr_cycles         = 20000,

        ## we turn off a few effects of the SOA to focus on the linear frequency-dependent gain

        # the following almost completely turns off the gaussian noise in the SOA
        # (a value of zero causes numerical issues in the simulation code)
        spontaneous_emission_coupling_amp = 1e-20, 
        # turn off gain saturation
        epsilon1_amp = 0.0,
        epsilon2_amp = 0.0,
        two_photon_absorption_amp = 0.0,
        # turn off kerr effect of SOA
        n2_index_amp = 0.0,

        # video_N           = 10,
        # video_start       = 3000
    )

    soa_current = np.linspace(0.01, 0.2, 8)
    soa_params = SoaParams(soa_current=0, soa_segments=100, lowpass_w=Judo1A_Weights)
    signal = SignalInputParams("chirp", first_pulse=400e-12, input_size=1.2e-9, complex_chirp=True, chirp_min=1e9, chirp_max=20e12)

    # set nr_cycles to be sufficient to cover the whole chirp
    time_step = sim_params.simulation_time_step()
    sim_params.nr_cycles = signal.input_size/sim_params.simulation_time_step() + soa_params.total_segments() + 1

    configs = []
    for i, soa_cur in enumerate(soa_current):
        for j in ("mzi", "lowpass"):
            test_config = TestSOASetup(sim_params, 
                                       replace(soa_params, soa_current=soa_cur, use_lowpass=(j=="lowpass"), use_mzi=(j=="mzi")), 
                                       signal, key=(i, j))
            configs.append(test_config)

    results = phid.PHIsim_run_concurrent(configs, work_folder, executables)

    # plot output power and spectrum
    soa_len_text = f"{fmt_eng(sim_params.simulation_segment_length(soa_params.soa_segments))}m"
    fig_signal, axs_signal = plt.subplots(4, 2, constrained_layout=True)
    fig_signal.suptitle(f"Single SOA [{soa_len_text}] Output [input = chirp]")	

    time_out = None 
    for (cfg, result) in results.items():
        assert isinstance(cfg, TestSOASetup)

        data = result.P_LR_out
        if time_out is None: # only need to do this once, all output should have same length
            time_out = np.linspace(0, time_step * (len(data) - 1), len(data))

        # the time it takes for the signal to pass through the system, in indexes of the output array
        start_index = int(signal.first_pulse/time_step) + cfg.soa_params.total_segments() + 1

        local_chirp = np.linspace(signal.chirp_min, signal.chirp_max, len(time_out[start_index:]))

        row = cfg.key[0] // 2
        col = cfg.key[0] % 2
        ax = axs_signal[row, col]
        ax.set_title(f"SOA current = {fmt_eng(cfg.soa_params.soa_current)}A")

        if (col == 0):
            ax.set_ylabel("Power (W)")
        if (row == 3):
            ax.set_xlabel("Frequency (THz)")

        # leave a bit of space, the first points in the response contain a bit of transient, so we ignore them
        transient_offset = 200 
        if cfg.key[1] == "mzi":
            color = 'b'
            label = "mzi"
        else:
            color = 'r'
            label = f"lowpass"

        ax.plot(local_chirp[transient_offset:]/1e12, data[start_index+transient_offset:], color, label=label)

    ax.legend()
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Power after SOA (W)")


def plot_soa_gain_saturation(plot_curve_fit=False):
    # run a number of tests with increasing pulse power to determine gain saturation
    sim_params = phip.PHIsim_params_InGaAsP_ridge.copy(
        n_wavelen_segment = 8,
        nr_cycles         = 20000,

        # the following almost completely turns off the gaussian noise in the SOA
        # (a value of zero causes numerical issues in the simulation code)
        #spontaneous_emission_coupling_amp = 1e-12, 

        # turn off waveguide loss
        other_loss_pwg = 0.001,

        # video_N           = 10,
        # video_start       = 3000
    )

    soa_current = 0.2 # max
    soa_len = 1e-3 

    soa_segments = sim_params.length_to_num_segments(soa_len)
    soa_len = soa_segments * sim_params.simulation_segment_length() # update length to fix any rounding errors

    signal = SignalInputParams("sech2", 0.01e-12, 5e-12, 0.0, "left")

    pulse_energies = np.linspace(10e-15, 1e-12, 50)
    pulse_fwhms = [1e-12, 5e-12]

    configs = []
    for i, fwhm in enumerate(pulse_fwhms):
        for j, pulse_energy in enumerate(pulse_energies):
            signal_ij = replace(signal, pulse_energy=pulse_energy, pulse_FWHM=fwhm)
            test_config = TestSOASetup(sim_params, SoaParams(soa_current=soa_current, soa_segments=soa_segments), 
                                    signal_ij, key=(i, j),
                                    work_folder=f"gain_saturation_{fmt_eng(soa_current)}A_{fmt_eng(pulse_energy)}J_{fmt_eng(fwhm)}s")
            configs.append(test_config)

    results = phid.PHIsim_run_concurrent(configs, work_folder, executables, skip_simulation=False)

    gain = np.zeros(shape=(len(pulse_fwhms), len(pulse_energies)))

    time_out = None 
    for (cfg, result) in results.items():
        assert isinstance(cfg, TestSOASetup)

        data = result.P_LR_out
        if time_out is None: # only need to do this once, all output should have same length
            time_out = np.linspace(0, sim_params.simulation_time_step() * (len(data) - 1), len(data))

        # find the peak in the output and reduce the output to only the output pulse
        peak_idx = np.argmax(data)
        delta_idx = int(phisig.fwhm(range(0, len(data)), data) * PULSE_CLIP_FWHM_FACTOR)
        time_slice = slice(peak_idx-delta_idx, peak_idx+delta_idx)

        energy_out = np.sum(data[time_slice]) * sim_params.simulation_time_step()

        gain[cfg.key] = energy_out / cfg.signal_params.pulse_energy

    gain_dB = 10*np.log10(gain)

    def gain_fit(x, g_ss, E_sat):
        return g_ss / (1 + x/E_sat)
    
    g_ss = np.zeros(len(pulse_fwhms))
    E_sat = np.zeros(len(pulse_fwhms))

    for i, fwhm in enumerate(pulse_fwhms):
        (g_ss[i], E_sat[i]), pcov = scipy.optimize.curve_fit(gain_fit, pulse_energies, gain[i], 
                                                             p0=[gain[i, 0], 500e-15])
        
    if not plot_curve_fit:
        print("approximate g_ss, E_sat for each pulse fwhm: ")
        for (i, fwhm) in enumerate(pulse_fwhms):
            print(f"  {fwhm*1e12}ps: {10*np.log10(g_ss[i]):.1f}dB, {E_sat[i]*1e15:.0f}fJ")

    plt.figure()
    for i, fwhm in enumerate(pulse_fwhms):
        plt.plot(pulse_energies*1e15, 10*np.log10(gain[i]), label=f"FWHM = {fwhm*1e12}\\,ps")	
        if plot_curve_fit:
            plt.plot(pulse_energies*1e15, 10*np.log10(gain_fit(pulse_energies, g_ss[i], E_sat[i])), '--', label=f"Fit $g_{{ss}}$ = {g_ss[i]:.1f}\\,dB, $E_{{sat}}$ = {E_sat[i]*1e15:.0f}\\,fJ")
    plt.xlabel("pulse energy (fJ)")
    plt.ylabel("Gain (dB)")
    plt.legend()



def plot_filter_response_with_chirped_input():
    """Plot the power output for various laser powers."""
    sim_params = phip.PHIsim_params_InGaAsP_ridge.copy(
        n_wavelen_segment = 8,

        other_loss_pwg = 0.001, # turn off waveguide loss, so we only measure the filter response
        #nr_cycles         = 20000,
        # video_N           = 10,
        # video_start       = 3000
    )

    fs = 1/sim_params.simulation_time_step()

    # turn off the SOA, just measure the filter response
    soa_params = SoaParams(soa_current=0, soa_segments=0, lowpass_w=Judo1A_Weights)
    signal = SignalInputParams("chirp", first_pulse=400e-12, input_size=1.2e-9, complex_chirp=True, chirp_min=1e9, chirp_max=fs/2)

    # set nr_cycles to be sufficient to cover the whole chirp
    time_step = sim_params.simulation_time_step()
    sim_params.nr_cycles = signal.input_size/sim_params.simulation_time_step() + soa_params.total_segments() + 1

    configs = []
    #for (name, weights) in chain((("MZI", None),), All_Weights.items()):
    for (name, weights) in (All_Weights.items()):
        test_config = TestSOASetup(sim_params, replace(soa_params, 
                                                       use_lowpass=(name != "MZI"), use_mzi=(name == "MZI"),
                                                       lowpass_w=weights), 
                                    signal, key=name, work_folder=f"transmission_with_filter_{name}")	
        configs.append(test_config)

    results = phid.PHIsim_run_concurrent(configs, work_folder, executables)

    fig, ax = plt.subplots(constrained_layout=True)
    #fig.suptitle(f"Filter Output [input = chirp]")	

    markercolor = ("black", "olive")
    time_out = None 
    for (i, (cfg, result)) in enumerate(results.items()):
        assert isinstance(cfg, TestSOASetup)

        data = result.P_LR_out
        if time_out is None: # only need to do this once, all output should have same length
            time_out = np.linspace(0, time_step * (len(data) - 1), len(data))

        # the time it takes for the signal to pass through the system, in indexes of the output array
        start_index = int(signal.first_pulse/time_step) + cfg.soa_params.total_segments() + 1

        local_chirp = np.linspace(signal.chirp_min, signal.chirp_max, len(time_out[start_index:]))

        # leave a bit of space for the transient response to the start of the signal 
        # (it's a low-pass filter, so the start of the signal will be a bit leveled out)
        transient_offset = 10

        label = f"FIR {i+1} (PHIsim)" if cfg.key != "MZI" else f"MZI"
        ax.plot(local_chirp[transient_offset:]/1e12, data[start_index+transient_offset:], label=label)

        if cfg.key != "MZI":
            w, h = scipy.signal.freqz(b=All_Weights[cfg.key], worN=30)
            delta_f = w*fs/(2*np.pi)
            ax.plot(delta_f/1e12, np.abs(h)**2, linestyle="", marker='v', label=f"FIR {i+1} (ref)", color=markercolor[i])

    ax.legend()
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Transmission")


if __name__ == '__main__':
    plt.ion() # set plots to non-blocking
    #plot_latex_style(plt, 12.0)

    print(f"Running single SOA simulations in [work folder {work_folder}]")
    #phis.plot_gaussian_pulse(np.linspace(0, 200e-12, 1000), 10e-15, 50e-12, 100e-12, 0, plot_phase=False)
    #phis.plot_gaussian_pulse(np.linspace(0, 200e-12, 10000), 100e-12, 1e-12, 5e-12, 1, plot_phase=True, num_pulses=8, pulse_distance=20e-12)
    #phis.plot_chirped_signal(np.linspace(0, 500e-12, 10000), 0.1, 200e-12, 1e9, 100e9, plot_phase=True, complex_chirp=True)

    #plot_output_power_for_various_soa_currents_with_single_pulse()
    #plot_soa_gain_for_various_soa_currents_with_single_pulse()
    #plot_soa_gain_saturation(plot_curve_fit=True)
    #plot_output_power_for_various_soa_currents_with_pulse_train()
    #plot_output_power_for_various_soa_currents_with_chirped_input()
    plot_filter_response_with_chirped_input()

    plt.show(block=True) # final command to hold plots (otherwise they close if the script is done)




