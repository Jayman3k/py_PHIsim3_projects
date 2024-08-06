# Introduction

This project contains the code for the parallel simulation engine I developed in the context of my master's thesis. 
Its purpose is to speed up simulation work using PHIsim (https://sites.google.com/tue.nl/phisim/home). Note that this framework was designed specifically for PHIsimV3, and might require updates to work with different versions.

We achieve the parallelization by defining simulation objects that contain all necessary information for running a PHIsim instance; then we spin out these instances to separate processes.
The framework also allows you to define hooks to collect information from a simulation, so some of the data processing can also be parallelized. 

The framework was tested on computers with the Windows 10 and Windows 11 operating systems.

## Getting started

In a first step, before you start using this framework, it's best to have a working PHIsim simulation with (at least) one variation of the setup you want to test. In other words, you manually design your `device_input` file such that you have a device file that you know will work. The format of the `device_input` file is easy to understand, so it should be straightforward to create the `device_file` by hand. Then, you can use the python scripts published on the PHIsim website to run your simulation and debug any potential issues. If this is your first rodeo with PHIsim, this will allow you to familiarize yourself with the concepts used by the simulation.

## Using this framework

As a reminder, PHIsim actually consists out of 2 executables: 

- `PHsim_input.exe` which converts a textual description (which I called the `device_input` file earlier on) of your structure into a detailed device file
- `PHIsim.exe` which takes in the detailed device, along with a simulation parameters file and (optionally) an input signal file, and runs the actual simulation. After a simulation run, this executable writes 3 files: the optical output signals, and a carrier and photon density file. The former is generally what we're interested in, the latter 2 files can be used to continue the simulation with another run.

All of this is summarized in the following image (courtesy of prof. Erwin Bente):

<img src="doc/PHIsim_workflow.png" alt="Phisim workflow" width="400"/>

The parallelization framework aims to automate these tasks. There are a lot of degrees of freedom here, so there are quite a few things to set up before we have a fully contained simulation object. 

### 1. Creating a flexible `device_file`

Starting from your working setup, create a function to write a parameterized `device_file`. Let's say that, for example, you have a simple setup with some waveguides and an SOA. The file would look like this:

```
IO_left__  8		            # this always needs to be in the input file
paswg_L__  1   200              # passive waveguide on the left
soa_1____  2   100	  0         # 100 segm = amplifier 800 micron approx (100fs, 20wl segments)
paswg_R__  1   200              # passive waveguide on the right
IO_right_  9		            # this always needs to be in the input file
-1 -1               # end of component list - start of connections list
IO_left__  R0 paswg_L__  L0    #
paswg_L__  R0 soa_1____  L0    #
soa_1____  R0 paswg_R__  L0    #
paswg_R__  R0 IO_right_  L0    #
xxx  -1  xxx -1 	# end of connections list - start of current source list
0     0.1    # current in soa_1 in amps 0.2 amps = 13kA/cm2
-1 -1	# end of current source list
From this point on the file can contain any text.
```

Let's say we want to have a parameterized waveguide and SOA length, and driving current. The corresponding python function would then become (for example):

```python
    def initialize_device_input_file(wg_segments, soa_segments, soa_current, device_filename):
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
""".format(cur = soa_current, seg = soa_segments, wg = wg_segments)

        with open(device_filename, 'w') as f:
            f.write(device_input_content)
```

You see we put the device code in a string and replaced a few constants with variables (to have a variable device file) using `format` to fill those in. Then, we open a file and write the content there.

### 2. Defining the parameters of the simulation

The simulation has a lot of input parameters, which, broadly speaking, fall into two categories:
- The runtime parameters: how many cycles to run the simulation, the size of a segment, etc.
- The physical parameters: material gain, losses, all sorts of coefficients to describe physical process, ...
  
Most of these parameters are contained within the `parameter_input_file` shown in the image. Since you may want to test some of these parameters in a simulation sweep, the parallel framework has some facilities to define them and write the parameter file. This is defined in the `PHIsim_sim_params.py` file, which also contains a variable `PHIsim_params_InGaAsP_ridge` with some useful physical constants, relevant for an InP ridge platform (YMMV). An example on how to use this could be:

```python 
import tools.PHIsim_sim_params as phip

# define a local copy, overwrite a few parameters
local_sim_params = phip.PHIsim_params_InGaAsP_ridge.copy(
    n_wavelen_segment = 5,
    nr_cycles         = 20000,
    # (for example) test with very low waveguide loss
    other_loss_pwg    = 0.001,
)

# create the parameter_input_file
local_sim_params.write_to_file() 
```

### 3. Define a signal input file

This may not be applicable in some cases, for example, if you're simulating a self-starting laser. In those cases, you can just leave the input signal file empty. If you do need an input signal, you will need to create a function to write that signal to a file. The framework provides some helper functions for a few common signal inputs (for example, a sech2 pulse) in the file `PHIsim_signal_input_util.py`. Some examples on how to use this can be found the example simulation file, too.

### 4. tying everything together


