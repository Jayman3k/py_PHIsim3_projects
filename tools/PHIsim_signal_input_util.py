import numpy as np
from matplotlib import pyplot as plt

from scipy.signal._waveforms import _chirp_phase

from tools.PHIsim_constants import SPEED_OF_LIGHT as c_l
from tools.fmt_utils import fmt_eng


def keep_phase_in_range(phase):
    """Return copy of phase, with phase in the range [-pi, pi]"""	
    return np.mod(phase + np.pi, 2*np.pi) - np.pi


def write_signal_input_file(filename, pulse_power_LR, pulse_phase_LR, pulse_power_RL=None, pulse_phase_RL=None):
    """Utility method to write a signal input file"""

    nr_data_in = len(pulse_power_LR)

    ## sanity checks 
    assert len(pulse_phase_LR) == nr_data_in, \
        f"pulse_power_LR and pulse_phase_LR must have same length ({nr_data_in} vs. {len(pulse_phase_LR)})"
    assert pulse_power_RL is None or len(pulse_power_RL) == nr_data_in, \
        f"pulse_power_LR and pulse_power_RL must have same length ({nr_data_in} vs. {len(pulse_power_RL)})"
    assert pulse_phase_RL is None or len(pulse_phase_RL) == nr_data_in, \
        f"pulse_power_LR and pulse_phase_LR must have same length ({nr_data_in} vs. {len(pulse_phase_RL)})"
    assert (pulse_power_RL is None) == (pulse_phase_RL is None), \
        "pulse_power_RL and pulse_phase_RL must be both None or both not None" 

    if pulse_power_RL is None:
        pulse_power_RL = np.zeros(nr_data_in)
    if pulse_phase_RL is None: 
        pulse_phase_RL = np.zeros(nr_data_in)

    with open(filename, 'w') as f_par_out:  # open the file for writing
        wr_f='{:01.9E} '
        for i in range(0, nr_data_in):
            s=wr_f.format(pulse_power_LR[i]) + wr_f.format(pulse_phase_LR[i]) + \
              wr_f.format(pulse_power_RL[i]) + wr_f.format(pulse_phase_RL[i])+'\n'
            f_par_out.write(s)


def sech2_FWHM_to_tau(t_FWHM: float) -> float:
    """convert sech^2 FWHM to tau (FWHM = 1.76 tau (for a sech^2 pulse))"""
    return t_FWHM / (2 * np.log(np.sqrt(2) + 1))


def sech2_pulse_with_peak_power(time_array, pulse_peak_power, pulse_FWHM, pulse_center, pulse_chirp) -> tuple[np.array, np.array]:
    """create generic sech2 pulse with desired peak power
    time_array should contain an array of timestamps, with constant time-step
    """
    time_step = time_array[1] - time_array[0] 
    if pulse_FWHM < 10 * time_step:
        print(f"WARNING: pulse_FWHM [{pulse_FWHM:.2e}s] is small compared to timestep [{time_step:.2e}s] of the simulation, " +
               "pulse peak will not contain enough points for a good result")

    pulse_phase = np.zeros(len(time_array))
    pulse_tau  = sech2_FWHM_to_tau(pulse_FWHM)    
     # add small power to avoid filling in zeroes (?)
    pulse_power = np.zeros(len(time_array)) # 1e-12 * pulse_peak_power * np.abs(np.random.normal(size=len(time_array))) 
    no_phase_fwhm_factor = 4
    #zero_phase = -pulse_chirp * (8 * pulse_FWHM / pulse_tau)**2 # assume everything past 5*FWHM has zero phase

    for i in range(0,len(pulse_power)):
        t = time_array[i] - pulse_center # time shifted for correct pulse position

        # outside these bounds, the result is approximately zero, and this limit prevents a cosh overflow
        if t > -4*pulse_FWHM and t < 4*pulse_FWHM: 
            pulse_power[i] += pulse_peak_power * (1/np.cosh(t / pulse_tau))**2

        if (t > (-no_phase_fwhm_factor * pulse_FWHM)) and (t < (no_phase_fwhm_factor * pulse_FWHM)):
            pulse_phase[i] = -pulse_chirp * (t / pulse_tau)**2 
        else:
            pulse_phase[i] = -pulse_chirp * (no_phase_fwhm_factor * pulse_FWHM / pulse_tau)**2 # constant phase outside the pulse

        # add small number to avoid filling in zeroes, prevents (unphysical) sharp phase transition
        # pulse_phase[i] += np.random.normal() * 1e-15 

    pulse_phase = keep_phase_in_range(pulse_phase)

    return pulse_power, pulse_phase


def sech2_pulse_with_total_energy(time_array, pulse_energy, pulse_FWHM, pulse_center, pulse_chirp) -> tuple[np.array, np.array]:
    """
    create generic sech2 pulse and just scale the data to match desired pulse energy 
    (scaling a pulse does not change its FWHM)
    """
    time_step = time_array[1] - time_array[0] 
    pulse_power, pulse_phase = sech2_pulse_with_peak_power(time_array, 10, pulse_FWHM, pulse_center, pulse_chirp)
    total_energy = np.sum(pulse_power) * time_step
    pulse_power *= (pulse_energy / total_energy)
    return pulse_power, pulse_phase


def plot_sech2_pulse(time_array, pulse_energy, pulse_FWHM, pulse_center, pulse_chirp, 
                     plot_phase=False, plot_chirp_lambda=0, plot_chirp_omega=False, num_pulses=1, pulse_distance=0):
    """Utility method to check the shape of a pulse"""
    pulse_power, pulse_phase = sech2_pulse_with_total_energy(time_array, pulse_energy, pulse_FWHM, pulse_center, pulse_chirp)

    if (num_pulses > 1) and (pulse_distance > 0):
        pulse_power, pulse_phase = repeat_pulse(time_array, pulse_power, pulse_phase, num_pulses, pulse_distance)

    if plot_chirp_omega or plot_chirp_lambda != 0:
        pulse_local_chirp = np.zeros(len(time_array))
        pulse_tau  = sech2_FWHM_to_tau(pulse_FWHM) 
        for i in range(0,len(pulse_local_chirp)):
            t = time_array[i] - pulse_center 
            if t > -3*pulse_FWHM and t < 3*pulse_FWHM:
                pulse_local_chirp[i] = 2 * pulse_chirp * t /(pulse_tau)**2  # local chirp = -d(phi)/dt
            
    fig, ax1 = plt.subplots()
    #plt.suptitle(f"Input sech2 pulse [energy {pulse_energy*1e12:.0f} pJ ; FWHM={pulse_FWHM*1e12:.1f}ps ; chirp {pulse_chirp}]")

    ax1.set_ylabel("Power (W)" )
    ax1.set_xlabel("Time (ps)" )
    ax1.plot(time_array*1e12, pulse_power)
    #ax1.set_xlim([pulse_center*1e12 - 4, pulse_center*1e12 + 4])
    ax1.set_ylim(0)

    if plot_phase:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel("Phase (rad)", color="r")
        ax2.plot(time_array*1e12, pulse_phase, 'r')
        ax2.tick_params(axis='y', labelcolor='r')

    if plot_chirp_lambda != 0:
        ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax3.set_ylabel("Chirp $\\delta\\lambda$(nm)", color="r")
        ax3.plot(time_array*1e12, -(plot_chirp_lambda)**2 * pulse_local_chirp / (2 * np.pi * c_l) * 1e9 , 'r') # convert to wavelength in nm
        ax3.tick_params(axis='y', labelcolor='r')
    elif plot_chirp_omega:
        ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax3.set_ylabel("Chirp $\\delta\\omega$ (rad/s)", color="r")
        ax3.plot(time_array*1e12, pulse_local_chirp , 'r')
        ax3.tick_params(axis='y', labelcolor='r')
        
    plt.show()


def gaussian(t, tau, t0):
    return np.exp(-0.5*((t-t0)/tau)**2)


def dgaussian(t, tau, t0):
    """derivative of the gaussian function: d(gaussian(t, tau, t0))/dt, useful for plotting chirp"""	
    return - (t-t0) * np.exp(-0.5*((t-t0)/tau)**2)/(tau**2)


def gaussian_FWHM_to_tau(t_FWHM) -> float:
    return t_FWHM / (2 * np.sqrt(2*np.log(2)))


def gaussian_pulse(time_array: np.array, pulse_energy, pulse_FWHM, pulse_center, pulse_chirp) -> tuple[np.array, np.array]:
    """create generic gaussian pulse with desired peak power
    time_array should contain an array of timestamps, with constant time-step
    """
    time_step = time_array[1] - time_array[0] 
    if pulse_FWHM < 10 * time_step:
        print(f"WARNING: pulse_FWHM [{pulse_FWHM:.2e}s] is small compared to timestep [{time_step:.2e}s] of the simulation, " +
               "pulse peak will not contain enough points for a good result")

    pulse_tau   = gaussian_FWHM_to_tau(pulse_FWHM) 
    pulse_power = gaussian(time_array, pulse_tau, pulse_center) # + np.random.normal(size=len(time_array)) * 1e-15 # addd small number to avoid filling in zeroes
   
    pulse_phase = pulse_chirp * gaussian(time_array, pulse_tau, pulse_center) # + np.random.normal(size=len(time_array)) * 1e-15
    pulse_phase = keep_phase_in_range(pulse_phase)

    # rescale to desired output power
    tot_energy = np.sum(pulse_power) * time_step
    pulse_power *= (pulse_energy / tot_energy)

    return pulse_power, pulse_phase


def plot_gaussian_pulse(time_array, pulse_energy, pulse_FWHM, pulse_center, pulse_chirp, 
                        plot_phase=False, plot_chirp_lambda=0, plot_chirp_omega=False, num_pulses=1, pulse_distance=0):
    """Utility method to check the shape of a pulse"""
    pulse_power, pulse_phase = gaussian_pulse(time_array, pulse_energy, pulse_FWHM, pulse_center, pulse_chirp)

    if (num_pulses > 1) and (pulse_distance > 0):
        pulse_power, pulse_phase = repeat_pulse(time_array, pulse_power, pulse_phase, num_pulses, pulse_distance)

    if plot_chirp_lambda != 0 or plot_chirp_omega:
        # chirp = -d(phi)/dt
        pulse_local_chirp = -pulse_chirp*dgaussian(time_array, gaussian_FWHM_to_tau(pulse_FWHM), pulse_center)
            
    fig, ax1 = plt.subplots()
    #plt.suptitle(f"Input gaussian pulse [energy {fmt_eng(pulse_energy)}J ; FWHM={pulse_FWHM*1e12:.1f}ps ; chirp {pulse_chirp}]")

    ax1.set_ylabel("Power (W)" )
    ax1.set_xlabel("Time (ps)" )
    ax1.plot(time_array*1e12, pulse_power)
    #ax1.set_xlim([pulse_center*1e12 - 4, pulse_center*1e12 + 4])
    ax1.set_ylim(0)

    if plot_phase:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel("Phase (rad)", color="r")
        ax2.plot(time_array*1e12, pulse_phase, 'r')
        ax2.tick_params(axis='y', labelcolor='r')

    if plot_chirp_lambda != 0:
        ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax3.set_ylabel("Chirp $\\delta\\lambda$(nm)", color="r")
        ax3.plot(time_array*1e12, -(plot_chirp_lambda)**2 * pulse_local_chirp / (2 * np.pi * c_l) * 1e9 , 'r') # convert to wavelength in nm
        ax3.tick_params(axis='y', labelcolor='r')
    elif plot_chirp_omega:
        ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax3.set_ylabel("Chirp $\\delta\\omega$ (rad/s)", color="r")
        ax3.plot(time_array*1e12, pulse_local_chirp , 'r')
        ax3.tick_params(axis='y', labelcolor='r')
        
    plt.show()


def repeat_pulse(time_array, pulse_power, pulse_phase, times : int, distance : float):
    """Utility method to repeat a pulse a given number of times. 
    'times' includes the original pulse, for example, times=3 will result in a signal with 3 pulses of the same shape.

    It's assumed that points in time_array are equally spaced.
    If distance is not a multiple of the time step, small rounding errors will be introduced, 
    and the resulting pulses will not be shifted exactly by the given distance."""

    time_step = time_array[1] - time_array[0]

    # to sum pulses, we convert to electric field, sum the shifted electric fields and then convert back
    base_field = np.sqrt(pulse_power) * np.exp(1j * pulse_phase)
    pulse_field = np.copy(base_field)
    for i in range(1, times):
         # re-shift the base field on each iteration to reduce the rounding error in index_shift
        index_shift = int(i * distance / time_step) 
        pulse_field += np.roll(base_field, index_shift)

    pulse_power = np.abs(pulse_field) ** 2
    pulse_phase = np.angle(pulse_field)

    return pulse_power, pulse_phase


def complex_chirp(t, f0, t1, f1, method='linear', phi=0, vertex_zero=True, *, use_complex=True):
    """ workaround for the fact that scipy.signal.chirp doesn't work with complex signals
    see also https://github.com/scipy/scipy/pull/17318
    """ 
    phase = _chirp_phase(t, f0, t1, f1, method, vertex_zero)
    if use_complex:
        return np.exp(1j * (phase + phi)) 
    else:
        return np.sin(phase + phi)


def chirped_signal(time_array, peak_power, start_time, start_f, end_f, use_complex=True):
    """Utility method to create a chirped pulse
    time_array should contain an array of timestamps, with constant time-step
    """
    time_step = time_array[1] - time_array[0]
    start_idx = int(start_time / time_step)
    signal = complex_chirp(time_array[start_idx:]-start_time, start_f, time_array[-1]-start_time, end_f, use_complex=use_complex)

    if use_complex:
        signal *= np.sqrt(peak_power)
        signal_power = np.abs(signal) ** 2
        signal_phase = np.angle(signal)
    else:
        signal_power = signal**2 * peak_power
        signal_phase = np.zeros(len(signal))

    # pad with zeros up to start_idx
    signal_power = np.pad(signal_power, (start_idx, 0), 'constant')
    signal_phase = np.pad(signal_phase, (start_idx, 0), 'constant')

    return signal_power, signal_phase


def plot_chirped_signal(time_array, peak_power, start_time, start_f, end_f, plot_phase=False, complex_chirp=True):
    """Utility method to check the shape of a pulse"""
    pulse_power, pulse_phase = chirped_signal(time_array, peak_power, start_time, start_f, end_f, complex_chirp)
            
    fig, ax1 = plt.subplots()
    plt.suptitle(f"Input gaussian pulse [peak power {peak_power} W ; f0={fmt_eng(start_f)}Hz ; f1={fmt_eng(end_f)}Hz]")

    ax1.set_ylabel("Power (W)" )
    ax1.set_xlabel("Time (ps)" )
    ax1.plot(time_array*1e12, pulse_power)

    if plot_phase:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel("Phase (rad)", color="r")
        ax2.plot(time_array*1e12, pulse_phase, 'r')
        ax2.tick_params(axis='y', labelcolor='r')
        
    plt.show()

