import numpy as np
from matplotlib import pyplot as plt

from scipy import signal 
from scipy import optimize
from scipy.interpolate import UnivariateSpline

from tools.fmt_utils import fmt_eng
from tools.PHIsim_constants import SPEED_OF_LIGHT as c_l



def fftshift_if_needed(freqs, values) -> tuple:
    # if this is a centered dataset (which typically happens using fft) use fftshift to 
    # start from negative frequencies, this avoids the "ugly" line connecting the beginning 
    # and the end of the spectrum
    if freqs[0] == 0:
        freqs  = np.fft.fftshift(freqs)
        values = np.fft.fftshift(values)

    return freqs, values	


def calculate_spectrum(P_LR_out: np.array, F_LR_out: np.array, P_RL_out: np.array, F_RL_out: np.array, 
                       timestep: float, startch: int = 0, windowing=False):
    
    nr_data_out = len(P_LR_out)
    reduced_nr_data_out = nr_data_out - startch

    filter_P_LR_out = P_LR_out[startch:nr_data_out]
    filter_P_RL_out = P_RL_out[startch:nr_data_out]

    if windowing:
        # apply a filter to the fft
        filt_x = np.linspace(0, reduced_nr_data_out, reduced_nr_data_out)
        arg_f  = (filt_x - reduced_nr_data_out/2) / (0.1 * reduced_nr_data_out)
        
        filt_y =  np.exp(-arg_f * arg_f)
        filter_P_LR_out = filt_y * P_LR_out[startch:nr_data_out]
        filter_P_RL_out = filt_y * P_RL_out[startch:nr_data_out]

    # complex electric field
    c_outLR  = np.sqrt(filter_P_LR_out[:])*np.exp(F_LR_out[startch:nr_data_out]*1.0j)
    c_outRL  = np.sqrt(filter_P_RL_out[:])*np.exp(F_RL_out[startch:nr_data_out]*1.0j)

    # Fourier transform of the electric field
    fc_outLR = np.fft.fft(c_outLR) 
    fc_outRL = np.fft.fft(c_outRL)
    
    ffreq    = np.fft.fftfreq(reduced_nr_data_out, timestep) # calculate frequency grid in Hz

    # power spectral density (note the division by number of points - this is necessary due to the way numpy scales fft)
    li_outLR = np.abs(fc_outLR)**2.0 / reduced_nr_data_out  
    li_outRL = np.abs(fc_outRL)**2.0 / reduced_nr_data_out           

    return ffreq, li_outLR, li_outRL


def calculate_spectrum_input(P_LR_out: np.array, F_LR_out: np.array, timestep: float, startch: int = 0, windowing=False):
    
    nr_data_out = len(P_LR_out)
    reduced_nr_data_out = nr_data_out - startch

    filter_P_LR_out = P_LR_out[startch:nr_data_out]

    if windowing:
        # apply a filter to the fft
        filt_x = np.linspace(0, reduced_nr_data_out, reduced_nr_data_out)
        arg_f  = (filt_x - reduced_nr_data_out/2) / (0.1 * reduced_nr_data_out)
        
        filt_y =  np.exp(-arg_f * arg_f)
        filter_P_LR_out = filt_y * P_LR_out[startch:nr_data_out]

    # complex electric field
    c_outLR  = np.sqrt(filter_P_LR_out[:])*np.exp(F_LR_out[startch:nr_data_out]*1.0j)

    # Fourier transform of the electric field
    fc_outLR = np.fft.fft(c_outLR) 
    ffreq    = np.fft.fftfreq(reduced_nr_data_out, timestep) # calculate frequency grid in Hz

    # power spectral density (note the division by number of points - this is necessary due to the way numpy scales fft)
    li_outLR = np.abs(fc_outLR)**2.0 / reduced_nr_data_out  

    return ffreq, li_outLR


def plot_spectrum_LR_RL(ffreq, dB_outLR, dB_outRL, plot_name):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f'{plot_name}')	

    plot_spectrum(ffreq, dB_outLR, ax1, 'L->R')
    plot_spectrum(ffreq, dB_outRL, ax2, 'R->L')


def plot_spectrum(ffreq, psd_dB, ax=None, plot_name=""):
    if ax is None:
        _, ax = plt.subplots()

    ffreq, psd_dB = fftshift_if_needed(ffreq, psd_dB)

    ax.set_ylabel('Signal (dB)')
    ax.set_title(f'Optical Spectrum {plot_name}')
    ax.set_xlabel('Frequency (GHz)')
    ax.plot(ffreq/1e9, psd_dB,'r')


def calculate_delta_lambda(freq, psd, center_wavelength):
    # see Demongodin supplement II for the formula
    # note that here omega_0 = 0 as the spectrum is centered around f=0 due to the way the simulation is implemented

    integral_top = 0
    d_omega = (freq[1] - freq[0]) * 2*np.pi
    for (f, p) in zip(freq, psd):
        omega = 2*np.pi*f
        integral_top += omega**2 * p * d_omega

    psd_sum = np.sum(psd) * d_omega

    delta_omega_rms_sq = integral_top / psd_sum
    delta_lambda_rms = center_wavelength**2  / (2*np.pi*c_l) * np.sqrt(delta_omega_rms_sq)

    return delta_lambda_rms


def fwhm(x, y):
    """
    Modified from the code of user HYRY at stackoverflow:
    https://stackoverflow.com/questions/10582795/finding-the-full-width-half-maximum-of-a-peak/

    Determine full-with-half-maximum of a peaked set of points, x and y.

    Assumes that there is only one peak present in the datasset. The function uses a spline interpolation.
    Only works with a 'clean' signal, if the signal is noisy, you should consider using a 
    curve_fit with the expected pulse shape instead.
    """
    # TODO in hindsight, it's probably simpler to use signal.find_peaks()

    half_max = np.max(y)/2.0
    spline = UnivariateSpline(x, y-half_max, s=0)
    roots = spline.roots() # find the roots

    if len(roots) < 2:
        print("Warning: No peaks were found in the data set; likely the dataset is flat (e.g. all zeros).")
        return 0
    elif len(roots) > 2:
        print("Warning: The dataset appears to have multiple peaks, the FWHM can't be determined.")
    
    return abs(roots[1] - roots[0])


def calculate_spectral_gain(freq_in, psd_in, freq_out, psd_out, mode="linear") -> tuple:
    """
    Calculate the spectral gain between two datasets. 
    mode shoule be either "linear" or "log" 
     - linear: psd_in and psd_out should be linear scale, i.e. the gain is P_out/P_in 
     - log: psd_in and psd_out should be log scale, i.e. the gain is P_out-P_in 

    This function assumes freq_in and freq_out overlap, and we interpolate the dataset to the same frequency grid.
    """
    freq_in, psd_in = fftshift_if_needed(freq_in, psd_in)
    freq_out, psd_out = fftshift_if_needed(freq_out, psd_out)

    # calculate the min and max frequency 
    min_freq = min(min(freq_in), min(freq_out))
    max_freq = max(max(freq_in), max(freq_out))
    freqs = np.linspace(min_freq, max_freq, max(len(freq_in), len(freq_out)))

    # interpolate to the same frequency grid
    y_in  = np.interp(freqs, freq_in, psd_in)
    y_out = np.interp(freqs, freq_out, psd_out)

    if mode == "linear":
        gain = y_out / y_in
    elif mode == "log":
        gain = y_out - y_in
    else:
        raise ValueError("mode should be either 'linear' or 'log'")

    return freqs, gain


def calculate_sampled_fft(power, phase, time_step:float, sampling_multiplier:int=10):
    """Avoid calculating the fft for the entire signal at once; due to the noise generated by the simulation,
    This will not give good results. It's better to average over several windows, even though that reduces
    the resoltion of the result. For the laser simulation, we use a 10x oversampling factor, which seems to
    generate a reasonable result. 

    Note that the resulting frequency resolution is approximately sampling_multiplier/time_step 
    (approximately, because we adjust the points_per_fft to be a power of 2).

    returns a tuple (frequencies, psd), where psd is the power spectral density in linear scale
    """
    assert len(phase) == len(power)

    E_field = np.sqrt(power) * np.exp(phase * 1.0j)

    points_per_fft = len(phase)/sampling_multiplier
    # we round the points_per_fft to the nearest power of 2
    points_per_fft = int(2**(np.round(np.log2(points_per_fft))))
    fs = 1/time_step

    # not sure about the detrend="constant", but we get some weird behavior in the output at exactly 0Hz
    # may be better to ignore that point completely...
    return signal.welch(E_field, fs=fs, return_onesided=False, nperseg=points_per_fft, scaling="spectrum", detrend="constant")


def get_peak_indexes(power, time_step:float, expected_rep_rate:float, minimum_peak_power:float=1e-3) -> np.ndarray:
    """Get the indexes of the peaks in the power dataset.

    We use the expected rep rate to calculate an expected distance between the peaks.
    This is a trade-off here: if we use a smaller expected distance, we could detect 
    higher-order oscillations, but if we pick something too small, it may register
    a single pulse as multiple pulses if the pulse quality is poor. 
    For now, we pick a relatively large expected pulse distance and focus on 
    pulse repetition rates that are close to the expected value.
    """

    expected_peak_distance = int(1 / (expected_rep_rate * time_step)) # in nbr of points
    peak_idxs, _ = signal.find_peaks(power, prominence=minimum_peak_power, distance=int(expected_peak_distance * 0.4))
    return peak_idxs


def pulse_energy_and_repetition_rate(power, time_step:float, expected_rep_rate:float) -> tuple[float, float]:
    """returns (average_pulse_energy, pulse_energy_std, actual_pulse_rate)
    Pulse rate is returned as a "sanity check". If this differs significantly from the expected value,
    it means that the pulse quality is poor.
    """	
    peak_idxs = get_peak_indexes(power, time_step, expected_rep_rate)

    if len(peak_idxs) == 0:
        return 0, 0, 0

    pulse_energy_series = np.zeros(len(peak_idxs))
    for i, idx in enumerate(peak_idxs):
        # calculate the energy of the pulse
        window = int(1/(expected_rep_rate * time_step))//2
        pulse_energy_series[i] = np.sum(power[idx-window//2:idx+window//2]) * time_step

    average_pulse_energy = np.average(pulse_energy_series)
    pulse_energy_std = np.std(pulse_energy_series)
    actual_pulse_rate = len(peak_idxs) / (time_step * len(power)) 
    return average_pulse_energy, pulse_energy_std, actual_pulse_rate


def _scaled_sech2(t, tau, scale, offset_t):
    return scale * (1/np.cosh((t-offset_t)/tau))**2

def estimate_FWHM_and_pulse_quality(power, time_step:float, expected_rep_rate:float, avg_num_peaks:int=100, 
                                    visualize:bool=False, **vis_figure_args) -> tuple[float, float]:
    """Estimate the FWHM of the pulses in an optical power trace. 
    
    We first estimate the position of the peaks in the signal using the get_peak_indexes() function 
    (for considerations regarding time_step and expected_rep_rate, see description of get_peak_indexes()).
    Next, we curve_fit a sech^2 function to the peaks in the signal, and use that to estimate the FWHM
    of each peak. The quality of the fitting is determined by the relative overlap between the actual
    signal and the fitting, yielding a value between 0 (no fit) and 1 (best fit).

    avg_num_peaks determines how many peaks (starting from the end of the signal counting back) are used
    in the calculation.

    You may pass visualize=True to visualize the results of the curve_fit() function and the quality of the 
    fitting. If visualize==True, you may also pass additional arguments with vis_figure_args, which are passed
    on to the matplotlib.pyplot.subplots() call to create the plot.

    Returns (FWHM, FWHM_stdev, pulse_quality, pulse_quality_stdev)
    """
    INVALID_FITTING = (0, 0, 0, 0)

    peak_idxs = get_peak_indexes(power, time_step, expected_rep_rate)
    if len(peak_idxs) == 0:
        return INVALID_FITTING
    
    actual_pulse_rate = len(peak_idxs) / (time_step * len(power)) 
    
    if avg_num_peaks > len(peak_idxs):
        print(f"Warning: avg_num_peaks {avg_num_peaks} is larger than the number of peaks, setting it to {len(peak_idxs)}")	
        avg_num_peaks = len(peak_idxs)

    if visualize:
        visualize_idx = peak_idxs[-(avg_num_peaks+1)//2] # pick something in the middle

    try:
        tau_series = np.zeros(avg_num_peaks)
        overlap_series = np.zeros(avg_num_peaks)
        # skip last peak in data, just in case it's not a complete pulse
        for i, idx in enumerate(peak_idxs[-(avg_num_peaks+1):-1]): 
            window = (int(1/(actual_pulse_rate * time_step)) // 2) * 2 # make sure it's divisible by 2
            y_data = power[idx-window//2:idx+window//2]	
            x_data = np.linspace(0, (window-1) * time_step, window)

            (t, scale, offset), pcov = optimize.curve_fit(_scaled_sech2, x_data, y_data, 
                                                          p0=[2e-12, power[idx], window/2*time_step])

            # calculate quality of fit only if pulse fits in pulse window 
            # (otherwise the result will be non-sensical)
            if t < 1/actual_pulse_rate:
                # calculate quality of fit: we calculate the relative overlap between the ideal pulse and the actual pulse
                #   = 1 - sum(abs(y_data - y_fit)) / sum(y_data) 
                # which should give us a value between 0 and 1
                overlap = (1 - np.sum(np.abs(y_data - _scaled_sech2(x_data, t, scale, offset))) / np.sum(y_data))
            else:
                overlap = 0

            tau_series[i] = t
            overlap_series[i] = overlap

            if visualize and idx == visualize_idx:
                try:
                    __visualize_sech2_fit(x_data, y_data, _scaled_sech2(x_data, t, scale, offset), t, overlap, vis_figure_args)
                except:
                    # error plotting visualization, ignore and continue (avoid failing the curve fit)
                    print("Warning: Failed to visualise sech^2 fit")

    except RuntimeError as e:
        # curve fit failed
        # we could technically work with the rest of the data if some curve_fit() do work,
        # but the likelyhood of the data not containing good pulses is very high if even
        # only a single fit fails.
        return INVALID_FITTING
    except ValueError as e:
        return INVALID_FITTING

    tau = np.average(tau_series)
    tau_std = np.std(tau_series)
    overlap = np.average(overlap_series)
    overlap_std = np.std(overlap_series)

    #if FWHM > window, then the fit is poor and the pulse quality is poor
    if ((tau * 1.76) > 1/expected_rep_rate):
        return INVALID_FITTING

    # for a sech2 pulse, FWHM = 1.76 * tau
    return tau * 1.76, tau_std * 1.76, overlap, overlap_std


def __visualize_sech2_fit(x_data, y_data, sech2_fit, t, overlap, vis_figure_args):
    x_data_scaled = x_data * 1e12  # ps
    y_data_scaled = y_data * 1e3   # mW
    y_fit_scaled = sech2_fit * 1e3 # mW

    fig, ax = plt.subplots(**vis_figure_args)
    ax.plot(x_data_scaled, y_data_scaled)
    ax.fill_between(x_data_scaled, y_data_scaled, y_fit_scaled, facecolor='red', alpha=0.4)
    ax.annotate(f'FWHM = {fmt_eng(t*1.76)}s',
        xy=(0, 1), xycoords='axes fraction',
        xytext=(15, -15), textcoords='offset pixels', 
        horizontalalignment='left', verticalalignment='top')
    ax.annotate(f'Q = {overlap:2.2f}',
        xy=(0, 1), xycoords='axes fraction',
        xytext=(15, -32), textcoords='offset pixels', 
        horizontalalignment='left', verticalalignment='top')
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("power (mW)")
    plt.tight_layout()


def estimate_3db_20db_bandwidth(freqs, psd, psd_input="linear") -> tuple[float, float]:
    # calculate the 3dB and 20dB bandwidth of the psd
    # We implement a relatively simple algorithm, where we take the minimum and maximum
    # frequency that have a peak above the cut-off point. This works well as long as there's
    # only 1 broad gain peak in the spectrum. If the spectrum is more complicated, 
    # this method will over-estimate the 3dB and 20dB bandwidth.

    if psd_input == "linear":
        psd_log = 10 * np.log10(psd)
    elif psd_input == "log":
        psd_log = psd
    else:
        raise ValueError("psd_input should be either 'linear' or 'log'")

    psd_log_max = np.max(psd_log)
    min_3dB = np.squeeze(np.argwhere(psd_log > psd_log_max - 3))
    min_20dB = np.squeeze(np.argwhere(psd_log > psd_log_max - 20))

    if min_3dB.size >= 2:
        BW_3dB = freqs[min_3dB[-1]] - freqs[min_3dB[0]]
    else:
        BW_3dB = 0

    if min_20dB.size >= 2:
        BW_20dB = freqs[min_20dB[-1]] - freqs[min_20dB[0]]
    else:
        BW_20dB = 0

    return BW_3dB, BW_20dB








