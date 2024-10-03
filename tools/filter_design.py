import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
#import tikzplotlib # some version issues with tikzplotlib, doesn't work with latest matplotlib

from fmt_utils import fmt_eng, plot_latex_style

# NOTE1: approximate values for sampling freq in PHIsim, @ approx λ=1550nm, n=3.7
# (rule-of-thumb values! for estimates only, for calculations use the exact values)
# segment - T_seg -  f_s   - L_seg
#   2 OPL -  10fs - 100THz -  840nm
#   5 OPL -  25fs -  40THz -  2µm
#   8 OPL -  40fs -  25THz -  3.3µm
#  10 OPL -  50fs -  20THz -  4µm
#  20 OPL - 100fs -  10THz -  8µm

# NOTE2: around 1550nm, Δf = 1THz is approx Δλ = 8nm

# NOTE3: We apply the filter to the amplitude of the optical field, not the power,
#        therefore, we have to take the square of the amplitude filter response to get
#        the spectral power response
#
#        Even though the labels say "magnitude" and "amplitude", it could be plotting the 
#        spectral power (amplitude**2). I switched back and forth a couple of times,
#        it's a hassle to keep the label consistent, and I don't want to write code for that. 
#        --> Just check if the data is squared or not.

def _fmt_arr(arr) -> str:
    return np.array2string(arr, precision=12, floatmode='fixed', separator=", ")

fs = 273e12 # 25e12 
design = "fir"
simulated_soa_len = 1e-3 # m
save_tikz = False

VanGasseGainCurve = {
    # experimental data
    # approximate data from: Van Gasse et al. "27 dB gain III-V-on-silicon semiconductor optical amplifier with > 50 mW output power" (2018) figure 5(a)
    "lambda_nm"  : np.array([1553.6838066001535, 1557.3215656178052, 1560.4067536454336, 1564.3668457405986, 1569.6623177283193, 1573.4382194934765, 1577.168073676132, 1579.8388334612432, 1583.7068303914045, 1587.6669224865695, 1591.2586339217191, 1594.8963929393708, 1599.6853415195703, 1602.4021488871833, 1605.2110514198005]),
    "gain_dB"    : np.array([21.54871794871795, 22.943589743589744, 23.94871794871795, 24.851282051282052, 25.54871794871795, 25.671794871794873, 25.46666666666667, 25.138461538461538, 24.42051282051282, 23.312820512820515, 22.02051282051282, 20.44102564102564, 17.897435897435898, 16.256410256410255, 14.369230769230768]),
    "soa_len"    : 1.2e-3 # m
}

JuodawlkisGainCurve = {
    # experimental data
    # approximate data from: Juodawlkis et al. "Packaged 1.5-m Quantum-Well SOA With 0.8-W Output Power and 5.5-dB Noise Figure" (2009) figure 2
    "1A" : {
        "lambda_nm" : np.array([1494.892857142857, 1500.0357142857142, 1505.0714285714287, 1509.892857142857, 1514.8214285714287, 1519.75, 1524.892857142857, 1529.8214285714287, 1534.9642857142858, 1539.7857142857142, 1544.9285714285713, 1549.75, 1554.5714285714287, 1559.9285714285713, 1564.75]),
        "gain_dB"   : np.array([2.6345177664974635, 4.1979695431472095, 4.979695431472084, 5.725888324873097, 6.5786802030456855, 7.111675126903556, 7.289340101522846, 7.431472081218278, 7.53807106598985, 7.3248730964467015, 6.934010152284266, 6.436548223350254, 5.654822335025383, 4.730964467005077, 3.9137055837563466])
    },
    "2A" : {
        "lambda_nm" : np.array([1484.7142857142858, 1489.75, 1494.7857142857142, 1499.7142857142858, 1504.75, 1509.892857142857, 1514.607142857143, 1519.75, 1525, 1529.8214285714287, 1534.75, 1539.7857142857142, 1544.8214285714287, 1549.857142857143]),
        "gain_dB"   : np.array([10.274111675126907, 10.55837563451777, 11.16243654822335, 11.695431472081221, 11.837563451776653, 12.050761421319798, 12.157360406091371, 12.228426395939088, 11.979695431472084, 11.730964467005077, 11.482233502538072, 11.126903553299496, 10.416243654822338, 9.847715736040612])
    }
}

ConnelyGainCurve = {
    # theorical data
    # data from: Connely et al. "Wideband Semiconductor Optical Amplifier Steady-State Numerical Model" (2001) Fig 2
    "lambda_nm"     : np.array([1502.8082191780823, 1508.2876712328766, 1514.041095890411, 1520.3424657534247, 1525.6849315068494, 1532.2602739726028, 1537.4657534246576, 1542.5342465753424, 1547.876712328767, 1552.3972602739725, 1556.6438356164383, 1561.3013698630136]),
    "gain_10^4per_m": np.array([6.0494296577946765, 6.178707224334601, 6.273764258555133, 6.349809885931559, 6.3726235741444865, 6.3764258555133075, 6.35361216730038, 6.304182509505703, 6.224334600760456, 6.136882129277566, 6.0418250950570345, 5.916349809885931]),
}


def spline_approx(x, y) -> tuple:
    from scipy.interpolate import UnivariateSpline
    spline = UnivariateSpline(x, y)
    new_x = np.linspace(np.min(x), np.max(x), 1000)
    return new_x, spline(new_x)

def center_and_normalize(x, y_dB) -> tuple:
    # shift around the peak gain wavelength
    x -= x[np.argmax(y_dB)] 
    # normalize gain (i.e., max gain = 0dB)
    y_dB -= max(y_dB)
    return x, y_dB

def smooth_gain_curve(x, y_dB):
    #return center_and_normalize(x, y_dB)
    return center_and_normalize(*spline_approx(x, y_dB))


if design == "bessel":
    ## NOTE: I investigated the use of these types of filters for PHIsim,
    ## but they are not currently used in any of the models in the thesis.
    ## This section is therefor somewhat obsolete.

    ## lowpass bessel (IIR) filter
    # advantages: 
    #  - low complexity, only a few coefficients required for nice smooth lowpass signature,
    #    looking similar to a semiconductor gain curve
    #  - low group delay variations compared to other IIR filter designs
    # disadvantages:
    #  - group delay is not exactly zero, and cannot be further compensated with
    #    a low-order cascaded IIR filter.

    b, a = signal.bessel(5, 4e12, 'low', analog=False, fs=fs, norm='phase')

    print(f"b =  {_fmt_arr(b)}, a = {_fmt_arr(a)}")

    w, h = signal.freqz(b, a)
    plt.figure()
    #plt.plot(w, 20 * np.log10(np.abs(h)))
    delta_f = w*fs/(2*np.pi)
    plt.plot(delta_f[:200]/1e9, np.abs(h[:200]))
    plt.title('Bessel filter magnitude response')
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Amplitude')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.show(block=False)

    if save_tikz:
        tikzplotlib.save("bessel_magnitude.tex")

    w, gd = signal.group_delay((b, a))

    plt.figure()
    plt.title('Digital filter group delay')
    plt.plot(delta_f[1:200]/1e9, gd[1:200])
    plt.ylabel('Group delay [samples]')
    plt.xlabel('Frequency [GHz]')
    plt.grid(which='both', axis='both')
    plt.show(block=False)

    if save_tikz:
        tikzplotlib.save("bessel_grp_delay.tex")

if design == "fir":
    ## lowpass FIR filter
    # advantages: 
    #  - relatively low complexity (simple to implement)
    #  - can be constructed with *zero* group delay variations (symmetric weights)
    # disadvantages:
    #  - In general, FIR needs more weights than IIR (resulting in higher absolute 
    #    group delay and more required computations). In our particular case though, 
    #    because we need a smooth gain drop-off, this disadvantage is actually
    #    pretty small. 

    fc = 20e12 #1e12

    plt.figure()
    plt.title(f'FIR response [fs={fmt_eng(fs)}Hz, fc={fmt_eng(fc)}Hz]')
    plt.xlabel('Frequency [GHz]')

    plt.ylabel('Amplitude')
    #plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')

    fir_h = {}

    for order in (5, 7, 9):
        b = signal.firwin(order, fc, fs=fs)
        #b = signal.firls(order, [0, fc, fc*2, 0.5*fs], [1, 0.8, 0, 0], fs=fs)
        print(f"[N={order}]: b = {_fmt_arr(b)}")
        
        w, h = signal.freqz(b)

        fir_h[order] = h

        delta_f = w*fs/(2*np.pi)
        #plt.plot(delta_f/1e9, 10*np.log10(np.abs(h)), label=f"N={order}")
        plt.plot(delta_f/1e9, np.abs(h)**2, label=f"N={order}")

    # b = [0.25, 0.5, 0.25]
    # w, h = signal.freqz(b)
    # delta_f = w*fs/(2*np.pi)
    # plt.plot(delta_f/1e9, np.abs(h)**2, label=f"N={3}")

    fir_h[3] = h

    plt.legend()
    plt.show(block=False)

    # group-delay should be constant (order-1)/2
    # (enable the plot to check if you want)

    # w, gd = signal.group_delay((b, [1])) 
    # plt.figure()
    # plt.title('Digital filter group delay')
    # plt.plot(w[1:100]*fs/(2*np.pi)/1e9, gd[1:100])
    # plt.ylabel('Group delay [samples]')
    # plt.xlabel('Frequency [GHz]')
    # plt.grid(which='both', axis='both')
    # plt.show()

if design == "reference":
    # designed previously to approximate the spectral behavior of the SOAs
    fir_ref_b = {
        "FIR Van Gasse":     [0.012851639924, 0.041698958521, 0.119001787219, 0.205084278797,
                           0.242726671078, 0.205084278797, 0.119001787219, 0.041698958521,
                           0.012851639924],    
        "FIR Juodawlkis 1" : [0.035702134069, 0.241065536045, 0.446464659773, 0.241065536045,
                           0.035702134069],
        "FIR Juodawlkis 2" : [0.005955212032, 0.214523424100, 0.559042727736, 0.214523424100,
                           0.005955212032]
    }

    fir_ref_h = {}
    for (name, b) in fir_ref_b.items():
        w, h = signal.freqz(b)
        fir_ref_h[name] = h
        delta_f = w*fs/(2*np.pi)


# compare with gain curves found in literature
##############################################

# first, scale to wavelength
delta_lambda = 1550e-9**2 / 3e8 * delta_f

# smooth approx of input data
van_gasse_x, van_gasse_y = smooth_gain_curve(VanGasseGainCurve["lambda_nm"], VanGasseGainCurve["gain_dB"])

juo_x = {}
juo_y = {}
for current in ("1A", "2A"):
    juo_x[current], juo_y[current] = smooth_gain_curve(JuodawlkisGainCurve[current]["lambda_nm"], JuodawlkisGainCurve[current]["gain_dB"])

#plot_latex_style(plt, 13.0)

plt.figure()
#plt.title("Comparison between experimental gain and filter response")
plt.xlabel('$\\lambda - \\lambda_P$ (nm)')
plt.ylabel('Relative gain (dB)')

plt.plot(van_gasse_x[:-3], van_gasse_y[:-3], label="Van Gasse")
plt.plot(juo_x["1A"], juo_y["1A"], label="Juodawlkis (1)")
plt.plot(juo_x["2A"], juo_y["2A"], label="Juodawlkis (2)")
plt.grid(which='both', axis='both')

# take chunk of data around central wavelength, mirror the filter response, scale x to nm, y to dB
f_points = 190
filter_x = np.concatenate((-np.flip(delta_lambda[0:f_points]), delta_lambda[0:f_points]))/1e-9 
if design == "bessel":
    values = 10*np.log10(np.abs(h[0:f_points])**2)
    iir_y = np.concatenate((np.flip(values), values))
    plt.plot(filter_x, iir_y, '.', label="Bessel (IIR)")
elif design == "fir":	
    for order, h in fir_h.items():
        values = 10*np.log10(np.abs(h[0:f_points])**2)
        fir_y = np.concatenate((np.flip(values), values))
        plt.plot(filter_x, fir_y, linestyle='dotted', label=f"FIR (order = {order})")
elif design == "reference":
    for name, h in fir_ref_h.items():
        values = 10*np.log10(np.abs(h[0:f_points])**2)
        fir_y = np.concatenate((np.flip(values), values))
        plt.plot(filter_x, fir_y, linestyle='dotted', label=f"{name} (order = {len(fir_ref_b[name])})")

plt.legend()

if save_tikz:
    tikzplotlib.save("filter_comparison_spectrum.tex")

plt.show(block=True)


