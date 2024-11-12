from prefixed import Float  # Float with better formatting options. To install, run: pip install prefixed

### thanks to Yatharth Agarwal @ https://stackoverflow.com/a/10970888/4431955
SI_prefix_scaling = {
    'y': 1e-24,  # yocto
    'z': 1e-21,  # zepto
    'a': 1e-18,  # atto
    'f': 1e-15,  # femto
    'p': 1e-12,  # pico
    'n': 1e-9,   # nano
    'μ': 1e-6,   # micro (original author put 'u' here, but Python should have no issues with 'μ')
    'm': 1e-3,   # mili
    'c': 1e-2,   # centi
    'd': 1e-1,   # deci
    'k': 1e3,    # kilo
    'M': 1e6,    # mega
    'G': 1e9,    # giga
    'T': 1e12,   # tera
    'P': 1e15,   # peta
    'E': 1e18,   # exa
    'Z': 1e21,   # zetta
    'Y': 1e24,   # yotta

    "yocto" : 1e-24,
    "zepto" : 1e-21,
    "atto"  : 1e-18,
    "femto" : 1e-15,
    "pico"  : 1e-12,
    "nano"  : 1e-9,
    "micro" : 1e-6,
    "mili"  : 1e-3,
    "centi" : 1e-2,
    "deci"  : 1e-1,
    "kilo"  : 1e3, 
    "mega"  : 1e6, 
    "giga"  : 1e9, 
    "tera"  : 1e12,
    "peta"  : 1e15,
    "exa"   : 1e18,
    "zetta" : 1e21,
    "yotta" : 1e24,
}


def fmt_pow10(val: float, precision: int = 2, times="times") -> str:
    """formatting function for matplotlib labels
    converts to powers-of-10, for example, 1.1e-16 to '1.10 x 10^-16', or 1230 to '1.23 x 10^3', ...
    """
    base, power = f'{val:.{precision}e}'.split('e')
    # +16 -> 16, +03 -> 3, -03 -> -3
    power = power.removeprefix('+').removeprefix('0').replace('-0', '-') 
    return f'${base}\\{times}10^{{{power}}}$' 


def fmt_eng(val: float, digits: int = 3, trailing_zeroes: bool=True) -> str:
    """formatting function for labels 
    converts numbers to SI (power of 3) magnitudes, for example, 1.1e-2 to '11m', 11000 to '11k', ...
    (note that this adds the 'μ' for micro, which has some issues when used in filenames)
    """
    tz = '#' if trailing_zeroes else ''
    return f'{Float(val):{tz}.{digits}H}'


def plot_latex_style(pyplot, font_size=11.0):
    """makes plots looks prettier with LaTeX fonts
    Use this for creating pubishable plots - it's a bit slower than regular style plotting,
    so you probably want to avoid using this for everyday use.
    Note: this requires a working LaTeX installation! 
    """
    pyplot.rcParams.update({
        "font.size":         font_size,
        "font.family":       "serif",
        "font.serif":        "Palatino",
        "axes.titlesize":    "medium",
        "figure.titlesize":  "medium",
        "text.usetex":        True,
        "text.latex.preamble": "\\DeclareUnicodeCharacter{03BC}{\\textmu}\n" + # fix for "μ is not set up" error
                               "\\usepackage{amsmath}\\usepackage{amssymb}\\usepackage{siunitx}[=v2]",
    })


