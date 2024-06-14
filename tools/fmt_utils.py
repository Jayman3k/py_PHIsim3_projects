from prefixed import Float  # Float with better formatting options. To install, run: pip install prefixed


def fmt_pow10(val: float, precision: int = 2, times="times") -> str:
    """formatting function for matplotlib labels
    converts to powers-of-10, for example, 1.1e-16 to '1.10 x 10^-16', or 1230 to '1.23 x 10^3', ...
    """
    base, power = f'{val:.{precision}e}'.split('e')
    # +16 -> 16, +03 -> 3, -03 -> -3
    power = power.removeprefix('+').removeprefix('0').replace('-0', '-') 
    return f'${base}\\{times}10^{{{power}}}$' 


def fmt_eng(val: float, digits: int = 3) -> str:
    """formatting function for labels 
    converts numbers to SI (power of 3) magnitudes, for example, 1.1e-2 to '11m', 11000 to '11k', ...
    (note that this adds the 'μ' for micro, which has some issues when used in filenames)
    """
    return f'{Float(val):.{digits}H}'


def plot_latex_style(pyplot, font_size=11.0):
    """makes plots looks prettier with LaTeX fonts
    Use this for creating pubishable plots - it's a bit slower than regular style plotting,
    so you probably want to avoid using this for everyday use.
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
