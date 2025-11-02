import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.collections as mcoll
import numpy as np

pgf_with_latex = {
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    # ~ "font.serif": [],
    # ~ "font.sans-serif": [],
    # ~ "font.monospace": [],
    "axes.labelsize": 13,
    "font.size": 13,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "pgf.preamble": r"\usepackage[utf8x]{inputenc}\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{amssymb}\usepackage{amsfonts}"
    }
    
mpl.rcParams.update(pgf_with_latex)
mpl.rcParams['mathtext.fontset'] = 'cm'

colors   = ['steelblue', 'seagreen', 'firebrick', 'orange', 'silver', 'lightsalmon', 'navy']
insetFontSize = 10

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def colorline(ax,
        x, y, z=None, cmap='copper', norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, "__iter__"):
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax.add_collection(lc)

    return lc

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

cmap_transition = matplotlib.colors.LinearSegmentedColormap.from_list("", ["steelblue","grey","orange"])
cmap_transition_r = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orange","grey","steelblue"])

