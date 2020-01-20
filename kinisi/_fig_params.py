"""
Figure parameters for kinisi to help make nice plots.

Copyright (c) Andrew R. McCluskey and Benjamin J. Morgan

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

from collections import OrderedDict
from matplotlib import rcParams, cycler

TABLEAU = OrderedDict(
    [
        ("blue", "#0173B2"),
        ("orange", "#DE8F05"),
        ("green", "#029E73"),
        ("red", "#D55E00"),
        ("purple", "#CC78BC"),
        ("brown", "#CA9161"),
        ("pink", "#FBAFE4"),
        ("grey", "#949494"),
        ("yellow", "#ECE133"),
        ("turquoise", "#56B4E9"),
    ]
)

FONTSIZE = 20
NEARLY_BLACK = "#161616"
LIGHT_GREY = "#F5F5F5"
WHITE = "#ffffff"

MASTER_FORMATTING = {
    "axes.formatter.limits": (-3, 3),
    "xtick.major.pad": 7,
    "ytick.major.pad": 7,
    "ytick.color": NEARLY_BLACK,
    "xtick.color": NEARLY_BLACK,
    "axes.labelcolor": NEARLY_BLACK,
    "axes.spines.bottom": False,
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "custom",
    "font.size": FONTSIZE,
    "font.family": "serif",
    # 'font.serif' : 'Source Serif Pro',
    "text.usetex": True,
    "savefig.bbox": "tight",
    "axes.facecolor": LIGHT_GREY,
    "axes.labelpad": 10.0,
    "axes.labelsize": FONTSIZE * 0.8,
    "axes.titlepad": 30,
    "axes.titlesize": FONTSIZE,
    "axes.grid": False,
    "grid.color": WHITE,
    "lines.markersize": 7.0,
    "lines.scale_dashes": False,
    "xtick.labelsize": FONTSIZE * 0.8,
    "ytick.labelsize": FONTSIZE * 0.8,
    "legend.fontsize": FONTSIZE * 0.8,
    "lines.linewidth": 4,
}

for k, v in MASTER_FORMATTING.items():
    rcParams[k] = v

COLOR_CYCLE = TABLEAU.values()

rcParams["axes.prop_cycle"] = cycler(color=COLOR_CYCLE)
