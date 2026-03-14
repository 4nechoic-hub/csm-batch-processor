"""
Publication Style
==================
Unified publication-quality style configuration for all CSM Processor figures.

Inspired by the clean, scientific aesthetic of the Acoular library's example
gallery.  Uses a restrained colour palette, proper typographic hierarchy, and
LaTeX-style axis labels where Matplotlib's mathtext supports it.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ─── Colour palette ─────────────────────────────────────────────────────────
# A muted, colour-blind-friendly palette suitable for journal reproduction.
PALETTE = {
    "blue": "#2563EB",
    "red": "#DC2626",
    "green": "#059669",
    "amber": "#D97706",
    "purple": "#7C3AED",
    "slate": "#475569",
    "sky": "#0EA5E9",
    "rose": "#E11D48",
}

# Sequential for multi-channel overlays
CHANNEL_COLORS = [
    "#2563EB",  # blue
    "#DC2626",  # red
    "#059669",  # green
    "#D97706",  # amber
    "#7C3AED",  # purple
    "#0EA5E9",  # sky
    "#E11D48",  # rose
    "#475569",  # slate
]

# Anomaly-specific
NORMAL_COLOR = "#2563EB"
ANOMALY_COLOR = "#DC2626"
THRESHOLD_COLOR = "#D97706"

# Feature category colours (for grouped bar charts)
CATEGORY_COLORS = {
    "power": "#2563EB",
    "spectral": "#059669",
    "band": "#7C3AED",
    "coherence": "#D97706",
}

# ─── Matplotlib RC configuration ────────────────────────────────────────────

STYLE = {
    # Figure
    "figure.facecolor": "white",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    # Axes
    "axes.facecolor": "white",
    "axes.edgecolor": "#334155",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.titlepad": 10,
    "axes.labelsize": 11,
    "axes.labelpad": 6,
    "axes.labelweight": "medium",
    "axes.labelcolor": "#1E293B",
    # Ticks
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    # Grid
    "grid.color": "#CBD5E1",
    "grid.alpha": 0.5,
    "grid.linewidth": 0.4,
    "grid.linestyle": "-",
    # Legend
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "#CBD5E1",
    "legend.fancybox": True,
    "legend.borderpad": 0.5,
    "legend.handlelength": 1.8,
    # Lines
    "lines.linewidth": 1.4,
    "lines.markersize": 5,
    # Font
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "mathtext.fontset": "dejavusans",
}


def apply_style():
    """Apply the publication style globally."""
    plt.rcParams.update(STYLE)


def get_channel_color(idx: int) -> str:
    """Return a colour for a given channel index (wraps around)."""
    return CHANNEL_COLORS[idx % len(CHANNEL_COLORS)]


def get_feature_category(name: str) -> str:
    """Classify a feature name into a display category."""
    if any(k in name for k in ("total_power", "oaspl", "peak_psd", "peak_freq")):
        return "power"
    if any(k in name for k in ("centroid", "bandwidth", "slope", "flatness", "crest", "kurtosis")):
        return "spectral"
    if any(k in name for k in ("band",)):
        return "band"
    if any(k in name for k in ("coherence", "phase")):
        return "coherence"
    return "power"


def categorical_cmap(values: np.ndarray, vmin=None, vmax=None, cmap_name="YlOrRd"):
    """Return RGBA array for a set of values mapped to a colourmap."""
    cmap = plt.cm.get_cmap(cmap_name)
    vmin = vmin if vmin is not None else values.min()
    vmax = vmax if vmax is not None else values.max()
    rng = vmax - vmin if (vmax - vmin) > 0 else 1.0
    normed = (values - vmin) / rng
    return cmap(normed)
