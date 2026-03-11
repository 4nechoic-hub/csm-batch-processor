"""
CSM Batch Processor — Python Edition
=====================================
Cross-Spectral Matrix calculator and spectral analysis toolkit.

Ported from MATLAB App Designer project by TZ.
Supports CSV, MAT, and TDMS input formats.

Features:
  - Narrowband CSM computation (Welch-style block averaging with Hanning window)
  - Octave-band / fractional-octave binning
  - Auto- and cross-correlation
  - Spectral visualisation helpers
"""

from .csm_calculator import csm_calculator
from .log_binning import log_freq_bin
from .correlation import compute_correlation
from .io_utils import load_data, save_results
from .plotting import (
    plot_autospectra,
    plot_cross_spectra,
    plot_coherence,
    plot_correlation,
)

__version__ = "1.0.0"
__all__ = [
    "csm_calculator",
    "log_freq_bin",
    "compute_correlation",
    "load_data",
    "save_results",
    "plot_autospectra",
    "plot_cross_spectra",
    "plot_coherence",
    "plot_correlation",
]
