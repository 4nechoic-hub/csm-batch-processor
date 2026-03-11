"""
CSM Batch Processor — Python Edition
=====================================
Cross-Spectral Matrix calculator and spectral analysis toolkit.

Designed by TZ.
Supports CSV, MAT, and TDMS input formats.

Features:
  - Narrowband CSM computation (Welch-style block averaging with Hanning window)
  - Octave-band / fractional-octave binning
  - Auto- and cross-correlation
  - Spectral visualisation helpers
  - ML-ready feature extraction from spectral data
  - Unsupervised anomaly detection (Isolation Forest, Mahalanobis, LOF)
"""

from .csm_calculator import csm_calculator
from .log_binning import log_freq_bin
from .correlation import compute_correlation
from .io_utils import load_data, save_results
from .feature_extraction import extract_features, extract_features_batch
from .anomaly_detection import SpectralAnomalyDetector, compare_methods
from .plotting import (
    plot_autospectra,
    plot_cross_spectra,
    plot_coherence,
    plot_correlation,
)
from .anomaly_plotting import (
    plot_anomaly_scores,
    plot_feature_importance,
    plot_anomaly_spectra,
    plot_method_comparison,
)

__version__ = "1.1.0"
__all__ = [
    "csm_calculator",
    "log_freq_bin",
    "compute_correlation",
    "load_data",
    "save_results",
    "extract_features",
    "extract_features_batch",
    "SpectralAnomalyDetector",
    "compare_methods",
    "plot_autospectra",
    "plot_cross_spectra",
    "plot_coherence",
    "plot_correlation",
    "plot_anomaly_scores",
    "plot_feature_importance",
    "plot_anomaly_spectra",
    "plot_method_comparison",
]