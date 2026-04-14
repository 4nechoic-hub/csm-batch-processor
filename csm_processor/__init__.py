from ._version import __version__
from .csm_calculator import csm_calculator
from .log_binning import log_freq_bin, bin_csm
from .correlation import compute_correlation
from .io_utils import load_data, save_results
from .feature_extraction import extract_features, extract_features_batch
from .anomaly_detection import SpectralAnomalyDetector, compare_methods
from .plotting import (
    plot_autospectra,
    plot_cross_spectra,
    plot_coherence,
    plot_correlation,
    plot_csm_matrix,
    plot_coherence_matrix,
)
from .anomaly_plotting import (
    plot_anomaly_scores,
    plot_feature_importance,
    plot_anomaly_spectra,
    plot_method_comparison,
    plot_detection_summary,
)

__all__ = [
    "__version__",
    "csm_calculator",
    "log_freq_bin",
    "bin_csm",
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
    "plot_csm_matrix",
    "plot_coherence_matrix",
    "plot_anomaly_scores",
    "plot_feature_importance",
    "plot_anomaly_spectra",
    "plot_method_comparison",
    "plot_detection_summary",
]