from importlib import import_module

from ._version import __version__
from .anomaly_detection import SpectralAnomalyDetector, compare_methods
from .correlation import compute_correlation
from .csm_calculator import csm_calculator
from .feature_extraction import extract_features, extract_features_batch
from .io_utils import load_data, save_results
from .log_binning import bin_csm, log_freq_bin

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

_PLOTTING_EXPORTS = {
    "plot_autospectra",
    "plot_cross_spectra",
    "plot_coherence",
    "plot_correlation",
    "plot_csm_matrix",
    "plot_coherence_matrix",
}

_ANOMALY_PLOTTING_EXPORTS = {
    "plot_anomaly_scores",
    "plot_feature_importance",
    "plot_anomaly_spectra",
    "plot_method_comparison",
    "plot_detection_summary",
}


def __getattr__(name):
    if name in _PLOTTING_EXPORTS:
        module = import_module(".plotting", __name__)
        return getattr(module, name)

    if name in _ANOMALY_PLOTTING_EXPORTS:
        module = import_module(".anomaly_plotting", __name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))