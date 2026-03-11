"""
Anomaly Detection Visualisation
=================================
Publication-quality plots for anomaly detection results.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional

from .anomaly_detection import AnomalyResult


def plot_anomaly_scores(
    result: AnomalyResult,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot anomaly scores as a time-series with threshold line.

    Anomalous samples are highlighted in red.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_facecolor("#fafbfc")

    n = len(result.scores)
    x = np.arange(n)

    # Normal points
    normal = result.labels == 1
    ax.scatter(
        x[normal], result.scores[normal],
        c="#3b82f6", s=20, alpha=0.6, label="Normal", zorder=3,
    )

    # Anomalies
    anomaly = result.labels == -1
    ax.scatter(
        x[anomaly], result.scores[anomaly],
        c="#ef4444", s=40, alpha=0.9, label="Anomaly",
        edgecolors="#991b1b", linewidths=0.8, zorder=4,
    )

    # Threshold
    ax.axhline(
        result.threshold, color="#f59e0b", linestyle="--",
        linewidth=1.5, label=f"Threshold ({result.threshold:.3f})",
    )

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Anomaly Score")
    ax.set_title(title or f"Anomaly Scores — {result.method}")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_feature_importance(
    result: AnomalyResult,
    top_n: int = 15,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Horizontal bar chart of top feature importances (contribution to anomaly).
    """
    if result.feature_importances is None:
        return None

    names = result.feature_names or [
        f"Feature {i}" for i in range(len(result.feature_importances))
    ]

    # Sort by importance
    idx = np.argsort(result.feature_importances)[::-1][:top_n]
    top_names = [names[i] for i in idx]
    top_values = result.feature_importances[idx]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.set_facecolor("#fafbfc")

    bars = ax.barh(
        range(len(top_names)),
        top_values[::-1],
        color="#6366f1",
        edgecolor="#4338ca",
        linewidth=0.5,
        height=0.7,
    )
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9, fontfamily="monospace")
    ax.set_xlabel("Importance")
    ax.set_title(title or f"Feature Importance — {result.method}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_anomaly_spectra(
    freq: np.ndarray,
    spectra_normal: np.ndarray,
    spectra_anomaly: np.ndarray,
    channel: int = 0,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Overlay normal vs anomalous auto-spectra for visual comparison.

    Parameters
    ----------
    freq : np.ndarray
    spectra_normal : np.ndarray or list of np.ndarray
        One or more CSMs classified as normal.
    spectra_anomaly : np.ndarray or list of np.ndarray
        One or more CSMs classified as anomalous.
    channel : int
        Which channel's auto-spectrum to plot.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    half = len(freq) // 2
    f = freq[1:half]

    def _plot_spectra(spectra_list, color, alpha, label_prefix):
        if isinstance(spectra_list, np.ndarray) and spectra_list.ndim == 3:
            spectra_list = [spectra_list]
        for i, s in enumerate(spectra_list):
            psd = np.real(s[1:half, channel, channel])
            psd_db = 10 * np.log10(np.maximum(psd, 1e-30))
            label = f"{label_prefix}" if i == 0 else None
            ax.semilogx(f, psd_db, color=color, alpha=alpha, linewidth=0.8, label=label)

    _plot_spectra(spectra_normal, "#3b82f6", 0.3, "Normal")
    _plot_spectra(spectra_anomaly, "#ef4444", 0.7, "Anomaly")

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [dB]")
    ax.set_title(title or f"Normal vs Anomalous Spectra — Ch {channel + 1}")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_method_comparison(
    results: dict[str, AnomalyResult],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side comparison of anomaly scores from multiple methods.
    """
    methods = list(results.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4), sharey=False)
    if n_methods == 1:
        axes = [axes]

    colors = {"isolation_forest": "#3b82f6", "mahalanobis": "#10b981", "lof": "#f59e0b"}

    for ax, method in zip(axes, methods):
        r = results[method]
        n = len(r.scores)
        x = np.arange(n)
        color = colors.get(method, "#6366f1")

        normal = r.labels == 1
        ax.scatter(x[normal], r.scores[normal], c=color, s=15, alpha=0.5)
        anomaly = r.labels == -1
        ax.scatter(x[anomaly], r.scores[anomaly], c="#ef4444", s=30, alpha=0.9,
                   edgecolors="#991b1b", linewidths=0.5)
        ax.axhline(r.threshold, color="#f59e0b", linestyle="--", linewidth=1)

        ax.set_title(f"{r.method}\n({r.n_anomalies} anomalies)", fontsize=11)
        ax.set_xlabel("Sample")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Anomaly Score")
    fig.suptitle(title or "Method Comparison", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig