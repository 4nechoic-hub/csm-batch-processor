"""
Anomaly Detection Visualisation
=================================
Publication-quality plots for anomaly detection results.

All figures follow the unified style in :mod:`csm_processor.style` and are
designed for journal / portfolio presentation quality.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional

from .anomaly_detection import AnomalyResult
from .style import (
    apply_style,
    NORMAL_COLOR,
    ANOMALY_COLOR,
    THRESHOLD_COLOR,
    CHANNEL_COLORS,
    get_channel_color,
    get_feature_category,
    CATEGORY_COLORS,
    categorical_cmap,
)


# --------------------------------------------------------------------------- #
#  Anomaly Scores  (shaded zones + refined markers)
# --------------------------------------------------------------------------- #


def plot_anomaly_scores(
    result: AnomalyResult,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot anomaly scores as a time-series with threshold and shaded zones.

    Normal samples are shown as small circles; anomalies as larger diamonds
    with edge highlights.  If the threshold is far outside the score range
    (common with Isolation Forest), the plot clips to data and shows an
    off-chart indicator.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4.2))

    n = len(result.scores)
    x = np.arange(n)
    normal = result.labels == 1
    anomaly = result.labels == -1

    score_max = np.max(result.scores)
    score_min = np.min(result.scores)
    data_range = score_max - score_min if score_max != score_min else 1.0
    pad = 0.12 * data_range

    # Decide if threshold is within plottable range
    # Use tight tolerance: threshold should be within 50% of data range
    # beyond the data extremes to be shown in-view
    threshold_in_view = (
        result.threshold >= score_min - 0.5 * data_range
        and result.threshold <= score_max + 0.5 * data_range
    )

    if threshold_in_view:
        y_lo = min(score_min - pad, result.threshold - pad)
        y_hi = max(score_max + pad, result.threshold + pad)
    else:
        y_lo = score_min - pad
        y_hi = score_max + pad

    # ── Shaded zones ─────────────────────────────────────────────────
    if threshold_in_view:
        ax.axhspan(result.threshold, y_hi, color=ANOMALY_COLOR, alpha=0.06, zorder=0)
        ax.axhspan(y_lo, result.threshold, color=NORMAL_COLOR, alpha=0.04, zorder=0)
        ax.axhline(
            result.threshold, color=THRESHOLD_COLOR, linestyle="--",
            linewidth=1.2, zorder=5, alpha=0.9,
        )
        ax.text(
            n + 0.3, result.threshold, f"θ = {result.threshold:.3f}",
            va="center", ha="left", fontsize=8, color=THRESHOLD_COLOR,
            fontweight="medium",
        )
    else:
        # Threshold off-chart: shade by labels instead
        if np.any(normal):
            n_scores = result.scores[normal]
            ax.axhspan(
                np.min(n_scores) - pad, np.max(n_scores) + pad * 0.5,
                color=NORMAL_COLOR, alpha=0.04, zorder=0,
            )
        if np.any(anomaly):
            a_scores = result.scores[anomaly]
            ax.axhspan(
                np.min(a_scores) - pad * 0.5, np.max(a_scores) + pad,
                color=ANOMALY_COLOR, alpha=0.06, zorder=0,
            )
        # Arrow pointing toward threshold direction
        direction = "↑" if result.threshold > score_max else "↓"
        ax.annotate(
            f"θ = {result.threshold:.3f} {direction}",
            xy=(n * 0.85, y_hi - 0.15 * (y_hi - y_lo)),
            fontsize=8.5, fontweight="medium", color=THRESHOLD_COLOR,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=THRESHOLD_COLOR, alpha=0.85),
        )

    # ── Normal points ────────────────────────────────────────────────
    ax.scatter(
        x[normal], result.scores[normal],
        c=NORMAL_COLOR, s=28, alpha=0.65, zorder=3,
        edgecolors="white", linewidths=0.4, label="Normal",
    )

    # ── Anomaly points ───────────────────────────────────────────────
    ax.scatter(
        x[anomaly], result.scores[anomaly],
        c=ANOMALY_COLOR, s=55, alpha=0.9, zorder=4,
        marker="D", edgecolors="#7F1D1D", linewidths=0.7,
        label=f"Anomaly ({result.n_anomalies})",
    )

    # ── Axes ─────────────────────────────────────────────────────────
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Anomaly Score")
    ax.set_title(title or f"Anomaly Scores — {result.method}")
    ax.set_xlim(-0.8, n + 2)
    ax.set_ylim(y_lo, y_hi)
    ax.legend(loc="upper left", framealpha=0.9)

    fig.subplots_adjust(left=0.1, right=0.92, top=0.9, bottom=0.15)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
#  Feature Importance  (gradient bars + category colours)
# --------------------------------------------------------------------------- #


def plot_feature_importance(
    result: AnomalyResult,
    top_n: int = 15,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    by_category: bool = True,
) -> Optional[plt.Figure]:
    """
    Horizontal bar chart of top feature importances.

    If *by_category* is True (default), bars are coloured by feature
    category (power / spectral / band / coherence).  Otherwise a
    sequential colourmap maps importance values.
    """
    if result.feature_importances is None:
        return None

    apply_style()

    names = result.feature_names or [
        f"Feature {i}" for i in range(len(result.feature_importances))
    ]

    idx = np.argsort(result.feature_importances)[::-1][:top_n]
    top_names = [names[i] for i in idx]
    top_values = result.feature_importances[idx]

    fig, ax = plt.subplots(figsize=(7.5, max(3.8, top_n * 0.32)))

    # Reverse for bottom→top ordering
    y_pos = range(len(top_names))
    disp_names = top_names[::-1]
    disp_vals = top_values[::-1]

    if by_category:
        colors = [CATEGORY_COLORS.get(get_feature_category(n), "#6366f1") for n in disp_names]
    else:
        colors = categorical_cmap(disp_vals, cmap_name="YlOrRd")

    bars = ax.barh(
        y_pos, disp_vals,
        color=colors, edgecolor="white", linewidth=0.5, height=0.72,
    )

    ax.set_yticks(range(len(disp_names)))
    ax.set_yticklabels(disp_names, fontsize=8.5, fontfamily="monospace")
    ax.set_xlabel("Importance")
    ax.set_title(title or f"Feature Importance — {result.method}")

    # Value labels on bars
    max_val = disp_vals.max()
    for bar, val in zip(bars, disp_vals):
        if val > 0.3 * max_val:
            ax.text(
                val - 0.005 * max_val, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="right", fontsize=7.5,
                color="white", fontweight="medium",
            )
        else:
            ax.text(
                val + 0.008 * max_val, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=7.5,
                color="#475569",
            )

    # Category legend
    if by_category:
        legend_handles = [
            mpatches.Patch(color=c, label=cat.title())
            for cat, c in CATEGORY_COLORS.items()
        ]
        ax.legend(
            handles=legend_handles, loc="lower right",
            fontsize=8, framealpha=0.9, title="Category", title_fontsize=8.5,
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
#  Normal vs Anomalous Spectra  (mean ± std envelopes + annotations)
# --------------------------------------------------------------------------- #


def plot_anomaly_spectra(
    freq: np.ndarray,
    spectra_normal,
    spectra_anomaly,
    channel: int = 0,
    title: Optional[str] = None,
    annotations: Optional[list[dict]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Overlay normal vs anomalous auto-spectra using **mean ± 1 std shaded
    envelopes** — much cleaner than plotting every individual trace.

    Parameters
    ----------
    freq : np.ndarray
    spectra_normal : list of np.ndarray  (each shape N_freq × M × M)
    spectra_anomaly : list of np.ndarray
    channel : int
    annotations : list of dict, optional
        Each dict has keys ``freq`` (Hz), ``label`` (str), and optionally
        ``offset_db`` (vertical nudge for label placement).
    """
    apply_style()

    if isinstance(spectra_normal, np.ndarray) and spectra_normal.ndim == 3:
        spectra_normal = [spectra_normal]
    if isinstance(spectra_anomaly, np.ndarray) and spectra_anomaly.ndim == 3:
        spectra_anomaly = [spectra_anomaly]

    half = len(freq) // 2
    f = freq[1:half]

    def _extract_db(spectra_list):
        all_db = []
        for s in spectra_list:
            psd = np.real(s[1:half, channel, channel])
            psd_db = 10 * np.log10(np.maximum(psd, 1e-30))
            all_db.append(psd_db)
        return np.array(all_db)

    normal_db = _extract_db(spectra_normal)
    anomaly_db = _extract_db(spectra_anomaly)

    fig, ax = plt.subplots(figsize=(9, 5))

    # ── Normal envelope ──────────────────────────────────────────────
    n_mean = np.mean(normal_db, axis=0)
    n_std = np.std(normal_db, axis=0)
    ax.semilogx(f, n_mean, color=NORMAL_COLOR, linewidth=1.5, label="Normal (mean)")
    ax.fill_between(
        f, n_mean - n_std, n_mean + n_std,
        color=NORMAL_COLOR, alpha=0.18, linewidth=0,
        label="Normal (±1σ)",
    )

    # ── Anomaly envelope ─────────────────────────────────────────────
    a_mean = np.mean(anomaly_db, axis=0)
    a_std = np.std(anomaly_db, axis=0)
    ax.semilogx(f, a_mean, color=ANOMALY_COLOR, linewidth=1.5, label="Anomaly (mean)")
    ax.fill_between(
        f, a_mean - a_std, a_mean + a_std,
        color=ANOMALY_COLOR, alpha=0.18, linewidth=0,
        label="Anomaly (±1σ)",
    )

    # ── Annotations (e.g. tonal peaks, separation humps) ────────────
    if annotations:
        for ann in annotations:
            af = ann["freq"]
            lbl = ann["label"]
            offset = ann.get("offset_db", 3)
            # Find nearest frequency
            fidx = np.argmin(np.abs(f - af))
            y_peak = max(a_mean[fidx], n_mean[fidx])
            # Place label below the peak to avoid title/legend area
            y_text = y_peak - offset
            ax.annotate(
                lbl,
                xy=(f[fidx], y_peak),
                xytext=(f[fidx] * 0.55, y_text),
                fontsize=8.5, fontweight="medium", color="#1E293B",
                arrowprops=dict(
                    arrowstyle="-|>", color="#64748B",
                    connectionstyle="arc3,rad=-0.15", lw=0.8,
                ),
                bbox=dict(
                    boxstyle="round,pad=0.3", fc="white",
                    ec="#CBD5E1", alpha=0.9,
                ),
            )

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [dB]")
    ax.set_title(title or f"Normal vs Anomalous Auto-Spectra — Ch {channel + 1}")
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
#  Method Comparison  (normalised scores, unified layout)
# --------------------------------------------------------------------------- #


def plot_method_comparison(
    results: dict[str, AnomalyResult],
    normalise: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side comparison of anomaly scores from multiple methods.

    When *normalise* is True (default), each method's scores are mapped to
    [0, 1] via min–max scaling, so all panels share a comparable y-axis.
    """
    apply_style()

    methods = list(results.keys())
    n_methods = len(methods)
    method_colors = [CHANNEL_COLORS[i % len(CHANNEL_COLORS)] for i in range(n_methods)]

    fig, axes = plt.subplots(
        1, n_methods, figsize=(4.2 * n_methods, 4.5), sharey=normalise,
    )
    if n_methods == 1:
        axes = [axes]

    for ax, method, mc in zip(axes, methods, method_colors):
        r = results[method]
        n = len(r.scores)
        x = np.arange(n)

        scores = r.scores.copy()
        threshold = r.threshold

        if normalise:
            s_min, s_max = scores.min(), scores.max()
            rng = s_max - s_min if s_max != s_min else 1.0
            scores = (scores - s_min) / rng
            threshold = (r.threshold - s_min) / rng

        normal = r.labels == 1
        anomaly = r.labels == -1

        # Shaded zones
        y_lo = min(scores.min(), 0) - 0.05
        y_hi = max(scores.max(), 1) + 0.05 if normalise else scores.max() * 1.08
        ax.axhspan(threshold, y_hi, color=ANOMALY_COLOR, alpha=0.05, zorder=0)
        ax.axhspan(y_lo, threshold, color=NORMAL_COLOR, alpha=0.03, zorder=0)

        # Threshold
        ax.axhline(threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.0, zorder=5)

        # Normal
        ax.scatter(
            x[normal], scores[normal], c=mc, s=22, alpha=0.55, zorder=3,
            edgecolors="white", linewidths=0.3,
        )
        # Anomalies
        ax.scatter(
            x[anomaly], scores[anomaly], c=ANOMALY_COLOR, s=45, alpha=0.9,
            zorder=4, marker="D", edgecolors="#7F1D1D", linewidths=0.6,
        )

        # Title: method name + count
        nice_name = r.method.replace("_", " ").title()
        nice_name = nice_name.replace("Lof", "LOF").replace("Cmf", "CMF")
        ax.set_title(f"{nice_name}\n({r.n_anomalies} detected)", fontsize=10.5)
        ax.set_xlabel("Sample")
        ax.set_xlim(-0.8, n + 0.5)
        if normalise:
            ax.set_ylim(-0.08, 1.12)

    axes[0].set_ylabel("Normalised Score" if normalise else "Anomaly Score")
    fig.suptitle(
        title or "Anomaly Detection — Method Comparison",
        fontsize=13, fontweight="semibold",
    )
    fig.subplots_adjust(top=0.85, wspace=0.15)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
#  Confusion-style Summary Bar
# --------------------------------------------------------------------------- #


def plot_detection_summary(
    results: dict[str, AnomalyResult],
    true_labels: np.ndarray,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing precision, recall, and F1 across methods.
    """
    apply_style()

    methods = list(results.keys())
    metrics = {"Precision": [], "Recall": [], "F1": []}

    for method in methods:
        r = results[method]
        tp = np.sum((r.labels == -1) & (true_labels == -1))
        fp = np.sum((r.labels == -1) & (true_labels == 1))
        fn = np.sum((r.labels == 1) & (true_labels == -1))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        metrics["Precision"].append(prec)
        metrics["Recall"].append(rec)
        metrics["F1"].append(f1)

    x = np.arange(len(methods))
    width = 0.22
    colors = [NORMAL_COLOR, ANOMALY_COLOR, THRESHOLD_COLOR]

    fig, ax = plt.subplots(figsize=(max(5, 2.5 * len(methods)), 4))

    for i, (metric_name, values) in enumerate(metrics.items()):
        bars = ax.bar(
            x + i * width, values, width,
            label=metric_name, color=colors[i], edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="medium",
            )

    nice_names = [m.replace("_", " ").title() for m in methods]
    # Fix common acronyms
    nice_names = [n.replace("Lof", "LOF").replace("Cmf", "CMF") for n in nice_names]
    ax.set_xticks(x + width)
    ax.set_xticklabels(nice_names, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.18)
    ax.set_title(title or "Detection Performance Summary")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
