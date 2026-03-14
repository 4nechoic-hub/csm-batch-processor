#!/usr/bin/env python3
"""
Anomaly Detection Demo
=======================
End-to-end demonstration of the CSM → Feature Extraction → Anomaly Detection
pipeline using synthetic aeroacoustic data.

Scenario
--------
Simulates a microphone array monitoring flow over an airfoil:
  - **Normal**: broadband turbulent boundary layer noise + tonal trailing
    edge noise at 2 kHz.
  - **Anomalous**: flow separation event causes a spectral hump at 800 Hz,
    increased broadband levels, and loss of coherence between sensors.

This is representative of real-world aeroacoustic monitoring where spectral
signatures change when flow conditions deteriorate.

Usage
-----
    python examples/anomaly_demo.py

Outputs (8 figures)
-------------------
    examples/output/anomaly_scores.png
    examples/output/feature_importance.png
    examples/output/normal_vs_anomaly_spectra.png
    examples/output/method_comparison.png
    examples/output/csm_matrix_normal.png
    examples/output/csm_matrix_anomaly.png
    examples/output/coherence_matrix_comparison.png
    examples/output/detection_summary.png
"""

import os
import sys
import numpy as np

# Add parent to path if running from examples/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from csm_processor import csm_calculator, extract_features_batch
from csm_processor.anomaly_detection import SpectralAnomalyDetector, compare_methods
from csm_processor.anomaly_plotting import (
    plot_anomaly_scores,
    plot_feature_importance,
    plot_anomaly_spectra,
    plot_method_comparison,
    plot_detection_summary,
)
from csm_processor.plotting import (
    plot_csm_matrix,
    plot_coherence_matrix,
)


def generate_normal_signal(fs, duration, n_channels, seed):
    """Simulate normal aeroacoustic data: broadband + 2 kHz tone."""
    rng = np.random.RandomState(seed)
    N = int(fs * duration)
    t = np.arange(N) / fs

    data = np.zeros((N, n_channels))
    for ch in range(n_channels):
        # Broadband turbulent BL noise (shaped: rolls off at high freq)
        noise = rng.randn(N)
        # Simple low-pass shaping via cumulative sum + normalisation
        shaped = np.cumsum(noise)
        shaped = shaped / np.std(shaped) * 0.3

        # Tonal trailing edge noise at 2 kHz (coherent across channels)
        tone = 0.8 * np.sin(2 * np.pi * 2000 * t + rng.uniform(0, 0.3))

        # Add some channel-specific broadband
        data[:, ch] = shaped + tone + 0.1 * rng.randn(N)

    return data


def generate_anomalous_signal(fs, duration, n_channels, seed):
    """Simulate anomalous data: flow separation with spectral hump at 800 Hz."""
    rng = np.random.RandomState(seed)
    N = int(fs * duration)
    t = np.arange(N) / fs

    data = np.zeros((N, n_channels))
    for ch in range(n_channels):
        # Elevated broadband (flow separation increases turbulence)
        noise = rng.randn(N)
        shaped = np.cumsum(noise)
        shaped = shaped / np.std(shaped) * 0.6  # 2x the normal level

        # Original tone weakened (separation disrupts trailing edge mechanism)
        tone = 0.2 * np.sin(2 * np.pi * 2000 * t + rng.uniform(0, 0.5))

        # New spectral hump at 800 Hz (vortex shedding from separation)
        hump = 1.2 * np.sin(2 * np.pi * 800 * t + rng.uniform(0, 2 * np.pi))
        hump += 0.4 * np.sin(2 * np.pi * 820 * t + rng.uniform(0, 2 * np.pi))

        # Channels less correlated (separation is spatially incoherent)
        data[:, ch] = shaped + tone + hump + 0.4 * rng.randn(N)

    return data


def main():
    # ── Parameters ───────────────────────────────────────────────────
    fs = 20000       # 20 kHz sampling
    duration = 0.5   # 0.5 sec per snapshot
    n_channels = 3   # 3-mic array
    n_rec = 2048     # FFT block size
    overlap = 50.0   # 50% overlap

    n_normal = 80    # baseline snapshots
    n_anomaly = 12   # anomalous snapshots

    outdir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(outdir, exist_ok=True)

    print("=" * 60)
    print("CSM Anomaly Detection Demo")
    print("=" * 60)
    print(f"  Channels:    {n_channels}")
    print(f"  fs:          {fs} Hz")
    print(f"  Nrec:        {n_rec}")
    print(f"  Normal:      {n_normal} snapshots")
    print(f"  Anomalous:   {n_anomaly} snapshots")
    print()

    # ── Step 1: Generate synthetic data & compute CSMs ───────────────
    print("[1/5] Generating synthetic data and computing CSMs...")

    normal_csms = []
    for i in range(n_normal):
        data = generate_normal_signal(fs, duration, n_channels, seed=i)
        spectra, freq = csm_calculator(data, fs, n_rec, overlap)
        normal_csms.append(spectra)

    anomaly_csms = []
    for i in range(n_anomaly):
        data = generate_anomalous_signal(fs, duration, n_channels, seed=1000 + i)
        spectra, freq = csm_calculator(data, fs, n_rec, overlap)
        anomaly_csms.append(spectra)

    print(f"  ✓ {n_normal} normal CSMs, {n_anomaly} anomalous CSMs")

    # ── Step 2: Extract features ─────────────────────────────────────
    print("[2/5] Extracting spectral features...")

    X_normal, feature_names = extract_features_batch(normal_csms, freq, fs)
    X_anomaly, _ = extract_features_batch(anomaly_csms, freq, fs)

    print(f"  ✓ {X_normal.shape[1]} features per snapshot")
    print(f"  ✓ Normal feature matrix:  {X_normal.shape}")
    print(f"  ✓ Anomaly feature matrix: {X_anomaly.shape}")

    # ── Step 3: Fit detector & predict ───────────────────────────────
    print("[3/5] Fitting anomaly detectors...")

    # Combined test set (normal + anomalous, shuffled)
    X_test = np.vstack([X_normal[60:], X_anomaly])  # hold out 20 normal for test
    true_labels = np.array([1] * 20 + [-1] * n_anomaly)

    # Use first 60 normal samples as training baseline
    X_train = X_normal[:60]

    # Single method (Isolation Forest)
    detector = SpectralAnomalyDetector(method="isolation_forest", contamination=0.15)
    detector.fit(X_train, feature_names=feature_names)
    result_if = detector.predict(X_test)

    print(f"\n  {result_if.summary()}")

    # Accuracy check
    correct = np.sum(result_if.labels == true_labels)
    print(f"\n  Accuracy: {correct}/{len(true_labels)} ({correct/len(true_labels):.1%})")

    # Compare all methods
    all_results = compare_methods(X_train, X_test, feature_names, contamination=0.15)

    for method, r in all_results.items():
        correct = np.sum(r.labels == true_labels)
        print(f"  {r.method}: {r.n_anomalies} detected, accuracy {correct/len(true_labels):.1%}")

    # ── Step 4: Generate spectral and matrix plots ───────────────────
    print("\n[4/5] Generating spectral & matrix visualisations...")

    # Normal vs anomalous spectra with annotations
    fig = plot_anomaly_spectra(
        freq,
        normal_csms[:10],
        anomaly_csms[:5],
        channel=0,
        title="Normal vs Anomalous Auto-Spectra — Ch 1",
        annotations=[
            {"freq": 800, "label": "Separation hump\n(~800 Hz)", "offset_db": 8},
            {"freq": 2000, "label": "TE tone (2 kHz)", "offset_db": 12},
        ],
    )
    fig.savefig(os.path.join(outdir, "normal_vs_anomaly_spectra.png"), dpi=300)
    print("  ✓ normal_vs_anomaly_spectra.png")

    # CSM matrix heatmaps — normal vs anomalous at 800 Hz
    # Average a few CSMs for a cleaner matrix
    avg_normal_csm = np.mean(normal_csms[:10], axis=0)
    avg_anomaly_csm = np.mean(anomaly_csms[:5], axis=0)

    fig = plot_csm_matrix(
        avg_normal_csm, freq, target_freq=800,
        title="CSM Magnitude — Normal (800 Hz)",
    )
    fig.savefig(os.path.join(outdir, "csm_matrix_normal.png"), dpi=300)
    print("  ✓ csm_matrix_normal.png")

    fig = plot_csm_matrix(
        avg_anomaly_csm, freq, target_freq=800,
        title="CSM Magnitude — Anomalous (800 Hz)",
    )
    fig.savefig(os.path.join(outdir, "csm_matrix_anomaly.png"), dpi=300)
    print("  ✓ csm_matrix_anomaly.png")

    # Coherence matrix side-by-side at the tonal frequency
    import matplotlib.pyplot as plt
    from csm_processor.style import apply_style
    apply_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Normal coherence at 2 kHz
    idx_2k = np.argmin(np.abs(freq - 2000))
    csm_n = avg_normal_csm[idx_2k, :, :]
    csm_a = avg_anomaly_csm[idx_2k, :, :]

    def _coh_matrix(csm_slice):
        n = csm_slice.shape[0]
        coh = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                denom = np.real(csm_slice[i, i]) * np.real(csm_slice[j, j])
                coh[i, j] = np.abs(csm_slice[i, j])**2 / max(denom, 1e-30)
        return coh

    coh_n = _coh_matrix(csm_n)
    coh_a = _coh_matrix(csm_a)
    ch_labels = [f"Ch {i+1}" for i in range(n_channels)]

    for ax, coh, label in [(ax1, coh_n, "Normal"), (ax2, coh_a, "Anomalous")]:
        im = ax.imshow(coh, origin="lower", cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(n_channels))
        ax.set_xticklabels(ch_labels, fontsize=8.5)
        ax.set_yticks(range(n_channels))
        ax.set_yticklabels(ch_labels, fontsize=8.5)
        ax.set_title(f"{label}", fontsize=11)
        # Annotate
        for i in range(n_channels):
            for j in range(n_channels):
                val = coh[i, j]
                tc = "white" if val > 0.65 else "#334155"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=tc, fontweight="medium")

    fig.suptitle(
        f"Coherence Matrix $\\gamma^2$ at {freq[idx_2k]:.0f} Hz",
        fontsize=12, fontweight="semibold",
    )
    cbar = fig.colorbar(im, ax=[ax1, ax2], shrink=0.85, pad=0.03)
    cbar.set_label(r"$\gamma^2$", fontsize=10)
    fig.subplots_adjust(top=0.88, wspace=0.25)
    fig.savefig(os.path.join(outdir, "coherence_matrix_comparison.png"), dpi=300, bbox_inches="tight")
    print("  ✓ coherence_matrix_comparison.png")

    # ── Step 5: Generate anomaly detection plots ─────────────────────
    print("\n[5/5] Generating anomaly detection visualisations...")

    # Anomaly scores
    fig = plot_anomaly_scores(result_if, title="Anomaly Scores — Wind Tunnel Monitoring")
    fig.savefig(os.path.join(outdir, "anomaly_scores.png"), dpi=300)
    print("  ✓ anomaly_scores.png")

    # Feature importance (Mahalanobis has the clearest importances)
    result_mah = all_results["mahalanobis"]
    fig = plot_feature_importance(result_mah, top_n=15)
    if fig:
        fig.savefig(os.path.join(outdir, "feature_importance.png"), dpi=300)
        print("  ✓ feature_importance.png")

    # Method comparison (normalised)
    fig = plot_method_comparison(all_results, normalise=True)
    fig.savefig(os.path.join(outdir, "method_comparison.png"), dpi=300)
    print("  ✓ method_comparison.png")

    # Detection performance summary
    fig = plot_detection_summary(all_results, true_labels)
    fig.savefig(os.path.join(outdir, "detection_summary.png"), dpi=300)
    print("  ✓ detection_summary.png")

    print(f"\n✅ All outputs saved to {outdir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
