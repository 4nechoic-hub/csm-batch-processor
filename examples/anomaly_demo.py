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

Outputs
-------
    examples/output/anomaly_scores.png
    examples/output/feature_importance.png
    examples/output/normal_vs_anomaly_spectra.png
    examples/output/method_comparison.png
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
    print("[1/4] Generating synthetic data and computing CSMs...")

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
    print("[2/4] Extracting spectral features...")

    X_normal, feature_names = extract_features_batch(normal_csms, freq, fs)
    X_anomaly, _ = extract_features_batch(anomaly_csms, freq, fs)

    print(f"  ✓ {X_normal.shape[1]} features per snapshot")
    print(f"  ✓ Normal feature matrix:  {X_normal.shape}")
    print(f"  ✓ Anomaly feature matrix: {X_anomaly.shape}")

    # ── Step 3: Fit detector & predict ───────────────────────────────
    print("[3/4] Fitting anomaly detectors...")

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

    # ── Step 4: Generate plots ───────────────────────────────────────
    print("\n[4/4] Generating plots...")

    # Anomaly scores
    fig = plot_anomaly_scores(result_if, title="Anomaly Scores — Wind Tunnel Monitoring")
    fig.savefig(os.path.join(outdir, "anomaly_scores.png"), dpi=200)
    print(f"  ✓ anomaly_scores.png")

    # Feature importance (Mahalanobis has the clearest importances)
    result_mah = all_results["mahalanobis"]
    fig = plot_feature_importance(result_mah, top_n=15)
    if fig:
        fig.savefig(os.path.join(outdir, "feature_importance.png"), dpi=200)
        print(f"  ✓ feature_importance.png")

    # Normal vs anomalous spectra overlay
    fig = plot_anomaly_spectra(
        freq,
        normal_csms[:10],
        anomaly_csms[:5],
        channel=0,
        title="Normal vs Anomalous Auto-Spectra — Ch 1",
    )
    fig.savefig(os.path.join(outdir, "normal_vs_anomaly_spectra.png"), dpi=200)
    print(f"  ✓ normal_vs_anomaly_spectra.png")

    # Method comparison
    fig = plot_method_comparison(all_results)
    fig.savefig(os.path.join(outdir, "method_comparison.png"), dpi=200)
    print(f"  ✓ method_comparison.png")

    print(f"\n✅ All outputs saved to {outdir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()