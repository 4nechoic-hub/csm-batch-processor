"""
Spectral Feature Extraction
=============================
Extracts ML-ready features from cross-spectral matrix outputs.

Designed for aeroacoustic and vibration monitoring applications where
changes in spectral shape, energy distribution, or inter-channel coherence
can indicate anomalous conditions (e.g. flow separation, structural
resonance, sensor degradation).

Each feature function takes a CSM output and returns a 1-D feature vector
suitable for downstream ML models.
"""

import numpy as np
from typing import Optional


def extract_features(
    spectra: np.ndarray,
    freq: np.ndarray,
    fs: float,
    bands: Optional[list[tuple[float, float]]] = None,
) -> dict[str, float]:
    """
    Extract a comprehensive feature set from a CSM.

    Parameters
    ----------
    spectra : np.ndarray, shape (N_freq, M, M)
        Cross-spectral matrix.
    freq : np.ndarray, shape (N_freq,)
        Frequency vector in Hz.
    fs : float
        Sampling rate in Hz.
    bands : list of (f_low, f_high) tuples, optional
        Frequency bands for band-energy features.
        Default: [(0, fs/8), (fs/8, fs/4), (fs/4, 3*fs/8), (3*fs/8, fs/2)]

    Returns
    -------
    features : dict[str, float]
        Named feature dictionary. Keys follow the pattern
        ``{feature_type}_ch{channel}`` for single-channel features and
        ``{feature_type}_ch{i}_ch{j}`` for cross-channel features.
    """
    n_freq, n_ch, _ = spectra.shape
    half = n_freq // 2
    freq_h = freq[:half]
    features = {}

    if bands is None:
        nyq = fs / 2
        bands = [
            (0, nyq / 4),
            (nyq / 4, nyq / 2),
            (nyq / 2, 3 * nyq / 4),
            (3 * nyq / 4, nyq),
        ]

    for ch in range(n_ch):
        psd = np.real(spectra[:half, ch, ch])
        psd_pos = np.maximum(psd, 1e-30)

        # ── Broadband features ──────────────────────────────────────
        total_power = np.sum(psd_pos)
        features[f"total_power_ch{ch}"] = total_power

        # Overall SPL (dB)
        features[f"oaspl_ch{ch}"] = 10 * np.log10(total_power)

        # Peak frequency and magnitude
        peak_idx = np.argmax(psd_pos)
        features[f"peak_freq_ch{ch}"] = freq_h[peak_idx]
        features[f"peak_psd_db_ch{ch}"] = 10 * np.log10(psd_pos[peak_idx])

        # Spectral centroid (centre of mass)
        features[f"spectral_centroid_ch{ch}"] = (
            np.sum(freq_h * psd_pos) / total_power
        )

        # Spectral bandwidth (spread around centroid)
        centroid = features[f"spectral_centroid_ch{ch}"]
        features[f"spectral_bandwidth_ch{ch}"] = np.sqrt(
            np.sum(((freq_h - centroid) ** 2) * psd_pos) / total_power
        )

        # Spectral slope (linear regression of log-PSD vs log-freq)
        valid = freq_h > 0
        if np.sum(valid) > 2:
            log_f = np.log10(freq_h[valid])
            log_p = np.log10(psd_pos[valid])
            coeffs = np.polyfit(log_f, log_p, 1)
            features[f"spectral_slope_ch{ch}"] = coeffs[0]
        else:
            features[f"spectral_slope_ch{ch}"] = 0.0

        # Spectral flatness (Wiener entropy) — ratio of geometric to
        # arithmetic mean.  Flat (white noise) → 1, tonal → 0.
        log_mean = np.mean(np.log(psd_pos))
        arith_mean = np.mean(psd_pos)
        features[f"spectral_flatness_ch{ch}"] = (
            np.exp(log_mean) / arith_mean if arith_mean > 0 else 0.0
        )

        # Spectral crest factor (peak / RMS)
        rms = np.sqrt(np.mean(psd_pos**2))
        features[f"spectral_crest_ch{ch}"] = (
            np.max(psd_pos) / rms if rms > 0 else 0.0
        )

        # Spectral kurtosis (peakedness of distribution)
        mean_p = np.mean(psd_pos)
        std_p = np.std(psd_pos)
        if std_p > 0:
            features[f"spectral_kurtosis_ch{ch}"] = (
                np.mean(((psd_pos - mean_p) / std_p) ** 4) - 3.0
            )
        else:
            features[f"spectral_kurtosis_ch{ch}"] = 0.0

        # ── Band energy features ────────────────────────────────────
        for bi, (f_lo, f_hi) in enumerate(bands):
            mask = (freq_h >= f_lo) & (freq_h < f_hi)
            band_power = np.sum(psd_pos[mask]) if np.any(mask) else 0.0
            features[f"band{bi}_power_ch{ch}"] = band_power
            # Band energy ratio
            features[f"band{bi}_ratio_ch{ch}"] = (
                band_power / total_power if total_power > 0 else 0.0
            )

    # ── Cross-channel features ──────────────────────────────────────
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            Gij = spectra[:half, i, j]
            Gii = np.real(spectra[:half, i, i])
            Gjj = np.real(spectra[:half, j, j])
            denom = np.maximum(Gii * Gjj, 1e-30)
            coh2 = np.abs(Gij) ** 2 / denom

            # Mean coherence
            features[f"mean_coherence_ch{i}_ch{j}"] = np.mean(coh2)

            # Peak coherence
            features[f"peak_coherence_ch{i}_ch{j}"] = np.max(coh2)

            # Coherence-weighted phase (dominant phase relationship)
            phase = np.angle(Gij)
            features[f"weighted_phase_ch{i}_ch{j}"] = (
                np.sum(coh2 * phase) / np.sum(coh2) if np.sum(coh2) > 0 else 0.0
            )

    return features


def features_to_array(features: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    """
    Convert a feature dictionary to a NumPy array + ordered key list.

    Returns
    -------
    array : np.ndarray, shape (n_features,)
    names : list[str]
    """
    names = sorted(features.keys())
    array = np.array([features[k] for k in names])
    return array, names


def extract_features_batch(
    spectra_list: list[np.ndarray],
    freq: np.ndarray,
    fs: float,
    **kwargs,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract features from multiple CSM snapshots into a feature matrix.

    Parameters
    ----------
    spectra_list : list of np.ndarray
        Each element is a CSM array, shape (N_freq, M, M).
    freq : np.ndarray
    fs : float

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix suitable for scikit-learn.
    feature_names : list[str]
    """
    all_features = []
    names = None
    for spectra in spectra_list:
        feat_dict = extract_features(spectra, freq, fs, **kwargs)
        arr, n = features_to_array(feat_dict)
        all_features.append(arr)
        if names is None:
            names = n

    return np.vstack(all_features), names