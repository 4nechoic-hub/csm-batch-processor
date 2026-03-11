"""
Cross-Spectral Matrix (CSM) Calculator
========================================
Direct port of CSM_Calculator.m

Computes the cross-spectral matrix from multi-channel time-series data
using Welch's block-averaging method with a Hanning window.

Parameters
----------
data : np.ndarray, shape (N, M)
    Time-series data matrix. N = number of temporal samples, M = number of
    channels. A single-channel signal should be shape (N, 1).
fs : float
    Sampling rate in Hz.
n_rec : int
    Number of samples per block (record length / FFT segment size).
overlap : float
    Overlap percentage (0–100). E.g. 50 means 50 % overlap.

Returns
-------
spectra : np.ndarray, shape (n_rec, M, M)
    Cross-spectral density matrix.  spectra[:, i, j] gives the cross-spectrum
    between channel *i* and channel *j*.  Auto-spectra live on the diagonal
    spectra[:, i, i].
freq : np.ndarray, shape (n_rec,)
    Corresponding frequency vector in Hz.
"""

import numpy as np


def csm_calculator(
    data: np.ndarray,
    fs: float,
    n_rec: int,
    overlap: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the narrowband cross-spectral matrix."""

    data = np.atleast_2d(data)
    if data.shape[0] < data.shape[1]:
        # Assume user passed (M, N) instead of (N, M)
        data = data.T

    n_samples, n_channels = data.shape

    # --- Overlap & block count ------------------------------------------------
    overlap_frac = overlap / 100.0
    n_blocks = int(np.floor((n_samples - n_rec) / (n_rec * (1 - overlap_frac))) + 1)
    if n_blocks < 1:
        raise ValueError(
            f"Not enough data ({n_samples} samples) for even one block of "
            f"length {n_rec} with {overlap}% overlap."
        )

    # --- Hanning window (periodic, energy-normalised) -------------------------
    # MATLAB: hanning(N,'periodic') / sqrt(0.375)
    window = np.hanning(n_rec + 1)[:-1]  # periodic Hanning
    window /= np.sqrt(0.375)  # energy normalisation

    # --- Block loop (vectorised inner product) --------------------------------
    csm_sum = np.zeros((n_rec, n_channels, n_channels), dtype=complex)
    step = int(n_rec * (1 - overlap_frac))

    for n in range(n_blocks):
        start = int(np.floor(n * n_rec * (1 - overlap_frac)))
        block = data[start : start + n_rec, :] * window[:, np.newaxis]  # (n_rec, M)
        S = np.fft.fft(block, axis=0)  # (n_rec, M)

        # Outer product at each frequency: csm[f, i, j] = S[f,i] * conj(S[f,j])
        # Equivalent to the MATLAB reshape + permute + element-wise multiply
        csm_sum += S[:, :, np.newaxis] * np.conj(S[:, np.newaxis, :])

    spectra = 2.0 * csm_sum / (n_rec * fs * n_blocks)

    # --- Frequency vector -----------------------------------------------------
    df = fs / n_rec
    freq = np.arange(n_rec) * df  # 0, df, 2*df, …, fs - df

    return spectra, freq
