"""
Auto- and Cross-Correlation
=============================
Computes normalised auto- and cross-correlation functions for multi-channel
time-series data, mirroring the MATLAB ``xcorr(…, 'normalized')`` behaviour.

Parameters
----------
data : np.ndarray, shape (N, M)
    Time-series data (N samples × M channels).
fs : float
    Sampling rate in Hz.

Returns
-------
tau : np.ndarray
    Time-lag vector in seconds.
corr_matrix : np.ndarray, shape (2*N-1, M, M)
    Normalised correlation matrix. ``corr_matrix[:, i, j]`` is the
    cross-correlation between channels *i* and *j*.
"""

import numpy as np
from scipy.signal import correlate


def compute_correlation(
    data: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalised auto/cross-correlation for all channel pairs."""

    data = np.atleast_2d(data)
    if data.shape[0] < data.shape[1]:
        data = data.T

    n_samples, n_channels = data.shape
    n_corr = 2 * n_samples - 1

    # Time-lag vector
    dt = 1.0 / fs
    tau = np.arange(-(n_samples - 1), n_samples) * dt

    corr_matrix = np.zeros((n_corr, n_channels, n_channels))

    for i in range(n_channels):
        for j in range(n_channels):
            cc = correlate(data[:, i], data[:, j], mode="full")
            # Normalise (equivalent to MATLAB 'normalized')
            norm = np.sqrt(np.sum(data[:, i] ** 2) * np.sum(data[:, j] ** 2))
            corr_matrix[:, i, j] = cc / norm if norm > 0 else cc

    return tau, corr_matrix
