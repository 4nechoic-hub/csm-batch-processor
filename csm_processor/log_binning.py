"""
Logarithmic Frequency Binning
===============================
Port of logfnan.m — combines narrowband spectral bins into fractional-octave
(logarithmically spaced) bands.

Parameters
----------
df : float
    Frequency spacing of the narrowband spectrum (Hz).
spectrum : np.ndarray
    1-D narrowband spectrum starting at *df* (i.e. DC component excluded).
    Length = number of narrowband bins.
bins_per_octave : int
    Number of logarithmic bins per octave (e.g. 3 for 1/3-octave, 12 for
    1/12-octave).

Returns
-------
freq_binned : np.ndarray
    Centre frequencies of the logarithmic bins (Hz).
spectrum_binned : np.ndarray
    Bin-averaged spectral values, with NaN where no valid data exist.
"""

import numpy as np


def log_freq_bin(
    df: float,
    spectrum: np.ndarray,
    bins_per_octave: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine narrowband spectrum into fractional-octave bands."""

    spectrum = np.asarray(spectrum).ravel()
    n_bins_log = int(np.floor(np.log2(len(spectrum)) * bins_per_octave))

    # Linear frequency vector (starting at df, no DC)
    freq_linear = np.arange(1, len(spectrum) + 1) * df

    # Logarithmically spaced bin edges (as indices into freq_linear)
    m = np.round(2.0 ** (np.arange(1, n_bins_log + 1) / bins_per_octave)).astype(int)
    m = np.unique(m)
    n = len(m)

    freq_binned = np.empty(n - 1)
    spectrum_binned = np.empty(n - 1, dtype=complex)

    for j in range(n - 1):
        lo, hi = m[j] - 1, m[j + 1] - 1  # convert to 0-based
        freq_binned[j] = np.mean(freq_linear[lo:hi])

        chunk = spectrum[lo:hi]
        valid = chunk[~np.isnan(chunk)]
        spectrum_binned[j] = np.mean(valid) if len(valid) >= 1 else np.nan

    return freq_binned, spectrum_binned


def bin_csm(
    df: float,
    spectra: np.ndarray,
    bins_per_octave: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply log-frequency binning to every element of a CSM.

    Parameters
    ----------
    df : float
        Narrowband frequency spacing (Hz).
    spectra : np.ndarray, shape (N_freq, M, M)
        Narrowband cross-spectral matrix (DC excluded from first axis).
    bins_per_octave : int
        Bins per octave for output.

    Returns
    -------
    freq_binned : np.ndarray
        Binned centre frequencies.
    spectra_binned : np.ndarray, shape (N_binned, M, M)
        Binned CSM.
    """
    n_channels = spectra.shape[1]
    freq_binned = None
    spectra_binned = None

    for i in range(n_channels):
        for j in range(n_channels):
            fb, sb = log_freq_bin(df, spectra[1:, i, j], bins_per_octave)
            if spectra_binned is None:
                spectra_binned = np.zeros(
                    (len(fb), n_channels, n_channels), dtype=complex
                )
                freq_binned = fb
            spectra_binned[:, i, j] = sb

    return freq_binned, spectra_binned
