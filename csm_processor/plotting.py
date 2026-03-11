"""
Plotting Utilities
===================
Publication-quality spectral and correlation plots using Matplotlib.
"""

from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
#  Style defaults
# --------------------------------------------------------------------------- #

_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
}


def _apply_style():
    plt.rcParams.update(_STYLE)


# --------------------------------------------------------------------------- #
#  Auto-spectra
# --------------------------------------------------------------------------- #


def plot_autospectra(
    freq: np.ndarray,
    spectra: np.ndarray,
    channels: Optional[list[int]] = None,
    db_ref: float = 1.0,
    log_freq: bool = True,
    title: str = "Auto-Spectra",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot auto-spectral densities (diagonal of CSM) in dB.

    Parameters
    ----------
    freq : array, shape (N_freq,)
    spectra : array, shape (N_freq, M, M)
    channels : list of int, optional
        Which channels to plot (0-indexed). Default: all.
    db_ref : float
        Reference value for dB conversion.
    """
    _apply_style()
    n_channels = spectra.shape[1]
    channels = channels or list(range(n_channels))

    fig, ax = plt.subplots(figsize=(10, 5))

    for ch in channels:
        psd = np.real(spectra[:, ch, ch])
        psd_db = 10 * np.log10(np.maximum(psd, 1e-30) / db_ref**2)
        label = f"Ch {ch + 1}"
        if log_freq:
            ax.semilogx(freq[1:], psd_db[1:], label=label, linewidth=1.2)
        else:
            ax.plot(freq[1:], psd_db[1:], label=label, linewidth=1.2)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [dB re {:.0e}]".format(db_ref))
    ax.set_title(title)
    ax.legend(fontsize=9, ncol=min(4, len(channels)))
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


# --------------------------------------------------------------------------- #
#  Cross-spectra (magnitude + phase)
# --------------------------------------------------------------------------- #


def plot_cross_spectra(
    freq: np.ndarray,
    spectra: np.ndarray,
    ch_i: int,
    ch_j: int,
    db_ref: float = 1.0,
    log_freq: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot cross-spectrum magnitude (dB) and phase between two channels."""
    _apply_style()

    csd = spectra[:, ch_i, ch_j]
    mag_db = 10 * np.log10(np.maximum(np.abs(csd), 1e-30) / db_ref**2)
    phase_deg = np.angle(csd, deg=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    f = freq[1:]
    plot_fn = ax1.semilogx if log_freq else ax1.plot

    plot_fn(f, mag_db[1:], color="#2563eb", linewidth=1.2)
    ax1.set_ylabel("Magnitude [dB]")
    ax1.set_title(title or f"Cross-Spectrum  Ch {ch_i+1} × Ch {ch_j+1}")

    plot_fn2 = ax2.semilogx if log_freq else ax2.plot
    plot_fn2(f, phase_deg[1:], color="#dc2626", linewidth=1.0)
    ax2.set_ylabel("Phase [°]")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylim(-180, 180)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
#  Coherence
# --------------------------------------------------------------------------- #


def plot_coherence(
    freq: np.ndarray,
    spectra: np.ndarray,
    ch_i: int,
    ch_j: int,
    log_freq: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot magnitude-squared coherence γ² between two channels.

    γ²(f) = |G_ij(f)|² / (G_ii(f) · G_jj(f))
    """
    _apply_style()

    Gij = spectra[:, ch_i, ch_j]
    Gii = np.real(spectra[:, ch_i, ch_i])
    Gjj = np.real(spectra[:, ch_j, ch_j])

    denom = Gii * Gjj
    coh2 = np.abs(Gij) ** 2 / np.maximum(denom, 1e-30)

    fig, ax = plt.subplots(figsize=(10, 4))
    f = freq[1:]
    plot_fn = ax.semilogx if log_freq else ax.plot
    plot_fn(f, coh2[1:], color="#7c3aed", linewidth=1.2)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Coherence γ²")
    ax.set_ylim(0, 1.05)
    ax.set_title(title or f"Coherence  Ch {ch_i+1} – Ch {ch_j+1}")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
#  Correlation
# --------------------------------------------------------------------------- #


def plot_correlation(
    tau: np.ndarray,
    corr_matrix: np.ndarray,
    channels: Optional[list[int]] = None,
    auto_only: bool = True,
    title: str = "Auto-Correlation",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot normalised auto- (or cross-) correlation functions."""
    _apply_style()
    n_channels = corr_matrix.shape[1]
    channels = channels or list(range(n_channels))

    fig, ax = plt.subplots(figsize=(10, 5))

    if auto_only:
        for ch in channels:
            ax.plot(tau * 1e3, corr_matrix[:, ch, ch], label=f"Ch {ch+1}", linewidth=1.0)
    else:
        for i in channels:
            for j in channels:
                if i <= j:
                    label = f"Ch {i+1}–{j+1}" if i != j else f"Ch {i+1} auto"
                    ax.plot(tau * 1e3, corr_matrix[:, i, j], label=label, linewidth=0.9)

    ax.set_xlabel("Lag τ [ms]")
    ax.set_ylabel("Normalised Correlation")
    ax.set_title(title)
    ax.legend(fontsize=9, ncol=min(4, len(channels)))
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
