"""
Plotting Utilities
===================
Publication-quality spectral, correlation, and matrix plots using Matplotlib.

Styling follows the unified configuration in :mod:`csm_processor.style`,
inspired by the clean scientific aesthetic of the `Acoular
<https://www.acoular.org/auto_examples/>`_ example gallery.
"""

from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from .style import apply_style, get_channel_color


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
    apply_style()
    n_channels = spectra.shape[1]
    channels = channels or list(range(n_channels))

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for ch in channels:
        psd = np.real(spectra[:, ch, ch])
        psd_db = 10 * np.log10(np.maximum(psd, 1e-30) / db_ref**2)
        color = get_channel_color(ch)
        label = f"Ch {ch + 1}"
        f = freq[1:]
        if log_freq:
            ax.semilogx(f, psd_db[1:], label=label, color=color, linewidth=1.3)
        else:
            ax.plot(f, psd_db[1:], label=label, color=color, linewidth=1.3)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(f"PSD [dB re {db_ref:.0e}]")
    ax.set_title(title)
    ax.legend(ncol=min(4, len(channels)), loc="best")
    ax.minorticks_on()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
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
    apply_style()

    csd = spectra[:, ch_i, ch_j]
    mag_db = 10 * np.log10(np.maximum(np.abs(csd), 1e-30) / db_ref**2)
    phase_deg = np.angle(csd, deg=True)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
    )

    f = freq[1:]
    plot_fn1 = ax1.semilogx if log_freq else ax1.plot
    plot_fn2 = ax2.semilogx if log_freq else ax2.plot

    plot_fn1(f, mag_db[1:], color=get_channel_color(0), linewidth=1.3)
    ax1.set_ylabel("Magnitude [dB]")
    ax1.set_title(title or f"Cross-Spectrum  Ch {ch_i+1} × Ch {ch_j+1}")
    ax1.minorticks_on()

    plot_fn2(f, phase_deg[1:], color=get_channel_color(1), linewidth=1.0, alpha=0.8)
    ax2.set_ylabel("Phase [°]")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylim(-180, 180)
    ax2.set_yticks([-180, -90, 0, 90, 180])
    ax2.minorticks_on()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
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
    r"""
    Plot magnitude-squared coherence γ² between two channels.

    γ²(f) = |G_ij(f)|² / (G_ii(f) · G_jj(f))
    """
    apply_style()

    Gij = spectra[:, ch_i, ch_j]
    Gii = np.real(spectra[:, ch_i, ch_i])
    Gjj = np.real(spectra[:, ch_j, ch_j])
    denom = Gii * Gjj
    coh2 = np.abs(Gij) ** 2 / np.maximum(denom, 1e-30)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    f = freq[1:]
    plot_fn = ax.semilogx if log_freq else ax.plot
    plot_fn(f, coh2[1:], color=get_channel_color(4), linewidth=1.3)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"Coherence $\gamma^2$")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title or f"Coherence  Ch {ch_i+1} – Ch {ch_j+1}")
    ax.minorticks_on()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
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
    apply_style()
    n_channels = corr_matrix.shape[1]
    channels = channels or list(range(n_channels))

    fig, ax = plt.subplots(figsize=(9, 4.5))

    if auto_only:
        for ch in channels:
            ax.plot(
                tau * 1e3, corr_matrix[:, ch, ch],
                label=f"Ch {ch+1}", color=get_channel_color(ch), linewidth=1.2,
            )
    else:
        idx = 0
        for i in channels:
            for j in channels:
                if i <= j:
                    label = f"Ch {i+1}–{j+1}" if i != j else f"Ch {i+1} auto"
                    ax.plot(
                        tau * 1e3, corr_matrix[:, i, j],
                        label=label, color=get_channel_color(idx), linewidth=0.9,
                    )
                    idx += 1

    ax.set_xlabel("Lag τ [ms]")
    ax.set_ylabel("Normalised Correlation")
    ax.set_title(title)
    ax.legend(fontsize=9, ncol=min(4, len(channels)))
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
#  CSM Magnitude Matrix (Acoular-inspired)
# --------------------------------------------------------------------------- #


def plot_csm_matrix(
    spectra: np.ndarray,
    freq: np.ndarray,
    target_freq: float,
    db: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the cross-spectral matrix magnitude at a single frequency as a
    channel × channel heatmap.

    This is the canonical CSM visualisation used extensively in microphone
    array processing (cf. Acoular ``PowerSpectraImport`` examples).

    Parameters
    ----------
    spectra : array, shape (N_freq, M, M)
    freq : array, shape (N_freq,)
    target_freq : float
        Centre frequency in Hz to display.
    db : bool
        If True, display in dB (10 log10).
    """
    apply_style()

    # Find nearest frequency bin
    idx = np.argmin(np.abs(freq - target_freq))
    actual_f = freq[idx]
    csm_slice = spectra[idx, :, :]
    mag = np.abs(csm_slice)
    n_ch = mag.shape[0]

    if db:
        data = 10 * np.log10(np.maximum(mag, 1e-30))
        cbar_label = "Magnitude [dB]"
    else:
        data = mag
        cbar_label = "Magnitude"

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(
        data, origin="lower", interpolation="nearest",
        cmap="inferno", aspect="equal",
    )

    ch_labels = [f"Ch {i+1}" for i in range(n_ch)]
    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(ch_labels, fontsize=8.5)
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels(ch_labels, fontsize=8.5)

    ax.set_xlabel("Channel")
    ax.set_ylabel("Channel")
    ax.set_title(title or f"CSM Magnitude at {actual_f:.0f} Hz")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=8.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
#  Coherence Matrix Heatmap
# --------------------------------------------------------------------------- #


def plot_coherence_matrix(
    spectra: np.ndarray,
    freq: np.ndarray,
    target_freq: float,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    r"""
    Plot the magnitude-squared coherence matrix at a single frequency.

    Parameters
    ----------
    spectra : array, shape (N_freq, M, M)
    freq : array, shape (N_freq,)
    target_freq : float
        Centre frequency in Hz.
    """
    apply_style()

    idx = np.argmin(np.abs(freq - target_freq))
    actual_f = freq[idx]
    csm_slice = spectra[idx, :, :]
    n_ch = csm_slice.shape[0]

    coh = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(n_ch):
            Gii = np.real(csm_slice[i, i])
            Gjj = np.real(csm_slice[j, j])
            denom = Gii * Gjj
            coh[i, j] = np.abs(csm_slice[i, j]) ** 2 / max(denom, 1e-30)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(
        coh, origin="lower", interpolation="nearest",
        cmap="YlOrRd", aspect="equal", vmin=0, vmax=1,
    )

    ch_labels = [f"Ch {i+1}" for i in range(n_ch)]
    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(ch_labels, fontsize=8.5)
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels(ch_labels, fontsize=8.5)

    for i in range(n_ch):
        for j in range(n_ch):
            val = coh[i, j]
            text_color = "white" if val > 0.65 else "#334155"
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=8, color=text_color, fontweight="medium",
            )

    ax.set_xlabel("Channel")
    ax.set_ylabel("Channel")
    ax.set_title(title or f"Coherence Matrix $\\gamma^2$ at {actual_f:.0f} Hz")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.04)
    cbar.set_label(r"$\gamma^2$", fontsize=10)
    cbar.ax.tick_params(labelsize=8.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
