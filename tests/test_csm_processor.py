"""
Unit tests for csm_processor
==============================
Tests core functionality against known analytical results.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import os
import tempfile

from csm_processor.csm_calculator import csm_calculator
from csm_processor.log_binning import log_freq_bin, bin_csm
from csm_processor.correlation import compute_correlation
from csm_processor.io_utils import load_data, save_results


# ─── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def sine_data():
    """Single 1 kHz sine wave, 1 second at 10 kHz sampling."""
    fs = 10000
    t = np.arange(0, 1.0, 1 / fs)
    signal = np.sin(2 * np.pi * 1000 * t)
    return signal[:, np.newaxis], fs


@pytest.fixture
def multichannel_data():
    """3-channel synthetic data with known tones."""
    fs = 10000
    t = np.arange(0, 1.0, 1 / fs)
    np.random.seed(42)
    ch1 = np.sin(2 * np.pi * 500 * t)
    ch2 = np.sin(2 * np.pi * 500 * t + np.pi / 4) + np.sin(2 * np.pi * 2000 * t)
    ch3 = 0.5 * np.random.randn(len(t))
    return np.column_stack([ch1, ch2, ch3]), fs


# ─── CSM Calculator Tests ───────────────────────────────────────────────

class TestCSMCalculator:

    def test_output_shape_single_channel(self, sine_data):
        data, fs = sine_data
        spectra, freq = csm_calculator(data, fs, n_rec=1024, overlap=50)
        assert spectra.shape == (1024, 1, 1)
        assert freq.shape == (1024,)

    def test_output_shape_multichannel(self, multichannel_data):
        data, fs = multichannel_data
        spectra, freq = csm_calculator(data, fs, n_rec=1024, overlap=50)
        assert spectra.shape == (1024, 3, 3)
        assert freq.shape == (1024,)

    def test_frequency_vector(self, sine_data):
        data, fs = sine_data
        spectra, freq = csm_calculator(data, fs, n_rec=1024, overlap=50)
        df = fs / 1024
        assert freq[0] == 0.0
        assert abs(freq[1] - df) < 1e-10
        assert abs(freq[-1] - (fs - df)) < 1e-6

    def test_peak_frequency(self, sine_data):
        """PSD of a 1 kHz sine should peak at 1 kHz."""
        data, fs = sine_data
        spectra, freq = csm_calculator(data, fs, n_rec=1024, overlap=50)
        half = len(freq) // 2
        psd = np.real(spectra[:half, 0, 0])
        peak_freq = freq[np.argmax(psd)]
        assert abs(peak_freq - 1000) < 20  # within ~2 bins

    def test_multichannel_peaks(self, multichannel_data):
        """Ch1 should peak at 500 Hz, Ch2 at 500 or 2000 Hz."""
        data, fs = multichannel_data
        spectra, freq = csm_calculator(data, fs, n_rec=1024, overlap=50)
        half = len(freq) // 2
        peak_ch1 = freq[np.argmax(np.real(spectra[:half, 0, 0]))]
        assert abs(peak_ch1 - 500) < 20

    def test_autospectra_positive(self, multichannel_data):
        """Auto-spectra (diagonal) should be real and non-negative."""
        data, fs = multichannel_data
        spectra, freq = csm_calculator(data, fs, n_rec=1024, overlap=50)
        for ch in range(3):
            auto = spectra[:, ch, ch]
            assert np.allclose(auto.imag, 0, atol=1e-15)
            assert np.all(np.real(auto) >= -1e-15)

    def test_hermitian_symmetry(self, multichannel_data):
        """CSM should be Hermitian: G_ij = conj(G_ji)."""
        data, fs = multichannel_data
        spectra, freq = csm_calculator(data, fs, n_rec=1024, overlap=50)
        for f_idx in [10, 50, 200]:
            mat = spectra[f_idx, :, :]
            assert np.allclose(mat, mat.conj().T, atol=1e-14)

    def test_overlap_zero(self, sine_data):
        """Should work with 0% overlap."""
        data, fs = sine_data
        spectra, freq = csm_calculator(data, fs, n_rec=1024, overlap=0)
        assert spectra.shape[0] == 1024

    def test_overlap_75(self, sine_data):
        """Should work with 75% overlap (more blocks)."""
        data, fs = sine_data
        spectra, freq = csm_calculator(data, fs, n_rec=1024, overlap=75)
        assert spectra.shape[0] == 1024

    def test_insufficient_data_raises(self):
        """Should raise if data is shorter than one block."""
        data = np.random.randn(100, 1)
        with pytest.raises(ValueError, match="Not enough data"):
            csm_calculator(data, fs=1000, n_rec=1024, overlap=50)

    def test_transposed_input(self, sine_data):
        """Should auto-transpose (M, N) input to (N, M)."""
        data, fs = sine_data
        spectra1, _ = csm_calculator(data, fs, n_rec=1024, overlap=50)
        spectra2, _ = csm_calculator(data.T, fs, n_rec=1024, overlap=50)
        assert np.allclose(spectra1, spectra2)


# ─── Log Binning Tests ──────────────────────────────────────────────────

class TestLogBinning:

    def test_output_length(self):
        """Binned output should have fewer points than input."""
        spectrum = np.random.rand(512)
        df = 10.0
        freq_b, spec_b = log_freq_bin(df, spectrum, bins_per_octave=3)
        assert len(freq_b) == len(spec_b)
        assert len(freq_b) < len(spectrum)

    def test_frequencies_monotonic(self):
        """Binned frequencies should be strictly increasing."""
        spectrum = np.random.rand(512)
        freq_b, _ = log_freq_bin(10.0, spectrum, bins_per_octave=3)
        assert np.all(np.diff(freq_b) > 0)

    def test_nan_handling(self):
        """Should handle NaN values gracefully."""
        spectrum = np.random.rand(512)
        spectrum[100:120] = np.nan
        freq_b, spec_b = log_freq_bin(10.0, spectrum, bins_per_octave=3)
        # Most bins should still be valid
        assert np.sum(~np.isnan(spec_b)) > len(spec_b) * 0.8

    def test_bin_csm_shape(self, multichannel_data):
        """bin_csm should preserve channel dimensions."""
        data, fs = multichannel_data
        spectra, freq = csm_calculator(data, fs, n_rec=1024, overlap=50)
        df = fs / 1024
        freq_b, spec_b = bin_csm(df, spectra, bins_per_octave=3)
        assert spec_b.shape[1] == 3
        assert spec_b.shape[2] == 3
        assert len(freq_b) == spec_b.shape[0]


# ─── Correlation Tests ───────────────────────────────────────────────────

class TestCorrelation:

    def test_autocorrelation_peak_at_zero(self):
        """Auto-correlation should peak at τ=0 with value 1.0."""
        data = np.random.randn(5000, 2)
        tau, corr = compute_correlation(data, fs=1000)
        zero_idx = len(tau) // 2
        for ch in range(2):
            assert abs(corr[zero_idx, ch, ch] - 1.0) < 1e-10

    def test_correlation_symmetry(self):
        """Auto-correlation should be symmetric about τ=0."""
        np.random.seed(0)
        data = np.random.randn(5000, 1)
        tau, corr = compute_correlation(data, fs=1000)
        mid = len(tau) // 2
        left = corr[:mid, 0, 0]
        right = corr[mid + 1 :, 0, 0][::-1]
        min_len = min(len(left), len(right))
        assert np.allclose(left[-min_len:], right[-min_len:], atol=1e-10)

    def test_correlated_channels(self):
        """Correlated channels should show high cross-correlation at τ=0."""
        np.random.seed(0)
        x = np.random.randn(5000)
        data = np.column_stack([x, 0.9 * x + 0.1 * np.random.randn(5000)])
        tau, corr = compute_correlation(data, fs=1000)
        zero_idx = len(tau) // 2
        cross_corr = corr[zero_idx, 0, 1]
        assert cross_corr > 0.8

    def test_tau_vector(self):
        """Tau should be centred on zero with correct spacing."""
        data = np.random.randn(1000, 1)
        fs = 2000
        tau, _ = compute_correlation(data, fs)
        dt = 1.0 / fs
        assert abs(tau[len(tau) // 2]) < dt  # centre is near zero


# ─── I/O Tests ───────────────────────────────────────────────────────────

class TestIO:

    def test_csv_roundtrip(self, multichannel_data):
        """Save as CSV, load back, verify shape."""
        data, fs = multichannel_data
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            np.savetxt(f, data, delimiter=",")
            tmppath = f.name
        try:
            loaded = load_data(tmppath)
            assert loaded.shape == data.shape
            assert np.allclose(loaded, data, atol=1e-6)
        finally:
            os.unlink(tmppath)

    def test_csv_with_header(self):
        """Should skip a header row automatically."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("ch1,ch2,ch3\n")
            for _ in range(100):
                f.write("1.0,2.0,3.0\n")
            tmppath = f.name
        try:
            loaded = load_data(tmppath)
            assert loaded.shape == (100, 3)
        finally:
            os.unlink(tmppath)

    def test_save_npz(self, multichannel_data):
        """save_results should produce a valid .npz file."""
        data, fs = multichannel_data
        spectra, freq = csm_calculator(data, fs, n_rec=1024, overlap=50)
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = save_results(
                os.path.join(tmpdir, "test"), spectra, freq, fs, 1024, 50, fmt="npz"
            )
            assert outpath.endswith(".npz")
            loaded = np.load(outpath)
            assert "spectra" in loaded
            assert "freq" in loaded
            assert np.allclose(loaded["spectra"], spectra)

    def test_save_mat(self, multichannel_data):
        """save_results should produce a valid .mat file."""
        data, fs = multichannel_data
        spectra, freq = csm_calculator(data, fs, n_rec=1024, overlap=50)
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = save_results(
                os.path.join(tmpdir, "test"), spectra, freq, fs, 1024, 50, fmt="mat"
            )
            assert outpath.endswith(".mat")
            import scipy.io as sio
            loaded = sio.loadmat(outpath)
            assert "spectra" in loaded

    def test_unsupported_format_raises(self):
        """Should raise on unknown file extension."""
        with pytest.raises(ValueError, match="Unsupported"):
            load_data("data.xyz")