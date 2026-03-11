"""
Tests for feature extraction and anomaly detection modules.
"""

import numpy as np
import pytest

from csm_processor.csm_calculator import csm_calculator
from csm_processor.feature_extraction import (
    extract_features,
    features_to_array,
    extract_features_batch,
)
from csm_processor.anomaly_detection import (
    SpectralAnomalyDetector,
    compare_methods,
)


# ─── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def sample_csm():
    """Generate a CSM from synthetic 2-channel data."""
    fs = 10000
    t = np.arange(0, 0.5, 1 / fs)
    np.random.seed(42)
    data = np.column_stack([
        np.sin(2 * np.pi * 1000 * t) + 0.1 * np.random.randn(len(t)),
        np.sin(2 * np.pi * 2000 * t) + 0.1 * np.random.randn(len(t)),
    ])
    spectra, freq = csm_calculator(data, fs, n_rec=1024, overlap=50)
    return spectra, freq, fs


@pytest.fixture
def baseline_and_anomaly():
    """
    Generate normal and anomalous CSM sets for detection testing.
    Normal: 1 kHz tone. Anomalous: shifted to 3 kHz + elevated broadband.
    """
    fs = 10000
    n_rec = 1024

    normal_csms = []
    for seed in range(40):
        rng = np.random.RandomState(seed)
        t = np.arange(0, 0.5, 1 / fs)
        data = np.column_stack([
            np.sin(2 * np.pi * 1000 * t) + 0.1 * rng.randn(len(t)),
            0.8 * np.sin(2 * np.pi * 1000 * t + 0.2) + 0.1 * rng.randn(len(t)),
        ])
        spectra, freq = csm_calculator(data, fs, n_rec, 50)
        normal_csms.append(spectra)

    anomaly_csms = []
    for seed in range(500, 508):
        rng = np.random.RandomState(seed)
        t = np.arange(0, 0.5, 1 / fs)
        data = np.column_stack([
            np.sin(2 * np.pi * 3000 * t) + 0.5 * rng.randn(len(t)),
            0.3 * rng.randn(len(t)),
        ])
        spectra, freq = csm_calculator(data, fs, n_rec, 50)
        anomaly_csms.append(spectra)

    return normal_csms, anomaly_csms, freq, fs


# ─── Feature Extraction Tests ────────────────────────────────────────────

class TestFeatureExtraction:

    def test_feature_dict_keys(self, sample_csm):
        spectra, freq, fs = sample_csm
        features = extract_features(spectra, freq, fs)
        assert isinstance(features, dict)
        assert len(features) > 0
        # Should have auto-spectral features for both channels
        assert "total_power_ch0" in features
        assert "total_power_ch1" in features
        assert "peak_freq_ch0" in features
        assert "spectral_centroid_ch0" in features

    def test_cross_channel_features(self, sample_csm):
        spectra, freq, fs = sample_csm
        features = extract_features(spectra, freq, fs)
        assert "mean_coherence_ch0_ch1" in features
        assert "peak_coherence_ch0_ch1" in features
        assert "weighted_phase_ch0_ch1" in features

    def test_band_energy_features(self, sample_csm):
        spectra, freq, fs = sample_csm
        features = extract_features(spectra, freq, fs)
        assert "band0_power_ch0" in features
        assert "band0_ratio_ch0" in features
        # Ratios should sum to ~1
        ratios = sum(features[f"band{i}_ratio_ch0"] for i in range(4))
        assert abs(ratios - 1.0) < 0.01

    def test_peak_frequency_correct(self, sample_csm):
        spectra, freq, fs = sample_csm
        features = extract_features(spectra, freq, fs)
        # Ch0 has a 1 kHz tone
        assert abs(features["peak_freq_ch0"] - 1000) < 50

    def test_spectral_flatness_range(self, sample_csm):
        spectra, freq, fs = sample_csm
        features = extract_features(spectra, freq, fs)
        # Flatness should be between 0 and 1
        for ch in range(2):
            assert 0 <= features[f"spectral_flatness_ch{ch}"] <= 1

    def test_features_to_array(self, sample_csm):
        spectra, freq, fs = sample_csm
        features = extract_features(spectra, freq, fs)
        arr, names = features_to_array(features)
        assert arr.shape == (len(features),)
        assert len(names) == len(features)
        assert sorted(names) == names  # should be alphabetically sorted

    def test_batch_extraction(self, baseline_and_anomaly):
        normal_csms, _, freq, fs = baseline_and_anomaly
        X, names = extract_features_batch(normal_csms[:5], freq, fs)
        assert X.shape[0] == 5
        assert X.shape[1] == len(names)
        assert X.shape[1] > 0

    def test_custom_bands(self, sample_csm):
        spectra, freq, fs = sample_csm
        bands = [(0, 500), (500, 2000), (2000, 5000)]
        features = extract_features(spectra, freq, fs, bands=bands)
        assert "band0_power_ch0" in features
        assert "band2_power_ch0" in features

    def test_all_features_finite(self, sample_csm):
        spectra, freq, fs = sample_csm
        features = extract_features(spectra, freq, fs)
        for key, val in features.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"


# ─── Anomaly Detection Tests ─────────────────────────────────────────────

class TestAnomalyDetection:

    def test_isolation_forest_fit_predict(self, baseline_and_anomaly):
        normal_csms, anomaly_csms, freq, fs = baseline_and_anomaly
        X_train, names = extract_features_batch(normal_csms[:30], freq, fs)
        X_test, _ = extract_features_batch(normal_csms[30:] + anomaly_csms, freq, fs)

        detector = SpectralAnomalyDetector(method="isolation_forest", contamination=0.2)
        detector.fit(X_train, feature_names=names)
        result = detector.predict(X_test)

        assert len(result.scores) == len(X_test)
        assert len(result.labels) == len(X_test)
        assert result.method == "Isolation Forest"
        assert set(result.labels).issubset({-1, 1})

    def test_mahalanobis_fit_predict(self, baseline_and_anomaly):
        normal_csms, anomaly_csms, freq, fs = baseline_and_anomaly
        X_train, names = extract_features_batch(normal_csms[:30], freq, fs)
        X_test, _ = extract_features_batch(anomaly_csms, freq, fs)

        detector = SpectralAnomalyDetector(method="mahalanobis", contamination=0.1)
        detector.fit(X_train, feature_names=names)
        result = detector.predict(X_test)

        assert len(result.scores) == len(X_test)
        assert result.feature_importances is not None
        assert len(result.feature_importances) == X_train.shape[1]

    def test_lof_fit_predict(self, baseline_and_anomaly):
        normal_csms, anomaly_csms, freq, fs = baseline_and_anomaly
        X_train, names = extract_features_batch(normal_csms[:30], freq, fs)
        X_test, _ = extract_features_batch(normal_csms[30:] + anomaly_csms, freq, fs)

        detector = SpectralAnomalyDetector(method="lof", contamination=0.2)
        detector.fit(X_train, feature_names=names)
        result = detector.predict(X_test)

        assert len(result.scores) == len(X_test)
        assert result.method == "Local Outlier Factor"

    def test_anomalies_detected(self, baseline_and_anomaly):
        """Anomalous samples should be flagged more often than normal ones."""
        normal_csms, anomaly_csms, freq, fs = baseline_and_anomaly
        X_train, names = extract_features_batch(normal_csms[:30], freq, fs)
        X_normal_test, _ = extract_features_batch(normal_csms[30:], freq, fs)
        X_anomaly_test, _ = extract_features_batch(anomaly_csms, freq, fs)

        detector = SpectralAnomalyDetector(method="isolation_forest", contamination=0.15)
        detector.fit(X_train, feature_names=names)

        result_normal = detector.predict(X_normal_test)
        result_anomaly = detector.predict(X_anomaly_test)

        # Anomalous set should have higher mean score
        assert np.mean(result_anomaly.scores) > np.mean(result_normal.scores)

    def test_fit_predict_shortcut(self, baseline_and_anomaly):
        normal_csms, _, freq, fs = baseline_and_anomaly
        X, names = extract_features_batch(normal_csms, freq, fs)

        detector = SpectralAnomalyDetector(method="isolation_forest")
        result = detector.fit_predict(X, feature_names=names)
        assert len(result.labels) == len(X)

    def test_compare_methods(self, baseline_and_anomaly):
        normal_csms, anomaly_csms, freq, fs = baseline_and_anomaly
        X_train, names = extract_features_batch(normal_csms[:30], freq, fs)
        X_test, _ = extract_features_batch(normal_csms[30:] + anomaly_csms, freq, fs)

        results = compare_methods(X_train, X_test, names, contamination=0.2)
        assert "isolation_forest" in results
        assert "mahalanobis" in results
        assert "lof" in results

    def test_unfitted_raises(self):
        detector = SpectralAnomalyDetector()
        with pytest.raises(RuntimeError, match="not been fitted"):
            detector.predict(np.random.randn(5, 10))

    def test_unknown_method_raises(self):
        detector = SpectralAnomalyDetector(method="invalid")
        with pytest.raises(ValueError, match="Unknown method"):
            detector.fit(np.random.randn(20, 5))

    def test_result_summary(self, baseline_and_anomaly):
        normal_csms, anomaly_csms, freq, fs = baseline_and_anomaly
        X_train, names = extract_features_batch(normal_csms[:30], freq, fs)
        X_test, _ = extract_features_batch(anomaly_csms, freq, fs)

        detector = SpectralAnomalyDetector(method="isolation_forest", contamination=0.3)
        detector.fit(X_train, feature_names=names)
        result = detector.predict(X_test)

        summary = result.summary()
        assert "Anomaly Detection Results" in summary
        assert "Samples:" in summary

    def test_anomaly_indices(self, baseline_and_anomaly):
        normal_csms, _, freq, fs = baseline_and_anomaly
        X, names = extract_features_batch(normal_csms, freq, fs)
        detector = SpectralAnomalyDetector(method="isolation_forest", contamination=0.1)
        result = detector.fit_predict(X, names)
        assert len(result.anomaly_indices) == result.n_anomalies