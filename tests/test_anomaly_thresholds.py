import numpy as np

from csm_processor.anomaly_detection import SpectralAnomalyDetector


def _toy_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    X_train = rng.normal(size=(100, 8))
    X_test = np.vstack([
        rng.normal(size=(20, 8)),
        rng.normal(loc=4.0, scale=1.0, size=(5, 8)),
    ])
    return X_train, X_test


def test_isolation_forest_threshold_matches_inverted_score_space():
    X_train, X_test = _toy_data()
    detector = SpectralAnomalyDetector(
        method="isolation_forest",
        contamination=0.1,
        random_state=0,
    )
    detector.fit(X_train)
    result = detector.predict(X_test)

    assert result.threshold == 0.0
    expected = np.where(result.scores > 0.0, -1, 1)
    assert np.array_equal(result.labels, expected)


def test_lof_threshold_matches_inverted_score_space():
    X_train, X_test = _toy_data()
    detector = SpectralAnomalyDetector(
        method="lof",
        contamination=0.1,
        random_state=0,
    )
    detector.fit(X_train)
    result = detector.predict(X_test)

    assert result.threshold == 0.0
    expected = np.where(result.scores > 0.0, -1, 1)
    assert np.array_equal(result.labels, expected)