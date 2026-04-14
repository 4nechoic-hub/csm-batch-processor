"""
Spectral Anomaly Detection
============================
Unsupervised anomaly detection on CSM-derived spectral features.

Designed for monitoring applications where a baseline of "normal" operating
conditions is established, and subsequent measurements are scored against it.

Typical workflow
----------------
1. Collect baseline data (e.g. 50–200 normal CSM snapshots)
2. Extract features → feature matrix X_train
3. Fit detector on X_train
4. Score new measurements → anomaly scores + labels

Three complementary methods are provided:

- **Isolation Forest** — tree-based, handles high-dimensional features well,
  robust to irrelevant features.  Good general-purpose choice.
- **Mahalanobis distance** — parametric, assumes multivariate Gaussian.
  Interpretable threshold via chi-squared distribution.  Works well when
  feature distributions are approximately normal.
- **Local Outlier Factor (LOF)** — density-based, catches anomalies that
  live in locally sparse regions.  Good for non-Gaussian distributions.

All methods return a consistent interface: anomaly scores (higher = more
anomalous) and binary labels (-1 = anomaly, 1 = normal).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2


@dataclass
class AnomalyResult:
    """Container for anomaly detection results."""

    scores: np.ndarray
    """Anomaly scores (higher = more anomalous)."""

    labels: np.ndarray
    """Binary labels: -1 = anomaly, 1 = normal."""

    threshold: float
    """Decision threshold used."""

    method: str
    """Detection method name."""

    feature_names: Optional[list[str]] = None
    """Feature names (for interpretability)."""

    feature_importances: Optional[np.ndarray] = None
    """Per-feature contribution to anomaly score (where available)."""

    @property
    def anomaly_indices(self) -> np.ndarray:
        """Indices of detected anomalies."""
        return np.where(self.labels == -1)[0]

    @property
    def n_anomalies(self) -> int:
        return int(np.sum(self.labels == -1))

    @property
    def anomaly_rate(self) -> float:
        return self.n_anomalies / len(self.labels)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Anomaly Detection Results ({self.method})",
            f"  Samples:     {len(self.labels)}",
            f"  Anomalies:   {self.n_anomalies} ({self.anomaly_rate:.1%})",
            f"  Threshold:   {self.threshold:.4f}",
            f"  Score range: [{self.scores.min():.4f}, {self.scores.max():.4f}]",
        ]
        if self.n_anomalies > 0 and self.n_anomalies <= 20:
            lines.append(f"  Indices:     {list(self.anomaly_indices)}")
        return "\n".join(lines)


class SpectralAnomalyDetector:
    """
    Unsupervised anomaly detector for CSM-derived spectral features.

    Parameters
    ----------
    method : str
        Detection method: ``'isolation_forest'``, ``'mahalanobis'``, or
        ``'lof'`` (Local Outlier Factor).
    contamination : float
        Expected fraction of anomalies (0–0.5). Used to set the decision
        threshold.  Default 0.05 (5%).
    random_state : int
        Random seed for reproducibility.

    Examples
    --------
    >>> detector = SpectralAnomalyDetector(method="isolation_forest")
    >>> detector.fit(X_train, feature_names=names)
    >>> result = detector.predict(X_test)
    >>> print(result.summary())
    """

    def __init__(
        self,
        method: Literal[
            "isolation_forest", "mahalanobis", "lof"
        ] = "isolation_forest",
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.method = method
        self.contamination = contamination
        self.random_state = random_state

        self.scaler_ = StandardScaler()
        self.is_fitted_ = False
        self.feature_names_: Optional[list[str]] = None

        # Method-specific internals
        self._model = None
        self._mean = None
        self._cov_inv = None
        self._threshold = None

    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> "SpectralAnomalyDetector":
        """
        Fit the detector on baseline (normal) data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix from normal operating conditions.
        feature_names : list[str], optional
            Feature names for interpretability.
        """
        self.feature_names_ = feature_names
        X_scaled = self.scaler_.fit_transform(X)

        if self.method == "isolation_forest":
            self._fit_isolation_forest(X_scaled)
        elif self.method == "mahalanobis":
            self._fit_mahalanobis(X_scaled)
        elif self.method == "lof":
            self._fit_lof(X_scaled)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> AnomalyResult:
        """
        Score and classify new samples.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix to evaluate.

        Returns
        -------
        AnomalyResult
        """
        if not self.is_fitted_:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")

        X_scaled = self.scaler_.transform(X)

        if self.method == "isolation_forest":
            return self._predict_isolation_forest(X_scaled)
        elif self.method == "mahalanobis":
            return self._predict_mahalanobis(X_scaled)
        elif self.method == "lof":
            return self._predict_lof(X_scaled)

    def fit_predict(
        self,
        X: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> AnomalyResult:
        """Fit on X and return predictions for the same data."""
        self.fit(X, feature_names)
        return self.predict(X)

    # ── Isolation Forest ──────────────────────────────────────────────

    def _fit_isolation_forest(self, X: np.ndarray):
        self._model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=200,
        )
        self._model.fit(X)

    def _predict_isolation_forest(self, X: np.ndarray) -> AnomalyResult:
        raw_scores = self._model.decision_function(X)
        scores = -raw_scores
        labels = self._model.predict(X)
        threshold = 0.0

        importances = None
        if hasattr(self._model, "feature_importances_"):
            importances = self._model.feature_importances_

        return AnomalyResult(
            scores=scores,
            labels=labels,
            threshold=threshold,
            method="Isolation Forest",
            feature_names=self.feature_names_,
            feature_importances=importances,
        )

    # ── Mahalanobis Distance ──────────────────────────────────────────

    def _fit_mahalanobis(self, X: np.ndarray):
        self._mean = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        # Regularise for numerical stability
        cov += np.eye(cov.shape[0]) * 1e-6
        self._cov_inv = np.linalg.inv(cov)
        # Threshold via chi-squared distribution
        n_features = X.shape[1]
        self._threshold = chi2.ppf(1 - self.contamination, df=n_features)

    def _predict_mahalanobis(self, X: np.ndarray) -> AnomalyResult:
        scores = np.array(
            [mahalanobis(x, self._mean, self._cov_inv) ** 2 for x in X]
        )
        labels = np.where(scores > self._threshold, -1, 1)

        # Per-feature contribution: (x - μ)² weighted by inverse covariance
        diff = X - self._mean
        contributions = np.abs(diff @ self._cov_inv) * np.abs(diff)
        importances = np.mean(contributions, axis=0)
        importances /= importances.sum() + 1e-30

        return AnomalyResult(
            scores=scores,
            labels=labels,
            threshold=self._threshold,
            method="Mahalanobis Distance",
            feature_names=self.feature_names_,
            feature_importances=importances,
        )

    # ── Local Outlier Factor ──────────────────────────────────────────

    def _fit_lof(self, X: np.ndarray):
        self._model = LocalOutlierFactor(
            contamination=self.contamination,
            novelty=True,
            n_neighbors=min(20, X.shape[0] - 1),
        )
        self._model.fit(X)

    def _predict_lof(self, X: np.ndarray) -> AnomalyResult:
        raw_scores = self._model.decision_function(X)
        scores = -raw_scores
        labels = self._model.predict(X)
        threshold = 0.0

        return AnomalyResult(
            scores=scores,
            labels=labels,
            threshold=threshold,
            method="Local Outlier Factor",
            feature_names=self.feature_names_,
        )


def compare_methods(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: Optional[list[str]] = None,
    contamination: float = 0.05,
) -> dict[str, AnomalyResult]:
    """
    Run all three detection methods and return results for comparison.

    Parameters
    ----------
    X_train : np.ndarray
        Baseline (normal) feature matrix.
    X_test : np.ndarray
        Test feature matrix to evaluate.
    feature_names : list[str], optional

    Returns
    -------
    results : dict[str, AnomalyResult]
        Keyed by method name.
    """
    results = {}
    for method in ["isolation_forest", "mahalanobis", "lof"]:
        detector = SpectralAnomalyDetector(
            method=method, contamination=contamination
        )
        detector.fit(X_train, feature_names=feature_names)
        results[method] = detector.predict(X_test)
    return results