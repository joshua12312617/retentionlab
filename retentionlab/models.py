"""Model comparison with stratified cross-validation.

Three models are implemented from scratch with numpy (logistic regression via
mini-batch gradient descent, Gaussian Naive Bayes, k-nearest neighbors) — they
run everywhere.  When scikit-learn is installed, tuned RandomForest and
GradientBoosting models join the comparison automatically.

All evaluation metrics (AUC, precision/recall at the Youden threshold) are
computed from first principles in :mod:`retentionlab.stats`, so the comparison
is identical across backends.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd

from . import config
from .features import FeatureSet
from .stats import roc_auc, youden_threshold

try:  # optional dependency
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

    HAS_SKLEARN = True
except ImportError:  # pragma: no cover - exercised only without sklearn
    HAS_SKLEARN = False


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))


class LogisticRegressionNumpy:
    """Binary logistic regression with L2 regularization (mini-batch GD)."""

    def __init__(self, l2: float = 1.0, learning_rate: float = 0.5,
                 epochs: int = 300, seed: int = 0) -> None:
        self.l2 = l2
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionNumpy":
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.w = rng.normal(0.0, 0.01, d)
        self.b = 0.0
        batch = max(32, n // 8)
        for _ in range(self.epochs):
            perm = rng.permutation(n)
            for start in range(0, n, batch):
                idx = perm[start : start + batch]
                p = _sigmoid(X[idx] @ self.w + self.b)
                error = p - y[idx]
                grad_w = X[idx].T @ error / len(idx) + self.l2 * self.w
                self.w -= self.learning_rate * grad_w
                self.b -= self.learning_rate * error.mean()
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class GaussianNB:
    """Gaussian Naive Bayes with class priors estimated from the data."""

    def __init__(self) -> None:
        self.classes: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNB":
        self.classes = np.unique(y)
        self.means = np.stack([X[y == cls].mean(axis=0) for cls in self.classes])
        self.vars = np.stack([X[y == cls].var(axis=0) + 1e-9 for cls in self.classes])
        self.priors = np.array([(y == cls).mean() for cls in self.classes])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_likelihood = np.stack(
            [
                -0.5 * np.sum(np.log(2 * np.pi * self.vars[i]) + (X - self.means[i]) ** 2 / self.vars[i], axis=1)
                + np.log(self.priors[i])
                for i in range(len(self.classes))
            ],
            axis=1,
        )
        log_likelihood -= log_likelihood.max(axis=1, keepdims=True)
        probabilities = np.exp(log_likelihood)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class KNNClassifier:
    """k-nearest neighbors with probability estimates."""

    def __init__(self, k: int = 15) -> None:
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self.X = X
        self.y = y
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # squared euclidean via the expansion ||a - b||^2 = |a|^2 + |b|^2 - 2ab
        distances = (
            np.sum(X**2, axis=1)[:, None]
            + np.sum(self.X**2, axis=1)[None, :]
            - 2.0 * X @ self.X.T
        )
        neighbors = np.argsort(distances, axis=1)[:, : self.k]
        fraction = self.y[neighbors].mean(axis=1)
        return np.column_stack([1.0 - fraction, fraction])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def build_model_registry() -> dict[str, object]:
    """Return ``{model_name: factory}`` for every available model."""
    registry: dict[str, object] = {
        "logistic (numpy)": lambda: LogisticRegressionNumpy(l2=1.0, seed=42),
        "gaussian naive bayes": GaussianNB,
        "k-nearest neighbors": lambda: KNNClassifier(k=15),
    }
    if HAS_SKLEARN:
        registry.update(
            {
                "random forest (sklearn)": lambda: RandomForestClassifier(
                    n_estimators=300, max_depth=8, min_samples_leaf=5,
                    random_state=42, n_jobs=-1,
                ),
                "gradient boosting (sklearn)": lambda: GradientBoostingClassifier(
                    n_estimators=150, learning_rate=0.05, max_depth=3,
                    random_state=42,
                ),
                "logistic (sklearn)": lambda: SklearnLogisticRegression(
                    max_iter=2000, random_state=42,
                ),
            }
        )
    return registry


def _cv_predictions(
    model: object, fs: FeatureSet, folds: int, seed: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Stratified CV, returning out-of-fold probabilities and mean fit time."""
    rng = np.random.default_rng(seed)
    y = fs.y_train
    n = len(y)
    fold_ids = np.zeros(n, dtype=int)
    for label in np.unique(y):
        group = np.where(y == label)[0]
        rng.shuffle(group)
        fold_ids[group] = np.arange(len(group)) % folds
    scores = np.zeros(n)
    started = time.perf_counter()
    for fold in range(folds):
        train_mask = fold_ids != fold
        if hasattr(model, "get_params"):
            cloned = model.__class__(**model.get_params())  # sklearn estimators
        else:
            cloned = copy.deepcopy(model)
        cloned.fit(fs.X_train[train_mask], y[train_mask])
        scores[~train_mask] = cloned.predict_proba(fs.X_train[~train_mask])[:, 1]
    elapsed = (time.perf_counter() - started) / folds
    return scores, y, elapsed


def _metrics_at_threshold(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "auc": roc_auc(y, scores),
        "threshold": threshold,
        "accuracy": (pred == y).mean(),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def compare_models(fs: FeatureSet, folds: int = 5, seed: int = config.DEFAULT_SEED) -> tuple[pd.DataFrame, str]:
    """Cross-validate every model and return ``(summary, champion_name)``."""
    rows = []
    y = fs.y_train
    baseline = float((y == 1).mean())
    rows.append(
        {
            "model": "dummy baseline (majority)",
            "auc": 0.5,
            "threshold": 0.5,
            "accuracy": max(baseline, 1 - baseline),
            "precision": baseline,
            "recall": 1.0 if baseline >= 0.5 else 0.0,
            "f1": 2 * baseline / (1 + baseline),
            "fit_seconds": 0.0,
        }
    )
    for name, factory in build_model_registry().items():
        try:
            model = factory()
            scores, _, seconds = _cv_predictions(model, fs, folds, seed)
            threshold, _ = youden_threshold(y, scores)
            metrics = _metrics_at_threshold(y, scores, threshold)
            rows.append({"model": name, "fit_seconds": seconds, **metrics})
        except Exception as exc:  # a broken optional backend must not sink the run
            rows.append({"model": name, "auc": float("nan"), "threshold": float("nan"),
                         "accuracy": float("nan"), "precision": float("nan"),
                         "recall": float("nan"), "f1": float("nan"),
                         "fit_seconds": 0.0, "error": str(exc)})
    summary = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    champion = str(summary.iloc[0]["model"])
    return summary, champion
