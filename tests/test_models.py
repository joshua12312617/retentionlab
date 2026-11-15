"""Tests for the from-scratch models and the CV comparison harness."""

import unittest

import numpy as np
import pandas as pd

from retentionlab.features import FeatureSet
from retentionlab.models import (
    GaussianNB,
    KNNClassifier,
    LogisticRegressionNumpy,
    build_model_registry,
    compare_models,
)
from retentionlab.stats import roc_auc


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))


def make_feature_set(n: int = 400, seed: int = 0) -> FeatureSet:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    logit = 1.4 * X[:, 0] - 0.9 * X[:, 1] + 0.5 * X[:, 2]
    y = (rng.random(n) < _sigmoid(logit)).astype(int)
    split = int(n * 0.75)
    return FeatureSet(
        X_train=X[:split], y_train=y[:split],
        X_test=X[split:], y_test=y[split:],
        feature_names=[f"f{i}" for i in range(6)],
        train_index=pd.Index(range(split)),
        test_index=pd.Index(range(split, n)),
        train_mean=X[:split].mean(axis=0),
        train_std=X[:split].std(axis=0),
    )


class NumpyModelTests(unittest.TestCase):
    def test_logistic_regression_learns(self):
        fs = make_feature_set()
        model = LogisticRegressionNumpy(l2=1.0, seed=0).fit(fs.X_train, fs.y_train)
        scores = model.predict_proba(fs.X_test)[:, 1]
        self.assertGreater(roc_auc(fs.y_test, scores), 0.8)
        self.assertEqual((len(fs.X_test), 2), model.predict_proba(fs.X_test).shape)

    def test_gaussian_nb_smoke(self):
        fs = make_feature_set()
        model = GaussianNB().fit(fs.X_train, fs.y_train)
        probabilities = model.predict_proba(fs.X_test)
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(np.abs(probabilities.sum(axis=1) - 1) < 1e-9))
        self.assertGreater(roc_auc(fs.y_test, probabilities[:, 1]), 0.7)

    def test_knn_smoke(self):
        fs = make_feature_set()
        model = KNNClassifier(k=15).fit(fs.X_train, fs.y_train)
        scores = model.predict_proba(fs.X_test)[:, 1]
        self.assertGreater(roc_auc(fs.y_test, scores), 0.7)


class ComparisonTests(unittest.TestCase):
    def test_registry_always_contains_numpy_models(self):
        registry = build_model_registry()
        self.assertIn("logistic (numpy)", registry)
        self.assertIn("gaussian naive bayes", registry)
        self.assertIn("k-nearest neighbors", registry)

    def test_compare_models_returns_expected_frame(self):
        fs = make_feature_set(n=300, seed=5)
        summary, champion = compare_models(fs, folds=4, seed=5)
        required = {"model", "auc", "threshold", "accuracy", "precision", "recall", "f1"}
        self.assertTrue(required.issubset(summary.columns))
        self.assertIn("dummy baseline (majority)", set(summary["model"]))
        self.assertIn("logistic (numpy)", set(summary["model"]))
        self.assertEqual(summary.iloc[0]["model"], champion)
        self.assertTrue((summary["auc"].dropna() >= 0).all())
        self.assertTrue((summary["auc"].dropna() <= 1).all())


if __name__ == "__main__":
    unittest.main()
