"""Evaluation: curves, thresholds, calibration, and the business view.

Beyond raw accuracy, a churn model is a *decision system*: someone has to
pick a probability threshold and pay for retention offers.  This module closes
that loop with a profit curve — the expected profit per customer, given a cost
per targeted offer and the value of a retained customer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .stats import pr_curve, roc_auc, roc_curve, youden_threshold


def confusion_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    return tn, fp, fn, tp


def calibration_bins(
    y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Binned predicted-vs-observed probability (calibration curve points)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, observed = [], []
    for i in range(n_bins):
        mask = (scores >= edges[i]) & (scores < edges[i + 1])
        if i == n_bins - 1:
            mask |= scores == 1.0
        if mask.sum() >= 10:
            centers.append(float(scores[mask].mean()))
            observed.append(float(y_true[mask].mean()))
    return np.array(centers), np.array(observed)


def profit_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    cost: float = config.COST_OF_RETENTION_OFFER,
    value: float = config.VALUE_OF_RETAINED_CUSTOMER,
) -> tuple[np.ndarray, np.ndarray]:
    """Expected profit per customer at every decision threshold.

    ``profit = TP * value - (TP + FP) * cost`` — targeting churners earns the
    retention value; targeting non-churners costs the offer.
    """
    thresholds = np.unique(scores)
    profits = []
    for threshold in thresholds:
        _, fp, _, tp = confusion_at_threshold(y_true, scores, threshold)
        profits.append(tp * value - (tp + fp) * cost)
    return thresholds, np.array(profits) / len(y_true)


def univariate_importance(X: np.ndarray, y: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Model-agnostic feature importance: per-feature ROC AUC strength.

    Each feature is used alone as a churn score; ``strength = 2 * |AUC - 0.5|``
    lies in [0, 1] and is comparable across model types.
    """
    strengths = []
    for index, name in enumerate(names):
        strength = 2.0 * abs(roc_auc(y, X[:, index]) - 0.5)
        strengths.append((name, round(strength, 4)))
    frame = pd.DataFrame(strengths, columns=["feature", "strength"])
    return frame.sort_values("strength", ascending=False).reset_index(drop=True)


def coefficient_importance(model, names: list[str]) -> pd.DataFrame | None:
    """Standardized coefficient magnitudes for linear models, if available."""
    coefficients = getattr(model, "coef_", None)
    if coefficients is None:
        return None
    flat = np.asarray(coefficients).reshape(-1)
    if len(flat) != len(names):
        return None
    frame = pd.DataFrame({"feature": names, "coefficient": np.abs(flat)})
    return frame.sort_values("coefficient", ascending=False).reset_index(drop=True)


def evaluate_champion(
    fs, champion_name: str, fitted_model: object
) -> dict:
    """Run the full evaluation battery for the champion on the holdout set."""
    y_test = fs.y_test
    scores = fitted_model.predict_proba(fs.X_test)[:, 1]
    threshold, youden = youden_threshold(y_test, scores)
    tn, fp, fn, tp = confusion_at_threshold(y_test, scores, threshold)
    fpr, tpr, _ = roc_curve(y_test, scores)
    precision, recall, _ = pr_curve(y_test, scores)
    centers, observed = calibration_bins(y_test, scores)
    thresholds, profits = profit_curve(y_test, scores)
    best_profit = float(profits.max())
    best_threshold = float(thresholds[int(np.argmax(profits))])
    return {
        "model": champion_name,
        "test_auc": roc_auc(y_test, scores),
        "youden_threshold": threshold,
        "youden_j": youden,
        "confusion": (tn, fp, fn, tp),
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn),
        "roc": (fpr, tpr),
        "pr": (precision, recall),
        "calibration": (centers, observed),
        "profit_thresholds": thresholds,
        "profit": profits,
        "best_profit_per_customer": best_profit,
        "best_threshold": best_threshold,
        "importance_univariate": univariate_importance(fs.X_train, fs.y_train, fs.feature_names),
        "importance_model": coefficient_importance(fitted_model, fs.feature_names),
    }
