"""Effect sizes and small statistics implemented with numpy.

The pipeline reports *effect sizes* (Cramér's V, point-biserial r, Cohen's d)
rather than raw p-values: with thousands of rows almost everything is
"significant", while effect size tells us what actually matters.  scipy is
used opportunistically when installed (for exact p-values and distribution
functions); the numpy implementations are self-contained and tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # optional dependency
    from scipy import stats as _scipy_stats

    HAS_SCIPY = True
except ImportError:  # pragma: no cover - exercised only without scipy
    _scipy_stats = None
    HAS_SCIPY = False


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Cramér's V measure of association between two categorical series."""
    table = pd.crosstab(a, b).values.astype(float)
    row_totals = table.sum(axis=1, keepdims=True)
    col_totals = table.sum(axis=0, keepdims=True)
    n = table.sum()
    if n == 0:
        return 0.0
    expected = row_totals @ col_totals / n
    chi2 = float(np.sum((table - expected) ** 2 / np.maximum(expected, 1e-12)))
    min_dim = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else 0.0


def chi2_pvalue(a: pd.Series, b: pd.Series) -> float | None:
    """Exact chi-square p-value when scipy is available, otherwise ``None``."""
    if not HAS_SCIPY:
        return None
    table = pd.crosstab(a, b).values
    result = _scipy_stats.chi2_contingency(table, correction=False)
    return float(result.pvalue)


def point_biserial(binary: pd.Series, continuous: pd.Series) -> float:
    """Point-biserial correlation between a binary and a continuous series."""
    x = binary.astype(float).to_numpy()
    y = continuous.astype(float).to_numpy()
    sd = float(y.std())
    if sd == 0:
        return 0.0
    p = float(x.mean())
    m1 = float(y[x == 1].mean())
    m0 = float(y[x == 0].mean())
    return ((m1 - m0) / sd) * np.sqrt(p * (1 - p))


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cohen's d for two independent groups (pooled standard deviation)."""
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """Area under a curve sampled at points ``(x, y)`` (trapezoid rule)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return 0.0
    order = np.argsort(x)
    return float(np.trapezoid(y[order], x[order]))


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC from labels and predicted scores (same ranking as any AUC)."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return auc(fpr, tpr)


def roc_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FPR/TPR/thresholds for a ROC curve (descending score order)."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s)
    y = y[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1.0 - y)
    total_pos = tps[-1]
    total_neg = fps[-1]
    if total_pos == 0 or total_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
    tpr = np.concatenate([[0.0], tps / total_pos])
    fpr = np.concatenate([[0.0], fps / total_neg])
    return fpr, tpr, s[order]


def pr_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision/recall/thresholds for a precision-recall curve."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s)
    y = y[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1.0 - y)
    precision = tps / np.maximum(tps + fps, 1e-12)
    recall = tps / max(tps[-1], 1e-12)
    return np.concatenate([[1.0], precision]), np.concatenate([[0.0], recall]), s[order]


def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Threshold maximizing Youden's J (TPR - FPR) and the J value itself."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr
    best = int(np.argmax(j))
    return float(thresholds[best - 1]) if best > 0 else float(thresholds[0]), float(j[best])
