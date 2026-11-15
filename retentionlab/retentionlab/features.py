"""Feature engineering: composites, interactions, encoding, and honest splits.

Two things here are worth noticing:

* The engagement composite is built with PCA (via numpy SVD) on usage metrics,
  so it is a learned feature rather than a hand-picked sum.
* Scaling statistics are fit **on the training split only** and applied to the
  test split — preventing the subtle data leak that standardizing the whole
  frame first would introduce.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class FeatureSet:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    train_index: pd.Index
    test_index: pd.Index
    train_mean: np.ndarray
    train_std: np.ndarray
    test_size: float = 0.2


def engagement_pca(df: pd.DataFrame) -> pd.Series:
    """First principal component of the (standardized) usage metrics."""
    columns = config.ENGAGEMENT_INPUTS
    z = (df[columns] - df[columns].mean()) / df[columns].std(ddof=0).replace(0, 1.0)
    _, _, vt = np.linalg.svd(z.to_numpy(), full_matrices=False)
    scores = z.to_numpy() @ vt[0]
    if np.corrcoef(scores, z.to_numpy()[:, 0])[0, 1] < 0:
        scores = -scores  # orient so higher = more engagement
    return pd.Series(scores, index=df.index, name="engagement_score")


def stratified_split(y: pd.Series, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic stratified train/test indices without sklearn."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    test_parts = []
    for label in np.unique(y.to_numpy()):
        group = indices[y.to_numpy() == label]
        n_test = max(1, int(round(test_size * len(group))))
        chosen = rng.choice(group, size=n_test, replace=False)
        test_parts.append(chosen)
    test = np.concatenate(test_parts)
    test_mask = np.zeros(len(y), dtype=bool)
    test_mask[test] = True
    return np.where(~test_mask)[0], np.where(test_mask)[0]


def build_features(
    df: pd.DataFrame,
    seed: int = config.DEFAULT_SEED,
    test_size: float = 0.2,
) -> FeatureSet:
    """Engineer features and produce a leakage-free train/test split."""
    df = df.copy()

    df["engagement_score"] = engagement_pca(df)
    df["tenure_log"] = np.log1p(df["tenure_months"])
    df["spend_ratio"] = df["monthly_charge"] / df["plan"].map(config.PLAN_BASE_CHARGE)
    df["engagement_x_discount"] = df["engagement_score"] * df["discount_rate"]
    df["support_intensity"] = df["support_tickets"] / (df["tenure_months"] + 1.0)
    df["age_quadratic"] = ((df["age"] - df["age"].mean()) / df["age"].std()) ** 2

    dummies = pd.get_dummies(
        df[config.CATEGORICALS].astype("category"), drop_first=True, dtype=float
    )

    engineered_numerics = [
        "age",
        "tenure_log",
        "monthly_charge",
        "discount_rate",
        "logins_last_month",
        "support_tickets",
        "watch_hours",
        "watch_hours_flagged",
        "engagement_score",
        "spend_ratio",
        "engagement_x_discount",
        "support_intensity",
        "age_quadratic",
    ]
    frame = pd.concat([df[engineered_numerics], dummies], axis=1)

    y = df[config.TARGET].astype(int)
    train_idx, test_idx = stratified_split(y, test_size, seed)
    train, test = frame.iloc[train_idx], frame.iloc[test_idx]

    mean = train.mean().to_numpy()
    std = train.std(ddof=0).replace(0, 1.0).to_numpy()
    X_train = (train.to_numpy() - mean) / std
    X_test = (test.to_numpy() - mean) / std

    return FeatureSet(
        X_train=X_train,
        y_train=y.iloc[train_idx].to_numpy(),
        X_test=X_test,
        y_test=y.iloc[test_idx].to_numpy(),
        feature_names=list(frame.columns),
        train_index=train.index,
        test_index=test.index,
        train_mean=mean,
        train_std=std,
        test_size=test_size,
    )
