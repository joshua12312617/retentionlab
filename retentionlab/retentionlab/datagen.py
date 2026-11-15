"""Generate a realistic (but synthetic) subscription-churn dataset.

The data-generating process (DGP) is fully specified here so that the rest of
the project runs against *known ground truth*: we know which defects were
injected, which variable leaks the target, and roughly how strong each signal
is.  That makes this a controlled experiment in which the pipeline can be
judged honestly — the same reason pharmacologists test against a known dose.

The fictional product is a subscription streaming service.  Churn log-odds
are:

    logit(churn) = -1.15
        - 0.90 * z(engagement)            # engaged customers stay
        - 0.45 * z(log1p(tenure))         # long-tenure customers stay
        + 0.50 * z(support_tickets)       # friction drives churn
        + 0.35 * discount_rate            # promo-only customers churn more
        + 0.30 * z(age) ** 2              # the young and old churn most
        + region_effect + plan_effect
        + 0.25 * z(monthly_charge) * discount_rate   # interaction

Defects injected on purpose, to exercise the cleaning pipeline:

* 15 exact duplicate rows
* watch_hours missing for ~8% of rows, *not at random* (churners more likely
  to have missing watch data); support_tickets and monthly_charge missing MAR
* a handful of impossible values (age = 0, tenure = -1)
* ~1% of watch_hours multiplied by 10–50 (data-entry outliers)
* inconsistent region/plan/channel labels ("North East", "premium ", "ads ")
* a `risk_score` column computed FROM the target — the classic leakage trap
  that looks great in cross-validation and useless in production
* a constant `data_source` column and a high-cardinality `customer_id`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _zscore(values: np.ndarray) -> np.ndarray:
    std = values.std()
    return (values - values.mean()) / std if std > 0 else np.zeros_like(values)


def generate_customers(n: int = config.DEFAULT_N_CUSTOMERS, seed: int = config.DEFAULT_SEED) -> pd.DataFrame:
    """Generate ``n`` raw customer rows with realistic defects included."""
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------- demographics
    age = np.clip(rng.normal(38.0, 12.0, n), 18, 78)
    region = rng.choice(config.REGIONS, n, p=[0.30, 0.25, 0.25, 0.20])
    channel = rng.choice(config.CHANNELS, n, p=[0.35, 0.20, 0.20, 0.25])
    plan = rng.choice(config.PLANS, n, p=[0.30, 0.45, 0.25])
    tenure = np.clip(rng.gamma(shape=1.6, scale=14.0, size=n), 0, 60).astype(int)

    base_charge = np.array([config.PLAN_BASE_CHARGE[p] for p in plan])
    monthly_charge = base_charge * (1.0 + rng.normal(0.0, 0.06, n))
    discount_rate = rng.choice([0.0, 0.10, 0.20, 0.30], n, p=[0.45, 0.25, 0.20, 0.10])

    # -------------------------------------------------------------- engagement
    # Latent engagement is a blend of demographics and unobserved noise.
    region_effect = {"north": 0.00, "south": -0.10, "east": 0.15, "west": -0.05}
    channel_effect = {"organic": 0.25, "referral": 0.05, "partner": -0.15, "ads": -0.20}
    plan_effect = {"basic": -0.20, "standard": 0.05, "premium": 0.25}
    engagement = (
        0.35 * _zscore(age)
        + np.array([region_effect[r] for r in region])
        + np.array([channel_effect[c] for c in channel])
        + np.array([plan_effect[p] for p in plan])
        + rng.normal(0.0, 0.75, n)
    )

    login_rate = np.clip(np.exp(0.55 * engagement + np.log(3.2)), 0.2, 60.0)
    logins = rng.poisson(login_rate).astype(int)
    watch_hours = rng.gamma(shape=2.2, scale=2.5 * np.exp(0.40 * engagement), size=n)
    ticket_rate = np.clip(1.1 - 0.30 * engagement, 0.2, 6.0)
    tickets = np.where(rng.random(n) < 0.35, 0, rng.poisson(ticket_rate)).astype(int)

    # -------------------------------------------------------------------- churn
    region_churn = {"north": 0.00, "south": 0.25, "east": -0.30, "west": 0.15}
    plan_churn = {"basic": 0.15, "standard": 0.00, "premium": -0.35}
    logit = (
        -1.15
        - 0.90 * _zscore(engagement)
        - 0.45 * _zscore(np.log1p(tenure))
        + 0.50 * _zscore(tickets.astype(float))
        + 0.35 * discount_rate
        + 0.30 * _zscore(age) ** 2
        + np.array([region_churn[r] for r in region])
        + np.array([plan_churn[p] for p in plan])
        + 0.25 * _zscore(monthly_charge) * discount_rate
    )
    churned = rng.random(n) < _sigmoid(logit)

    df = pd.DataFrame(
        {
            config.ID_COLUMN: [f"C{i + 1:05d}" for i in range(n)],
            "age": np.round(age, 1),
            "region": region,
            "signup_channel": channel,
            "plan": plan,
            "tenure_months": tenure,
            "monthly_charge": np.round(monthly_charge, 2),
            "discount_rate": discount_rate,
            "logins_last_month": logins,
            "support_tickets": tickets,
            "watch_hours": np.round(watch_hours, 1),
            config.TARGET: churned.astype(int),
        }
    )

    # ------------------------------------------------------- leakage & noise
    df["risk_score"] = np.clip(0.55 * churned + 0.45 * rng.random(n), 0.0, 1.0)
    df["data_source"] = "batch-2025"

    # ---------------------------------------------------- label inconsistency
    region_variants = {"north": "North East", "south": "south-east", "east": "East Coast"}
    mask = rng.random(n) < 0.10
    df.loc[mask, "region"] = df.loc[mask, "region"].map(
        lambda value: region_variants.get(value, value)
    )
    mask = rng.random(n) < 0.08
    df.loc[mask, "plan"] = df.loc[mask, "plan"].str.upper()
    mask = rng.random(n) < 0.05
    df.loc[mask, "signup_channel"] = df.loc[mask, "signup_channel"] + " "

    # ------------------------------------------- missingness (MNAR and MAR)
    df.loc[rng.random(len(df)) < 0.05 + 0.06 * df[config.TARGET], "watch_hours"] = np.nan
    df.loc[rng.random(len(df)) < 0.02, "support_tickets"] = np.nan
    df.loc[rng.random(len(df)) < 0.03, "monthly_charge"] = np.nan

    # ------------------------------------------------------ impossible values
    # The two injections use disjoint rows so exactly 9 rows are invalid.
    age_bad = rng.choice(len(df), size=5, replace=False)
    df.loc[age_bad, "age"] = 0.0
    remaining = df.index.difference(age_bad)
    tenure_bad = rng.choice(remaining, size=4, replace=False)
    df.loc[tenure_bad, "tenure_months"] = -1

    # ---------------------------------------------------------------- outliers
    outlier_mask = rng.random(len(df)) < 0.01
    df.loc[outlier_mask, "watch_hours"] = (
        df.loc[outlier_mask, "watch_hours"] * rng.uniform(10.0, 50.0, outlier_mask.sum())
    )

    # ------------------------------------------------------------ duplicates
    # Appended *after* every defect so each duplicate is an exact copy of its
    # original and the cleaner removes precisely 15 rows.
    duplicate_idx = rng.choice(n, size=15, replace=False)
    df = pd.concat([df, df.iloc[duplicate_idx]], ignore_index=True)

    return df


def save_raw(df: pd.DataFrame, path: Path) -> None:
    """Persist the raw dataset (used by the CLI ``generate`` command)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
