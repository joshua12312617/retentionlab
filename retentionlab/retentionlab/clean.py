"""The cleaning pipeline: from messy raw data to analysis-ready rows.

Every transformation is recorded in a :class:`CleaningReport` so the whole
story is reproducible and auditable — an important habit to demonstrate in a
portfolio project.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


@dataclass
class CleaningReport:
    rows_before: int = 0
    rows_after: int = 0
    duplicates_removed: int = 0
    rows_dropped_invalid: int = 0
    values_winsorized: int = 0
    values_imputed: dict[str, int] = field(default_factory=dict)
    labels_standardized: dict[str, int] = field(default_factory=dict)
    columns_dropped: list[tuple[str, str]] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            "=" * 70,
            "CLEANING REPORT",
            "=" * 70,
            f"Rows:           {self.rows_before:,} -> {self.rows_after:,}",
            f"Duplicates:     {self.duplicates_removed} removed",
            f"Invalid rows:   {self.rows_dropped_invalid} dropped "
            "(age <= 0 or > 100, negative tenure)",
            f"Winsorized:     {self.values_winsorized} watch_hours values capped "
            f"at the {config.WATCH_HOURS_CAP_QUANTILE:.0%} quantile",
            "",
            "Imputed values (median for skewed numerics, mode otherwise):",
        ]
        lines += [
            f"  {column:<20} {count:>6}" for column, count in sorted(self.values_imputed.items())
        ] or ["  (none)"]
        lines += ["", "Standardized labels:"]
        lines += [
            f"  {column:<20} {count:>6} values normalized"
            for column, count in sorted(self.labels_standardized.items())
        ] or ["  (none)"]
        lines += ["", "Dropped columns:"]
        lines += [f"  {column:<20} {reason}" for column, reason in self.columns_dropped] or [
            "  (none)"
        ]
        lines.append("=" * 70)
        return "\n".join(lines)


_REGION_MAP = {
    "north": "north",
    "south": "south",
    "east": "east",
    "west": "west",
    "north east": "north",
    "north-east": "north",
    "northeast": "north",
    "south-east": "south",
    "southeast": "south",
    "east coast": "east",
}
_PLAN_MAP = {
    "basic": "basic",
    "standard": "standard",
    "premium": "premium",
}


def _standardize_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    counts: dict[str, int] = {}
    df = df.copy()

    original_region = df["region"].astype(str)
    region = original_region.str.strip().str.lower()
    df["region"] = region.map(_REGION_MAP)
    counts["region"] = int((region.map(_REGION_MAP) != original_region).sum())

    original_plan = df["plan"].astype(str)
    plan = original_plan.str.strip().str.lower()
    df["plan"] = plan.map(_PLAN_MAP)
    counts["plan"] = int((df["plan"] != original_plan).sum())

    original_channel = df["signup_channel"].astype(str)
    channel = original_channel.str.strip().str.lower()
    df["signup_channel"] = channel
    counts["signup_channel"] = int((channel != original_channel).sum())

    # Unknown labels fall back to the most common category (mode).
    for column, allowed in (("region", config.REGIONS), ("plan", config.PLANS),
                            ("signup_channel", config.CHANNELS)):
        mode = df[column].mode().iloc[0]
        unknown = ~df[column].isin(allowed)
        counts[column] += int(unknown.sum())
        df.loc[unknown, column] = mode
    return df, counts


def _impute_numeric(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    counts: dict[str, int] = {}
    df = df.copy()
    for column in config.NUMERICS:
        missing = df[column].isna().sum()
        if missing == 0:
            continue
        counts[column] = int(missing)
        if abs(df[column].skew()) > 1.0:
            df[column] = df[column].fillna(df[column].median())
        else:
            df[column] = df[column].fillna(df[column].mean())
    return df, counts


def clean_customers(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean the raw dataset and return ``(cleaned, report)``."""
    report = CleaningReport(rows_before=len(raw))
    df = raw.copy()

    # 1. Exact duplicates (the generator appends 15 of them on purpose).
    before = len(df)
    df = df.drop_duplicates(ignore_index=True)
    report.duplicates_removed = before - len(df)

    # 2. Drop rows with impossible values.
    invalid = (df["age"] <= 0) | (df["age"] > 100) | (df["tenure_months"] < 0)
    report.rows_dropped_invalid = int(invalid.sum())
    df = df.loc[~invalid].reset_index(drop=True)

    # 3. Normalize categorical labels.
    df, report.labels_standardized = _standardize_labels(df)

    # 4. Winsorize extreme watch-hours and flag the affected rows.
    cap = df["watch_hours"].quantile(config.WATCH_HOURS_CAP_QUANTILE)
    above = df["watch_hours"] > cap
    report.values_winsorized = int(above.sum())
    df["watch_hours_flagged"] = above.astype(int)
    df.loc[above, "watch_hours"] = cap

    # 5. Impute missing numeric values.
    df, report.values_imputed = _impute_numeric(df)

    # 6. Drop constant, identifier, and leaking columns.
    for column in list(df.columns):
        if column == config.TARGET:
            continue
        unique = df[column].nunique(dropna=True)
        if unique <= 1:
            df = df.drop(columns=[column])
            report.columns_dropped.append((column, "constant column"))
        elif not pd.api.types.is_numeric_dtype(df[column]) and unique == len(df):
            df = df.drop(columns=[column])
            report.columns_dropped.append((column, "identifier (high cardinality)"))
        elif pd.api.types.is_numeric_dtype(df[column]):
            correlation = df[column].corr(df[config.TARGET])
            if abs(correlation) >= config.LEAKAGE_CORRELATION:
                df = df.drop(columns=[column])
                report.columns_dropped.append(
                    (column, f"leakage: |r| with target = {correlation:.2f}")
                )

    report.rows_after = len(df)
    return df.reset_index(drop=True), report
