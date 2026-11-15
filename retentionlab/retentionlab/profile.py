"""Data profiling: a readable first look at the raw dataset."""

from __future__ import annotations

import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> str:
    """Return a human-readable profiling report for ``df``."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("RAW DATA PROFILE")
    lines.append("=" * 70)
    lines.append(f"Shape:        {df.shape[0]:,} rows x {df.shape[1]} columns")
    lines.append(f"Memory:       {df.memory_usage(deep=True).sum() / 1024:.1f} KiB")
    lines.append(f"Duplicates:   {int(df.duplicated().sum())} exact duplicate rows")
    lines.append("")
    lines.append("Missing values:")
    missing = df.isna().sum()
    if missing.sum() == 0:
        lines.append("  (none)")
    for column, count in missing.items():
        if count:
            lines.append(f"  {column:<20} {count:>6}  ({count / len(df):.1%})")
    lines.append("")
    lines.append("Column summary:")
    lines.append(f"  {'column':<20} {'dtype':<10} {'unique':>7}  notes")
    for column in df.columns:
        values = df[column]
        unique = values.nunique(dropna=True)
        notes = []
        if unique <= 1:
            notes.append("CONSTANT")
        elif not pd.api.types.is_numeric_dtype(values) and unique == len(df):
            notes.append("identifier-like (high cardinality)")
        if column != "churned" and pd.api.types.is_numeric_dtype(values):
            correlation = values.corr(df["churned"]) if "churned" in df else None
            if correlation is not None and abs(correlation) >= 0.6:
                notes.append(f"suspicious |r| with target = {correlation:.2f}")
        lines.append(
            f"  {column:<20} {str(values.dtype):<10} {unique:>7}  {', '.join(notes) or '-'}"
        )
    lines.append("=" * 70)
    return "\n".join(lines)
