"""Assemble the final self-contained HTML report with embedded figures."""

from __future__ import annotations

import base64
import html as html_lib
from pathlib import Path

import pandas as pd

from . import __version__, config
from .clean import CleaningReport

_CSS = """
:root { --ink:#26302a; --muted:#6b756d; --line:#e4e8e2; --bg:#f6f7f4;
        --card:#ffffff; --green:#5b8c5a; --orange:#d9822b; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:15px/1.6 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.wrap { max-width:1000px; margin:0 auto; padding:42px 24px 80px; }
.hero { border-bottom:1px solid var(--line); padding-bottom:22px; margin-bottom:26px; }
.hero h1 { font-size:30px; letter-spacing:-.6px; margin:0 0 6px; }
.hero .meta { color:var(--muted); font-size:13px; }
.cards { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:26px 0 8px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 18px; }
.card .value { font-size:25px; font-weight:700; letter-spacing:-.5px; }
.card .label { color:var(--muted); font-size:12px; margin-top:2px; }
h2 { font-size:20px; margin:40px 0 12px; letter-spacing:-.3px; }
p { margin:8px 0; }
figure { margin:0 0 18px; }
figure img { width:100%; border:1px solid var(--line); border-radius:12px; background:#fff; }
figcaption { color:var(--muted); font-size:12px; margin-top:6px; }
table { border-collapse:collapse; width:100%; font-size:13px; background:var(--card); }
th, td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:.6px; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
.note { background:#eef3ec; border:1px solid #dbe6d9; border-radius:10px; padding:12px 16px;
        font-size:13px; color:#3c5742; margin:14px 0; }
.warn { background:#faf0e4; border:1px solid #ecdcbf; color:#7c5a2a; }
footer { margin-top:46px; color:#9aa19b; font-size:12px;
         border-top:1px solid var(--line); padding-top:14px; }
@media (max-width:720px) { .cards { grid-template-columns:repeat(2,1fr); } }
"""

_FIGURES = [
    ("fig_01_target_balance.png", "The cleaned dataset is moderately imbalanced — fine for AUC-based evaluation, and the reason the report uses Youden's threshold instead of 0.5."),
    ("fig_02_categorical_vs_churn.png", "Churn rate by region, signup channel, and plan. The dashed line marks the overall rate."),
    ("fig_03_numeric_distributions.png", "Distributions after cleaning. Skewed usage metrics motivated median imputation."),
    ("fig_04_numeric_by_churn.png", "Boxplots split by churn status — engagement metrics separate the groups clearly."),
    ("fig_05_correlation_heatmap.png", "Pairwise correlations. None of the retained features is a proxy for the target."),
    ("fig_06_effect_sizes.png", "Effect sizes beat p-values at this sample size: Cramér's V and point-biserial r rank what actually drives churn."),
    ("fig_07_roc.png", "Receiver operating characteristic for every model on the holdout test split."),
    ("fig_08_pr.png", "Precision-recall curves; the baseline is the churn prevalence."),
    ("fig_09_confusion.png", "Confusion matrix of the champion model at the Youden-optimal threshold."),
    ("fig_10_calibration.png", "Calibration: how well predicted probabilities match observed churn frequencies."),
    ("fig_11_importance.png", "Model-agnostic feature importance — each feature's standalone AUC strength."),
    ("fig_12_profit.png", "The decision view: expected profit per customer across thresholds, combining offer cost and retention value."),
]


def _esc(value: object) -> str:
    return html_lib.escape(str(value), quote=True)


def _card(value: str, label: str) -> str:
    return f'<div class="card"><div class="value">{_esc(value)}</div><div class="label">{_esc(label)}</div></div>'


def _table(frame: pd.DataFrame, numeric: set[int] | None = None) -> str:
    numeric = numeric or set()
    head = "".join(
        f'<th class="{"num" if i in numeric else ""}">{_esc(col)}</th>'
        for i, col in enumerate(frame.columns)
    )
    body = []
    for _, row in frame.iterrows():
        cells = "".join(
            f'<td class="{"num" if i in numeric else ""}">{_esc(value)}</td>'
            for i, value in enumerate(row)
        )
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _embed_image(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_report(
    output_dir: Path,
    *,
    comparison: pd.DataFrame,
    champion_name: str,
    evaluation: dict,
    cleaning_report: CleaningReport,
    effect_frame: pd.DataFrame,
    churn_rate: float,
) -> Path:
    """Write ``report.html`` into ``output_dir`` and return its path."""
    figures_html = []
    for filename, caption in _FIGURES:
        image_path = output_dir / filename
        if image_path.exists():
            figures_html.append(
                f'<figure><img src="{_embed_image(image_path)}" alt="{_esc(filename)}">'
                f"<figcaption>{_esc(caption)}</figcaption></figure>"
            )

    leakage = [column for column, reason in cleaning_report.columns_dropped if "leakage" in reason]
    metrics = evaluation
    test_auc = metrics["test_auc"]

    comparison_display = comparison.copy()
    if "error" in comparison_display.columns:
        comparison_display = comparison_display.drop(columns=["error"])
    numeric_cols = {i for i, col in enumerate(comparison_display.columns)
                    if col not in ("model",)}

    effect_display = effect_frame.drop(columns=["abs_value"])

    cards = (
        _card(f"{churn_rate:.1%}", "churn rate (cleaned data)")
        + _card(f"{test_auc:.3f}", "champion test AUC")
        + _card(f"{metrics['precision']:.2f} / {metrics['recall']:.2f}", "precision / recall")
        + _card(f"${metrics['best_profit_per_customer']:.2f}", "best profit per customer")
    )

    top_features = metrics["importance_univariate"].head(5)
    features_list = ", ".join(
        f"{row.feature} ({row.strength:.2f})" for row in top_features.itertuples()
    )

    leakage_note = ""
    if leakage:
        leakage_note = (
            f'<div class="note warn"><strong>Leakage found and removed:</strong> '
            f"{_esc(', '.join(leakage))}. These columns were derived from the target — "
            "they score brilliantly in cross-validation but would be unavailable (or "
            "meaningless) in production. Dropping them is what makes the reported "
            "metrics honest.</div>"
        )

    report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RetentionLab report</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <h1>RetentionLab — customer churn study</h1>
    <div class="meta">Reproducible end-to-end pipeline: synthetic data with known
      ground truth → cleaning → EDA → feature engineering → model comparison →
      evaluation · retentionlab v{_esc(__version__)}</div>
  </div>

  <div class="cards">{cards}</div>

  {leakage_note}

  <div class="note">Champion model: <strong>{_esc(champion_name)}</strong>.
  The five strongest features are {features_list}. At the Youden threshold of
  {metrics['youden_threshold']:.2f}, the model finds {metrics['recall']:.0%} of
  churners at {metrics['precision']:.0%} precision.</div>

  <h2>1 · The data</h2>
  {figures_html[0] if len(figures_html) > 0 else ""}
  {figures_html[1] if len(figures_html) > 1 else ""}
  {figures_html[2] if len(figures_html) > 2 else ""}

  <h2>2 · Exploration</h2>
  {figures_html[3] if len(figures_html) > 3 else ""}
  {figures_html[4] if len(figures_html) > 4 else ""}
  {figures_html[5] if len(figures_html) > 5 else ""}
  <h2>3 · Effect sizes</h2>
  {_table(effect_display, numeric={3})}

  <h2>4 · Model comparison (5-fold CV)</h2>
  {_table(comparison_display, numeric=numeric_cols)}

  <h2>5 · Evaluation</h2>
  {figures_html[6] if len(figures_html) > 6 else ""}
  {figures_html[7] if len(figures_html) > 7 else ""}
  {figures_html[8] if len(figures_html) > 8 else ""}
  {figures_html[9] if len(figures_html) > 9 else ""}

  <h2>6 · Interpretation &amp; decisions</h2>
  {figures_html[10] if len(figures_html) > 10 else ""}
  {figures_html[11] if len(figures_html) > 11 else ""}

  <h2>7 · Cleaning audit</h2>
  <pre style="background:var(--card);border:1px solid var(--line);border-radius:12px;
              padding:16px;font-size:12px;overflow-x:auto;">{_esc(cleaning_report.to_text())}</pre>

  <footer>Generated by RetentionLab {_esc(__version__)} · figures: matplotlib + seaborn ·
  metrics: numpy from first principles · models: numpy implementations plus scikit-learn
  when available</footer>
</div>
</body>
</html>
"""
    target = output_dir / config.REPORT_FILE
    target.write_text(report, encoding="utf-8")
    return target
