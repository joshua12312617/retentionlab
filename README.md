# RetentionLab

**A reproducible, end-to-end data science study** — from a messy, synthetic
customer dataset with *known ground truth* all the way to model comparison,
evaluation, and a self-contained HTML report.

Built with **pandas · numpy · matplotlib · seaborn** (plus **scipy** and
**scikit-learn** when installed). Three of the models are implemented from
scratch with numpy, and every metric (ROC/PR AUC, Youden's threshold) is
computed from first principles — so the project shows both *using* the
standard stack and *understanding* what it does under the hood.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-30%20passing-brightgreen)](tests/)

---

## The idea

Most tutorial projects download a clean CSV and fit a model. RetentionLab is
the opposite: the **data-generating process (DGP) is fully specified in code**
([`retentionlab/datagen.py`](retentionlab/datagen.py)), including the churn
log-odds equation — and a set of deliberately injected defects:

- 15 exact duplicate rows
- missing values that are **not at random** (churners are more likely to have
  missing watch-time data) plus MAR gaps in other columns
- impossible values (`age = 0`, `tenure = -1`)
- 1% data-entry outliers in watch hours
- inconsistent labels (`"North East"`, `"PREMIUM"`, `"ads "`)
- a **leakage column** (`risk_score`) computed *from the target* — the classic
  trap that looks brilliant in cross-validation and is useless in production
- a constant column and a high-cardinality identifier

Because the truth is known, every stage of the pipeline can be judged honestly:
did the cleaner catch the leakage? did it impute sanely? does the model recover
the engineered signals?

## Quick start

```sh
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # Windows
# source .venv/bin/python -m pip install -r requirements.txt   # macOS/Linux

.venv\Scripts\python -m retentionlab pipeline --n 6000
# then open outputs/report.html in a browser
```

The pipeline has nine stages — generate → profile → clean → EDA → features →
model comparison → evaluation → report — and each can also be run on its own
(`generate`, `profile`, `clean`, `eda`, `features`, `models`, `evaluate`,
`report`; every stage accepts `--data-dir`, `--outputs`, and `--seed`).

## What the pipeline produces

```text
outputs/
├── 01_profile.txt              raw-data audit (missingness, suspicious columns)
├── 02_cleaning_report.txt      every cleaning decision, with counts
├── 05_effect_sizes.csv         Cramér's V and point-biserial r per feature
├── 04_model_comparison.csv     5-fold CV results for every model
├── evaluation.json             curves, thresholds, calibration, profit
├── fig_01 … fig_12 .png        publication-style figures (matplotlib + seaborn)
├── summary.txt                 headline numbers
└── report.html                 the whole study in one shareable file
```

## Real results (n = 6,000, seed 42)

**Cleaning audit:** 6,015 rows → 5,991 · 15 duplicates removed · 9 impossible
rows dropped · 56 outliers winsorized and flagged · 716 values imputed
(median for skewed features) · 1,224 inconsistent labels normalized ·
`risk_score` **dropped as leakage** (|r| with target = 0.89), `customer_id`
and the constant `data_source` dropped.

**Effect sizes** (ranked): support tickets (r = .26) and engagement metrics
(logins r = −.20, watch hours r = −.14) lead; the `discount_rate` effect hides
behind an interaction with spend — a nice example of why effect sizes need
multivariate models.

**Model comparison (5-fold stratified CV):**

| Model | CV AUC |
|---|---|
| logistic (sklearn) | **0.746** |
| random forest (sklearn) | 0.733 |
| gradient boosting (sklearn) | 0.732 |
| gaussian naive bayes (numpy) | 0.712 |
| k-nearest neighbors (numpy) | 0.694 |
| logistic (numpy) | 0.640 |
| dummy baseline | 0.500 |

**Champion on the holdout test split:** AUC **0.777** · recall 78.7% ·
precision 51.2% at the Youden threshold (0.30) · F1 0.62.

**The decision view:** with a $15 retention-offer cost and $120 value per
saved customer, the best threshold yields **$27.07 expected profit per
customer** — the pipeline finds the profit-maximizing operating point instead
of stopping at accuracy.

## Features worth talking about 

- **Honest splits:** scaling statistics are fit on the training split only.
- **Leakage detection:** numeric features with |corr| ≥ 0.6 against the
  target are dropped automatically and reported.
- **Effect sizes over p-values:** at n = 6,000 everything is "significant";
  the pipeline ranks what *matters* (Cramér's V, point-biserial r).
- **Metrics from scratch:** ROC/PR curves, AUC, and the Youden threshold are
  numpy implementations in [`retentionlab/stats.py`](retentionlab/stats.py) —
  identical across backends.
- **From-scratch models:** logistic regression (mini-batch gradient descent
  with L2), Gaussian Naive Bayes, and k-NN run everywhere; scikit-learn's
  random forest and gradient boosting join automatically when installed.
- **Business framing:** the profit curve turns the classifier into a decision
  system with a cost/value trade-off.
- **Reproducibility:** fixed seed, deterministic stages, every artifact saved.

## Project structure

```text
retentionlab/
├── config.py        paths, constants, schema, business parameters
├── datagen.py       the fully specified data-generating process + defects
├── profile.py       raw-data audit
├── clean.py         dedupe → invalid rows → labels → winsorize → impute → leakage
├── stats.py         Cramér's V, point-biserial r, AUC, ROC/PR, Youden (numpy)
├── features.py      PCA engagement score, interactions, honest stratified split
├── models.py        numpy models + optional sklearn models + CV comparison
├── evaluate.py      confusion, calibration, profit curve, feature importance
├── plots.py         the figure theme and all 12 charts
├── report.py        self-contained HTML report with embedded figures
└── cli.py           the nine pipeline commands
tests/               30 tests: DGP, cleaning, stats, features, models, pipeline
```

## Development

```sh
.venv\Scripts\python -m unittest discover -s tests -v
```

The suite covers the injected defects exactly (15 duplicates, 9 invalid rows),
statistics against known values, from-scratch models on separable data, and a
full end-to-end pipeline run asserting every artifact appears.

## Using it on real data

The pipeline is deliberately decoupled from the synthetic generator. Drop any
CSV with the columns in [`retentionlab/config.py`](retentionlab/config.py)
(`customer_id`, `age`, `region`, `signup_channel`, `plan`, `tenure_months`,
`monthly_charge`, `discount_rate`, `logins_last_month`, `support_tickets`,
`watch_hours`, `churned`) into `data/raw_customers.csv` and run
`retentionlab clean` onward — the cleaning, EDA, modeling, and reporting stages
work unchanged.

## Limitations

- The dataset is synthetic: results are illustrative of *methodology*, not of
  any real customer base.
- Effects are engineered into the DGP, so the study demonstrates that the
  pipeline *recovers* known signals — the right way to validate a pipeline.
- For production work you would add: temporal (backtesting) splits instead of
  random ones, hyperparameter search for the tree models, SHAP-style
  attributions, and drift monitoring.

## License

MIT — see [LICENSE](LICENSE).
