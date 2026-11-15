"""Publication-style figures built with matplotlib + seaborn.

All figures use the Agg backend so they render identically in scripts, CI,
and notebooks.  Colors, grids, and typography follow one theme defined here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config
from .stats import cramers_v, point_biserial

PALETTE = ["#5b8c5a", "#d9822b", "#3e7c9e", "#b5678f", "#8a7fb5", "#c2a143"]
CHURN_COLORS = {"stayed": "#5b8c5a", "churned": "#d9822b"}


def apply_theme() -> None:
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, path: Path, name: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")
    return path


def fig_target_balance(df: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    counts = df[config.TARGET].value_counts().sort_index()
    labels = ["stayed", "churned"]
    bars = ax.bar(labels, counts.values, color=[CHURN_COLORS["stayed"], CHURN_COLORS["churned"]])
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 12,
                f"{count:,}\n({count / len(df):.1%})", ha="center", fontsize=9)
    ax.set_title("Churn rate in the cleaned dataset")
    ax.set_ylabel("customers")
    ax.margins(y=0.18)
    return save_figure(fig, path, "fig_01_target_balance.png")


def fig_categorical_vs_churn(df: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    overall = df[config.TARGET].mean()
    for ax, column in zip(axes, config.CATEGORICALS):
        sns.barplot(data=df, x=column, y=config.TARGET, ax=ax, hue=column,
                    legend=False, palette=PALETTE[: df[column].nunique()])
        ax.axhline(overall, color="#8a948b", linestyle="--", linewidth=1)
        ax.set_title(f"churn rate by {column}")
        ax.set_ylabel("churn rate")
        ax.text(0.02, 0.95, f"overall {overall:.1%}", transform=ax.transAxes,
                fontsize=8, va="top", color="#6b756d")
    fig.suptitle("Categorical features vs churn", fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_figure(fig, path, "fig_02_categorical_vs_churn.png")


def fig_numeric_distributions(df: pd.DataFrame, path: Path) -> Path:
    columns = [col for col in config.NUMERICS if col in df]
    rows = (len(columns) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(13, 2.6 * rows))
    for index, column in enumerate(columns):
        ax = axes.flat[index]
        sns.histplot(data=df, x=column, kde=True, ax=ax, color=PALETTE[index % len(PALETTE)],
                     linewidth=0)
        ax.set_title(column.replace("_", " "))
    for ax in axes.flat[len(columns):]:
        ax.axis("off")
    fig.suptitle("Distributions of numeric features", fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_figure(fig, path, "fig_03_numeric_distributions.png")


def fig_numeric_by_churn(df: pd.DataFrame, path: Path) -> Path:
    columns = [col for col in config.NUMERICS if col in df]
    rows = (len(columns) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(13, 2.6 * rows))
    for index, column in enumerate(columns):
        ax = axes.flat[index]
        sns.boxplot(data=df, x=config.TARGET, y=column, ax=ax, hue=config.TARGET,
                    legend=False,
                    palette=[CHURN_COLORS["stayed"], CHURN_COLORS["churned"]],
                    fliersize=1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["stayed", "churned"])
        ax.set_title(column.replace("_", " "))
    for ax in axes.flat[len(columns):]:
        ax.axis("off")
    fig.suptitle("Numeric features by churn status", fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_figure(fig, path, "fig_04_numeric_by_churn.png")


def fig_correlation_heatmap(df: pd.DataFrame, path: Path) -> Path:
    columns = [col for col in config.NUMERICS if col in df] + [config.TARGET]
    corr = df[columns].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(9.5, 7))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation matrix (numeric features and target)")
    return save_figure(fig, path, "fig_05_correlation_heatmap.png")


def effect_sizes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in config.CATEGORICALS:
        v = cramers_v(df[column], df[config.TARGET])
        rows.append({"feature": column, "kind": "categorical", "effect": "Cramér's V",
                     "value": round(v, 4)})
    for column in config.NUMERICS:
        r = point_biserial(df[config.TARGET], df[column])
        rows.append({"feature": column, "kind": "numeric", "effect": "point-biserial r",
                     "value": round(r, 4)})
    frame = pd.DataFrame(rows)
    frame["abs_value"] = frame["value"].abs()
    return frame.sort_values("abs_value", ascending=False).reset_index(drop=True)


def fig_effect_sizes(frame: pd.DataFrame, path: Path) -> Path:
    plot = frame.sort_values("abs_value")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [PALETTE[0] if kind == "categorical" else PALETTE[2]
              for kind in plot["kind"]]
    ax.barh(plot["feature"], plot["abs_value"], color=colors)
    for y, row in enumerate(plot.itertuples()):
        ax.text(row.abs_value + 0.004, y, f"{row.abs_value:.3f}", va="center", fontsize=8)
    ax.set_xlabel("|effect size|")
    ax.set_title("Feature association with churn (effect sizes)")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=PALETTE[0], label="Cramér's V (categorical)"),
        Patch(color=PALETTE[2], label="point-biserial r (numeric)"),
    ], loc="lower right")
    ax.set_xlim(0, plot["abs_value"].max() * 1.12)
    return save_figure(fig, path, "fig_06_effect_sizes.png")


def fig_roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5.4))
    ax.plot([0, 1], [0, 1], "--", color="#b9bfb9", label="random (AUC 0.50)")
    for index, (name, (fpr, tpr)) in enumerate(curves.items()):
        ax.plot(fpr, tpr, color=PALETTE[index % len(PALETTE)], linewidth=2,
                label=f"{name} (AUC {np.trapezoid(tpr, fpr):.3f})")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curves — test split")
    ax.legend(loc="lower right", fontsize=8)
    return save_figure(fig, path, "fig_07_roc.png")


def fig_pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], prevalence: float,
                  path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5.4))
    ax.axhline(prevalence, color="#b9bfb9", linestyle="--",
               label=f"no-skill baseline ({prevalence:.1%})")
    for index, (name, (precision, recall)) in enumerate(curves.items()):
        ax.plot(recall, precision, color=PALETTE[index % len(PALETTE)], linewidth=2,
                label=name)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-recall curves — test split")
    ax.legend(loc="lower left", fontsize=8)
    return save_figure(fig, path, "fig_08_pr.png")


def fig_confusion_matrix(confusion: tuple[int, int, int, int], path: Path) -> Path:
    tn, fp, fn, tp = confusion
    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Greens", cbar=False,
                xticklabels=["predicted stay", "predicted churn"],
                yticklabels=["actual stay", "actual churn"], ax=ax, annot_kws={"fontsize": 13})
    ax.set_title("Confusion matrix at the Youden threshold")
    return save_figure(fig, path, "fig_09_confusion.png")


def fig_calibration(centers: np.ndarray, observed: np.ndarray, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.plot([0, 1], [0, 1], "--", color="#b9bfb9", label="perfect calibration")
    ax.plot(centers, observed, "o-", color=PALETTE[0], linewidth=2, label="model")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed churn frequency")
    ax.set_title("Calibration curve (10 bins, test split)")
    ax.legend(loc="upper left")
    return save_figure(fig, path, "fig_10_calibration.png")


def fig_importance(importance: pd.DataFrame, path: Path, top: int = 15) -> Path:
    plot = importance.head(top).sort_values("strength")
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.barh(plot["feature"], plot["strength"], color=PALETTE[4])
    for y, row in enumerate(plot.itertuples()):
        ax.text(row.strength + 0.004, y, f"{row.strength:.3f}", va="center", fontsize=8)
    ax.set_xlabel("strength = 2 · |AUC − 0.5|")
    ax.set_title("Feature importance (univariate AUC, model-agnostic)")
    ax.set_xlim(0, plot["strength"].max() * 1.12)
    return save_figure(fig, path, "fig_11_importance.png")


def fig_profit(thresholds: np.ndarray, profits: np.ndarray, path: Path) -> Path:
    best = int(np.argmax(profits))
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(thresholds, profits, color=PALETTE[0], linewidth=2)
    ax.plot(thresholds[best], profits[best], "o", color=PALETTE[1], markersize=7)
    ax.annotate(
        f"best: threshold {thresholds[best]:.2f}\nprofit {profits[best]:.2f}/customer",
        xy=(thresholds[best], profits[best]),
        xytext=(thresholds[best] - 0.22, profits[best] - (profits.max() - profits.min()) * 0.35),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "#6b756d"},
    )
    ax.axhline(0, color="#b9bfb9", linewidth=1)
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("expected profit per customer ($)")
    ax.set_title(
        "Profit curve — value of a saved customer vs cost of a retention offer"
    )
    return save_figure(fig, path, "fig_12_profit.png")
