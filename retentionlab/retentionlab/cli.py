"""Command-line interface: run the whole study or one stage at a time."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, config
from .clean import clean_customers
from .datagen import generate_customers, save_raw
from .evaluate import evaluate_champion
from .features import build_features
from .models import build_model_registry, compare_models
from .plots import (
    apply_theme,
    effect_sizes,
    fig_calibration,
    fig_categorical_vs_churn,
    fig_confusion_matrix,
    fig_correlation_heatmap,
    fig_effect_sizes,
    fig_importance,
    fig_numeric_by_churn,
    fig_numeric_distributions,
    fig_pr_curves,
    fig_profit,
    fig_roc_curves,
    fig_target_balance,
)
from .profile import profile_dataframe
from .report import build_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retentionlab",
        description="RetentionLab — a reproducible, end-to-end customer-churn data science study.",
    )
    parser.add_argument("--version", action="version", version=f"retentionlab {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=Path, default=config.DEFAULT_DATA_DIR,
                        help="directory for the dataset (default: ./data)")
    common.add_argument("--outputs", type=Path, default=config.DEFAULT_OUTPUT_DIR,
                        help="directory for figures, tables, and reports (default: ./outputs)")
    common.add_argument("--seed", type=int, default=config.DEFAULT_SEED,
                        help="random seed (default: 42)")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    generate = sub.add_parser("generate", parents=[common], help="generate the synthetic dataset")
    generate.add_argument("--n", type=int, default=config.DEFAULT_N_CUSTOMERS,
                          help="number of customers (default: 6000)")

    sub.add_parser("profile", parents=[common], help="profile the raw dataset")
    sub.add_parser("clean", parents=[common], help="clean the raw dataset")
    sub.add_parser("eda", parents=[common], help="exploratory figures and effect sizes")
    sub.add_parser("features", parents=[common], help="engineer features and split data")
    sub.add_parser("models", parents=[common], help="compare models with cross-validation")
    sub.add_parser("evaluate", parents=[common], help="evaluate the champion on the test split")
    sub.add_parser("report", parents=[common], help="assemble the HTML report")
    pipeline = sub.add_parser("pipeline", parents=[common], help="run every stage end to end")
    pipeline.add_argument("--n", type=int, default=config.DEFAULT_N_CUSTOMERS,
                          help="number of customers to generate (default: 6000)")
    return parser


def _paths(args: argparse.Namespace) -> tuple[Path, Path]:
    return args.data_dir, args.outputs


def _load_raw(data_dir: Path) -> pd.DataFrame:
    raw_path = data_dir / config.RAW_FILE
    if not raw_path.exists():
        print(f"Missing {raw_path} — run `retentionlab generate` first.", file=sys.stderr)
        raise SystemExit(3)
    return pd.read_csv(raw_path)


def _load_clean(data_dir: Path) -> pd.DataFrame:
    clean_path = data_dir / config.CLEAN_FILE
    if not clean_path.exists():
        print(f"Missing {clean_path} — run `retentionlab clean` first.", file=sys.stderr)
        raise SystemExit(3)
    return pd.read_csv(clean_path)


def _load_comparison(outputs: Path) -> tuple[pd.DataFrame, str]:
    path = outputs / config.MODEL_COMPARISON_FILE
    if not path.exists():
        print(f"Missing {path} — run `retentionlab models` first.", file=sys.stderr)
        raise SystemExit(3)
    comparison = pd.read_csv(path)
    champion = str(comparison.iloc[0]["model"])
    return comparison, champion


def cmd_generate(args: argparse.Namespace) -> int:
    data_dir, _ = _paths(args)
    df = generate_customers(n=args.n, seed=args.seed)
    save_raw(df, data_dir / config.RAW_FILE)
    print(f"Generated {len(df):,} raw rows -> {data_dir / config.RAW_FILE}")
    print(f"True churn rate (before defects): {df[config.TARGET].mean():.1%}")
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    data_dir, outputs = _paths(args)
    report = profile_dataframe(_load_raw(data_dir))
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / config.PROFILE_FILE).write_text(report, encoding="utf-8")
    print(report)
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    data_dir, outputs = _paths(args)
    cleaned, report = clean_customers(_load_raw(data_dir))
    cleaned.to_csv(data_dir / config.CLEAN_FILE, index=False)
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / config.CLEANING_REPORT_FILE).write_text(report.to_text(), encoding="utf-8")
    print(report.to_text())
    return 0


def cmd_eda(args: argparse.Namespace) -> int:
    data_dir, outputs = _paths(args)
    df = _load_clean(data_dir)
    outputs.mkdir(parents=True, exist_ok=True)
    apply_theme()
    fig_target_balance(df, outputs / "fig_01_target_balance.png")
    fig_categorical_vs_churn(df, outputs / "fig_02_categorical_vs_churn.png")
    fig_numeric_distributions(df, outputs / "fig_03_numeric_distributions.png")
    fig_numeric_by_churn(df, outputs / "fig_04_numeric_by_churn.png")
    fig_correlation_heatmap(df, outputs / "fig_05_correlation_heatmap.png")
    frame = effect_sizes(df)
    frame.to_csv(outputs / config.EFFECT_SIZES_FILE, index=False)
    fig_effect_sizes(frame, outputs / "fig_06_effect_sizes.png")
    print("Exploratory analysis complete.")
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    data_dir, _ = _paths(args)
    df = _load_clean(data_dir)
    fs = build_features(df, seed=args.seed)
    feature_frame = pd.DataFrame(fs.X_train, columns=fs.feature_names)
    feature_frame["is_test"] = False
    test_frame = pd.DataFrame(fs.X_test, columns=fs.feature_names)
    test_frame["is_test"] = True
    pd.concat([feature_frame, test_frame]).to_csv(data_dir / config.FEATURES_FILE, index=False)
    print(f"Engineered {fs.X_train.shape[1]} features; "
          f"train {fs.X_train.shape[0]:,} / test {fs.X_test.shape[0]:,} rows "
          f"(stratified, test {fs.test_size:.0%}).")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    data_dir, outputs = _paths(args)
    fs = build_features(_load_clean(data_dir), seed=args.seed)
    comparison, champion = compare_models(fs, seed=args.seed)
    outputs.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(outputs / config.MODEL_COMPARISON_FILE, index=False)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nChampion: {champion}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    data_dir, outputs = _paths(args)
    df = _load_clean(data_dir)
    fs = build_features(df, seed=args.seed)
    _, champion_name = _load_comparison(outputs)
    champion_model = build_model_registry()[champion_name]()
    champion_model.fit(fs.X_train, fs.y_train)
    evaluation = evaluate_champion(fs, champion_name, champion_model)

    apply_theme()
    fpr, tpr = evaluation["roc"]
    precision, recall = evaluation["pr"]
    fig_roc_curves({champion_name: (fpr, tpr)}, outputs / "fig_07_roc.png")
    fig_pr_curves({champion_name: (precision, recall)}, float(fs.y_test.mean()),
                  outputs / "fig_08_pr.png")
    fig_confusion_matrix(evaluation["confusion"], outputs / "fig_09_confusion.png")
    centers, observed = evaluation["calibration"]
    fig_calibration(centers, observed, outputs / "fig_10_calibration.png")
    importance = evaluation["importance_univariate"]
    fig_importance(importance, outputs / "fig_11_importance.png")
    fig_profit(evaluation["profit_thresholds"], evaluation["profit"],
               outputs / "fig_12_profit.png")

    summary_lines = [
        "=" * 70,
        "EVALUATION SUMMARY",
        "=" * 70,
        f"Model:            {champion_name}",
        f"Test AUC:         {evaluation['test_auc']:.4f}",
        f"Youden threshold: {evaluation['youden_threshold']:.3f}",
        f"Accuracy:         {evaluation['accuracy']:.4f}",
        f"Precision:        {evaluation['precision']:.4f}",
        f"Recall:           {evaluation['recall']:.4f}",
        f"F1:               {evaluation['f1']:.4f}",
        f"Best profit:      ${evaluation['best_profit_per_customer']:.2f} per customer",
        f"                   at threshold {evaluation['best_threshold']:.2f}",
        "=" * 70,
    ]
    print("\n".join(summary_lines))
    (outputs / config.SUMMARY_FILE).write_text("\n".join(summary_lines), encoding="utf-8")

    def _to_json(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, tuple):
            return [_to_json(item) for item in value]
        return value

    serializable = {
        key: _to_json(value)
        for key, value in evaluation.items()
        if key not in ("importance_univariate", "importance_model")
    }
    with (outputs / "evaluation.json").open("w") as handle:
        json.dump(serializable, handle)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    data_dir, outputs = _paths(args)
    df = _load_clean(data_dir)
    _, cleaning_report = clean_customers(_load_raw(data_dir))
    comparison, champion_name = _load_comparison(outputs)
    effect_frame = pd.read_csv(outputs / config.EFFECT_SIZES_FILE)

    # Rebuild the champion evaluation deterministically (same seed and
    # hyperparameters as the `evaluate` stage) so the report is self-contained.
    fs = build_features(df, seed=args.seed)
    champion_model = build_model_registry()[champion_name]()
    champion_model.fit(fs.X_train, fs.y_train)
    evaluation = evaluate_champion(fs, champion_name, champion_model)

    target = build_report(
        outputs,
        comparison=comparison,
        champion_name=champion_name,
        evaluation=evaluation,
        cleaning_report=cleaning_report,
        effect_frame=effect_frame,
        churn_rate=float(df[config.TARGET].mean()),
    )
    print(f"Report written to {target}")
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    data_dir, outputs = _paths(args)
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    print("1/8 generating synthetic dataset ...")
    cmd_generate(args)
    print("\n2/8 profiling raw data ...")
    cmd_profile(args)
    print("\n3/8 cleaning ...")
    cmd_clean(args)
    print("\n4/8 exploratory analysis ...")
    cmd_eda(args)
    print("\n5/8 feature engineering ...")
    cmd_features(args)
    print("\n6/8 model comparison (5-fold CV) ...")
    cmd_models(args)
    print("\n7/8 champion evaluation ...")
    cmd_evaluate(args)
    print("\n8/8 assembling report ...")
    cmd_report(args)
    print(f"\nDone. Open {outputs / config.REPORT_FILE} in a browser.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {
        "generate": cmd_generate,
        "profile": cmd_profile,
        "clean": cmd_clean,
        "eda": cmd_eda,
        "features": cmd_features,
        "models": cmd_models,
        "evaluate": cmd_evaluate,
        "report": cmd_report,
        "pipeline": cmd_pipeline,
    }
    try:
        return commands[args.command](args)
    except KeyError:
        print("Unknown command.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
