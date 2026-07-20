#!/usr/bin/env python3
"""Feature-family ablation for CICIoV2024-to-HCRL transfer.

This exploratory study trains nine class-balanced logistic-regression models:

1. all 63 engineered features;
2. each of four feature families by itself; and
3. all features with each family removed in turn.

Every model is fitted only on the CICIoV2024 group-disjoint training split.
Its decision threshold is selected only on the CICIoV2024 group-disjoint
validation split.  The frozen model and threshold are then evaluated on the
CICIoV2024 test split and the complete HCRL Car-Hacking external dataset.
No HCRL label is used for fitting, scaling, or threshold selection.

Because the ablations were designed after inspecting the first HCRL zero-shot
result, treat this as exploratory diagnosis rather than a new confirmatory
external validation.  A separate unseen capture is still required later.

Run from D:\\ztav_project after Steps 13 and 14:

    .\\.venv\\Scripts\\python.exe src\\15_feature_family_ablation.py

This is a research prototype, not production automotive safety software.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
WINDOW_SIZE = 100
PREDICTION_BATCH_ROWS = 20_000
FAMILY_ORDER = (
    "identifier_and_frame_structure",
    "payload_extremes",
    "payload_dynamics",
    "payload_statistics",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CICIoV-to-HCRL feature-family ablation study."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--no-save-models",
        action="store_true",
        help="Do not retain the nine exploratory fitted pipelines.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def feature_family(feature: str) -> str:
    if feature.startswith("id_") or feature.startswith("frame_") or feature.startswith(
        ("dominant_id_", "consecutive_frame_")
    ):
        return "identifier_and_frame_structure"
    if feature.endswith(("_zero_fraction", "_ff_fraction")):
        return "payload_extremes"
    if feature.endswith("_mean_abs_change"):
        return "payload_dynamics"
    if feature.startswith("data_"):
        return "payload_statistics"
    raise ValueError(f"Feature does not belong to a known family: {feature}")


def build_feature_sets(feature_names: Sequence[str]) -> tuple[
    dict[str, list[str]], dict[str, list[str]]
]:
    families = {
        family: [name for name in feature_names if feature_family(name) == family]
        for family in FAMILY_ORDER
    }
    empty = [family for family, names in families.items() if not names]
    if empty:
        raise ValueError(f"Empty feature families: {empty}")
    if sum(len(names) for names in families.values()) != len(feature_names):
        raise RuntimeError("Feature-family allocation is incomplete")

    feature_sets: dict[str, list[str]] = {"all_features": list(feature_names)}
    for family in FAMILY_ORDER:
        feature_sets[f"only_{family}"] = families[family]
    for family in FAMILY_ORDER:
        excluded = set(families[family])
        feature_sets[f"without_{family}"] = [
            name for name in feature_names if name not in excluded
        ]
    return feature_sets, families


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def choose_f1_threshold(truth: np.ndarray, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(truth, probability)
    if not len(thresholds):
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-12))
    # Prefer the highest threshold among exact F1 ties to reduce false alarms.
    return float(thresholds[best[np.argmax(thresholds[best])]])


def predict_in_batches(
    model: Pipeline,
    frame: pd.DataFrame,
    feature_names: Sequence[str],
) -> np.ndarray:
    batches: list[np.ndarray] = []
    for start in range(0, len(frame), PREDICTION_BATCH_ROWS):
        end = min(start + PREDICTION_BATCH_ROWS, len(frame))
        values = frame.iloc[start:end][list(feature_names)].to_numpy(dtype=np.float64)
        batches.append(model.predict_proba(values)[:, 1])
    return np.concatenate(batches)


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def metric_row(
    ablation: str,
    dataset: str,
    scope: str,
    truth: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    prediction = (probability >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    both_classes = len(np.unique(truth)) == 2
    if both_classes:
        roc_auc = float(roc_auc_score(truth, probability))
        pr_auc = float(average_precision_score(truth, probability))
        orientation = "attack_if_high" if roc_auc >= 0.5 else "attack_if_low_inverted"
    else:
        roc_auc = math.nan
        pr_auc = math.nan
        orientation = "not_applicable_single_class"
    return {
        "ablation": ablation,
        "dataset": dataset,
        "scope": scope,
        "threshold": threshold,
        "windows": len(truth),
        "benign_windows": int((truth == 0).sum()),
        "attack_windows": int((truth == 1).sum()),
        "accuracy": accuracy_score(truth, prediction),
        "balanced_accuracy": (
            balanced_accuracy_score(truth, prediction) if both_classes else math.nan
        ),
        "precision": precision_score(truth, prediction, zero_division=0),
        "recall": recall_score(truth, prediction, zero_division=0),
        "f1": f1_score(truth, prediction, zero_division=0),
        "mcc": matthews_corrcoef(truth, prediction) if both_classes else math.nan,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "score_orientation_diagnostic": orientation,
        "false_positive_rate": safe_rate(int(fp), int(fp + tn)),
        "false_negative_rate": safe_rate(int(fn), int(fn + tp)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "probability_mean_benign": (
            float(probability[truth == 0].mean()) if np.any(truth == 0) else math.nan
        ),
        "probability_mean_attack": (
            float(probability[truth == 1].mean()) if np.any(truth == 1) else math.nan
        ),
    }


def validate_binary_partition(frame: pd.DataFrame, name: str) -> None:
    if frame.empty:
        raise ValueError(f"{name} is empty")
    classes = set(frame["binary_target"].dropna().unique())
    if classes != {0, 1}:
        raise ValueError(f"{name} must contain both classes; found {sorted(classes)}")


def coefficient_rows(
    ablation: str,
    model: Pipeline,
    feature_names: Sequence[str],
) -> list[dict[str, object]]:
    classifier = model.named_steps["classifier"]
    coefficients = classifier.coef_[0]
    rows = [
        {
            "ablation": ablation,
            "feature": feature,
            "feature_family": feature_family(feature),
            "standardized_coefficient": float(coefficient),
            "absolute_standardized_coefficient": float(abs(coefficient)),
        }
        for feature, coefficient in zip(feature_names, coefficients)
    ]
    return sorted(
        rows,
        key=lambda row: float(row["absolute_standardized_coefficient"]),
        reverse=True,
    )


def plot_ablation_summary(summary: pd.DataFrame, output: Path) -> None:
    frame = summary.sort_values("external_macro_source_roc_auc", ascending=True)
    positions = np.arange(len(frame))
    height = 0.25
    figure, axis = plt.subplots(figsize=(12, 7.5))
    axis.barh(
        positions - height,
        frame["ciciov_test_pr_auc"],
        height,
        label="CICIoV test PR-AUC",
    )
    axis.barh(
        positions,
        frame["external_macro_source_roc_auc"],
        height,
        label="HCRL macro source ROC-AUC",
    )
    axis.barh(
        positions + height,
        frame["external_worst_source_roc_auc"],
        height,
        label="HCRL worst-source ROC-AUC",
    )
    axis.set_yticks(positions, frame["ablation"])
    axis.set_xlim(0, 1.02)
    axis.set_xlabel("Metric value")
    axis.set_title("Feature-family ablation: in-domain vs cross-domain ranking")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    processed_dir = project_root / "data" / "processed"
    prefix = processed_dir / f"ciciov2024_windows_w{WINDOW_SIZE}_group_disjoint"
    paths = {
        "train": Path(f"{prefix}_train.csv"),
        "validation": Path(f"{prefix}_validation.csv"),
        "test": Path(f"{prefix}_test.csv"),
        "external": (
            processed_dir
            / "external_car_hacking"
            / "car_hacking_windows_w100_predictions.csv"
        ),
        "feature_columns": (
            project_root
            / "results"
            / "external_car_hacking_zero_shot"
            / "external_feature_columns.json"
        ),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    with paths["feature_columns"].open(encoding="utf-8") as handle:
        feature_names = list(json.load(handle))
    feature_sets, families = build_feature_sets(feature_names)
    required_ciciov = ["binary_target", *feature_names]
    required_external = [
        "source_file",
        "source_capture_class",
        "binary_target",
        *feature_names,
    ]

    print("Loading CICIoV2024 group-disjoint partitions ...")
    partitions = {
        name: pd.read_csv(path, usecols=required_ciciov)
        for name, path in paths.items()
        if name in {"train", "validation", "test"}
    }
    for name, frame in partitions.items():
        validate_binary_partition(frame, f"CICIoV {name}")
        if frame[feature_names].isna().any().any():
            raise ValueError(f"CICIoV {name} contains missing feature values")

    print("Loading HCRL external windows ...")
    external = pd.read_csv(paths["external"], usecols=required_external)
    validate_binary_partition(external, "HCRL external")
    if external[feature_names].isna().any().any():
        raise ValueError("HCRL external data contains missing feature values")

    results_dir = project_root / "results" / "feature_family_ablation_w100"
    models_dir = project_root / "models" / "feature_family_ablation_w100"
    results_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save_models:
        models_dir.mkdir(parents=True, exist_ok=True)

    all_metric_rows: list[dict[str, object]] = []
    per_source_rows: list[dict[str, object]] = []
    coefficient_output: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    feature_manifest_rows: list[dict[str, object]] = []

    y_train = partitions["train"]["binary_target"].to_numpy(dtype=np.uint8)
    y_validation = partitions["validation"]["binary_target"].to_numpy(dtype=np.uint8)
    y_test = partitions["test"]["binary_target"].to_numpy(dtype=np.uint8)
    external_truth = external["binary_target"].to_numpy(dtype=np.uint8)

    print(f"\nTraining {len(feature_sets)} logistic-regression ablations ...")
    for index, (ablation, selected_features) in enumerate(feature_sets.items(), start=1):
        print(f"  [{index}/{len(feature_sets)}] {ablation} ({len(selected_features)} features)")
        model = make_model()
        started = time.perf_counter()
        model.fit(
            partitions["train"][selected_features].to_numpy(dtype=np.float64),
            y_train,
        )
        train_seconds = time.perf_counter() - started

        validation_probability = model.predict_proba(
            partitions["validation"][selected_features].to_numpy(dtype=np.float64)
        )[:, 1]
        threshold = choose_f1_threshold(y_validation, validation_probability)
        validation_metric = metric_row(
            ablation,
            "CICIoV2024",
            "validation",
            y_validation,
            validation_probability,
            threshold,
        )
        test_probability = model.predict_proba(
            partitions["test"][selected_features].to_numpy(dtype=np.float64)
        )[:, 1]
        test_metric = metric_row(
            ablation,
            "CICIoV2024",
            "test",
            y_test,
            test_probability,
            threshold,
        )
        external_probability = predict_in_batches(model, external, selected_features)
        external_metric = metric_row(
            ablation,
            "HCRL_Car_Hacking",
            "all_external_windows",
            external_truth,
            external_probability,
            threshold,
        )
        for row in (validation_metric, test_metric, external_metric):
            row["feature_count"] = len(selected_features)
            row["threshold_source"] = "CICIoV2024 group-disjoint validation"
            row["train_seconds"] = train_seconds
            all_metric_rows.append(row)

        source_auc_values: list[float] = []
        for source_file, source_frame in external.groupby("source_file", sort=True):
            positions = source_frame.index.to_numpy(dtype=np.int64)
            source_truth = source_frame["binary_target"].to_numpy(dtype=np.uint8)
            source_probability = external_probability[positions]
            source_metric = metric_row(
                ablation,
                "HCRL_Car_Hacking",
                f"source:{source_file}",
                source_truth,
                source_probability,
                threshold,
            )
            source_metric["feature_count"] = len(selected_features)
            source_metric["threshold_source"] = (
                "CICIoV2024 group-disjoint validation"
            )
            per_source_rows.append(source_metric)
            if not pd.isna(source_metric["roc_auc"]):
                source_auc_values.append(float(source_metric["roc_auc"]))

        coefficient_output.extend(
            coefficient_rows(ablation, model, selected_features)
        )
        summary_rows.append(
            {
                "ablation": ablation,
                "feature_count": len(selected_features),
                "ciciov_validation_threshold": threshold,
                "ciciov_validation_f1": validation_metric["f1"],
                "ciciov_test_f1": test_metric["f1"],
                "ciciov_test_pr_auc": test_metric["pr_auc"],
                "ciciov_test_roc_auc": test_metric["roc_auc"],
                "external_f1_at_ciciov_threshold": external_metric["f1"],
                "external_precision_at_ciciov_threshold": external_metric["precision"],
                "external_recall_at_ciciov_threshold": external_metric["recall"],
                "external_fpr_at_ciciov_threshold": external_metric[
                    "false_positive_rate"
                ],
                "external_pr_auc": external_metric["pr_auc"],
                "external_roc_auc": external_metric["roc_auc"],
                "external_macro_source_roc_auc": float(np.mean(source_auc_values)),
                "external_worst_source_roc_auc": float(np.min(source_auc_values)),
                "external_source_roc_auc_std": float(np.std(source_auc_values)),
                "ciciov_to_external_pr_auc_gap": float(test_metric["pr_auc"])
                - float(external_metric["pr_auc"]),
                "train_seconds": train_seconds,
            }
        )
        included = set(selected_features)
        for family in FAMILY_ORDER:
            family_features = families[family]
            feature_manifest_rows.append(
                {
                    "ablation": ablation,
                    "feature_family": family,
                    "family_total_features": len(family_features),
                    "family_included_features": sum(
                        feature in included for feature in family_features
                    ),
                    "included": all(feature in included for feature in family_features),
                    "selected_feature_count": len(selected_features),
                    "selected_features_json": json.dumps(selected_features),
                }
            )

        if not args.no_save_models:
            joblib.dump(model, models_dir / f"{ablation}.joblib")
        print(
            f"      CIC test F1={float(test_metric['f1']):.4f}, "
            f"external PR-AUC={float(external_metric['pr_auc']):.4f}, "
            f"macro source ROC-AUC={float(np.mean(source_auc_values)):.4f}, "
            f"worst source ROC-AUC={float(np.min(source_auc_values)):.4f}"
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["external_macro_source_roc_auc", "external_worst_source_roc_auc"],
        ascending=False,
    )
    summary.to_csv(results_dir / "ablation_cross_domain_summary.csv", index=False)
    write_csv(results_dir / "ablation_all_metrics.csv", all_metric_rows)
    write_csv(results_dir / "ablation_external_per_source_metrics.csv", per_source_rows)
    write_csv(results_dir / "ablation_feature_coefficients.csv", coefficient_output)
    write_csv(results_dir / "ablation_feature_manifest.csv", feature_manifest_rows)
    plot_ablation_summary(summary, results_dir / "ablation_cross_domain_summary.png")

    manifest: list[dict[str, object]] = [
        {"item": "experiment_type", "value": "feature-family ablation"},
        {"item": "models", "value": len(feature_sets)},
        {"item": "classifier", "value": "class-balanced logistic regression"},
        {"item": "training_data", "value": "CICIoV2024 group-disjoint train"},
        {
            "item": "threshold_selection",
            "value": "maximum F1 on CICIoV2024 group-disjoint validation only",
        },
        {"item": "external_dataset", "value": "HCRL Car-Hacking Dataset"},
        {"item": "external_training", "value": "none"},
        {"item": "external_threshold_tuning", "value": "none"},
        {
            "item": "study_status",
            "value": "exploratory; designed after inspecting initial HCRL result",
        },
        {
            "item": "confirmatory_requirement",
            "value": "evaluate chosen design on a separate unseen capture/dataset",
        },
        {
            "item": "primary_cross_domain_metrics",
            "value": "per-source ROC-AUC and PR-AUC; threshold metrics are secondary",
        },
    ]
    write_csv(results_dir / "ablation_manifest.csv", manifest)

    display_columns = [
        "ablation",
        "feature_count",
        "ciciov_test_f1",
        "external_pr_auc",
        "external_macro_source_roc_auc",
        "external_worst_source_roc_auc",
    ]
    print("\n" + "=" * 100)
    print("Feature-family ablation completed successfully.")
    print(summary[display_columns].to_string(index=False))
    print(f"\nResults directory: {results_dir}")
    if not args.no_save_models:
        print(f"Models directory: {models_dir}")
    print(
        "\nNext: inspect per-source improvements and select a drift-aware CAN trust "
        "gate for confirmatory testing."
    )


if __name__ == "__main__":
    main()
