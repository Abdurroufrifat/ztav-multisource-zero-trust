#!/usr/bin/env python3
"""Diagnose HCRL domain shift and test external threshold calibration.

Step 13 is the primary, untouched cross-dataset zero-shot experiment.  This
script answers a separate question: if a small labelled HCRL calibration set
is available, can the frozen CICIoV2024 CAN model obtain useful performance by
changing only its decision threshold?

To reduce optimistic leakage, exact 63-feature signatures are assigned wholly
to either calibration or test.  The split is stratified at signature-group
level, deterministic, and global across HCRL source files.  No model fitting or
feature scaling is performed here.  A calibrated result must therefore be
reported as external calibration, never as zero-shot performance.

This is a research prototype, not production automotive safety software.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
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
    roc_curve,
)


DEFAULT_CALIBRATION_FRACTION = 0.20
DEFAULT_FPR_LIMIT = 0.05
DEFAULT_SEED = 42
REQUIRED_PREDICTION_COLUMNS = {
    "source_file",
    "source_capture_class",
    "window_index",
    "binary_target",
    "attack_probability",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="External HCRL domain-shift and threshold-calibration audit."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=DEFAULT_CALIBRATION_FRACTION,
    )
    parser.add_argument("--fpr-limit", type=float, default=DEFAULT_FPR_LIMIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if not 0.05 <= args.calibration_fraction <= 0.50:
        parser.error("--calibration-fraction must be between 0.05 and 0.50")
    if not 0.0 < args.fpr_limit < 1.0:
        parser.error("--fpr-limit must be between 0 and 1")
    return args


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_inputs(data: pd.DataFrame, feature_names: Sequence[str]) -> None:
    missing = REQUIRED_PREDICTION_COLUMNS - set(data.columns)
    missing_features = set(feature_names) - set(data.columns)
    if missing or missing_features:
        raise ValueError(
            "External prediction schema mismatch: "
            f"missing required={sorted(missing)}, "
            f"missing features={sorted(missing_features)}"
        )
    if data.empty:
        raise ValueError("External prediction dataset is empty")
    if not set(data["binary_target"].dropna().unique()).issubset({0, 1}):
        raise ValueError("binary_target must contain only 0 and 1")
    probability = data["attack_probability"].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("attack_probability contains invalid values")


def add_feature_signatures(
    data: pd.DataFrame,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    output = data.copy()
    hashed = pd.util.hash_pandas_object(
        output[list(feature_names)],
        index=False,
        categorize=True,
    ).to_numpy(dtype=np.uint64)
    output["feature_signature"] = [f"{int(value):016x}" for value in hashed]
    return output


def signature_group_table(data: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        data.groupby("feature_signature", sort=False)
        .agg(
            rows=("binary_target", "size"),
            attack_rows=("binary_target", "sum"),
            source_count=("source_file", "nunique"),
        )
        .reset_index()
    )
    grouped["benign_rows"] = grouped["rows"] - grouped["attack_rows"]
    grouped["label_stratum"] = np.select(
        [grouped["attack_rows"] == 0, grouped["benign_rows"] == 0],
        ["benign_only", "attack_only"],
        default="label_conflict",
    )
    return grouped


def allocate_signature_groups(
    groups: pd.DataFrame,
    calibration_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Assign each exact feature signature to one deterministic split."""

    rng = np.random.default_rng(seed)
    allocated: list[pd.DataFrame] = []
    for stratum, subset in groups.groupby("label_stratum", sort=True):
        subset = subset.copy().reset_index(drop=True)
        order = rng.permutation(len(subset))
        number = int(round(calibration_fraction * len(subset)))
        if len(subset) >= 2:
            number = min(max(number, 1), len(subset) - 1)
        else:
            number = len(subset)
        calibration_indices = set(order[:number].tolist())
        subset["split"] = [
            "calibration" if index in calibration_indices else "test"
            for index in range(len(subset))
        ]
        allocated.append(subset)
    result = pd.concat(allocated, ignore_index=True)
    if result["feature_signature"].duplicated().any():
        raise RuntimeError("A feature signature was allocated more than once")
    return result


def attach_split(data: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    output = data.merge(
        groups[["feature_signature", "split"]],
        on="feature_signature",
        how="left",
        validate="many_to_one",
    )
    if output["split"].isna().any():
        raise RuntimeError("Some external windows were not allocated")
    crossing = output.groupby("feature_signature")["split"].nunique()
    if int((crossing > 1).sum()) != 0:
        raise RuntimeError("Feature signatures cross calibration and test")
    return output


def choose_max_f1_threshold(truth: np.ndarray, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(truth, probability)
    if len(thresholds) == 0:
        raise ValueError("Cannot select an F1 threshold from constant probabilities")
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


def choose_fpr_limited_threshold(
    truth: np.ndarray,
    probability: np.ndarray,
    fpr_limit: float,
) -> float:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(truth, probability)
    eligible = np.isfinite(thresholds) & (false_positive_rate <= fpr_limit)
    if not eligible.any():
        raise ValueError("No finite threshold satisfies the requested FPR limit")
    best_recall = true_positive_rate[eligible].max()
    candidate_indices = np.flatnonzero(
        eligible & np.isclose(true_positive_rate, best_recall, rtol=0, atol=1e-12)
    )
    # Prefer the largest threshold when recall is tied.
    return float(thresholds[candidate_indices[np.argmax(thresholds[candidate_indices])]])


def metric_row(
    frame: pd.DataFrame,
    threshold: float,
    method: str,
    evaluation_split: str,
    scope: str,
) -> dict[str, object]:
    truth = frame["binary_target"].to_numpy(dtype=np.uint8)
    probability = frame["attack_probability"].to_numpy(dtype=float)
    prediction = (probability >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    both_classes = len(np.unique(truth)) == 2
    return {
        "method": method,
        "evaluation_split": evaluation_split,
        "scope": scope,
        "threshold": threshold,
        "windows": len(frame),
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
        "roc_auc": roc_auc_score(truth, probability) if both_classes else math.nan,
        "pr_auc": average_precision_score(truth, probability) if both_classes else math.nan,
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


def split_summary_rows(data: pd.DataFrame) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for (split, source_file), frame in data.groupby(["split", "source_file"], sort=True):
        output.append(
            {
                "split": split,
                "source_file": source_file,
                "windows": len(frame),
                "benign_windows": int((frame["binary_target"] == 0).sum()),
                "attack_windows": int((frame["binary_target"] == 1).sum()),
                "unique_feature_signatures": frame["feature_signature"].nunique(),
            }
        )
    return output


def source_ranking_rows(test: pd.DataFrame) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for source_file, frame in test.groupby("source_file", sort=True):
        truth = frame["binary_target"].to_numpy(dtype=np.uint8)
        probability = frame["attack_probability"].to_numpy(dtype=float)
        if len(np.unique(truth)) < 2:
            auc = math.nan
            pr_auc = math.nan
            orientation = "not_applicable_single_class"
            oriented_auc = math.nan
            oriented_pr_auc = math.nan
        else:
            auc = float(roc_auc_score(truth, probability))
            pr_auc = float(average_precision_score(truth, probability))
            orientation = "attack_if_high" if auc >= 0.5 else "attack_if_low_inverted"
            oriented_probability = probability if auc >= 0.5 else 1.0 - probability
            oriented_auc = float(roc_auc_score(truth, oriented_probability))
            oriented_pr_auc = float(average_precision_score(truth, oriented_probability))
        output.append(
            {
                "source_file": source_file,
                "test_windows": len(frame),
                "benign_windows": int((truth == 0).sum()),
                "attack_windows": int((truth == 1).sum()),
                "roc_auc": auc,
                "pr_auc": pr_auc,
                "score_orientation_diagnostic": orientation,
                "oriented_roc_auc_diagnostic_only": oriented_auc,
                "oriented_pr_auc_diagnostic_only": oriented_pr_auc,
                "probability_mean_benign": (
                    float(probability[truth == 0].mean())
                    if np.any(truth == 0)
                    else math.nan
                ),
                "probability_mean_attack": (
                    float(probability[truth == 1].mean())
                    if np.any(truth == 1)
                    else math.nan
                ),
            }
        )
    return output


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
    return "other"


def feature_family_shift_rows(shift: pd.DataFrame) -> list[dict[str, object]]:
    required = {
        "feature",
        "standardized_mean_difference",
        "external_fraction_outside_ciciov_train_range",
    }
    missing = required - set(shift.columns)
    if missing:
        raise ValueError(f"Feature-shift file is missing columns: {sorted(missing)}")
    frame = shift.copy()
    frame["feature_family"] = frame["feature"].map(feature_family)
    frame["absolute_standardized_mean_difference"] = frame[
        "standardized_mean_difference"
    ].abs()
    output: list[dict[str, object]] = []
    for family, group in frame.groupby("feature_family", sort=True):
        largest_index = group["absolute_standardized_mean_difference"].idxmax()
        largest = frame.loc[largest_index]
        output.append(
            {
                "feature_family": family,
                "feature_count": len(group),
                "mean_absolute_standardized_mean_difference": group[
                    "absolute_standardized_mean_difference"
                ].mean(),
                "maximum_absolute_standardized_mean_difference": group[
                    "absolute_standardized_mean_difference"
                ].max(),
                "most_shifted_feature": largest["feature"],
                "mean_external_fraction_outside_training_range": group[
                    "external_fraction_outside_ciciov_train_range"
                ].mean(),
                "maximum_external_fraction_outside_training_range": group[
                    "external_fraction_outside_ciciov_train_range"
                ].max(),
            }
        )
    return sorted(
        output,
        key=lambda row: float(row["mean_absolute_standardized_mean_difference"]),
        reverse=True,
    )


def require_both_classes(frame: pd.DataFrame, name: str) -> None:
    counts = frame["binary_target"].value_counts()
    if not {0, 1}.issubset(set(counts.index)):
        raise ValueError(f"{name} split does not contain both classes: {counts.to_dict()}")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    zero_shot_dir = project_root / "results" / "external_car_hacking_zero_shot"
    predictions_path = (
        project_root
        / "data"
        / "processed"
        / "external_car_hacking"
        / "car_hacking_windows_w100_predictions.csv"
    )
    feature_columns_path = zero_shot_dir / "external_feature_columns.json"
    zero_shot_metrics_path = zero_shot_dir / "external_zero_shot_metrics.csv"
    shift_path = zero_shot_dir / "external_feature_domain_shift.csv"
    required_paths = (
        predictions_path,
        feature_columns_path,
        zero_shot_metrics_path,
        shift_path,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Step 13 output: {path}. Run Step 13 successfully first."
            )

    print(f"Loading external predictions: {predictions_path}")
    data = pd.read_csv(predictions_path)
    with feature_columns_path.open(encoding="utf-8") as handle:
        feature_names = list(json.load(handle))
    validate_inputs(data, feature_names)

    zero_shot_metrics = pd.read_csv(zero_shot_metrics_path)
    overall = zero_shot_metrics[zero_shot_metrics["scope"] == "all_external_windows"]
    if len(overall) != 1:
        raise ValueError("Could not identify the Step 13 overall zero-shot metric row")
    original_threshold = float(overall.iloc[0]["threshold"])

    data = add_feature_signatures(data, feature_names)
    groups = signature_group_table(data)
    groups = allocate_signature_groups(
        groups,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    data = attach_split(data, groups)
    calibration = data[data["split"] == "calibration"].copy()
    test = data[data["split"] == "test"].copy()
    require_both_classes(calibration, "Calibration")
    require_both_classes(test, "Test")

    calibration_truth = calibration["binary_target"].to_numpy(dtype=np.uint8)
    calibration_probability = calibration["attack_probability"].to_numpy(dtype=float)
    max_f1_threshold = choose_max_f1_threshold(
        calibration_truth,
        calibration_probability,
    )
    fpr_limited_threshold = choose_fpr_limited_threshold(
        calibration_truth,
        calibration_probability,
        fpr_limit=args.fpr_limit,
    )
    methods = (
        (
            "original_ciciov_zero_shot_threshold",
            original_threshold,
            "CICIoV2024 validation; no HCRL labels",
        ),
        (
            "external_calibration_max_f1",
            max_f1_threshold,
            "maximum F1 on signature-disjoint HCRL calibration split",
        ),
        (
            f"external_calibration_recall_at_fpr_le_{args.fpr_limit:.3f}",
            fpr_limited_threshold,
            "maximum calibration recall under the specified calibration FPR limit",
        ),
    )

    threshold_rows: list[dict[str, object]] = []
    test_metric_rows: list[dict[str, object]] = []
    per_source_rows: list[dict[str, object]] = []
    for method, threshold, objective in methods:
        calibration_metric = metric_row(
            calibration,
            threshold,
            method,
            "calibration",
            "all_external_windows",
        )
        threshold_rows.append(
            {
                "method": method,
                "threshold": threshold,
                "selection_objective": objective,
                "calibration_precision": calibration_metric["precision"],
                "calibration_recall": calibration_metric["recall"],
                "calibration_f1": calibration_metric["f1"],
                "calibration_false_positive_rate": calibration_metric[
                    "false_positive_rate"
                ],
            }
        )
        test_metric_rows.append(
            metric_row(test, threshold, method, "test", "all_external_windows")
        )
        for source_file, frame in test.groupby("source_file", sort=True):
            per_source_rows.append(
                metric_row(
                    frame,
                    threshold,
                    method,
                    "test",
                    f"source:{source_file}",
                )
            )

    shift = pd.read_csv(shift_path)
    family_shift = feature_family_shift_rows(shift)
    split_rows = split_summary_rows(data)
    source_ranking = source_ranking_rows(test)

    crossing_signatures = int(
        (data.groupby("feature_signature")["split"].nunique() > 1).sum()
    )
    output_dir = project_root / "results" / "external_domain_shift_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "external_calibration_split_summary.csv", split_rows)
    write_csv(output_dir / "external_selected_thresholds.csv", threshold_rows)
    write_csv(output_dir / "external_test_threshold_metrics.csv", test_metric_rows)
    write_csv(output_dir / "external_test_per_source_metrics.csv", per_source_rows)
    write_csv(output_dir / "external_source_ranking_diagnostics.csv", source_ranking)
    write_csv(output_dir / "external_feature_family_shift.csv", family_shift)
    groups.to_csv(
        output_dir / "external_calibration_signature_manifest.csv",
        index=False,
    )

    manifest: list[dict[str, object]] = [
        {"item": "experiment_type", "value": "external threshold calibration"},
        {
            "item": "zero_shot_result_status",
            "value": "preserved unchanged in external_car_hacking_zero_shot",
        },
        {"item": "model_retraining", "value": "none; frozen CICIoV2024 model"},
        {"item": "external_labels_used", "value": "calibration split only"},
        {"item": "split_unit", "value": "exact 63-feature signature"},
        {"item": "calibration_fraction", "value": args.calibration_fraction},
        {"item": "random_seed", "value": args.seed},
        {"item": "calibration_rows", "value": len(calibration)},
        {"item": "test_rows", "value": len(test)},
        {"item": "unique_signatures", "value": len(groups)},
        {
            "item": "conflicting_label_signatures",
            "value": int((groups["label_stratum"] == "label_conflict").sum()),
        },
        {"item": "signatures_crossing_splits", "value": crossing_signatures},
        {
            "item": "limitation",
            "value": (
                "random signature-group split within one external dataset; a future "
                "independent vehicle/capture dataset is still required"
            ),
        },
    ]
    write_csv(output_dir / "external_calibration_manifest.csv", manifest)

    print("\n" + "=" * 88)
    print("External domain-shift and calibration audit completed successfully.")
    print(
        f"Rows: calibration={len(calibration):,}, test={len(test):,}; "
        f"unique signatures={len(groups):,}; crossing signatures={crossing_signatures}"
    )
    for row in test_metric_rows:
        print(
            f"{row['method']}: threshold={float(row['threshold']):.6f}, "
            f"test precision={float(row['precision']):.4f}, "
            f"recall={float(row['recall']):.4f}, F1={float(row['f1']):.4f}, "
            f"FPR={float(row['false_positive_rate']):.4f}"
        )
    print(
        "Most shifted feature family: "
        f"{family_shift[0]['feature_family']} "
        f"(mean |standardized shift|="
        f"{float(family_shift[0]['mean_absolute_standardized_mean_difference']):.3f})"
    )
    print(f"Results directory: {output_dir}")
    print("\nNext: use these diagnostics to design a feature-family ablation study.")


if __name__ == "__main__":
    main()
