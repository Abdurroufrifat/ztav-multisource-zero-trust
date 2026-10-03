#!/usr/bin/env python3
"""Build a drift-aware CAN trust gate using healthy target-vehicle enrollment.

Step 15 showed that the seven identifier/frame-structure features transfer
better than payload features, but the CICIoV-supervised score is inverted for
HCRL Fuzzy attacks.  A one-sided attack-probability threshold cannot detect
both unusually high and unusually low deviations.

This script keeps the Step 15 identifier-only classifier frozen and adds two
unsupervised target-vehicle checks:

1. a two-sided deviation gate on the classifier log-odds; and
2. a robust multivariate distance over the seven identifier/frame features.

The HCRL attack-free capture is divided chronologically into reference (50%),
healthy calibration (25%), and healthy holdout (25%).  Robust reference
statistics are fitted on the first segment.  Gate thresholds are selected on
the healthy calibration segment to target a configurable false-alarm rate.
Attack labels are used only after threshold selection for evaluation.

This is healthy-enrollment domain adaptation, not zero-shot evaluation.  It is
also exploratory because HCRL results motivated the design.  Confirm the final
gate on a separate unseen vehicle/capture before making a thesis claim about
generalization.

This is a research prototype, not production automotive safety software.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


WINDOW_SIZE = 100
PREDICTION_BATCH_ROWS = 20_000
DEFAULT_TARGET_HEALTHY_FPR = 0.01
REFERENCE_FRACTION = 0.50
CALIBRATION_FRACTION = 0.25
NORMAL_SOURCE = "normal_run_data.txt"
EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Healthy-enrollment drift-aware CAN trust gate."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--target-healthy-fpr",
        type=float,
        default=DEFAULT_TARGET_HEALTHY_FPR,
        help="False-alarm target used only on healthy calibration windows.",
    )
    args = parser.parse_args()
    if not 0.001 <= args.target_healthy_fpr <= 0.10:
        parser.error("--target-healthy-fpr must be between 0.001 and 0.10")
    return args


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
    raise ValueError(f"Unknown feature family: {feature}")


def predict_in_batches(
    model: object,
    frame: pd.DataFrame,
    features: Sequence[str],
) -> np.ndarray:
    probabilities: list[np.ndarray] = []
    for start in range(0, len(frame), PREDICTION_BATCH_ROWS):
        end = min(start + PREDICTION_BATCH_ROWS, len(frame))
        values = frame.iloc[start:end][list(features)].to_numpy(dtype=np.float64)
        probabilities.append(model.predict_proba(values)[:, 1])
    return np.concatenate(probabilities)


def split_healthy_capture(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    output["evaluation_role"] = "external_attack_capture_evaluation"
    normal = output[output["source_file"] == NORMAL_SOURCE].sort_values(
        "window_index"
    )
    if len(normal) < 1_000:
        raise ValueError(
            f"At least 1,000 {NORMAL_SOURCE} windows are required; found {len(normal)}"
        )
    reference_end = int(math.floor(REFERENCE_FRACTION * len(normal)))
    calibration_end = int(
        math.floor((REFERENCE_FRACTION + CALIBRATION_FRACTION) * len(normal))
    )
    output.loc[normal.index[:reference_end], "evaluation_role"] = "healthy_reference"
    output.loc[
        normal.index[reference_end:calibration_end], "evaluation_role"
    ] = "healthy_calibration"
    output.loc[normal.index[calibration_end:], "evaluation_role"] = "healthy_holdout"
    return output


def robust_reference_statistics(
    reference: pd.DataFrame,
    features: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    values = reference[list(features)].to_numpy(dtype=np.float64)
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    robust_scale = 1.4826 * mad
    standard_deviation = values.std(axis=0)
    scale = np.where(
        robust_scale > EPSILON,
        robust_scale,
        np.where(standard_deviation > EPSILON, standard_deviation, 1.0),
    )
    rows = [
        {
            "feature": feature,
            "reference_median": float(median[index]),
            "reference_mad": float(mad[index]),
            "reference_robust_scale": float(robust_scale[index]),
            "reference_standard_deviation": float(standard_deviation[index]),
            "scale_used": float(scale[index]),
            "fallback_scale_used": bool(robust_scale[index] <= EPSILON),
        }
        for index, feature in enumerate(features)
    ]
    return median, scale, rows


def robust_feature_distance(
    frame: pd.DataFrame,
    features: Sequence[str],
    median: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    values = frame[list(features)].to_numpy(dtype=np.float64)
    absolute_z = np.abs((values - median) / scale)
    absolute_z = np.clip(absolute_z, 0, 1_000)
    # The 75th percentile requires several structural features to deviate and
    # is less sensitive to one noisy feature than a maximum-distance rule.
    return np.quantile(absolute_z, 0.75, axis=1)


def probability_to_log_odds(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def two_sided_ratio(
    values: np.ndarray,
    center: float,
    lower: float,
    upper: float,
) -> np.ndarray:
    lower_scale = max(center - lower, EPSILON)
    upper_scale = max(upper - center, EPSILON)
    return np.where(
        values < center,
        (center - values) / lower_scale,
        (values - center) / upper_scale,
    )


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def metric_row(
    method: str,
    scope: str,
    truth: np.ndarray,
    alarm: np.ndarray,
    anomaly_score: np.ndarray,
    threshold_description: str,
) -> dict[str, object]:
    prediction = alarm.astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    both_classes = len(np.unique(truth)) == 2
    return {
        "method": method,
        "scope": scope,
        "threshold_description": threshold_description,
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
        "roc_auc": (
            roc_auc_score(truth, anomaly_score) if both_classes else math.nan
        ),
        "pr_auc": (
            average_precision_score(truth, anomaly_score) if both_classes else math.nan
        ),
        "false_positive_rate": safe_rate(int(fp), int(fp + tn)),
        "false_negative_rate": safe_rate(int(fn), int(fn + tp)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "mean_anomaly_score_benign": (
            float(anomaly_score[truth == 0].mean())
            if np.any(truth == 0)
            else math.nan
        ),
        "mean_anomaly_score_attack": (
            float(anomaly_score[truth == 1].mean())
            if np.any(truth == 1)
            else math.nan
        ),
    }


def attack_density_rows(
    evaluation: pd.DataFrame,
    methods: dict[str, tuple[np.ndarray, np.ndarray, str]],
) -> list[dict[str, object]]:
    attacked_positions = np.flatnonzero(
        evaluation["binary_target"].to_numpy(dtype=np.uint8) == 1
    )
    attack_counts = evaluation.iloc[attacked_positions]["attack_frame_count"].to_numpy(
        dtype=int
    )
    bins = (
        ("1", 1, 1),
        ("2-5", 2, 5),
        ("6-20", 6, 20),
        ("21-50", 21, 50),
        ("51-99", 51, 99),
        ("100", 100, 100),
    )
    output: list[dict[str, object]] = []
    for method, (alarm, score, _) in methods.items():
        attacked_alarm = alarm[attacked_positions]
        attacked_score = score[attacked_positions]
        for label, lower, upper in bins:
            selected = (attack_counts >= lower) & (attack_counts <= upper)
            output.append(
                {
                    "method": method,
                    "attack_frames_per_window": label,
                    "windows": int(selected.sum()),
                    "recall": (
                        float(attacked_alarm[selected].mean())
                        if selected.any()
                        else math.nan
                    ),
                    "mean_anomaly_score": (
                        float(attacked_score[selected].mean())
                        if selected.any()
                        else math.nan
                    ),
                }
            )
    return output


def plot_per_source_recall(
    per_source: pd.DataFrame,
    output: Path,
) -> None:
    attack_sources = per_source[per_source["attack_windows"] > 0].copy()
    pivot = attack_sources.pivot(index="scope", columns="method", values="recall")
    axis = pivot.plot(kind="bar", figsize=(12, 6.5), width=0.82)
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("HCRL source capture")
    axis.set_ylabel("Attack-window recall")
    axis.set_title("Drift-aware CAN gate recall by attack source")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Gate", loc="lower right")
    axis.figure.tight_layout()
    axis.figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(axis.figure)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    prediction_path = (
        project_root
        / "data"
        / "processed"
        / "external_car_hacking"
        / "car_hacking_windows_w100_predictions.csv"
    )
    feature_columns_path = (
        project_root
        / "results"
        / "external_car_hacking_zero_shot"
        / "external_feature_columns.json"
    )
    ablation_summary_path = (
        project_root
        / "results"
        / "feature_family_ablation_w100"
        / "ablation_cross_domain_summary.csv"
    )
    model_path = (
        project_root
        / "models"
        / "feature_family_ablation_w100"
        / "only_identifier_and_frame_structure.joblib"
    )
    for path in (
        prediction_path,
        feature_columns_path,
        ablation_summary_path,
        model_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing required Step 13/15 asset: {path}")

    with feature_columns_path.open(encoding="utf-8") as handle:
        all_features = list(json.load(handle))
    identifier_features = [
        feature
        for feature in all_features
        if feature_family(feature) == "identifier_and_frame_structure"
    ]
    if len(identifier_features) != 7:
        raise ValueError(
            "Expected seven identifier/frame features; "
            f"found {len(identifier_features)}: {identifier_features}"
        )
    required_columns = [
        "source_file",
        "source_capture_class",
        "window_index",
        "binary_target",
        "attack_frame_count",
        "attack_frame_fraction",
        *identifier_features,
    ]
    print(f"Loading external windows: {prediction_path}")
    data = pd.read_csv(prediction_path, usecols=required_columns)
    if data[required_columns].isna().any().any():
        raise ValueError("Required external columns contain missing values")
    if set(data["binary_target"].unique()) != {0, 1}:
        raise ValueError("External data must contain benign and attack windows")
    data = split_healthy_capture(data)

    model = joblib.load(model_path)
    model_feature_count = getattr(model, "n_features_in_", len(identifier_features))
    if int(model_feature_count) != len(identifier_features):
        raise ValueError(
            f"Identifier model expects {model_feature_count} features, "
            f"but {len(identifier_features)} were selected"
        )
    data["identifier_attack_probability"] = predict_in_batches(
        model,
        data,
        identifier_features,
    )
    data["identifier_log_odds"] = probability_to_log_odds(
        data["identifier_attack_probability"].to_numpy(dtype=float)
    )

    reference = data[data["evaluation_role"] == "healthy_reference"]
    calibration = data[data["evaluation_role"] == "healthy_calibration"]
    evaluation = data[
        data["evaluation_role"].isin(
            ["healthy_holdout", "external_attack_capture_evaluation"]
        )
    ].copy()
    for name, frame in (
        ("reference", reference),
        ("calibration", calibration),
        ("evaluation", evaluation),
    ):
        if frame.empty:
            raise ValueError(f"Healthy-enrollment {name} partition is empty")

    reference_median, reference_scale, reference_rows = robust_reference_statistics(
        reference,
        identifier_features,
    )
    data["identifier_feature_distance"] = robust_feature_distance(
        data,
        identifier_features,
        reference_median,
        reference_scale,
    )

    lower_quantile = args.target_healthy_fpr / 2.0
    upper_quantile = 1.0 - args.target_healthy_fpr / 2.0
    calibration_log_odds = data.loc[
        calibration.index, "identifier_log_odds"
    ].to_numpy(dtype=float)
    score_center = float(np.median(calibration_log_odds))
    score_lower = float(np.quantile(calibration_log_odds, lower_quantile))
    score_upper = float(np.quantile(calibration_log_odds, upper_quantile))
    data["score_two_sided_ratio"] = two_sided_ratio(
        data["identifier_log_odds"].to_numpy(dtype=float),
        score_center,
        score_lower,
        score_upper,
    )

    calibration_feature_distance = data.loc[
        calibration.index, "identifier_feature_distance"
    ].to_numpy(dtype=float)
    feature_threshold = float(
        np.quantile(calibration_feature_distance, 1.0 - args.target_healthy_fpr)
    )
    data["feature_distance_ratio"] = (
        data["identifier_feature_distance"] / max(feature_threshold, EPSILON)
    )
    data["combined_raw_ratio"] = np.maximum(
        data["score_two_sided_ratio"],
        data["feature_distance_ratio"],
    )
    combined_threshold = float(
        np.quantile(
            data.loc[calibration.index, "combined_raw_ratio"].to_numpy(dtype=float),
            1.0 - args.target_healthy_fpr,
        )
    )
    data["combined_risk_ratio"] = data["combined_raw_ratio"] / max(
        combined_threshold, EPSILON
    )
    data["can_continuous_trust"] = 1.0 / (
        1.0 + np.square(data["combined_risk_ratio"])
    )

    ablation_summary = pd.read_csv(ablation_summary_path)
    selected = ablation_summary[
        ablation_summary["ablation"] == "only_identifier_and_frame_structure"
    ]
    if len(selected) != 1:
        raise ValueError("Could not find identifier-only ablation threshold")
    ciciov_threshold = float(selected.iloc[0]["ciciov_validation_threshold"])

    evaluation_positions = evaluation.index.to_numpy(dtype=np.int64)
    probability = data.loc[
        evaluation_positions, "identifier_attack_probability"
    ].to_numpy(dtype=float)
    score_ratio = data.loc[
        evaluation_positions, "score_two_sided_ratio"
    ].to_numpy(dtype=float)
    feature_distance = data.loc[
        evaluation_positions, "identifier_feature_distance"
    ].to_numpy(dtype=float)
    combined_ratio = data.loc[
        evaluation_positions, "combined_raw_ratio"
    ].to_numpy(dtype=float)
    methods: dict[str, tuple[np.ndarray, np.ndarray, str]] = {
        "ciciov_supervised_threshold": (
            probability >= ciciov_threshold,
            probability,
            f"probability >= {ciciov_threshold:.12g}; selected on CICIoV validation",
        ),
        "healthy_score_two_sided": (
            score_ratio >= 1.0,
            score_ratio,
            (
                f"healthy log-odds outside [{score_lower:.6g}, {score_upper:.6g}]; "
                f"target calibration FPR={args.target_healthy_fpr:.4f}"
            ),
        ),
        "healthy_feature_robust": (
            feature_distance >= feature_threshold,
            feature_distance,
            (
                f"robust feature distance >= {feature_threshold:.6g}; "
                f"target calibration FPR={args.target_healthy_fpr:.4f}"
            ),
        ),
        "drift_aware_combined": (
            combined_ratio >= combined_threshold,
            combined_ratio,
            (
                f"combined ratio >= {combined_threshold:.6g}; "
                f"target calibration FPR={args.target_healthy_fpr:.4f}"
            ),
        ),
    }

    truth = evaluation["binary_target"].to_numpy(dtype=np.uint8)
    overall_rows: list[dict[str, object]] = []
    per_source_rows: list[dict[str, object]] = []
    for method, (alarm, anomaly_score, description) in methods.items():
        overall_rows.append(
            metric_row(
                method,
                "all_evaluation_windows",
                truth,
                alarm,
                anomaly_score,
                description,
            )
        )
        for source_file, source_frame in evaluation.groupby("source_file", sort=True):
            local_positions = evaluation.index.get_indexer(source_frame.index)
            per_source_rows.append(
                metric_row(
                    method,
                    f"source:{source_file}",
                    source_frame["binary_target"].to_numpy(dtype=np.uint8),
                    alarm[local_positions],
                    anomaly_score[local_positions],
                    description,
                )
            )

    calibration_positions = calibration.index.to_numpy(dtype=np.int64)
    calibration_threshold_rows: list[dict[str, object]] = [
        {
            "method": "ciciov_supervised_threshold",
            "threshold_type": "one_sided_probability",
            "lower_threshold": math.nan,
            "upper_threshold": ciciov_threshold,
            "combined_threshold": math.nan,
            "target_healthy_fpr": args.target_healthy_fpr,
            "empirical_calibration_fpr": float(
                (
                    data.loc[
                        calibration_positions, "identifier_attack_probability"
                    ].to_numpy(dtype=float)
                    >= ciciov_threshold
                ).mean()
            ),
        },
        {
            "method": "healthy_score_two_sided",
            "threshold_type": "two_sided_log_odds",
            "lower_threshold": score_lower,
            "upper_threshold": score_upper,
            "combined_threshold": math.nan,
            "target_healthy_fpr": args.target_healthy_fpr,
            "empirical_calibration_fpr": float(
                (
                    data.loc[
                        calibration_positions, "score_two_sided_ratio"
                    ].to_numpy(dtype=float)
                    >= 1.0
                ).mean()
            ),
        },
        {
            "method": "healthy_feature_robust",
            "threshold_type": "upper_robust_feature_distance",
            "lower_threshold": math.nan,
            "upper_threshold": feature_threshold,
            "combined_threshold": math.nan,
            "target_healthy_fpr": args.target_healthy_fpr,
            "empirical_calibration_fpr": float(
                (
                    data.loc[
                        calibration_positions, "identifier_feature_distance"
                    ].to_numpy(dtype=float)
                    >= feature_threshold
                ).mean()
            ),
        },
        {
            "method": "drift_aware_combined",
            "threshold_type": "upper_combined_ratio",
            "lower_threshold": math.nan,
            "upper_threshold": math.nan,
            "combined_threshold": combined_threshold,
            "target_healthy_fpr": args.target_healthy_fpr,
            "empirical_calibration_fpr": float(
                (
                    data.loc[
                        calibration_positions, "combined_raw_ratio"
                    ].to_numpy(dtype=float)
                    >= combined_threshold
                ).mean()
            ),
        },
    ]

    role_rows: list[dict[str, object]] = []
    for role, frame in data.groupby("evaluation_role", sort=True):
        role_rows.append(
            {
                "evaluation_role": role,
                "windows": len(frame),
                "benign_windows": int((frame["binary_target"] == 0).sum()),
                "attack_windows": int((frame["binary_target"] == 1).sum()),
                "first_window_index": int(frame["window_index"].min()),
                "last_window_index": int(frame["window_index"].max()),
                "source_files": frame["source_file"].nunique(),
            }
        )

    output_dir = project_root / "results" / "drift_aware_can_gate_w100"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "drift_gate_thresholds.csv", calibration_threshold_rows)
    write_csv(output_dir / "drift_gate_overall_metrics.csv", overall_rows)
    write_csv(output_dir / "drift_gate_per_source_metrics.csv", per_source_rows)
    write_csv(
        output_dir / "drift_gate_attack_density_recall.csv",
        attack_density_rows(evaluation, methods),
    )
    write_csv(output_dir / "healthy_enrollment_split_summary.csv", role_rows)
    write_csv(output_dir / "identifier_reference_statistics.csv", reference_rows)

    prediction_columns = [
        "source_file",
        "source_capture_class",
        "window_index",
        "binary_target",
        "attack_frame_count",
        "attack_frame_fraction",
        "evaluation_role",
        "identifier_attack_probability",
        "identifier_log_odds",
        "score_two_sided_ratio",
        "identifier_feature_distance",
        "feature_distance_ratio",
        "combined_raw_ratio",
        "combined_risk_ratio",
        "can_continuous_trust",
    ]
    for method, (alarm, _, _) in methods.items():
        full_alarm = np.full(len(data), False, dtype=bool)
        full_alarm[evaluation_positions] = alarm
        data[f"alarm_{method}"] = full_alarm.astype(np.uint8)
        prediction_columns.append(f"alarm_{method}")
    data[prediction_columns].to_csv(
        output_dir / "drift_gate_predictions.csv",
        index=False,
    )
    per_source_frame = pd.DataFrame(per_source_rows)
    plot_per_source_recall(
        per_source_frame,
        output_dir / "drift_gate_per_source_recall.png",
    )

    manifest: list[dict[str, object]] = [
        {"item": "experiment_type", "value": "healthy-enrollment CAN drift gate"},
        {
            "item": "frozen_supervised_model",
            "value": "Step 15 identifier/frame-only logistic regression",
        },
        {"item": "healthy_source", "value": NORMAL_SOURCE},
        {"item": "reference_fraction", "value": REFERENCE_FRACTION},
        {"item": "calibration_fraction", "value": CALIBRATION_FRACTION},
        {
            "item": "healthy_holdout_fraction",
            "value": 1.0 - REFERENCE_FRACTION - CALIBRATION_FRACTION,
        },
        {"item": "target_healthy_fpr", "value": args.target_healthy_fpr},
        {"item": "attack_labels_used_for_thresholds", "value": "none"},
        {
            "item": "adaptation_status",
            "value": "target-domain healthy enrollment; not zero-shot",
        },
        {
            "item": "study_status",
            "value": "exploratory; HCRL findings motivated the gate design",
        },
        {
            "item": "confirmatory_requirement",
            "value": "freeze the gate and evaluate on a separate unseen capture",
        },
        {
            "item": "continuous_trust_mapping",
            "value": "1 / (1 + combined_risk_ratio^2); trust=0.5 at alarm boundary",
        },
    ]
    write_csv(output_dir / "drift_gate_manifest.csv", manifest)

    print("\n" + "=" * 96)
    print("Drift-aware CAN trust-gate evaluation completed successfully.")
    print(
        f"Healthy windows: reference={len(reference):,}, "
        f"calibration={len(calibration):,}, holdout="
        f"{int((data['evaluation_role'] == 'healthy_holdout').sum()):,}"
    )
    display = pd.DataFrame(overall_rows)[
        ["method", "precision", "recall", "f1", "pr_auc", "false_positive_rate"]
    ]
    print(display.to_string(index=False))
    print(f"\nResults directory: {output_dir}")
    print(
        "\nNext: inspect capture-specific false alarms, then freeze the selected CAN "
        "gate for integration into the multi-source Zero Trust policy."
    )


if __name__ == "__main__":
    main()
