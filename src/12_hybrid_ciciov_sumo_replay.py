#!/usr/bin/env python3
"""Run hybrid replay with real CICIoV2024 CAN-model predictions and SUMO context.

The previous SUMO experiments used an explicit synthetic CAN attack score. This
stage replaces that score with predictions from the trained group-disjoint
CICIoV2024 logistic-regression pipeline. Held-out benign CAN windows are paired
with non-CAN SUMO phases, while held-out attack windows are paired with the CAN
injection and combined-attack phases. Pairing is without replacement per seed.

This is hybrid replay validation, not an external-dataset result: the CAN model
and replay windows still originate from CICIoV2024, and SUMO GNSS/V2X evidence
is simulated. The original sensor-control consistency field is retained as a
separate cross-source signal and identified in the manifest.

Run from D:\\ztav_project after Steps 06, 09 and 10:

    .\\.venv\\Scripts\\python.exe src\\12_hybrid_ciciov_sumo_replay.py

This is a research prototype, not production automotive safety software.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


NON_MODEL_COLUMNS = {
    "source_file",
    "window_index",
    "start_row",
    "end_row",
    "split",
    "chronological_split",
    "feature_signature",
    "binary_target",
    "multiclass_target",
}
CAN_ATTACK_PHASES = {"can_injection", "combined_attack"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid CICIoV2024 CAN and SUMO multi-source replay."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def load_script(path: Path, module_name: str) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find required script: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find required data: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty results: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_from_filename(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot extract seed from {path.name}") from exc


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(
    truth: Sequence[int], prediction: Sequence[int]
) -> dict[str, float | int]:
    if len(truth) != len(prediction):
        raise ValueError("truth and prediction lengths differ")
    tp = fp = tn = fn = 0
    for expected, predicted in zip(truth, prediction):
        if expected and predicted:
            tp += 1
        elif not expected and predicted:
            fp += 1
        elif not expected and not predicted:
            tn += 1
        else:
            fn += 1
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": safe_divide(fp, fp + tn),
        "accuracy": safe_divide(tp + tn, len(truth)),
    }


def load_can_assets(
    project_root: Path,
) -> tuple[pd.DataFrame, list[str], Any, float, dict[str, float | int]]:
    test_path = (
        project_root
        / "data"
        / "processed"
        / "ciciov2024_windows_w100_group_disjoint_test.csv"
    )
    model_path = (
        project_root
        / "models"
        / "group_disjoint_w100"
        / "group_disjoint_logistic_regression.joblib"
    )
    threshold_path = (
        project_root
        / "results"
        / "group_disjoint_w100"
        / "group_disjoint_thresholds.json"
    )
    for path in (test_path, model_path, threshold_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing Step 06 asset: {path}")

    data = pd.read_csv(test_path)
    feature_names = [column for column in data.columns if column not in NON_MODEL_COLUMNS]
    if not feature_names:
        raise ValueError("No CICIoV2024 model features found")
    if data[feature_names].isna().any().any():
        raise ValueError("CICIoV2024 test features contain missing values")
    model = joblib.load(model_path)
    with threshold_path.open(encoding="utf-8") as handle:
        thresholds = json.load(handle)
    threshold = float(thresholds["Logistic Regression"])

    x_test = data[feature_names].to_numpy(dtype=np.float64)
    truth = data["binary_target"].to_numpy(dtype=np.uint8)
    probability = model.predict_proba(x_test)[:, 1]
    prediction = (probability >= threshold).astype(np.uint8)
    data = data.copy()
    data["can_model_attack_probability"] = probability
    data["can_model_prediction"] = prediction
    subsystem_metrics: dict[str, float | int] = {
        "samples": len(data),
        "benign_samples": int((truth == 0).sum()),
        "attack_samples": int((truth == 1).sum()),
        "threshold": threshold,
        "accuracy": float(accuracy_score(truth, prediction)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, probability)),
        "pr_auc": float(average_precision_score(truth, probability)),
    }
    return data, feature_names, model, threshold, subsystem_metrics


def sample_replay_windows(
    can_test: pd.DataFrame,
    sumo_rows: Sequence[dict[str, str]],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    attack_count = sum(row["phase"] in CAN_ATTACK_PHASES for row in sumo_rows)
    benign_count = len(sumo_rows) - attack_count
    benign = can_test[can_test["binary_target"] == 0]
    attack = can_test[can_test["binary_target"] == 1]
    if benign_count > len(benign) or attack_count > len(attack):
        raise ValueError(
            "Not enough held-out CICIoV2024 windows for without-replacement replay: "
            f"need benign={benign_count}, attack={attack_count}; "
            f"available benign={len(benign)}, attack={len(attack)}"
        )
    benign_sample = benign.sample(n=benign_count, replace=False, random_state=seed)
    attack_sample = attack.sample(n=attack_count, replace=False, random_state=seed + 10000)
    return (
        benign_sample.sample(frac=1.0, random_state=seed + 20000).reset_index(drop=True),
        attack_sample.sample(frac=1.0, random_state=seed + 30000).reset_index(drop=True),
    )


def build_hybrid_rows(
    sumo_rows: Sequence[dict[str, str]],
    can_test: pd.DataFrame,
    seed: int,
) -> list[dict[str, str]]:
    benign, attack = sample_replay_windows(can_test, sumo_rows, seed)
    benign_index = 0
    attack_index = 0
    output: list[dict[str, str]] = []
    for sumo_row in sumo_rows:
        use_attack = sumo_row["phase"] in CAN_ATTACK_PHASES
        if use_attack:
            can_row = attack.iloc[attack_index]
            attack_index += 1
        else:
            can_row = benign.iloc[benign_index]
            benign_index += 1

        probability = float(can_row["can_model_attack_probability"])
        row = dict(sumo_row)
        row["original_synthetic_can_attack_probability"] = row.get(
            "can_attack_probability", ""
        )
        row["can_attack_probability"] = f"{probability:.12g}"
        row["can_behavior_score"] = f"{max(0.0, min(1.0, 1.0 - probability)):.12g}"
        row["can_replay_binary_target"] = str(int(can_row["binary_target"]))
        row["can_replay_model_prediction"] = str(
            int(can_row["can_model_prediction"])
        )
        row["can_replay_source_file"] = str(can_row.get("source_file", ""))
        row["can_replay_window_index"] = str(can_row.get("window_index", ""))
        row["can_replay_multiclass_target"] = str(
            can_row.get("multiclass_target", "")
        )
        output.append(row)
    return output


def aggregate_metrics(per_seed: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    metric_names = ("precision", "recall", "f1", "false_positive_rate", "accuracy")
    output: list[dict[str, object]] = []
    for metric in metric_names:
        values = [float(row[metric]) for row in per_seed]
        output.append(
            {
                "metric": metric,
                "runs": len(values),
                "mean": round(statistics.fmean(values), 6),
                "population_std": round(statistics.pstdev(values), 6),
                "minimum": round(min(values), 6),
                "maximum": round(max(values), 6),
            }
        )
    return output


def phase_summary(
    seed: int,
    hybrid_rows: Sequence[dict[str, str]],
    evaluated: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    if len(hybrid_rows) != len(evaluated):
        raise ValueError("Hybrid and evaluated rows differ in length")
    grouped: dict[str, list[tuple[dict[str, str], dict[str, object]]]] = defaultdict(list)
    phase_order: list[str] = []
    for hybrid, decision in zip(hybrid_rows, evaluated):
        phase = hybrid["phase"]
        if phase not in grouped:
            phase_order.append(phase)
        grouped[phase].append((hybrid, decision))

    output: list[dict[str, object]] = []
    for phase in phase_order:
        values = grouped[phase]
        probabilities = [float(row[0]["can_attack_probability"]) for row in values]
        active_sources: Counter[str] = Counter()
        for _, decision in values:
            for source in str(decision["active_anomaly_sources"]).split(";"):
                if source:
                    active_sources[source] += 1
        output.append(
            {
                "seed": seed,
                "phase": phase,
                "ground_truth_attack": int(values[0][0]["ground_truth_attack"]),
                "rows": len(values),
                "mean_can_model_attack_probability": round(
                    statistics.fmean(probabilities), 6
                ),
                "can_replay_attack_windows": sum(
                    int(row[0]["can_replay_binary_target"]) for row in values
                ),
                "context_aware_alarm_rate": round(
                    safe_divide(
                        sum(int(row[1]["context_aware_alarm"]) for row in values),
                        len(values),
                    ),
                    6,
                ),
                "can_active_rows": active_sources.get("can", 0),
                "sensor_control_active_rows": active_sources.get(
                    "sensor_control", 0
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    step09_path = project_root / "src" / "09_evaluate_context_aware_zero_trust.py"
    if not step09_path.exists():
        step09_path = project_root / step09_path.name
    step09 = load_script(step09_path, "ztav_step09_hybrid")

    can_test, feature_names, _model, threshold, subsystem_metrics = load_can_assets(
        project_root
    )
    sumo_dir = project_root / "data" / "processed" / "sumo_repeated_seeds"
    sumo_paths = sorted(sumo_dir.glob("sumo_context_attacks_seed_*.csv"))
    if not sumo_paths:
        raise FileNotFoundError(f"No repeated-seed SUMO contexts found in {sumo_dir}")

    hybrid_dir = project_root / "data" / "processed" / "hybrid_ciciov_sumo"
    results_dir = project_root / "results" / "hybrid_ciciov_sumo"
    per_seed_metrics: list[dict[str, object]] = []
    all_phase_rows: list[dict[str, object]] = []
    replay_counts: list[dict[str, object]] = []
    for sumo_path in sumo_paths:
        seed = seed_from_filename(sumo_path)
        sumo_rows = read_csv_rows(sumo_path)
        hybrid_rows = build_hybrid_rows(sumo_rows, can_test, seed)
        evaluated = step09.evaluate(hybrid_rows)
        truth = [int(row["ground_truth_attack"]) for row in hybrid_rows]
        alarms = [int(row["context_aware_alarm"]) for row in evaluated]
        metrics = binary_metrics(truth, alarms)
        per_seed_metrics.append(
            {
                "seed": seed,
                **{
                    key: round(value, 6) if isinstance(value, float) else value
                    for key, value in metrics.items()
                },
            }
        )
        all_phase_rows.extend(phase_summary(seed, hybrid_rows, evaluated))
        replay_counts.append(
            {
                "seed": seed,
                "sumo_rows": len(hybrid_rows),
                "replayed_benign_can_windows": sum(
                    int(row["can_replay_binary_target"]) == 0 for row in hybrid_rows
                ),
                "replayed_attack_can_windows": sum(
                    int(row["can_replay_binary_target"]) == 1 for row in hybrid_rows
                ),
                "unique_replayed_source_windows": len(
                    {
                        (row["can_replay_source_file"], row["can_replay_window_index"])
                        for row in hybrid_rows
                    }
                ),
            }
        )
        write_csv(
            hybrid_dir / f"hybrid_context_seed_{seed}.csv",
            [dict(row) for row in hybrid_rows],
        )

    subsystem_row: dict[str, object] = {
        "model": "group_disjoint_logistic_regression",
        **{
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in subsystem_metrics.items()
        },
    }
    manifest_rows: list[dict[str, object]] = [
        {"item": "evaluation_type", "value": "hybrid replay; not external validation"},
        {"item": "can_source", "value": "held-out group-disjoint CICIoV2024 windows"},
        {"item": "context_source", "value": "SUMO GNSS/IMU/V2X repeated-seed simulations"},
        {"item": "can_model", "value": "group_disjoint_logistic_regression.joblib"},
        {"item": "can_model_threshold", "value": threshold},
        {"item": "can_feature_count", "value": len(feature_names)},
        {
            "item": "pairing_rule",
            "value": "without replacement per seed; CAN/combined phases receive attack windows",
        },
        {
            "item": "retained_simulated_signal",
            "value": "sensor_control_consistency_score remains the Step 08 cross-source signal",
        },
        {
            "item": "known_limitation",
            "value": "CICIoV2024 capture bias and limited signature diversity remain",
        },
    ]

    write_csv(results_dir / "can_subsystem_test_metrics.csv", [subsystem_row])
    write_csv(results_dir / "hybrid_per_seed_metrics.csv", per_seed_metrics)
    write_csv(
        results_dir / "hybrid_aggregate_metrics.csv",
        aggregate_metrics(per_seed_metrics),
    )
    write_csv(results_dir / "hybrid_phase_summary.csv", all_phase_rows)
    write_csv(results_dir / "hybrid_replay_counts.csv", replay_counts)
    write_csv(results_dir / "hybrid_replay_manifest.csv", manifest_rows)
    with (results_dir / "can_feature_columns.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_names, handle, indent=2)

    f1_values = [float(row["f1"]) for row in per_seed_metrics]
    print("\n" + "=" * 80)
    print("Hybrid CICIoV2024 + SUMO replay completed successfully.")
    print(
        f"CAN subsystem test F1={float(subsystem_metrics['f1']):.4f}, "
        f"threshold={threshold:.6f}"
    )
    print(
        f"End-to-end context-aware F1 mean={statistics.fmean(f1_values):.4f}, "
        f"std={statistics.pstdev(f1_values):.4f}"
    )
    print(f"Results directory: {results_dir}")
    print("\nNext: external-dataset validation and ablation analysis.")


if __name__ == "__main__":
    main()
