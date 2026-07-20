#!/usr/bin/env python3
"""Sparse-event repair of the ROAD semantic signal gate.

Step 29 aggregated row-level semantic evidence with a 95th percentile inside
each 100-row window.  The ROAD label audit subsequently showed that the median
positive window contains only about 4.46 malicious rows (and the coolant attack
contains one).  A 95th percentile can therefore discard the attack rows by
construction.  This explicitly post-hoc *development* stage replaces q95 with
maximum/top-event evidence while preserving the important leakage controls:

* The 105-ID semantic profiles fitted by Step 29 are reused unchanged.
* Only Step 29's two ambient calibration captures select thresholds.
* The two benign holdout captures and all attack labels remain evaluation-only.
* The primary rule is predeclared here as an instantaneous maximum-event rule;
  it is intended to trigger VERIFY/RESTRICT, not a hard safety fallback.

Run from D:\\ztav_project after Step 29:

    .\\.venv\\Scripts\\python.exe src\\30_road_sparse_signal_event_gate.py

This is a research prototype, not production automotive safety software.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIMARY_METHOD = "sparse_event_max_any_instant"
METHODS = {
    "sparse_event_max_any_instant": ("risk_max_any", 1),
    "sparse_event_max_context_instant": ("risk_max_context", 1),
    "sparse_event_max_any_persistent_2": ("risk_max_any", 2),
    "sparse_event_top3_any_instant": ("risk_top3_any", 1),
}
EVIDENCE_NAMES = ("marginal", "context", "transition")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sparse-event repair for the ROAD semantic signal gate."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--chunk-rows", type=int, default=200_000)
    args = parser.parse_args()
    if args.chunk_rows < 1_000:
        parser.error("--chunk-rows must be at least 1000")
    return args


def locate_script(root: Path, name: str) -> Path:
    for candidate in (root / "src" / name, root / name):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find required script: {name}")


def load_script(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def restore_profiles(raw: dict[object, object], step29: ModuleType) -> dict[int, object]:
    profiles: dict[int, object] = {}
    for raw_id, payload in raw.items():
        can_id = int(raw_id)
        if isinstance(payload, step29.SignalProfile):
            profiles[can_id] = payload
        elif isinstance(payload, dict):
            profiles[can_id] = step29.SignalProfile(**payload)
        else:
            raise TypeError(f"Unsupported Step 29 profile payload for ID {can_id}")
    if not profiles:
        raise ValueError("Step 29 model contains no semantic profiles")
    return profiles


def top_k_mean(values: np.ndarray, k: int) -> np.ndarray:
    k = min(k, values.shape[1])
    selected = np.partition(values, values.shape[1] - k, axis=1)[:, -k:]
    return selected.mean(axis=1)


def sparse_windowize_capture(
    capture_name: str,
    path: Path,
    role: str,
    attack_family: str,
    profiles: dict[int, object],
    step29: ModuleType,
    chunk_rows: int,
    window_rows: int,
) -> pd.DataFrame:
    pending: dict[str, np.ndarray] = {
        "time": np.empty(0, dtype=float),
        "label": np.empty(0, dtype=np.int8),
        "marginal_score": np.empty(0, dtype=float),
        "context_score": np.empty(0, dtype=float),
        "transition_score": np.empty(0, dtype=float),
        "unknown_signal": np.empty(0, dtype=float),
    }
    last_by_id: dict[int, np.ndarray] = {}
    batches: list[pd.DataFrame] = []
    window_offset = 0
    for chunk in step29.read_chunks(path, chunk_rows):
        scored = step29.score_chunk(chunk, profiles, last_by_id)
        arrays = {
            "time": chunk["Time"].to_numpy(float),
            "label": pd.to_numeric(chunk["Label"], errors="coerce")
            .fillna(0)
            .gt(0)
            .to_numpy(np.int8),
            **scored,
        }
        arrays = {
            key: np.concatenate((pending[key], value))
            for key, value in arrays.items()
        }
        complete = (len(arrays["time"]) // window_rows) * window_rows
        if complete:
            count = complete // window_rows
            reshaped = {
                key: value[:complete].reshape(count, window_rows)
                for key, value in arrays.items()
            }
            batch = pd.DataFrame(
                {
                    "capture_name": capture_name,
                    "capture_role": role,
                    "attack_family": attack_family,
                    "window_index": np.arange(window_offset, window_offset + count),
                    "window_start_time": reshaped["time"][:, 0],
                    "window_end_time": reshaped["time"][:, -1],
                    "attack_target": reshaped["label"].max(axis=1).astype(np.int8),
                    "attack_row_fraction": reshaped["label"].mean(axis=1),
                    "marginal_max": reshaped["marginal_score"].max(axis=1),
                    "context_max": reshaped["context_score"].max(axis=1),
                    "transition_max": reshaped["transition_score"].max(axis=1),
                    "marginal_top3_mean": top_k_mean(reshaped["marginal_score"], 3),
                    "context_top3_mean": top_k_mean(reshaped["context_score"], 3),
                    "transition_top3_mean": top_k_mean(reshaped["transition_score"], 3),
                    "unknown_signal_fraction": reshaped["unknown_signal"].mean(axis=1),
                }
            )
            batches.append(batch)
            window_offset += count
        pending = {key: value[complete:] for key, value in arrays.items()}
    if not batches:
        raise ValueError(f"Capture {capture_name} produced no complete windows")
    output = pd.concat(batches, ignore_index=True)
    output["discarded_tail_rows"] = len(pending["time"])
    return output


def add_sparse_risks(
    frames: Sequence[pd.DataFrame],
    calibration: pd.DataFrame,
    step29: ModuleType,
) -> dict[str, np.ndarray]:
    references: dict[str, np.ndarray] = {}
    for suffix in ("max", "top3_mean"):
        columns = [f"{name}_{suffix}" for name in EVIDENCE_NAMES]
        suffix_references = {
            column: step29.balanced_reference(calibration, column) for column in columns
        }
        references.update(suffix_references)
        for frame in frames:
            tail_scores = np.column_stack(
                [
                    robust_tail_score(
                        suffix_references[column], frame[column].to_numpy(float)
                    )
                    for column in columns
                ]
            )
            ordered = np.sort(tail_scores, axis=1)
            if suffix == "max":
                frame["risk_max_any"] = ordered[:, 2]
                frame["risk_max_consensus"] = ordered[:, 1]
                frame["risk_max_context"] = (ordered[:, 1] + ordered[:, 2]) / 2.0
            else:
                frame["risk_top3_any"] = ordered[:, 2]
                frame["risk_top3_context"] = (ordered[:, 1] + ordered[:, 2]) / 2.0
    return references


def robust_tail_score(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map non-negative evidence to an unbounded benign-relative tail score."""
    reference_log = np.log1p(np.maximum(np.asarray(reference, dtype=float), 0.0))
    value_log = np.log1p(np.maximum(np.asarray(values, dtype=float), 0.0))
    center = float(np.median(reference_log))
    q95 = float(np.quantile(reference_log, 0.95))
    mad = float(np.median(np.abs(reference_log - center)) * 1.4826)
    scale = max(q95 - center, mad, 1e-6)
    return (value_log - center) / scale


def select_benign_threshold(
    calibration: pd.DataFrame,
    risk_column: str,
    persistence: int,
    step29: ModuleType,
) -> dict[str, object]:
    """Choose the most sensitive threshold satisfying benign-only FPR limits."""
    groups = [
        group.sort_values("window_index")[risk_column].to_numpy(float)
        for _, group in calibration.groupby("capture_name", sort=True)
    ]
    pooled_values = np.concatenate(groups)
    quantiles = np.quantile(pooled_values, np.linspace(0.0, 1.0, 2_001))
    unique = np.unique(quantiles)
    candidates = np.unique(
        np.r_[unique, np.nextafter(unique, np.inf), np.nextafter(unique[-1], np.inf)]
    )

    def operating_point(threshold: float) -> tuple[bool, float, float, float, int, int]:
        rates: list[float] = []
        alarms = 0
        windows = 0
        for values in groups:
            alarm = step29.alarm_for_group(values, threshold, persistence)
            rates.append(float(alarm.mean()))
            alarms += int(alarm.sum())
            windows += len(alarm)
        pooled = alarms / max(windows, 1)
        macro = float(np.mean(rates))
        worst = float(np.max(rates))
        passed = (
            pooled <= step29.TARGET_CALIBRATION_FPR
            and macro <= step29.TARGET_CALIBRATION_FPR
            and worst <= step29.MAX_CALIBRATION_CAPTURE_FPR
        )
        return passed, pooled, macro, worst, alarms, windows

    low = 0
    high = len(candidates) - 1
    while low < high:
        middle = (low + high) // 2
        if operating_point(float(candidates[middle]))[0]:
            high = middle
        else:
            low = middle + 1
    threshold = float(candidates[low])
    passed, pooled, macro, worst, _, windows = operating_point(threshold)
    if not passed:
        raise RuntimeError("Could not satisfy benign-only threshold constraints")
    return {
        "threshold": threshold,
        "calibration_pooled_fpr": pooled,
        "calibration_macro_fpr": macro,
        "calibration_worst_capture_fpr": worst,
        "calibration_windows": windows,
    }


def evaluate(
    holdout: pd.DataFrame,
    attacks: pd.DataFrame,
    controls: pd.DataFrame,
    step29: ModuleType,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    overall: list[dict[str, object]] = []
    holdout_rows: list[dict[str, object]] = []
    family_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    endpoint = pd.concat((holdout, attacks), ignore_index=True)
    for method in METHODS:
        column = f"alarm_{method}"
        overall.append(
            step29.metric_row(endpoint, method, column, "benign_holdout_plus_signal_attacks")
        )
        for name, group in holdout.groupby("capture_name", sort=True):
            holdout_rows.append(
                step29.metric_row(group, method, column, f"holdout:{name}")
            )
        for family, group in attacks.groupby("attack_family", sort=True):
            family_rows.append(
                step29.metric_row(group, method, column, f"family:{family}")
            )
        if len(controls):
            for name, group in controls.groupby("capture_name", sort=True):
                control_rows.append(
                    {
                        "method": method,
                        "capture_name": name,
                        "windows": len(group),
                        "compromised_state_alarm_rate": float(group[column].mean()),
                        "note": "negative control; capture is compromised from startup",
                    }
                )
    return overall, holdout_rows, family_rows, control_rows


def acceptance_rows(
    overall: pd.DataFrame,
    holdout: pd.DataFrame,
    families: pd.DataFrame,
    attack_audit: list[dict[str, object]],
) -> list[dict[str, object]]:
    selected_overall = overall[overall["method"] == PRIMARY_METHOD].iloc[0]
    selected_holdout = holdout[holdout["method"] == PRIMARY_METHOD]
    selected_families = families[families["method"] == PRIMARY_METHOD]
    pooled = selected_holdout["fp"].sum() / max(selected_holdout["benign_windows"].sum(), 1)
    macro = float(selected_holdout["false_positive_rate"].mean())
    worst_holdout = float(selected_holdout["false_positive_rate"].max())
    worst_family = float(selected_families["recall"].min())
    primary_audit = [row for row in attack_audit if row["endpoint_role"] == "primary_signal_attack"]
    criteria = [
        ("holdout_pooled_fpr", pooled, "<=", 0.05),
        ("holdout_macro_fpr", macro, "<=", 0.05),
        ("holdout_worst_capture_fpr", worst_holdout, "<=", 0.10),
        (
            "endpoint_benign_window_fpr",
            float(selected_overall["false_positive_rate"]),
            "<=",
            0.05,
        ),
        ("primary_masquerade_recall", float(selected_overall["recall"]), ">=", 0.70),
        ("worst_attack_family_recall", worst_family, ">=", 0.50),
        (
            "primary_attack_captures_with_positive_labels",
            float(sum(int(row["positive_windows"]) > 0 for row in primary_audit)),
            "==",
            float(len(primary_audit)),
        ),
        ("attack_labels_used_for_profile_or_threshold_fit", 0.0, "==", 0.0),
    ]
    rows: list[dict[str, object]] = []
    passes: list[bool] = []
    for criterion, observed, operator, required in criteria:
        if operator == "<=":
            passed = observed <= required
        elif operator == ">=":
            passed = observed >= required
        else:
            passed = observed == required
        passes.append(bool(passed))
        rows.append(
            {
                "criterion": criterion,
                "observed_value": observed,
                "operator": operator,
                "required_value": required,
                "passed": bool(passed),
            }
        )
    rows.append(
        {
            "criterion": "all_predeclared_candidate_readiness_criteria",
            "observed_value": float(all(passes)),
            "operator": "==",
            "required_value": 1.0,
            "passed": bool(all(passes)),
        }
    )
    return rows


def plot_summary(
    family_metrics: pd.DataFrame,
    holdout_metrics: pd.DataFrame,
    path: Path,
) -> None:
    family = family_metrics[family_metrics["method"] == PRIMARY_METHOD].copy()
    family["family"] = family["scope"].str.replace("family:", "", regex=False)
    holdout = holdout_metrics[holdout_metrics["method"] == PRIMARY_METHOD].copy()
    holdout["capture"] = holdout["scope"].str.replace("holdout:", "", regex=False)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].bar(family["family"], family["recall"], color="#2ca02c")
    axes[0].axhline(0.50, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Attack-window recall")
    axes[0].set_title("Maximum-event semantic recall")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(holdout["capture"], holdout["false_positive_rate"], color="#d62728")
    axes[1].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylim(0, max(0.12, float(holdout["false_positive_rate"].max()) * 1.15))
    axes[1].set_ylabel("False-positive rate")
    axes[1].set_title("Untouched benign-capture availability")
    axes[1].tick_params(axis="x", rotation=25)
    figure.suptitle("ROAD sparse signal-event gate")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    step29 = load_script(
        locate_script(root, "29_road_signal_context_gate.py"), "ztav_step29_signal_context"
    )
    step29_model_path = (
        root / "models" / "road_signal_context_gate" / "road_signal_context_gate.joblib"
    )
    if not step29_model_path.exists():
        raise FileNotFoundError(
            f"Step 29 model not found: {step29_model_path}. Run Step 29 first."
        )
    step29_model = joblib.load(step29_model_path)
    profiles = restore_profiles(step29_model["profiles"], step29)
    split = {str(name): str(role) for name, role in step29_model["ambient_split"].items()}
    window_rows = int(step29_model["window_rows"])

    signal_root = step29.locate_signal_root(root)
    metadata_path = step29.locate_attack_metadata(signal_root)
    metadata = step29.load_metadata(metadata_path)
    ambient_files, attack_files = step29.list_files(signal_root, metadata)
    if set(split) != set(ambient_files):
        raise ValueError("Current ambient capture set differs from the frozen Step 29 split")
    calibration_files = sorted(
        (name, ambient_files[name]) for name in ambient_files if split[name] == "threshold_calibration"
    )
    holdout_files = sorted(
        (name, ambient_files[name]) for name in ambient_files if split[name] == "benign_holdout"
    )

    print(f"Reusing {len(profiles)} frozen Step 29 arbitration-ID profiles.")
    calibration_frames: list[pd.DataFrame] = []
    holdout_frames: list[pd.DataFrame] = []
    for name, path in calibration_files:
        print(f"Scoring benign calibration capture: {name}")
        calibration_frames.append(
            sparse_windowize_capture(
                name, path, "ambient_calibration", "ambient", profiles, step29,
                args.chunk_rows, window_rows,
            )
        )
    for name, path in holdout_files:
        print(f"Scoring untouched benign holdout: {name}")
        holdout_frames.append(
            sparse_windowize_capture(
                name, path, "ambient_holdout", "ambient", profiles, step29,
                args.chunk_rows, window_rows,
            )
        )
    calibration = pd.concat(calibration_frames, ignore_index=True)
    holdout = pd.concat(holdout_frames, ignore_index=True)

    attack_frames: list[pd.DataFrame] = []
    control_frames: list[pd.DataFrame] = []
    attack_audit: list[dict[str, object]] = []
    for index, name in enumerate(sorted(attack_files), start=1):
        family = step29.attack_family(name)
        interval = metadata[name].get("injection_interval")
        role = "compromised_start_control" if interval is None else "primary_signal_attack"
        print(f"[{index}/{len(attack_files)}] Scoring {role}: {name}")
        scored = sparse_windowize_capture(
            name, attack_files[name], role, family, profiles, step29,
            args.chunk_rows, window_rows,
        )
        attack_audit.append(
            {
                "capture_name": name,
                "attack_family": family,
                "endpoint_role": role,
                "windows": len(scored),
                "positive_windows": int(scored["attack_target"].sum()),
                "positive_rows_equivalent": float(
                    scored["attack_row_fraction"].sum() * window_rows
                ),
                "metadata_has_injection_interval": interval is not None,
            }
        )
        if role == "primary_signal_attack":
            attack_frames.append(scored)
        else:
            control_frames.append(scored)
    attacks = pd.concat(attack_frames, ignore_index=True)
    controls = pd.concat(control_frames, ignore_index=True) if control_frames else pd.DataFrame()

    all_frames = [calibration, holdout, attacks, *([controls] if len(controls) else [])]
    references = add_sparse_risks(all_frames, calibration, step29)
    threshold_rows: list[dict[str, object]] = []
    thresholds: dict[str, dict[str, object]] = {}
    for method, (risk_column, persistence) in METHODS.items():
        selected = select_benign_threshold(
            calibration, risk_column, persistence, step29
        )
        thresholds[method] = selected
        threshold_rows.append(
            {
                "method": method,
                "risk_column": risk_column,
                "persistence_windows": persistence,
                **selected,
                "attack_labels_used": False,
            }
        )
        for frame in all_frames:
            step29.add_alarm(
                frame,
                f"alarm_{method}",
                risk_column,
                float(selected["threshold"]),
                persistence,
            )

    overall, holdout_rows, family_rows, control_rows = evaluate(
        holdout, attacks, controls, step29
    )
    acceptance = acceptance_rows(
        pd.DataFrame(overall),
        pd.DataFrame(holdout_rows),
        pd.DataFrame(family_rows),
        attack_audit,
    )
    output_dir = root / "results" / "road_sparse_signal_event_gate"
    model_dir = root / "models" / "road_sparse_signal_event_gate"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction = pd.concat((calibration, holdout, attacks, controls), ignore_index=True)
    prediction.to_csv(output_dir / "sparse_signal_predictions.csv", index=False)
    write_csv(output_dir / "sparse_signal_thresholds.csv", threshold_rows)
    write_csv(output_dir / "sparse_signal_overall_metrics.csv", overall)
    write_csv(output_dir / "sparse_signal_holdout_fpr.csv", holdout_rows)
    write_csv(output_dir / "sparse_signal_per_family_metrics.csv", family_rows)
    write_csv(output_dir / "sparse_signal_attack_label_audit.csv", attack_audit)
    if control_rows:
        write_csv(output_dir / "sparse_signal_compromised_controls.csv", control_rows)
    write_csv(output_dir / "sparse_signal_acceptance_criteria.csv", acceptance)
    manifest = [
        {"item": "experiment_stage", "value": "post-hoc sparse aggregation development after Step 29"},
        {"item": "source_model", "value": str(step29_model_path)},
        {"item": "frozen_profiles", "value": len(profiles)},
        {"item": "window_rows", "value": window_rows},
        {"item": "primary_method_predeclared", "value": PRIMARY_METHOD},
        {"item": "attack_labels_used_for_profile_fit", "value": False},
        {"item": "attack_labels_used_for_threshold_calibration", "value": False},
        {"item": "development_reason", "value": "Step 29 q95 exceeded median malicious-row count per positive window"},
    ]
    write_csv(output_dir / "sparse_signal_manifest.csv", manifest)
    plot_summary(
        pd.DataFrame(family_rows),
        pd.DataFrame(holdout_rows),
        output_dir / "sparse_signal_summary.png",
    )
    joblib.dump(
        {
            "source_step29_model": str(step29_model_path),
            "thresholds": thresholds,
            "references": references,
            "methods": METHODS,
            "primary_method": PRIMARY_METHOD,
            "window_rows": window_rows,
            "manifest": manifest,
        },
        model_dir / "road_sparse_signal_event_gate.joblib",
        compress=3,
    )

    overall_frame = pd.DataFrame(overall).set_index("method")
    holdout_frame = pd.DataFrame(holdout_rows)
    family_frame = pd.DataFrame(family_rows)
    acceptance_frame = pd.DataFrame(acceptance).set_index("criterion")
    selected = overall_frame.loc[PRIMARY_METHOD]
    selected_holdout = holdout_frame[holdout_frame["method"] == PRIMARY_METHOD]
    selected_family = family_frame[family_frame["method"] == PRIMARY_METHOD]
    positive = [row for row in attack_audit if row["endpoint_role"] == "primary_signal_attack"]
    rows_per_positive_window = sum(float(row["positive_rows_equivalent"]) for row in positive) / max(
        sum(int(row["positive_windows"]) for row in positive), 1
    )
    print("\n" + "=" * 88)
    print("ROAD sparse signal-event gate completed successfully.")
    print(f"Frozen Step 29 profiles reused: {len(profiles)} IDs")
    print(f"Weighted malicious rows per positive 100-row window: {rows_per_positive_window:.3f}")
    print(f"Primary rule: {PRIMARY_METHOD}")
    print(
        f"Endpoint precision={selected['precision']:.4f}, recall={selected['recall']:.4f}, "
        f"F1={selected['f1']:.4f}, FPR={selected['false_positive_rate']:.4f}"
    )
    print(
        f"Untouched ambient pooled FPR={selected_holdout['fp'].sum() / max(selected_holdout['benign_windows'].sum(), 1):.4f}, "
        f"macro FPR={selected_holdout['false_positive_rate'].mean():.4f}"
    )
    print("\nPrimary recall by signal attack family:")
    for _, row in selected_family.iterrows():
        print(f"  {row['scope'].replace('family:', ''):<40} recall={row['recall']:.4f}")
    ready = bool(
        acceptance_frame.loc[
            "all_predeclared_candidate_readiness_criteria", "passed"
        ]
    )
    print(f"\nPredeclared candidate-readiness criteria passed: {ready}")
    print(f"Results directory: {output_dir}")
    print(f"Model directory: {model_dir}")
    print("\nNext: integrate only if recall and untouched-capture FPR both pass.")


if __name__ == "__main__":
    main()
