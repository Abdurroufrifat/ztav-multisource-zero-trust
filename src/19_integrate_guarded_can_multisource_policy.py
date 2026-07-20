#!/usr/bin/env python3
"""Integrate the guarded HCRL CAN gate with multi-source SUMO Zero Trust context.

This stage connects the principal components built in Steps 07--18:

* session-normalized HCRL CAN deviation and graded CAN actions (Step 17),
* the frozen startup-consistency guard stress-tested in Step 18, and
* SUMO GNSS/IMU, V2X, identity, device-posture, and vehicle-state context.

Four HCRL capture sources are replayed across every preserved SUMO seed under
four attack-density conditions.  Attack labels are used only to construct the
controlled replay conditions and to score the resulting decisions.  The
detector and policy read evidence fields, never ``ground_truth_attack`` or
``phase``.

The HCRL data and Step 17 gate are not an independent confirmatory test because
the same external corpus informed the exploratory gate design.  SUMO context is
simulated.  Treat this as a hybrid integration and ablation experiment.

Run from D:\\ztav_project after Steps 10, 17, and 18:

    .\\.venv\\Scripts\\python.exe src\\19_integrate_guarded_can_multisource_policy.py

This is a research prototype, not production automotive safety software.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import statistics
import sys
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CAN_ATTACK_PHASES = {"can_injection", "combined_attack"}
ATTACK_DENSITY_SCENARIOS = {
    "representative_all": (1, 100),
    "low_1_5": (1, 5),
    "medium_6_20": (6, 20),
    "high_21_100": (21, 100),
}
DENSITY_ORDER = tuple(ATTACK_DENSITY_SCENARIOS)
SOURCE_THRESHOLDS = {
    "gnss": ("gnss_imu_consistency_score", 0.30),
    "v2x": ("v2x_consistency_score", 0.30),
    "sensor_control": ("sensor_control_consistency_score", 0.35),
}
ATTACK_PERSISTENCE = 2
RECOVERY_PERSISTENCE = 3
METHOD_COLUMNS = {
    "legacy_weighted_threshold": "legacy_weighted_alarm",
    "guarded_can_only": "guarded_can_alarm",
    "context_without_can": "noncan_context_alarm",
    "proposed_without_vehicle_state": "proposed_without_vehicle_state_alarm",
    "proposed_multisource": "proposed_multisource_alarm",
}
REQUIRED_CAN_COLUMNS = {
    "source_file",
    "source_capture_class",
    "window_index",
    "binary_target",
    "attack_frame_count",
    "session_deviation_score",
}
REQUIRED_SUMO_COLUMNS = {
    "simulation_time_s",
    "phase",
    "ground_truth_attack",
    "security_alarm",
    "identity_critical_failure",
    "device_critical_failure",
    "identity_score",
    "device_posture_score",
    "can_behavior_score",
    "gnss_imu_consistency_score",
    "v2x_consistency_score",
    "sensor_control_consistency_score",
    "freshness_score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrate the guarded HCRL CAN gate with SUMO context."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/guarded_multisource_zero_trust_w100"),
    )
    return parser.parse_args()


def resolve_path(project_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else project_root / value


def locate_script(project_root: Path, name: str) -> Path:
    candidates = (project_root / "src" / name, project_root / name)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find required script: {name}")


def load_script(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find required CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def seed_from_filename(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot extract seed from {path.name}") from exc


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts).encode("utf-8")
    return zlib.crc32(text) & 0xFFFFFFFF


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(
    truth: Sequence[int], prediction: Sequence[int]
) -> dict[str, float | int]:
    if len(truth) != len(prediction):
        raise ValueError("Truth and prediction lengths differ")
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
        "false_negative_rate": safe_divide(fn, fn + tp),
        "accuracy": safe_divide(tp + tn, len(truth)),
    }


@dataclass
class PersistentMonitor:
    threshold: float
    attack_persistence: int = ATTACK_PERSISTENCE
    recovery_persistence: int = RECOVERY_PERSISTENCE
    active: bool = False
    bad_streak: int = 0
    good_streak: int = 0

    def update(self, score: float) -> bool:
        if score < self.threshold:
            self.bad_streak += 1
            self.good_streak = 0
            if self.bad_streak >= self.attack_persistence:
                self.active = True
        else:
            self.bad_streak = 0
            if self.active:
                self.good_streak += 1
                if self.good_streak >= self.recovery_persistence:
                    self.active = False
                    self.good_streak = 0
            else:
                self.good_streak = 0
        return self.active


@dataclass
class BooleanCriticalMonitor:
    recovery_persistence: int = RECOVERY_PERSISTENCE
    active: bool = False
    good_streak: int = 0

    def update(self, failure: bool) -> bool:
        if failure:
            self.active = True
            self.good_streak = 0
        elif self.active:
            self.good_streak += 1
            if self.good_streak >= self.recovery_persistence:
                self.active = False
                self.good_streak = 0
        return self.active


def load_operational_can_threshold(path: Path) -> float:
    rows = read_csv_rows(path)
    matches = [row for row in rows if math.isclose(float(row["calibration_target_fpr"]), 0.05)]
    if len(matches) != 1:
        raise ValueError(f"Expected one operational 0.05 threshold in {path}")
    return float(matches[0]["deviation_threshold"])


def load_clean_startup_guard(path: Path) -> dict[str, dict[str, object]]:
    rows = read_csv_rows(path)
    selected = [
        row
        for row in rows
        if int(row["contaminated_bootstrap_windows"]) == 0
        and int(row["repetition"]) == 0
        and row["method"] == "instant_verify_restrict"
    ]
    output: dict[str, dict[str, object]] = {}
    for row in selected:
        source = row["source_file"]
        if source in output:
            raise ValueError(f"Duplicate clean startup guard row for {source}")
        output[source] = {
            "guard_score": float(row["guard_score"]),
            "guard_threshold": float(row["guard_threshold"]),
            "guard_rejected": row["guard_rejected"].strip().lower() == "true",
        }
    if not output:
        raise ValueError(f"No clean startup guard rows found in {path}")
    return output


def validate_columns(actual: Iterable[str], required: set[str], label: str) -> None:
    missing = required - set(actual)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def sample_frame(
    frame: pd.DataFrame,
    count: int,
    random_seed: int,
) -> tuple[pd.DataFrame, bool]:
    if frame.empty:
        raise ValueError("Cannot sample an empty replay pool")
    replace = count > len(frame)
    rng = np.random.default_rng(random_seed)
    positions = rng.choice(len(frame), size=count, replace=replace)
    return frame.iloc[positions].reset_index(drop=True), replace


def build_replay_rows(
    sumo_rows: Sequence[dict[str, str]],
    can_predictions: pd.DataFrame,
    source_file: str,
    density_scenario: str,
    seed: int,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    source = can_predictions[can_predictions["source_file"] == source_file].copy()
    if source.empty:
        raise ValueError(f"No HCRL predictions found for {source_file}")
    lower, upper = ATTACK_DENSITY_SCENARIOS[density_scenario]
    benign_pool = source[source["binary_target"] == 0]
    attack_pool = source[
        (source["binary_target"] == 1)
        & source["attack_frame_count"].between(lower, upper)
    ]
    attack_count = sum(row["phase"] in CAN_ATTACK_PHASES for row in sumo_rows)
    benign_count = len(sumo_rows) - attack_count
    benign, benign_replacement = sample_frame(
        benign_pool,
        benign_count,
        stable_seed(seed, source_file, density_scenario, "benign"),
    )
    attack, attack_replacement = sample_frame(
        attack_pool,
        attack_count,
        stable_seed(seed, source_file, density_scenario, "attack"),
    )
    benign_i = attack_i = 0
    output: list[dict[str, str]] = []
    for sumo_row in sumo_rows:
        if sumo_row["phase"] in CAN_ATTACK_PHASES:
            can_row = attack.iloc[attack_i]
            attack_i += 1
        else:
            can_row = benign.iloc[benign_i]
            benign_i += 1
        row = dict(sumo_row)
        row.update(
            {
                "hcrl_source_file": str(can_row["source_file"]),
                "hcrl_source_capture_class": str(can_row["source_capture_class"]),
                "hcrl_window_index": str(int(can_row["window_index"])),
                "hcrl_binary_target": str(int(can_row["binary_target"])),
                "hcrl_attack_frame_count": str(int(can_row["attack_frame_count"])),
                "hcrl_session_deviation_score": f"{float(can_row['session_deviation_score']):.12g}",
            }
        )
        output.append(row)
    manifest = {
        "seed": seed,
        "hcrl_source_file": source_file,
        "density_scenario": density_scenario,
        "density_min_attack_frames": lower,
        "density_max_attack_frames": upper,
        "sumo_rows": len(sumo_rows),
        "hcrl_benign_pool_windows": len(benign_pool),
        "hcrl_attack_pool_windows": len(attack_pool),
        "replayed_benign_windows": benign_count,
        "replayed_attack_windows": attack_count,
        "benign_sampling_with_replacement": benign_replacement,
        "attack_sampling_with_replacement": attack_replacement,
        "unique_replayed_benign_windows": int(benign["window_index"].nunique()),
        "unique_replayed_attack_windows": int(attack["window_index"].nunique()),
    }
    return output, manifest


def local_policy(
    noncan_sources: set[str],
    can_instant: bool,
    can_persistent: bool,
    guard_rejected: bool,
) -> tuple[str, str, str]:
    if guard_rejected:
        return "SAFE_FALLBACK", "DENY_COOPERATIVE_ACTION", "DENY"
    active = set(noncan_sources)
    if can_instant:
        active.add("can")
    if not active:
        return "ALLOW", "ALLOW", "ALLOW"
    if (
        can_persistent
        or active & {"identity", "device_posture", "gnss", "sensor_control"}
    ):
        local = "SAFE_FALLBACK"
    elif can_instant:
        local = "VERIFY_RESTRICT"
    else:
        local = "ALLOW_LOCAL_ONLY"
    cooperative = (
        "DENY_COOPERATIVE_ACTION"
        if "v2x" in active
        else "REQUIRE_REVERIFICATION"
    )
    telemetry = "DENY" if active & {"identity", "device_posture"} else "RESTRICT"
    return local, cooperative, telemetry


def evaluate_replay(
    rows: Sequence[dict[str, str]],
    can_threshold: float,
    guard: dict[str, object],
    trust_module: ModuleType,
) -> list[dict[str, object]]:
    monitors = {
        source: PersistentMonitor(threshold)
        for source, (_, threshold) in SOURCE_THRESHOLDS.items()
    }
    identity_monitor = BooleanCriticalMonitor()
    device_monitor = BooleanCriticalMonitor()
    trust_engine = trust_module.ContinuousTrustEngine()
    previous_can_instant = False
    evaluated: list[dict[str, object]] = []

    for row in rows:
        deviation = float(row["hcrl_session_deviation_score"])
        can_instant = deviation >= can_threshold
        can_persistent = can_instant and previous_can_instant
        previous_can_instant = can_instant
        can_trust = 1.0 / (1.0 + (deviation / max(can_threshold, 1e-12)) ** 2)

        noncan_sources: set[str] = set()
        for source, (column, _) in SOURCE_THRESHOLDS.items():
            if monitors[source].update(float(row[column])):
                noncan_sources.add(source)
        identity_failure = bool(int(row["identity_critical_failure"]))
        device_failure = bool(int(row["device_critical_failure"]))
        if identity_monitor.update(identity_failure):
            noncan_sources.add("identity")
        if device_monitor.update(device_failure):
            noncan_sources.add("device_posture")

        guard_rejected = bool(guard["guard_rejected"])
        active_sources = set(noncan_sources)
        if can_instant:
            active_sources.add("can")
        if guard_rejected:
            active_sources.add("startup_guard")
        local, cooperative, telemetry = local_policy(
            noncan_sources,
            can_instant,
            can_persistent,
            guard_rejected,
        )

        evidence = {
            "identity": trust_module.Evidence(
                float(row["identity_score"]),
                critical_failure=identity_failure,
            ),
            "device_posture": trust_module.Evidence(
                float(row["device_posture_score"]),
                critical_failure=device_failure,
            ),
            "can_behavior": trust_module.Evidence(can_trust, quality=0.90),
            "gnss_imu_consistency": trust_module.Evidence(
                float(row["gnss_imu_consistency_score"])
            ),
            "v2x_consistency": trust_module.Evidence(
                float(row["v2x_consistency_score"])
            ),
            "sensor_control_consistency": trust_module.Evidence(
                float(row["sensor_control_consistency_score"])
            ),
            "freshness": trust_module.Evidence(float(row["freshness_score"])),
        }
        fused = trust_engine.evaluate(evidence)

        guarded_can_alarm = int(can_instant or guard_rejected)
        noncan_alarm = int(bool(noncan_sources))
        without_vehicle_state_sources = noncan_sources - {"sensor_control"}
        proposed_without_vehicle_state = int(
            can_instant or bool(without_vehicle_state_sources) or guard_rejected
        )
        proposed = int(can_instant or bool(noncan_sources) or guard_rejected)
        evaluated.append(
            {
                "simulation_time_s": float(row["simulation_time_s"]),
                "phase": row["phase"],
                "ground_truth_attack": int(row["ground_truth_attack"]),
                "hcrl_source_file": row["hcrl_source_file"],
                "hcrl_window_index": int(row["hcrl_window_index"]),
                "hcrl_binary_target": int(row["hcrl_binary_target"]),
                "hcrl_attack_frame_count": int(row["hcrl_attack_frame_count"]),
                "can_deviation_score": deviation,
                "can_deviation_threshold": can_threshold,
                "continuous_can_trust": can_trust,
                "can_alarm_instant": int(can_instant),
                "can_alarm_persistent_2": int(can_persistent),
                "startup_guard_score": float(guard["guard_score"]),
                "startup_guard_threshold": float(guard["guard_threshold"]),
                "startup_guard_rejected": int(guard_rejected),
                "active_anomaly_sources": ";".join(sorted(active_sources)),
                "legacy_weighted_alarm": int(row["security_alarm"]),
                "guarded_can_alarm": guarded_can_alarm,
                "noncan_context_alarm": noncan_alarm,
                "proposed_without_vehicle_state_alarm": proposed_without_vehicle_state,
                "proposed_multisource_alarm": proposed,
                "local_control_decision": local,
                "cooperative_action_decision": cooperative,
                "telemetry_decision": telemetry,
                "fused_instantaneous_trust": fused.instantaneous_trust,
                "fused_continuous_trust": fused.continuous_trust,
                "fused_risk": fused.risk,
            }
        )
    return evaluated


def run_metric_rows(
    seed: int,
    source_file: str,
    density_scenario: str,
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    truth = [int(row["ground_truth_attack"]) for row in rows]
    output: list[dict[str, object]] = []
    for method, column in METHOD_COLUMNS.items():
        prediction = [int(row[column]) for row in rows]
        metrics = binary_metrics(truth, prediction)
        output.append(
            {
                "seed": seed,
                "hcrl_source_file": source_file,
                "density_scenario": density_scenario,
                "method": method,
                **metrics,
            }
        )
    return output


def phase_rows(
    seed: int,
    source_file: str,
    density_scenario: str,
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    order: list[str] = []
    for row in rows:
        phase = str(row["phase"])
        if phase not in grouped:
            order.append(phase)
        grouped[phase].append(row)
    output: list[dict[str, object]] = []
    for phase in order:
        values = grouped[phase]
        active = Counter()
        for row in values:
            active.update(
                source
                for source in str(row["active_anomaly_sources"]).split(";")
                if source
            )
        output.append(
            {
                "seed": seed,
                "hcrl_source_file": source_file,
                "density_scenario": density_scenario,
                "phase": phase,
                "ground_truth_attack": int(values[0]["ground_truth_attack"]),
                "rows": len(values),
                "mean_can_attack_frames": statistics.fmean(
                    int(row["hcrl_attack_frame_count"]) for row in values
                ),
                "mean_continuous_can_trust": statistics.fmean(
                    float(row["continuous_can_trust"]) for row in values
                ),
                **{
                    f"{method}_alarm_rate": statistics.fmean(
                        int(row[column]) for row in values
                    )
                    for method, column in METHOD_COLUMNS.items()
                },
                "can_active_rows": active["can"],
                "gnss_active_rows": active["gnss"],
                "v2x_active_rows": active["v2x"],
                "identity_active_rows": active["identity"],
                "device_active_rows": active["device_posture"],
                "sensor_control_active_rows": active["sensor_control"],
                "safe_fallback_rate": statistics.fmean(
                    row["local_control_decision"] == "SAFE_FALLBACK" for row in values
                ),
            }
        )
    return output


def latency_rows(
    seed: int,
    source_file: str,
    density_scenario: str,
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if int(row["ground_truth_attack"]):
            grouped[str(row["phase"])].append(row)
    output: list[dict[str, object]] = []
    for phase, values in grouped.items():
        for method, column in METHOD_COLUMNS.items():
            offsets = [index for index, row in enumerate(values) if int(row[column])]
            output.append(
                {
                    "seed": seed,
                    "hcrl_source_file": source_file,
                    "density_scenario": density_scenario,
                    "phase": phase,
                    "method": method,
                    "phase_rows": len(values),
                    "detected": int(bool(offsets)),
                    "first_detection_offset_windows": offsets[0] if offsets else -1,
                }
            )
    return output


def source_attribution_rows(
    seed: int,
    source_file: str,
    density_scenario: str,
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        truth_key = "attack_rows" if int(row["ground_truth_attack"]) else "benign_rows"
        for source in str(row["active_anomaly_sources"]).split(";"):
            if source:
                counts[source]["active_rows"] += 1
                counts[source][truth_key] += 1
    return [
        {
            "seed": seed,
            "hcrl_source_file": source_file,
            "density_scenario": density_scenario,
            "active_source": active_source,
            "active_rows": values["active_rows"],
            "attack_rows": values["attack_rows"],
            "benign_rows": values["benign_rows"],
        }
        for active_source, values in sorted(counts.items())
    ]


def decision_distribution_rows(
    seed: int,
    source_file: str,
    density_scenario: str,
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    counts = Counter(str(row["local_control_decision"]) for row in rows)
    return [
        {
            "seed": seed,
            "hcrl_source_file": source_file,
            "density_scenario": density_scenario,
            "local_control_decision": decision,
            "rows": count,
            "percentage": 100.0 * count / len(rows),
        }
        for decision, count in sorted(counts.items())
    ]


def aggregate_metric_rows(per_run: pd.DataFrame) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    metric_names = (
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
        "accuracy",
    )
    grouped = per_run.groupby(["density_scenario", "method"], sort=False)
    for (density, method), group in grouped:
        for metric in metric_names:
            values = group[metric].to_numpy(dtype=float)
            output.append(
                {
                    "density_scenario": density,
                    "method": method,
                    "metric": metric,
                    "runs": len(values),
                    "mean": float(np.mean(values)),
                    "population_std": float(np.std(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )
    return output


def density_phase_summary(phase_frame: pd.DataFrame) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for (density, phase), group in phase_frame.groupby(
        ["density_scenario", "phase"], sort=False
    ):
        row: dict[str, object] = {
            "density_scenario": density,
            "phase": phase,
            "runs": len(group),
            "ground_truth_attack": int(group["ground_truth_attack"].iloc[0]),
            "mean_can_attack_frames": float(group["mean_can_attack_frames"].mean()),
            "mean_continuous_can_trust": float(
                group["mean_continuous_can_trust"].mean()
            ),
        }
        for method in METHOD_COLUMNS:
            column = f"{method}_alarm_rate"
            row[f"{method}_recall_or_alarm_rate_mean"] = float(group[column].mean())
            row[f"{method}_recall_or_alarm_rate_std"] = float(group[column].std(ddof=0))
        row["safe_fallback_rate_mean"] = float(group["safe_fallback_rate"].mean())
        output.append(row)
    return output


def plot_summary(
    per_run: pd.DataFrame,
    phase_frame: pd.DataFrame,
    output_path: Path,
) -> None:
    labels = {
        "representative_all": "All densities",
        "low_1_5": "1-5",
        "medium_6_20": "6-20",
        "high_21_100": "21-100",
    }
    methods = (
        "guarded_can_only",
        "proposed_without_vehicle_state",
        "proposed_multisource",
    )
    styles = {
        "guarded_can_only": ("Guarded CAN only", "o"),
        "proposed_without_vehicle_state": ("Multi-source without vehicle state", "s"),
        "proposed_multisource": ("Proposed multi-source", "^"),
    }
    x = np.arange(len(DENSITY_ORDER))
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    can_phase = phase_frame[phase_frame["phase"] == "can_injection"]
    for method in methods:
        phase_values = []
        f1_values = []
        for density in DENSITY_ORDER:
            phase_group = can_phase[can_phase["density_scenario"] == density]
            metric_group = per_run[
                (per_run["density_scenario"] == density)
                & (per_run["method"] == method)
            ]
            phase_values.append(float(phase_group[f"{method}_alarm_rate"].mean()))
            f1_values.append(float(metric_group["f1"].mean()))
        label, marker = styles[method]
        axes[0].plot(x, phase_values, marker=marker, linewidth=2, label=label)
        axes[1].plot(x, f1_values, marker=marker, linewidth=2, label=label)
    for axis in axes:
        axis.set_xticks(x, [labels[item] for item in DENSITY_ORDER])
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.3)
        axis.set_xlabel("HCRL malicious frames per 100-frame CAN window")
    axes[0].set_title("CAN-injection phase detection")
    axes[0].set_ylabel("Mean alarm rate / recall")
    axes[1].set_title("End-to-end attack detection")
    axes[1].set_ylabel("Mean F1")
    axes[1].legend(loc="lower right")
    figure.suptitle("Guarded multi-source Zero Trust integration")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    results_dir = resolve_path(project_root, args.results_dir)
    can_prediction_path = (
        project_root
        / "results"
        / "session_normalized_can_gate_w100"
        / "session_gate_predictions.csv"
    )
    can_threshold_path = (
        project_root
        / "results"
        / "session_normalized_can_gate_w100"
        / "session_gate_thresholds.csv"
    )
    guard_runs_path = (
        project_root
        / "results"
        / "startup_poisoning_stress_w100"
        / "bootstrap_poisoning_runs.csv"
    )
    sumo_dir = project_root / "data" / "processed" / "sumo_repeated_seeds"
    sumo_paths = sorted(sumo_dir.glob("sumo_context_attacks_seed_*.csv"))
    if not sumo_paths:
        raise FileNotFoundError(
            f"No repeated-seed SUMO files found in {sumo_dir}; run Step 10 first"
        )

    trust_module = load_script(locate_script(project_root, "ztav_phase0.py"), "ztav_core_step19")
    can_predictions = pd.read_csv(can_prediction_path)
    validate_columns(can_predictions.columns, REQUIRED_CAN_COLUMNS, "Step 17 predictions")
    can_threshold = load_operational_can_threshold(can_threshold_path)
    guards = load_clean_startup_guard(guard_runs_path)
    sources = sorted(
        source
        for source in can_predictions.loc[
            can_predictions["source_file"] != "normal_run_data.txt", "source_file"
        ].unique()
    )
    missing_guards = set(sources) - set(guards)
    if missing_guards:
        raise ValueError(f"Step 18 has no clean startup guard rows for {sorted(missing_guards)}")

    per_run_metrics: list[dict[str, object]] = []
    per_phase: list[dict[str, object]] = []
    latencies: list[dict[str, object]] = []
    attributions: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    replay_manifest: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []

    total_runs = len(sumo_paths) * len(sources) * len(DENSITY_ORDER)
    run_number = 0
    for sumo_path in sumo_paths:
        seed = seed_from_filename(sumo_path)
        sumo_rows = read_csv_rows(sumo_path)
        validate_columns(sumo_rows[0], REQUIRED_SUMO_COLUMNS, sumo_path.name)
        for source_file in sources:
            for density_scenario in DENSITY_ORDER:
                run_number += 1
                print(
                    f"[{run_number}/{total_runs}] seed={seed}, source={source_file}, "
                    f"density={density_scenario}"
                )
                replay, replay_audit = build_replay_rows(
                    sumo_rows,
                    can_predictions,
                    source_file,
                    density_scenario,
                    seed,
                )
                evaluated = evaluate_replay(
                    replay,
                    can_threshold,
                    guards[source_file],
                    trust_module,
                )
                per_run_metrics.extend(
                    run_metric_rows(seed, source_file, density_scenario, evaluated)
                )
                per_phase.extend(
                    phase_rows(seed, source_file, density_scenario, evaluated)
                )
                latencies.extend(
                    latency_rows(seed, source_file, density_scenario, evaluated)
                )
                attributions.extend(
                    source_attribution_rows(
                        seed, source_file, density_scenario, evaluated
                    )
                )
                decisions.extend(
                    decision_distribution_rows(
                        seed, source_file, density_scenario, evaluated
                    )
                )
                replay_manifest.append(
                    {
                        **replay_audit,
                        "can_deviation_threshold": can_threshold,
                        "startup_guard_score": guards[source_file]["guard_score"],
                        "startup_guard_threshold": guards[source_file]["guard_threshold"],
                        "startup_guard_rejected": guards[source_file]["guard_rejected"],
                    }
                )
                for row in evaluated:
                    decision_rows.append(
                        {
                            "seed": seed,
                            "density_scenario": density_scenario,
                            **row,
                        }
                    )

    per_run_frame = pd.DataFrame(per_run_metrics)
    phase_frame = pd.DataFrame(per_phase)
    aggregate = aggregate_metric_rows(per_run_frame)
    density_phases = density_phase_summary(phase_frame)
    manifest_rows: list[dict[str, object]] = [
        {"item": "experiment_type", "value": "guarded multi-source hybrid replay"},
        {"item": "hcrl_sources", "value": ";".join(sources)},
        {"item": "sumo_seeds", "value": ";".join(str(seed_from_filename(path)) for path in sumo_paths)},
        {"item": "total_replay_runs", "value": total_runs},
        {"item": "can_operational_target_fpr", "value": 0.05},
        {"item": "can_deviation_threshold", "value": can_threshold},
        {"item": "can_first_anomaly_action", "value": "VERIFY_RESTRICT"},
        {"item": "can_two_consecutive_anomalies_action", "value": "SAFE_FALLBACK"},
        {"item": "startup_guard_action", "value": "reject enrollment and SAFE_FALLBACK"},
        {"item": "density_conditions", "value": ";".join(DENSITY_ORDER)},
        {
            "item": "sampling_rule",
            "value": "source-specific HCRL windows; replacement only when a density pool is smaller than the replay demand",
        },
        {
            "item": "label_usage",
            "value": "labels select controlled replay conditions and score outputs; detector reads evidence only",
        },
        {
            "item": "vehicle_state_ablation",
            "value": "proposed_without_vehicle_state removes sensor_control from the alarm but retains other sources",
        },
        {
            "item": "external_validity_limit",
            "value": "HCRL informed exploratory gate design; this is not an independent confirmatory test",
        },
        {
            "item": "simulation_limit",
            "value": "GNSS, V2X, identity, device posture, and vehicle-state context are SUMO/synthetic",
        },
    ]

    write_csv(results_dir / "integrated_per_run_metrics.csv", per_run_metrics)
    write_csv(results_dir / "integrated_aggregate_metrics.csv", aggregate)
    write_csv(results_dir / "integrated_per_phase_metrics.csv", per_phase)
    write_csv(results_dir / "integrated_density_phase_summary.csv", density_phases)
    write_csv(results_dir / "integrated_detection_latencies.csv", latencies)
    write_csv(results_dir / "integrated_source_attribution.csv", attributions)
    write_csv(results_dir / "integrated_decision_distribution.csv", decisions)
    write_csv(results_dir / "integrated_replay_audit.csv", replay_manifest)
    write_csv(results_dir / "integrated_policy_decisions.csv", decision_rows)
    write_csv(results_dir / "integrated_manifest.csv", manifest_rows)
    plot_summary(
        per_run_frame,
        phase_frame,
        results_dir / "integrated_density_sensitivity.png",
    )

    print("\n" + "=" * 80)
    print("Guarded multi-source Zero Trust integration completed successfully.")
    print(f"Replay runs: {total_runs} ({len(sources)} HCRL sources x {len(sumo_paths)} seeds x {len(DENSITY_ORDER)} densities)")
    print(f"Step 17 CAN deviation threshold: {can_threshold:.6f}")
    rejected_guards = sum(bool(guards[source]["guard_rejected"]) for source in sources)
    print(
        "Clean startup guard decisions: "
        f"accepted={len(sources) - rejected_guards}, rejected={rejected_guards}"
    )
    print("\nMean end-to-end F1 by density:")
    for density in DENSITY_ORDER:
        values = per_run_frame[per_run_frame["density_scenario"] == density]
        can_f1 = values.loc[values["method"] == "guarded_can_only", "f1"].mean()
        no_state_f1 = values.loc[
            values["method"] == "proposed_without_vehicle_state", "f1"
        ].mean()
        proposed_f1 = values.loc[
            values["method"] == "proposed_multisource", "f1"
        ].mean()
        print(
            f"  {density:<21} CAN-only={can_f1:.4f}, "
            f"without-vehicle-state={no_state_f1:.4f}, "
            f"proposed={proposed_f1:.4f}"
        )
    print(f"\nResults directory: {results_dir}")
    print("\nNext: inspect sparse-attack attribution, then freeze the integrated policy for confirmatory evaluation.")


if __name__ == "__main__":
    main()
