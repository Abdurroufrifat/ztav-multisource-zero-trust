#!/usr/bin/env python3
"""Evaluate source-aware and action-aware Zero Trust policy offline.

Step 08 showed why a single weighted trust threshold is insufficient: a severe
anomaly in a lower-weight source (especially GNSS or V2X) may be diluted by
healthy sources.  This stage adds independent, persistent source monitors and
maps active anomalies to action-specific policy responses.

Important experimental rule: ``phase`` and ``ground_truth_attack`` are used
only after each decision for evaluation.  The detector reads evidence columns,
not attack labels.

This is a research prototype, not production automotive safety software.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_INPUT = Path("data/processed/sumo_context_attacks.csv")
DEFAULT_RESULTS = Path("results/context_aware_zero_trust")


# Fixed engineering thresholds: they are declared before evaluating test
# labels and are not optimized on the attack phase outcomes.
SOURCE_THRESHOLDS: dict[str, tuple[str, float]] = {
    "gnss": ("gnss_imu_consistency_score", 0.30),
    "v2x": ("v2x_consistency_score", 0.30),
    "can": ("can_behavior_score", 0.30),
    "sensor_control": ("sensor_control_consistency_score", 0.35),
}
ATTACK_PERSISTENCE = 2
RECOVERY_PERSISTENCE = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate source-aware, action-aware Zero Trust policy."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


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

    def update(self, critical_failure: bool) -> bool:
        if critical_failure:
            self.active = True
            self.good_streak = 0
        elif self.active:
            self.good_streak += 1
            if self.good_streak >= self.recovery_persistence:
                self.active = False
                self.good_streak = 0
        return self.active


def resolve_path(project_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else project_root / value


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find attack dataset: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")

    required = {
        "simulation_time_s",
        "phase",
        "ground_truth_attack",
        "security_alarm",
        "identity_critical_failure",
        "device_critical_failure",
        *(column for column, _ in SOURCE_THRESHOLDS.values()),
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return rows


def policy_for_sources(active_sources: set[str]) -> tuple[str, str, str]:
    """Return local-control, cooperative-action, and telemetry decisions."""

    if not active_sources:
        return "ALLOW", "ALLOW", "ALLOW"

    critical_local = active_sources & {
        "identity",
        "device_posture",
        "can",
        "sensor_control",
        "gnss",
    }
    if critical_local:
        local_control = "SAFE_FALLBACK"
    else:
        # A V2X-only failure does not justify disabling trusted local sensing
        # and control; it removes the external source from the trust boundary.
        local_control = "ALLOW_LOCAL_ONLY"

    cooperative_action = (
        "DENY_COOPERATIVE_ACTION"
        if "v2x" in active_sources
        else "REQUIRE_REVERIFICATION"
    )
    telemetry = (
        "DENY"
        if active_sources & {"identity", "device_posture"}
        else "RESTRICT"
    )
    return local_control, cooperative_action, telemetry


def evaluate(rows: Sequence[dict[str, str]]) -> list[dict[str, object]]:
    score_monitors = {
        source: PersistentMonitor(threshold)
        for source, (_, threshold) in SOURCE_THRESHOLDS.items()
    }
    identity_monitor = BooleanCriticalMonitor()
    device_monitor = BooleanCriticalMonitor()
    evaluated: list[dict[str, object]] = []

    for row in rows:
        active_sources: set[str] = set()
        for source, (column, _) in SOURCE_THRESHOLDS.items():
            if score_monitors[source].update(float(row[column])):
                active_sources.add(source)
        if identity_monitor.update(bool(int(row["identity_critical_failure"]))):
            active_sources.add("identity")
        if device_monitor.update(bool(int(row["device_critical_failure"]))):
            active_sources.add("device_posture")

        local_control, cooperative_action, telemetry = policy_for_sources(
            active_sources
        )
        evaluated.append(
            {
                "simulation_time_s": float(row["simulation_time_s"]),
                "phase": row["phase"],
                "ground_truth_attack": int(row["ground_truth_attack"]),
                "weighted_threshold_alarm": int(row["security_alarm"]),
                "context_aware_alarm": int(bool(active_sources)),
                "active_anomaly_sources": ";".join(sorted(active_sources)),
                "local_control_decision": local_control,
                "cooperative_action_decision": cooperative_action,
                "telemetry_decision": telemetry,
            }
        )
    return evaluated


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def metrics_for_alarm(
    rows: Sequence[dict[str, object]], alarm_column: str
) -> dict[str, float | int]:
    tp = fp = tn = fn = 0
    for row in rows:
        truth = bool(int(row["ground_truth_attack"]))
        alarm = bool(int(row[alarm_column]))
        if truth and alarm:
            tp += 1
        elif not truth and alarm:
            fp += 1
        elif not truth and not alarm:
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
        "accuracy": safe_divide(tp + tn, len(rows)),
    }


def write_evaluated_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_comparison(path: Path, rows: Sequence[dict[str, object]]) -> None:
    methods = (
        ("weighted_global_threshold", "weighted_threshold_alarm"),
        ("proposed_context_aware", "context_aware_alarm"),
    )
    output: list[dict[str, object]] = []
    for method, alarm_column in methods:
        metrics = metrics_for_alarm(rows, alarm_column)
        output.append(
            {
                "method": method,
                **{
                    key: round(value, 6) if isinstance(value, float) else value
                    for key, value in metrics.items()
                },
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


def write_phase_summary(path: Path, rows: Sequence[dict[str, object]]) -> None:
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
        active_source_counts: Counter[str] = Counter()
        for row in values:
            for source in str(row["active_anomaly_sources"]).split(";"):
                if source:
                    active_source_counts[source] += 1
        local_decisions = Counter(
            str(row["local_control_decision"]) for row in values
        )
        output.append(
            {
                "phase": phase,
                "ground_truth_attack": values[0]["ground_truth_attack"],
                "rows": len(values),
                "weighted_alarm_rate": round(
                    safe_divide(
                        sum(int(row["weighted_threshold_alarm"]) for row in values),
                        len(values),
                    ),
                    6,
                ),
                "context_aware_alarm_rate": round(
                    safe_divide(
                        sum(int(row["context_aware_alarm"]) for row in values),
                        len(values),
                    ),
                    6,
                ),
                "local_allow": local_decisions.get("ALLOW", 0),
                "local_only": local_decisions.get("ALLOW_LOCAL_ONLY", 0),
                "safe_fallback": local_decisions.get("SAFE_FALLBACK", 0),
                "gnss_active_rows": active_source_counts.get("gnss", 0),
                "v2x_active_rows": active_source_counts.get("v2x", 0),
                "can_active_rows": active_source_counts.get("can", 0),
                "identity_active_rows": active_source_counts.get("identity", 0),
                "device_active_rows": active_source_counts.get("device_posture", 0),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


def write_latencies(path: Path, rows: Sequence[dict[str, object]]) -> None:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    order: list[str] = []
    for row in rows:
        if not int(row["ground_truth_attack"]):
            continue
        phase = str(row["phase"])
        if phase not in grouped:
            order.append(phase)
        grouped[phase].append(row)

    output: list[dict[str, object]] = []
    for phase in order:
        values = grouped[phase]
        start = float(values[0]["simulation_time_s"])
        result: dict[str, object] = {"phase": phase}
        for method, alarm_column in (
            ("weighted", "weighted_threshold_alarm"),
            ("context_aware", "context_aware_alarm"),
        ):
            first = next(
                (
                    float(row["simulation_time_s"])
                    for row in values
                    if int(row[alarm_column])
                ),
                None,
            )
            result[f"{method}_latency_s"] = (
                "not_detected" if first is None else round(first - start, 3)
            )
        output.append(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


def write_thresholds(path: Path) -> None:
    rows = [
        {
            "source": source,
            "evidence_column": column,
            "anomaly_below": threshold,
            "attack_persistence_rows": ATTACK_PERSISTENCE,
            "recovery_persistence_rows": RECOVERY_PERSISTENCE,
        }
        for source, (column, threshold) in SOURCE_THRESHOLDS.items()
    ]
    rows.extend(
        {
            "source": source,
            "evidence_column": column,
            "anomaly_below": "critical_failure_true",
            "attack_persistence_rows": 1,
            "recovery_persistence_rows": RECOVERY_PERSISTENCE,
        }
        for source, column in (
            ("identity", "identity_critical_failure"),
            ("device_posture", "device_critical_failure"),
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    input_path = resolve_path(project_root, args.input)
    results_dir = resolve_path(project_root, args.results_dir)
    source_rows = read_rows(input_path)
    evaluated = evaluate(source_rows)

    evaluated_path = results_dir / "context_aware_row_decisions.csv"
    comparison_path = results_dir / "baseline_vs_context_aware.csv"
    phase_path = results_dir / "context_aware_phase_summary.csv"
    latency_path = results_dir / "detection_latency_comparison.csv"
    thresholds_path = results_dir / "source_thresholds.csv"
    write_evaluated_rows(evaluated_path, evaluated)
    write_comparison(comparison_path, evaluated)
    write_phase_summary(phase_path, evaluated)
    write_latencies(latency_path, evaluated)
    write_thresholds(thresholds_path)

    baseline = metrics_for_alarm(evaluated, "weighted_threshold_alarm")
    proposed = metrics_for_alarm(evaluated, "context_aware_alarm")
    print("\n" + "=" * 76)
    print("Context-aware Zero Trust evaluation completed successfully.")
    print(
        "Weighted threshold: "
        f"precision={baseline['precision']:.4f}, "
        f"recall={baseline['recall']:.4f}, F1={baseline['f1']:.4f}"
    )
    print(
        "Context-aware:     "
        f"precision={proposed['precision']:.4f}, "
        f"recall={proposed['recall']:.4f}, F1={proposed['f1']:.4f}"
    )
    print(f"Results directory: {results_dir}")
    print("\nNext: repeat across seeds and perform threshold sensitivity analysis.")


if __name__ == "__main__":
    main()
