#!/usr/bin/env python3
"""Run controlled SUMO attacks against the context-aware Zero Trust engine.

This stage reuses the network builder in 07_build_sumo_context_testbed.py and
runs a deterministic 720-second experiment containing healthy, attack, and
recovery intervals.  It injects research-testbed representations of:

* GNSS spoofing,
* V2X falsification/Sybil messages,
* CAN command/behaviour injection,
* identity compromise, and
* a combined multi-source attack.

The script records ground truth, evidence scores, continuous trust, two policy
decisions (safety-critical actuator control and non-critical telemetry), and
row-level detection metrics.  These attacks are simulation abstractions, not
production exploit implementations.

Run from D:\\ztav_project:

    .\\.venv\\Scripts\\python.exe src\\08_run_sumo_attack_experiments.py \
        --rebuild-scenario

Optional visualization:

    .\\.venv\\Scripts\\python.exe src\\08_run_sumo_attack_experiments.py --gui

This is a research prototype, not production automotive safety software.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Iterable, Sequence


EGO_ID = "ego"
STEP_LENGTH_S = 1.0
V2X_RADIUS_M = 200.0
DEFAULT_STEPS = 720
ALARM_THRESHOLD = 0.75


# Half-open step intervals make every row belong to exactly one phase.
PHASES: tuple[tuple[int, int, str, bool], ...] = (
    (0, 120, "healthy_baseline", False),
    (120, 200, "gps_spoofing", True),
    (200, 240, "recovery_after_gps", False),
    (240, 320, "v2x_falsification", True),
    (320, 360, "recovery_after_v2x", False),
    (360, 440, "can_injection", True),
    (440, 480, "recovery_after_can", False),
    (480, 560, "identity_compromise", True),
    (560, 600, "recovery_after_identity", False),
    (600, 680, "combined_attack", True),
    (680, DEFAULT_STEPS, "final_recovery", False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled SUMO multi-source attack experiments."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root (default: current directory).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Simulation steps; must be at least {DEFAULT_STEPS}.",
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--rebuild-scenario", action="store_true")
    args = parser.parse_args()
    if args.steps < DEFAULT_STEPS:
        parser.error(f"--steps must be at least {DEFAULT_STEPS}")
    return args


def load_step07(project_root: Path) -> ModuleType:
    candidates = (
        project_root / "src" / "07_build_sumo_context_testbed.py",
        project_root / "07_build_sumo_context_testbed.py",
    )
    script = next((path for path in candidates if path.exists()), None)
    if script is None:
        raise FileNotFoundError(
            "Cannot find 07_build_sumo_context_testbed.py in the project src "
            "directory. Complete Step 07 first."
        )
    spec = importlib.util.spec_from_file_location("ztav_step07", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def phase_for_step(step: int) -> tuple[str, bool, int]:
    for start, end, name, is_attack in PHASES:
        if start <= step < end:
            return name, is_attack, step - start
    return "final_recovery", False, max(0, step - DEFAULT_STEPS)


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def attack_parameters(phase: str, offset: int) -> dict[str, float | int | bool]:
    """Return deterministic attack controls for one experiment phase."""

    parameters: dict[str, float | int | bool] = {
        "gnss_bias_x_m": 0.0,
        "gnss_bias_y_m": 0.0,
        "sybil_messages": 0,
        "v2x_false_offset_m": 0.0,
        "can_attack_probability": 0.02,
        "identity_failure": False,
        "device_failure": False,
    }
    if phase == "gps_spoofing":
        # A ramp avoids an unrealistic instantaneous position jump only.
        parameters["gnss_bias_x_m"] = 20.0 + 0.55 * offset
        parameters["gnss_bias_y_m"] = 12.0 + 0.20 * offset
    elif phase == "v2x_falsification":
        parameters["sybil_messages"] = 6
        parameters["v2x_false_offset_m"] = 65.0
    elif phase == "can_injection":
        parameters["can_attack_probability"] = 0.98
    elif phase == "identity_compromise":
        parameters["identity_failure"] = True
    elif phase == "combined_attack":
        parameters["gnss_bias_x_m"] = 55.0 + 0.35 * offset
        parameters["gnss_bias_y_m"] = -35.0
        parameters["sybil_messages"] = 8
        parameters["v2x_false_offset_m"] = 90.0
        parameters["can_attack_probability"] = 0.99
        parameters["device_failure"] = True
    return parameters


def run_experiment(
    project_root: Path,
    base: ModuleType,
    config_file: Path,
    output_file: Path,
    steps: int,
    seed: int,
    use_gui: bool,
) -> list[dict[str, object]]:
    try:
        import traci  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "TraCI is unavailable. Run: "
            ".\\.venv\\Scripts\\python.exe -m pip install traci sumolib"
        ) from exc

    (
        AccessRequest,
        ContinuousTrustEngine,
        Evidence,
        SafetyAwarePolicyEngine,
    ) = base.load_zero_trust_classes(project_root)

    trust_engine = ContinuousTrustEngine()
    policy_engine = SafetyAwarePolicyEngine()
    actuator_request = AccessRequest(
        subject="ego_vehicle_controller",
        resource="longitudinal_and_lateral_control",
        action="apply_actuator_command",
        safety_critical=True,
    )
    telemetry_request = AccessRequest(
        subject="ego_vehicle_controller",
        resource="vehicle_telemetry",
        action="read",
        safety_critical=False,
    )

    rng = random.Random(seed)
    sumo_binary = base.resolve_executable("sumo-gui" if use_gui else "sumo")
    command = [
        str(sumo_binary),
        "-c",
        str(config_file),
        "--seed",
        str(seed),
        "--no-warnings",
        "true",
    ]

    rows: list[dict[str, object]] = []
    previous_speed: float | None = None
    previous_heading: float | None = None
    ego_seen = False

    print(f"Starting {'sumo-gui' if use_gui else 'sumo'} attack experiment ...")
    traci.start(command)
    try:
        for step in range(steps):
            traci.simulationStep()
            vehicle_ids = list(traci.vehicle.getIDList())
            if EGO_ID not in vehicle_ids:
                if ego_seen:
                    print("Ego route completed before the requested step limit.")
                    break
                continue
            ego_seen = True

            phase, ground_truth_attack, phase_offset = phase_for_step(step)
            attack = attack_parameters(phase, phase_offset)
            sim_time = float(traci.simulation.getTime())
            position = tuple(map(float, traci.vehicle.getPosition(EGO_ID)))
            speed = float(traci.vehicle.getSpeed(EGO_ID))
            acceleration = float(traci.vehicle.getAcceleration(EGO_ID))
            heading = float(traci.vehicle.getAngle(EGO_ID))
            lane_id = str(traci.vehicle.getLaneID(EGO_ID))
            road_id = str(traci.vehicle.getRoadID(EGO_ID))

            derived_acceleration = (
                0.0
                if previous_speed is None
                else (speed - previous_speed) / STEP_LENGTH_S
            )
            heading_delta = (
                0.0
                if previous_heading is None
                else ((heading - previous_heading + 180.0) % 360.0) - 180.0
            )
            imu_acceleration = derived_acceleration + rng.gauss(0.0, 0.08)
            imu_yaw_rate = heading_delta / STEP_LENGTH_S + rng.gauss(0.0, 0.15)

            gnss_position = (
                position[0]
                + rng.gauss(0.0, 0.8)
                + float(attack["gnss_bias_x_m"]),
                position[1]
                + rng.gauss(0.0, 0.8)
                + float(attack["gnss_bias_y_m"]),
            )
            gnss_error = euclidean(position, gnss_position)

            neighbor_data: list[tuple[float, float, float]] = []
            v2x_position_errors: list[float] = []
            for vehicle_id in vehicle_ids:
                if vehicle_id == EGO_ID:
                    continue
                neighbor_position = tuple(
                    map(float, traci.vehicle.getPosition(vehicle_id))
                )
                distance = euclidean(position, neighbor_position)
                if distance > V2X_RADIUS_M:
                    continue
                neighbor_speed = float(traci.vehicle.getSpeed(vehicle_id))
                reported_position = (
                    neighbor_position[0] + rng.gauss(0.0, 0.6),
                    neighbor_position[1] + rng.gauss(0.0, 0.6),
                )
                report_error = euclidean(neighbor_position, reported_position)
                neighbor_data.append((distance, neighbor_speed, report_error))
                v2x_position_errors.append(report_error)

            sybil_messages = int(attack["sybil_messages"])
            false_offset = float(attack["v2x_false_offset_m"])
            for sybil_index in range(sybil_messages):
                angle = 2.0 * math.pi * sybil_index / max(1, sybil_messages)
                fake_error = false_offset + 5.0 * math.sin(angle)
                fake_distance = 15.0 + 8.0 * sybil_index
                fake_speed = 35.0 if sybil_index % 2 == 0 else 0.0
                neighbor_data.append((fake_distance, fake_speed, fake_error))
                v2x_position_errors.append(abs(fake_error))

            nearest_distance = (
                min(item[0] for item in neighbor_data) if neighbor_data else -1.0
            )
            mean_neighbor_speed = mean_or_zero(item[1] for item in neighbor_data)
            v2x_rmse = (
                math.sqrt(mean_or_zero(error * error for error in v2x_position_errors))
                if v2x_position_errors
                else 0.0
            )

            can_attack_probability = float(attack["can_attack_probability"])
            identity_failure = bool(attack["identity_failure"])
            device_failure = bool(attack["device_failure"])
            acceleration_error = abs(acceleration - derived_acceleration)
            gnss_imu_score = clamp(
                math.exp(-gnss_error / 7.0)
                * math.exp(-abs(imu_acceleration - acceleration) / 3.0)
            )
            v2x_score = (
                clamp(math.exp(-v2x_rmse / 6.0)) if neighbor_data else 0.85
            )
            v2x_quality = 1.0 if neighbor_data else 0.45
            sensor_control_score = clamp(math.exp(-acceleration_error / 2.5))
            if phase in {"can_injection", "combined_attack"}:
                sensor_control_score = min(sensor_control_score, 0.20)
            can_behavior_score = clamp(1.0 - can_attack_probability)

            evidence = {
                "identity": Evidence(
                    0.02 if identity_failure else 0.99,
                    quality=1.0,
                    critical_failure=identity_failure,
                ),
                "device_posture": Evidence(
                    0.08 if device_failure else 0.97,
                    quality=1.0,
                    critical_failure=device_failure,
                ),
                "can_behavior": Evidence(can_behavior_score, quality=0.90),
                "gnss_imu_consistency": Evidence(gnss_imu_score, quality=1.0),
                "v2x_consistency": Evidence(v2x_score, quality=v2x_quality),
                "sensor_control_consistency": Evidence(
                    sensor_control_score, quality=1.0
                ),
                "freshness": Evidence(1.0, quality=1.0),
            }
            trust = trust_engine.evaluate(evidence)
            actuator_policy = policy_engine.decide(actuator_request, trust)
            telemetry_policy = policy_engine.decide(telemetry_request, trust)
            security_alarm = bool(
                trust.continuous_trust < ALARM_THRESHOLD or trust.critical_failures
            )

            rows.append(
                {
                    "simulation_time_s": round(sim_time, 3),
                    "phase": phase,
                    "ground_truth_attack": int(ground_truth_attack),
                    "security_alarm": int(security_alarm),
                    "ego_id": EGO_ID,
                    "x_m": round(position[0], 4),
                    "y_m": round(position[1], 4),
                    "speed_mps": round(speed, 4),
                    "acceleration_mps2": round(acceleration, 4),
                    "heading_deg": round(heading, 4),
                    "lane_id": lane_id,
                    "road_id": road_id,
                    "gnss_x_m": round(gnss_position[0], 4),
                    "gnss_y_m": round(gnss_position[1], 4),
                    "gnss_error_m": round(gnss_error, 4),
                    "gnss_bias_x_m": round(float(attack["gnss_bias_x_m"]), 4),
                    "gnss_bias_y_m": round(float(attack["gnss_bias_y_m"]), 4),
                    "imu_longitudinal_accel_mps2": round(imu_acceleration, 4),
                    "imu_yaw_rate_dps": round(imu_yaw_rate, 4),
                    "real_v2x_neighbor_count": len(neighbor_data) - sybil_messages,
                    "reported_v2x_neighbor_count": len(neighbor_data),
                    "sybil_message_count": sybil_messages,
                    "v2x_nearest_distance_m": round(nearest_distance, 4),
                    "v2x_mean_neighbor_speed_mps": round(mean_neighbor_speed, 4),
                    "v2x_position_rmse_m": round(v2x_rmse, 4),
                    "can_attack_probability": round(can_attack_probability, 6),
                    "identity_critical_failure": int(identity_failure),
                    "device_critical_failure": int(device_failure),
                    "identity_score": round(evidence["identity"].score, 6),
                    "device_posture_score": round(
                        evidence["device_posture"].score, 6
                    ),
                    "can_behavior_score": round(can_behavior_score, 6),
                    "gnss_imu_consistency_score": round(gnss_imu_score, 6),
                    "v2x_consistency_score": round(v2x_score, 6),
                    "sensor_control_consistency_score": round(
                        sensor_control_score, 6
                    ),
                    "freshness_score": 1.0,
                    "instantaneous_trust": round(trust.instantaneous_trust, 6),
                    "continuous_trust": round(trust.continuous_trust, 6),
                    "risk": round(trust.risk, 6),
                    "actuator_decision": actuator_policy.decision.value,
                    "telemetry_decision": telemetry_policy.decision.value,
                }
            )

            previous_speed = speed
            previous_heading = heading
            if (step + 1) % 80 == 0:
                print(
                    f"  step={step + 1}, phase={phase}, "
                    f"trust={trust.continuous_trust:.3f}, "
                    f"alarm={int(security_alarm)}, "
                    f"actuator={actuator_policy.decision.value}"
                )
    finally:
        traci.close()

    if not ego_seen or not rows:
        raise RuntimeError("Attack experiment completed without ego context rows")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def confusion_counts(rows: Sequence[dict[str, object]]) -> dict[str, int]:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for row in rows:
        truth = bool(int(row["ground_truth_attack"]))
        alarm = bool(int(row["security_alarm"]))
        if truth and alarm:
            counts["tp"] += 1
        elif not truth and alarm:
            counts["fp"] += 1
        elif not truth and not alarm:
            counts["tn"] += 1
        else:
            counts["fn"] += 1
    return counts


def write_phase_summary(path: Path, rows: Sequence[dict[str, object]]) -> None:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["phase"])].append(row)

    fieldnames = [
        "phase",
        "ground_truth_attack",
        "rows",
        "mean_continuous_trust",
        "minimum_continuous_trust",
        "maximum_risk",
        "alarm_rate",
        "actuator_safe_fallback_rate",
        "telemetry_allow",
        "telemetry_restrict",
        "telemetry_deny",
    ]
    phase_rows: list[dict[str, object]] = []
    phase_order = [item[2] for item in PHASES]
    for phase in phase_order:
        values = grouped.get(phase, [])
        if not values:
            continue
        n = len(values)
        telemetry = Counter(str(row["telemetry_decision"]) for row in values)
        phase_rows.append(
            {
                "phase": phase,
                "ground_truth_attack": values[0]["ground_truth_attack"],
                "rows": n,
                "mean_continuous_trust": round(
                    statistics.fmean(float(row["continuous_trust"]) for row in values),
                    6,
                ),
                "minimum_continuous_trust": round(
                    min(float(row["continuous_trust"]) for row in values), 6
                ),
                "maximum_risk": round(max(float(row["risk"]) for row in values), 6),
                "alarm_rate": round(
                    safe_divide(
                        sum(int(row["security_alarm"]) for row in values), n
                    ),
                    6,
                ),
                "actuator_safe_fallback_rate": round(
                    safe_divide(
                        sum(
                            str(row["actuator_decision"]) == "SAFE_FALLBACK"
                            for row in values
                        ),
                        n,
                    ),
                    6,
                ),
                "telemetry_allow": telemetry.get("ALLOW", 0),
                "telemetry_restrict": telemetry.get("RESTRICT", 0),
                "telemetry_deny": telemetry.get("DENY", 0),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(phase_rows)


def episode_latencies(rows: Sequence[dict[str, object]]) -> list[tuple[str, str]]:
    latencies: list[tuple[str, str]] = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if int(row["ground_truth_attack"]):
            grouped[str(row["phase"])].append(row)
    for phase in [item[2] for item in PHASES if item[3]]:
        values = grouped.get(phase, [])
        if not values:
            continue
        start = float(values[0]["simulation_time_s"])
        first_alarm = next(
            (
                float(row["simulation_time_s"])
                for row in values
                if int(row["security_alarm"])
            ),
            None,
        )
        latency = "not_detected" if first_alarm is None else f"{first_alarm - start:.3f}"
        latencies.append((f"detection_latency_{phase}_s", latency))
    return latencies


def write_detection_metrics(path: Path, rows: Sequence[dict[str, object]]) -> None:
    counts = confusion_counts(rows)
    precision = safe_divide(counts["tp"], counts["tp"] + counts["fp"])
    recall = safe_divide(counts["tp"], counts["tp"] + counts["fn"])
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    false_positive_rate = safe_divide(
        counts["fp"], counts["fp"] + counts["tn"]
    )
    accuracy = safe_divide(counts["tp"] + counts["tn"], len(rows))
    metrics: list[tuple[str, object]] = [
        ("alarm_threshold", ALARM_THRESHOLD),
        ("rows", len(rows)),
        ("true_positive", counts["tp"]),
        ("false_positive", counts["fp"]),
        ("true_negative", counts["tn"]),
        ("false_negative", counts["fn"]),
        ("precision", f"{precision:.6f}"),
        ("recall", f"{recall:.6f}"),
        ("f1", f"{f1:.6f}"),
        ("false_positive_rate", f"{false_positive_rate:.6f}"),
        ("accuracy", f"{accuracy:.6f}"),
    ]
    metrics.extend(episode_latencies(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    if not (project_root / "ztav_phase0.py").exists():
        raise FileNotFoundError(
            f"Expected {project_root / 'ztav_phase0.py'}. "
            "Run from D:\\ztav_project."
        )

    base = load_step07(project_root)
    scenario_dir = project_root / "sumo" / "scenario_attacks"
    output_file = project_root / "data" / "processed" / "sumo_context_attacks.csv"
    results_dir = project_root / "results" / "sumo_attack_experiments"
    phase_summary_file = results_dir / "attack_phase_summary.csv"
    detection_metrics_file = results_dir / "detection_metrics.csv"

    config_file = base.build_scenario(
        scenario_dir,
        steps=args.steps,
        seed=args.seed,
        rebuild=args.rebuild_scenario,
    )
    rows = run_experiment(
        project_root,
        base,
        config_file,
        output_file,
        steps=args.steps,
        seed=args.seed,
        use_gui=args.gui,
    )
    write_phase_summary(phase_summary_file, rows)
    write_detection_metrics(detection_metrics_file, rows)

    counts = confusion_counts(rows)
    precision = safe_divide(counts["tp"], counts["tp"] + counts["fp"])
    recall = safe_divide(counts["tp"], counts["tp"] + counts["fn"])
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    print("\n" + "=" * 76)
    print("SUMO controlled attack experiment completed successfully.")
    print(f"Rows: {len(rows):,}")
    print(f"Confusion counts: {counts}")
    print(f"Alarm precision={precision:.4f}, recall={recall:.4f}, F1={f1:.4f}")
    print(f"Context dataset: {output_file}")
    print(f"Phase summary: {phase_summary_file}")
    print(f"Detection metrics: {detection_metrics_file}")
    print("\nNext: inspect phase-level trust, latency, and recovery behaviour.")


if __name__ == "__main__":
    main()
