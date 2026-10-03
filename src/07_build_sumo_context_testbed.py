#!/usr/bin/env python3
"""Build and run the healthy SUMO multi-source context baseline.

This stage creates a reproducible microscopic traffic scenario and connects it
to the Phase-0 continuous Zero Trust engine.  At every simulation step it logs:

* SUMO vehicle dynamics (position, speed, acceleration, heading and lane),
* noisy GNSS and derived IMU observations,
* nearby-vehicle/V2X context,
* synthetic identity and device-attestation evidence,
* a motion-plausibility CAN-behaviour baseline, and
* instantaneous/continuous trust plus the safety-aware policy decision.

Identity, device posture and CAN evidence are explicit research-testbed
baselines; they are not measurements from production automotive hardware.  A
later stage will inject attacks and connect the trained CAN classifier.

This is a research prototype, not production automotive safety software.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


EGO_ID = "ego"
SCENARIO_NAME = "healthy_baseline"
STEP_LENGTH_S = 1.0
V2X_RADIUS_M = 120.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SUMO scenario and collect healthy multi-source context."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing ztav_phase0.py (default: current directory).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=600,
        help="Maximum simulation steps (default: 600).",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--gui", action="store_true", help="Run sumo-gui instead of command-line SUMO."
    )
    parser.add_argument(
        "--rebuild-scenario",
        action="store_true",
        help="Regenerate the network and traffic routes even if they exist.",
    )
    args = parser.parse_args()
    if args.steps < 10:
        parser.error("--steps must be at least 10")
    return args


def resolve_executable(name: str) -> Path:
    """Locate a SUMO executable from PATH or SUMO_HOME."""

    executable = f"{name}.exe" if os.name == "nt" else name
    from_path = shutil.which(executable) or shutil.which(name)
    if from_path:
        return Path(from_path)

    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidate = Path(sumo_home) / "bin" / executable
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot find {executable}. Add SUMO_HOME\\bin to PATH or set SUMO_HOME."
    )


def run_checked(command: Sequence[str], description: str) -> None:
    print(f"{description} ...")
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"{description} failed:\n{details}")


def find_directed_cycle(net_file: Path, max_edges: int = 12) -> list[str]:
    """Find a short directed cycle in a generated SUMO network."""

    root = ET.parse(net_file).getroot()
    adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        from_node = edge.get("from")
        to_node = edge.get("to")
        if not edge_id or not from_node or not to_node or edge_id.startswith(":"):
            continue
        if edge.get("function") in {"internal", "crossing", "walkingarea"}:
            continue
        adjacency[from_node].append((to_node, edge_id))

    def search(
        start: str,
        node: str,
        visited: set[str],
        path_edges: list[str],
    ) -> list[str] | None:
        if len(path_edges) >= max_edges:
            return None
        for next_node, edge_id in adjacency[node]:
            if next_node == start and len(path_edges) >= 3:
                return path_edges + [edge_id]
            if next_node in visited:
                continue
            result = search(
                start,
                next_node,
                visited | {next_node},
                path_edges + [edge_id],
            )
            if result:
                return result
        return None

    for start_node in sorted(adjacency):
        cycle = search(start_node, start_node, {start_node}, [])
        if cycle:
            return cycle
    raise RuntimeError("Could not find a directed cycle in the SUMO network")


def write_ego_routes(path: Path, cycle: Sequence[str], repeats: int = 35) -> None:
    route = " ".join(list(cycle) * repeats)
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="ego_type" accel="2.6" decel="4.5" sigma="0.0"
           length="4.8" minGap="2.5" maxSpeed="13.89"/>
    <vehicle id="{EGO_ID}" type="ego_type" depart="0"
             departLane="best" departSpeed="max">
        <route edges="{route}"/>
    </vehicle>
</routes>
"""
    path.write_text(content, encoding="utf-8")


def write_sumocfg(path: Path, end_time: int) -> None:
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="grid.net.xml"/>
        <route-files value="ego.rou.xml,traffic.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{end_time}"/>
        <step-length value="{STEP_LENGTH_S}"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
    </report>
</configuration>
"""
    path.write_text(content, encoding="utf-8")


def build_scenario(
    scenario_dir: Path,
    steps: int,
    seed: int,
    rebuild: bool,
) -> Path:
    scenario_dir.mkdir(parents=True, exist_ok=True)
    net_file = scenario_dir / "grid.net.xml"
    traffic_file = scenario_dir / "traffic.rou.xml"
    ego_file = scenario_dir / "ego.rou.xml"
    config_file = scenario_dir / "ztav_baseline.sumocfg"

    required = (net_file, traffic_file, ego_file, config_file)
    if not rebuild and all(path.exists() for path in required):
        print(f"Reusing scenario: {config_file}")
        return config_file

    netgenerate = resolve_executable("netgenerate")
    run_checked(
        [
            str(netgenerate),
            "--grid",
            "--grid.number",
            "6",
            "--grid.length",
            "200",
            "--default.speed",
            "13.89",
            "--output-file",
            str(net_file),
        ],
        "Generating a 6x6 SUMO road network",
    )

    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise EnvironmentError("SUMO_HOME is not set")
    random_trips = Path(sumo_home) / "tools" / "randomTrips.py"
    if not random_trips.exists():
        raise FileNotFoundError(f"Cannot find SUMO randomTrips.py at {random_trips}")

    trips_file = scenario_dir / "traffic.trips.xml"
    run_checked(
        [
            sys.executable,
            str(random_trips),
            "-n",
            str(net_file),
            "-o",
            str(trips_file),
            "-r",
            str(traffic_file),
            "-b",
            "0",
            "-e",
            str(steps),
            "-p",
            "3.0",
            "--prefix",
            "traffic_",
            "--seed",
            str(seed),
            "--validate",
        ],
        "Generating reproducible surrounding traffic",
    )

    cycle = find_directed_cycle(net_file)
    write_ego_routes(ego_file, cycle)
    write_sumocfg(config_file, steps)
    print(f"Ego cycle ({len(cycle)} edges): {' '.join(cycle)}")
    print(f"Saved SUMO scenario: {scenario_dir}")
    return config_file


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def load_zero_trust_classes(project_root: Path):
    sys.path.insert(0, str(project_root))
    try:
        from ztav_phase0 import (  # type: ignore
            AccessRequest,
            ContinuousTrustEngine,
            Evidence,
            SafetyAwarePolicyEngine,
        )
    except ImportError as exc:
        raise ImportError(
            f"Could not import {project_root / 'ztav_phase0.py'}. "
            "Run this script from D:\\ztav_project."
        ) from exc
    return AccessRequest, ContinuousTrustEngine, Evidence, SafetyAwarePolicyEngine


def run_simulation(
    project_root: Path,
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
    ) = load_zero_trust_classes(project_root)

    trust_engine = ContinuousTrustEngine()
    policy_engine = SafetyAwarePolicyEngine()
    request = AccessRequest(
        subject="ego_vehicle_controller",
        resource="longitudinal_and_lateral_control",
        action="maintain_planned_route",
        safety_critical=True,
    )
    rng = random.Random(seed)
    sumo_binary = resolve_executable("sumo-gui" if use_gui else "sumo")
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

    print(f"Starting {'sumo-gui' if use_gui else 'sumo'} ...")
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
                position[0] + rng.gauss(0.0, 0.8),
                position[1] + rng.gauss(0.0, 0.8),
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

            nearest_distance = (
                min(item[0] for item in neighbor_data) if neighbor_data else -1.0
            )
            mean_neighbor_speed = mean_or_zero(item[1] for item in neighbor_data)
            v2x_rmse = (
                math.sqrt(mean_or_zero(error * error for error in v2x_position_errors))
                if v2x_position_errors
                else 0.0
            )

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
            can_behavior_score = clamp(
                0.98
                - 0.03 * max(0.0, abs(acceleration) - 4.5)
                - 0.02 * max(0.0, speed - 14.0)
            )

            evidence = {
                "identity": Evidence(0.99, quality=1.0),
                "device_posture": Evidence(0.97, quality=1.0),
                "can_behavior": Evidence(can_behavior_score, quality=0.75),
                "gnss_imu_consistency": Evidence(gnss_imu_score, quality=1.0),
                "v2x_consistency": Evidence(v2x_score, quality=v2x_quality),
                "sensor_control_consistency": Evidence(
                    sensor_control_score, quality=1.0
                ),
                "freshness": Evidence(1.0, quality=1.0),
            }
            trust = trust_engine.evaluate(evidence)
            policy = policy_engine.decide(request, trust)

            rows.append(
                {
                    "simulation_time_s": round(sim_time, 3),
                    "scenario": SCENARIO_NAME,
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
                    "imu_longitudinal_accel_mps2": round(imu_acceleration, 4),
                    "imu_yaw_rate_dps": round(imu_yaw_rate, 4),
                    "v2x_neighbor_count": len(neighbor_data),
                    "v2x_nearest_distance_m": round(nearest_distance, 4),
                    "v2x_mean_neighbor_speed_mps": round(mean_neighbor_speed, 4),
                    "v2x_position_rmse_m": round(v2x_rmse, 4),
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
                    "decision": policy.decision.value,
                }
            )

            previous_speed = speed
            previous_heading = heading
            if (step + 1) % 100 == 0:
                print(
                    f"  step={step + 1}, rows={len(rows)}, "
                    f"trust={trust.continuous_trust:.3f}, "
                    f"decision={policy.decision.value}"
                )
    finally:
        traci.close()

    if not ego_seen:
        raise RuntimeError("The ego vehicle never entered the simulation")
    if not rows:
        raise RuntimeError("Simulation completed without context rows")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_summary(path: Path, rows: Sequence[dict[str, object]]) -> None:
    decisions = Counter(str(row["decision"]) for row in rows)
    trust_values = [float(row["continuous_trust"]) for row in rows]
    risk_values = [float(row["risk"]) for row in rows]
    neighbor_counts = [int(row["v2x_neighbor_count"]) for row in rows]
    summary = [
        ("scenario", SCENARIO_NAME),
        ("rows", len(rows)),
        ("start_time_s", rows[0]["simulation_time_s"]),
        ("end_time_s", rows[-1]["simulation_time_s"]),
        ("mean_continuous_trust", f"{statistics.fmean(trust_values):.6f}"),
        ("minimum_continuous_trust", f"{min(trust_values):.6f}"),
        ("maximum_risk", f"{max(risk_values):.6f}"),
        ("mean_v2x_neighbor_count", f"{statistics.fmean(neighbor_counts):.3f}"),
    ]
    summary.extend((f"decision_{name}", count) for name, count in sorted(decisions.items()))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(summary)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    phase0 = project_root / "ztav_phase0.py"
    if not phase0.exists():
        raise FileNotFoundError(
            f"Expected {phase0}. Run this script from the project root."
        )

    scenario_dir = project_root / "sumo" / "scenario_baseline"
    output_file = project_root / "data" / "processed" / "sumo_context_baseline.csv"
    summary_file = (
        project_root / "results" / "sumo_context_baseline" / "baseline_summary.csv"
    )

    config_file = build_scenario(
        scenario_dir,
        steps=args.steps,
        seed=args.seed,
        rebuild=args.rebuild_scenario,
    )
    rows = run_simulation(
        project_root,
        config_file,
        output_file,
        steps=args.steps,
        seed=args.seed,
        use_gui=args.gui,
    )
    write_summary(summary_file, rows)

    decisions = Counter(str(row["decision"]) for row in rows)
    trust_values = [float(row["continuous_trust"]) for row in rows]
    print("\n" + "=" * 72)
    print("SUMO healthy-context baseline completed successfully.")
    print(f"Rows: {len(rows):,}")
    print(f"Mean continuous trust: {statistics.fmean(trust_values):.4f}")
    print(f"Minimum continuous trust: {min(trust_values):.4f}")
    print(f"Policy decisions: {dict(decisions)}")
    print(f"Context dataset: {output_file}")
    print(f"Summary: {summary_file}")
    print("\nNext: inspect this baseline before adding attack injection.")


if __name__ == "__main__":
    main()
