#!/usr/bin/env python3
r"""Benchmark frozen ZTAV inference cost on the executing research computer.

This additive publication stage never trains, recalibrates, edits, or freezes
any security component. It measures inference over already-engineered feature
and evidence rows. Raw CAN parsing, feature extraction, sensor acquisition,
network transport, and embedded-ECU execution are outside scope.

Run from D:\ztav_project after Steps 30A--30C:

    .\.venv\Scripts\python.exe .\src\30D_publication_efficiency_benchmark.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import psutil
except ImportError:  # optional instrumentation
    psutil = None  # type: ignore[assignment]


POLICY_COLUMNS = (
    "multiscale_alarm_instant",
    "multiscale_alarm_persistent_2",
    "startup_quality_warning",
    "active_noncan_sources",
)
CRITICAL_LOCAL_SOURCES = {"identity", "device_posture", "gnss"}
DEFAULT_BATCH_SIZES = (1, 32, 256, 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publication runtime/resource benchmark for frozen ZTAV components."
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--sample-rows", type=int, default=10_000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--batch-repetitions", type=int, default=12)
    parser.add_argument("--cold-repetitions", type=int, default=5)
    parser.add_argument("--skip-cold-start", action="store_true")
    args = parser.parse_args()
    for name in (
        "sample_rows",
        "warmup",
        "iterations",
        "batch_repetitions",
        "cold_repetitions",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    return args


def discover_project_root(explicit: Path | None) -> Path:
    if explicit is not None:
        root = explicit.resolve()
        if not root.exists():
            raise FileNotFoundError(f"Project root does not exist: {root}")
        return root
    script = Path(__file__).resolve()
    for candidate in (script.parent.parent, Path.cwd(), script.parent):
        if (candidate / "models").is_dir() and (candidate / "results").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot identify a project root containing models/ and results/. "
        "Run from D:\\ztav_project or supply --project-root."
    )


def locate_unique(root: Path, preferred: str, filename: str) -> Path:
    preferred_path = root / preferred
    if preferred_path.is_file():
        return preferred_path
    matches = sorted(
        path
        for path in root.rglob(filename)
        if "publication_efficiency_benchmark" not in path.parts
        and "publication_readiness_audit" not in path.parts
    )
    if not matches:
        raise FileNotFoundError(f"Cannot find required artifact: {filename}")
    if len(matches) > 1:
        rendered = "\n  ".join(str(path) for path in matches)
        raise RuntimeError(
            f"Multiple candidates found for {filename}; keep the preferred path "
            f"or remove ambiguity:\n  {rendered}"
        )
    return matches[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


class MemoryMonitor:
    """Sample process RSS and Python allocations during a benchmark segment."""

    def __init__(self) -> None:
        self.process = psutil.Process() if psutil is not None else None
        self.rss_start = math.nan
        self.rss_peak = math.nan
        self.python_peak = math.nan
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        assert self.process is not None
        while not self._stop.wait(0.002):
            try:
                self.rss_peak = max(
                    self.rss_peak, float(self.process.memory_info().rss)
                )
            except Exception:
                return

    def __enter__(self) -> "MemoryMonitor":
        if self.process is not None:
            self.rss_start = float(self.process.memory_info().rss)
            self.rss_peak = self.rss_start
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        tracemalloc.start()
        return self

    def __exit__(self, *_args: object) -> None:
        _current, peak = tracemalloc.get_traced_memory()
        self.python_peak = float(peak)
        tracemalloc.stop()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.process is not None:
            try:
                self.rss_peak = max(
                    self.rss_peak, float(self.process.memory_info().rss)
                )
            except Exception:
                pass


def memory_fields(memory: MemoryMonitor) -> dict[str, float]:
    return {
        "python_allocation_peak_mb": memory.python_peak / (1024**2),
        "process_rss_start_mb": memory.rss_start / (1024**2),
        "process_rss_peak_mb": memory.rss_peak / (1024**2),
        "process_rss_increment_mb": (memory.rss_peak - memory.rss_start)
        / (1024**2),
    }


def benchmark_online(
    method: str,
    component: str,
    window_size: int,
    function: Callable[[Any], Any],
    samples: Sequence[Any],
    warmup: int,
    iterations: int,
    input_bytes: int,
    scope: str,
) -> dict[str, Any]:
    if not samples:
        raise ValueError(f"No samples supplied for {method}")
    for index in range(warmup):
        function(samples[index % len(samples)])
    wall_ms: list[float] = []
    cpu_ms: list[float] = []
    with MemoryMonitor() as memory:
        total_wall_start = time.perf_counter_ns()
        total_cpu_start = time.process_time_ns()
        for index in range(iterations):
            sample = samples[index % len(samples)]
            wall_start = time.perf_counter_ns()
            cpu_start = time.process_time_ns()
            function(sample)
            cpu_ms.append((time.process_time_ns() - cpu_start) / 1e6)
            wall_ms.append((time.perf_counter_ns() - wall_start) / 1e6)
        total_cpu_s = (time.process_time_ns() - total_cpu_start) / 1e9
        total_wall_s = (time.perf_counter_ns() - total_wall_start) / 1e9
    return {
        "method": method,
        "component": component,
        "window_size_frames": window_size,
        "execution_mode": "online_single_item",
        "batch_size": 1,
        "warmup_iterations": warmup,
        "measurement_repetitions": iterations,
        "measured_items": iterations,
        "wall_latency_median_ms": statistics.median(wall_ms),
        "wall_latency_p95_ms": percentile(wall_ms, 95),
        "wall_latency_p99_ms": percentile(wall_ms, 99),
        "cpu_latency_median_ms": statistics.median(cpu_ms),
        "throughput_items_per_second": iterations / max(total_wall_s, 1e-12),
        "process_cpu_core_equivalent_percent": 100.0
        * total_cpu_s
        / max(total_wall_s, 1e-12),
        **memory_fields(memory),
        "input_buffer_bytes": input_bytes,
        "measurement_scope": scope,
    }


def benchmark_model_batches(
    model: Any,
    x: np.ndarray,
    repetitions: int,
    batch_sizes: Iterable[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for requested in batch_sizes:
        batch_size = min(int(requested), len(x))
        batch = np.ascontiguousarray(x[:batch_size])
        model.predict_proba(batch)
        wall_per_item_ms: list[float] = []
        cpu_per_item_ms: list[float] = []
        total_wall = total_cpu = 0.0
        with MemoryMonitor() as memory:
            for _ in range(repetitions):
                wall_start = time.perf_counter_ns()
                cpu_start = time.process_time_ns()
                model.predict_proba(batch)
                cpu_elapsed = (time.process_time_ns() - cpu_start) / 1e9
                wall_elapsed = (time.perf_counter_ns() - wall_start) / 1e9
                total_wall += wall_elapsed
                total_cpu += cpu_elapsed
                wall_per_item_ms.append(1000.0 * wall_elapsed / batch_size)
                cpu_per_item_ms.append(1000.0 * cpu_elapsed / batch_size)
        items = repetitions * batch_size
        rows.append(
            {
                "method": "frozen_w100_logistic_regression",
                "component": "CAN classifier",
                "window_size_frames": 100,
                "execution_mode": "batch_inference",
                "batch_size": batch_size,
                "warmup_iterations": 1,
                "measurement_repetitions": repetitions,
                "measured_items": items,
                "wall_latency_median_ms": statistics.median(wall_per_item_ms),
                "wall_latency_p95_ms": percentile(wall_per_item_ms, 95),
                "wall_latency_p99_ms": percentile(wall_per_item_ms, 99),
                "cpu_latency_median_ms": statistics.median(cpu_per_item_ms),
                "throughput_items_per_second": items / max(total_wall, 1e-12),
                "process_cpu_core_equivalent_percent": 100.0
                * total_cpu
                / max(total_wall, 1e-12),
                **memory_fields(memory),
                "input_buffer_bytes": int(batch.nbytes),
                "measurement_scope": (
                    "predict_proba on engineered w100 features; "
                    "parsing/feature extraction excluded"
                ),
            }
        )
    return rows


def split_sources(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    return {item for item in str(value).split(";") if item}


def graded_policy_core(
    sample: tuple[int, int, int, object],
) -> tuple[str, str, str]:
    instant, persistent, warning, source_text = sample
    sources = split_sources(source_text) - {"sensor_control"}
    critical = bool(sources & CRITICAL_LOCAL_SOURCES)
    v2x_active = "v2x" in sources
    if persistent or critical:
        local = "SAFE_FALLBACK"
    elif instant:
        local = "MONITOR_VERIFY"
    elif warning:
        local = "DEGRADED_MONITORING"
    elif v2x_active:
        local = "ALLOW_LOCAL_ONLY"
    else:
        local = "ALLOW"
    cooperative = (
        "DENY_COOPERATIVE_ACTION"
        if v2x_active
        else "REQUIRE_REVERIFICATION"
        if warning
        else "ALLOW"
    )
    telemetry = (
        "DENY"
        if bool(sources & {"identity", "device_posture"})
        else "RESTRICT"
        if instant or warning
        else "ALLOW"
    )
    return local, cooperative, telemetry


def micro_gate_core(
    sample: tuple[float, bool], threshold: float
) -> tuple[bool, bool, float]:
    deviation, previous_alarm = sample
    instant = bool(deviation >= threshold)
    persistent = bool(instant and previous_alarm)
    trust = 1.0 / (1.0 + (deviation / max(threshold, 1e-12)) ** 2)
    return instant, persistent, trust


def load_inputs(root: Path, sample_rows: int) -> dict[str, Any]:
    paths = {
        "model": locate_unique(
            root,
            "models/group_disjoint_w100/group_disjoint_logistic_regression.joblib",
            "group_disjoint_logistic_regression.joblib",
        ),
        "test": locate_unique(
            root,
            "data/processed/ciciov2024_windows_w100_group_disjoint_test.csv",
            "ciciov2024_windows_w100_group_disjoint_test.csv",
        ),
        "thresholds": locate_unique(
            root,
            "results/group_disjoint_w100/group_disjoint_thresholds.json",
            "group_disjoint_thresholds.json",
        ),
        "features": locate_unique(
            root,
            "results/hybrid_ciciov_sumo/can_feature_columns.json",
            "can_feature_columns.json",
        ),
        "micro_predictions": locate_unique(
            root,
            "results/multiscale_sparse_can_gate/w20_micro_gate_predictions.csv",
            "w20_micro_gate_predictions.csv",
        ),
        "micro_calibration": locate_unique(
            root,
            "results/multiscale_sparse_can_gate/micro_calibration_summary.csv",
            "micro_calibration_summary.csv",
        ),
        "policy": locate_unique(
            root,
            "results/graded_zero_trust_policy/graded_policy_decisions.csv",
            "graded_policy_decisions.csv",
        ),
    }
    hashes_before = {name: sha256_file(path) for name, path in paths.items()}
    model = joblib.load(paths["model"])

    feature_names = json.loads(paths["features"].read_text(encoding="utf-8"))
    if not isinstance(feature_names, list) or not all(
        isinstance(item, str) for item in feature_names
    ):
        raise ValueError("can_feature_columns.json must contain a list of names")
    header = pd.read_csv(paths["test"], nrows=0)
    missing_features = set(feature_names) - set(header.columns)
    if missing_features:
        raise ValueError(
            f"Test CSV is missing model features: {sorted(missing_features)}"
        )
    features = pd.read_csv(
        paths["test"], usecols=feature_names, nrows=sample_rows
    )[feature_names]
    x = np.ascontiguousarray(features.to_numpy(dtype=np.float64))
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Model benchmark features are empty or non-finite")
    expected = getattr(model, "n_features_in_", x.shape[1])
    if int(expected) != x.shape[1]:
        raise ValueError(
            f"Model expects {expected} features but input has {x.shape[1]}"
        )
    thresholds = json.loads(paths["thresholds"].read_text(encoding="utf-8"))
    model_threshold = float(thresholds["Logistic Regression"])

    micro = pd.read_csv(
        paths["micro_predictions"],
        usecols=lambda name: name
        in {"micro_session_id", "micro_structural_deviation"},
        nrows=sample_rows,
    )
    required_micro = {"micro_session_id", "micro_structural_deviation"}
    missing_micro = required_micro - set(micro.columns)
    if missing_micro:
        raise ValueError(
            f"Micro predictions are missing: {sorted(missing_micro)}"
        )
    calibration = pd.read_csv(paths["micro_calibration"], nrows=1)
    micro_threshold = float(
        calibration.iloc[0]["micro_deviation_threshold"]
    )
    previous = (
        micro.groupby("micro_session_id", sort=False)[
            "micro_structural_deviation"
        ]
        .shift(1)
        .fillna(-math.inf)
    )
    micro_samples = list(
        zip(
            micro["micro_structural_deviation"].astype(float),
            (previous.astype(float) >= micro_threshold),
        )
    )

    policy = pd.read_csv(
        paths["policy"], usecols=list(POLICY_COLUMNS), nrows=sample_rows
    )
    missing_policy = set(POLICY_COLUMNS) - set(policy.columns)
    if missing_policy:
        raise ValueError(
            f"Policy decisions are missing: {sorted(missing_policy)}"
        )
    policy_samples = list(
        zip(
            policy["multiscale_alarm_instant"].astype(int),
            policy["multiscale_alarm_persistent_2"].astype(int),
            policy["startup_quality_warning"].astype(int),
            policy["active_noncan_sources"],
        )
    )
    return {
        "paths": paths,
        "hashes_before": hashes_before,
        "model": model,
        "x": x,
        "feature_names": feature_names,
        "features_memory": int(
            features.memory_usage(index=True, deep=True).sum()
        ),
        "model_threshold": model_threshold,
        "micro_threshold": micro_threshold,
        "micro_samples": micro_samples,
        "micro_memory": int(micro.memory_usage(index=True, deep=True).sum()),
        "policy_samples": policy_samples,
        "policy_memory": int(policy.memory_usage(index=True, deep=True).sum()),
    }


CHILD_RUNNER = r'''import json, sys, time
start = time.perf_counter_ns()
import joblib
import numpy as np
import_done = time.perf_counter_ns()
load_start = time.perf_counter_ns()
model = joblib.load(sys.argv[1])
load_done = time.perf_counter_ns()
x = np.load(sys.argv[2])
infer_start = time.perf_counter_ns()
model.predict_proba(x)
end = time.perf_counter_ns()
rss = None
try:
    import psutil
    rss = psutil.Process().memory_info().rss
except Exception:
    pass
print(json.dumps({
    "child_import_ms": (import_done-start)/1e6,
    "child_model_load_ms": (load_done-load_start)/1e6,
    "child_first_inference_ms": (end-infer_start)/1e6,
    "child_internal_total_ms": (end-start)/1e6,
    "child_process_rss_mb": None if rss is None else rss/(1024**2),
}))
'''


def cold_start_benchmark(
    out: Path, model_path: Path, sample: np.ndarray, repetitions: int
) -> list[dict[str, Any]]:
    runner = out / "cold_start_runner.py"
    sample_path = out / "cold_start_sample.npy"
    runner.write_text(CHILD_RUNNER, encoding="utf-8")
    np.save(sample_path, np.ascontiguousarray(sample.reshape(1, -1)))
    rows: list[dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        start = time.perf_counter_ns()
        completed = subprocess.run(
            [sys.executable, str(runner), str(model_path), str(sample_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        parent_ms = (time.perf_counter_ns() - start) / 1e6
        values = json.loads(completed.stdout.strip().splitlines()[-1])
        rows.append(
            {
                "repetition": repetition,
                "fresh_process_total_ms": parent_ms,
                **values,
                "cache_state": (
                    "new Python process; operating-system filesystem cache "
                    "not flushed"
                ),
            }
        )
    return rows


def hardware_manifest() -> dict[str, Any]:
    cpu_name = (
        os.environ.get("PROCESSOR_IDENTIFIER")
        or platform.processor()
        or platform.uname().processor
        or "not reported by operating system"
    )
    result: dict[str, Any] = {
        "operating_system": platform.platform(),
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "machine": platform.machine(),
        "processor": cpu_name,
        "logical_cpu_count": os.cpu_count(),
        "psutil_available": psutil is not None,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "joblib_version": joblib.__version__,
    }
    if psutil is not None:
        result.update(
            {
                "physical_cpu_count": psutil.cpu_count(logical=False),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
                "available_memory_gb_at_start": (
                    psutil.virtual_memory().available / (1024**3)
                ),
            }
        )
    return result


def plot_summary(runtime: pd.DataFrame, output: Path) -> None:
    online = runtime[
        runtime["execution_mode"] == "online_single_item"
    ].copy()
    batch = runtime[
        (runtime["execution_mode"] == "batch_inference")
        & (runtime["batch_size"] > 1)
    ].copy()
    figure, axes = plt.subplots(
        1, 2, figsize=(15, 5.6), constrained_layout=True
    )
    x = np.arange(len(online))
    width = 0.25
    for offset, column, label in (
        (-width, "wall_latency_median_ms", "Median"),
        (0.0, "wall_latency_p95_ms", "P95"),
        (width, "wall_latency_p99_ms", "P99"),
    ):
        axes[0].bar(x + offset, online[column], width, label=label)
    axes[0].set_xticks(x, online["method"], rotation=20, ha="right")
    axes[0].set_ylabel("Wall latency (ms / item)")
    axes[0].set_title("Warm online inference latency")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    if len(batch):
        axes[1].plot(
            batch["batch_size"],
            batch["throughput_items_per_second"],
            marker="o",
        )
        axes[1].set_xscale("log", base=2)
        axes[1].set_xticks(
            batch["batch_size"], batch["batch_size"].astype(int)
        )
    axes[1].set_xlabel("Batch size")
    axes[1].set_ylabel("100-frame windows / second")
    axes[1].set_title("Frozen CAN-model batch throughput")
    axes[1].grid(alpha=0.25)
    figure.suptitle("Step 30D publication efficiency benchmark")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    root = discover_project_root(args.project_root)
    run_id = datetime.now(timezone.utc).strftime(
        "run_%Y%m%dT%H%M%S_%fZ"
    )
    out = (
        root
        / "results"
        / "publication_efficiency_benchmark"
        / run_id
    )
    out.mkdir(parents=True, exist_ok=False)

    print("Loading frozen model and recorded evidence ...")
    inputs = load_inputs(root, args.sample_rows)
    model = inputs["model"]
    x: np.ndarray = inputs["x"]
    micro_samples = inputs["micro_samples"]
    policy_samples = inputs["policy_samples"]
    model_samples = [x[index : index + 1] for index in range(len(x))]

    model_function = lambda sample: model.predict_proba(sample)
    micro_function = lambda sample: micro_gate_core(
        sample, inputs["micro_threshold"]
    )
    combined_samples = [
        (
            model_samples[index % len(model_samples)],
            policy_samples[index % len(policy_samples)],
        )
        for index in range(min(len(model_samples), len(policy_samples)))
    ]

    def combined_function(
        sample: tuple[np.ndarray, tuple[int, int, int, object]]
    ) -> Any:
        feature_row, policy_row = sample
        probability = float(model.predict_proba(feature_row)[0, 1])
        model_alarm = int(probability >= inputs["model_threshold"])
        instant, persistent, warning, sources = policy_row
        return graded_policy_core(
            (
                int(bool(instant) or bool(model_alarm)),
                int(bool(persistent) or bool(model_alarm)),
                warning,
                sources,
            )
        )

    runtime_rows: list[dict[str, Any]] = []
    print("[1/4] Benchmarking frozen w100 model online inference ...")
    runtime_rows.append(
        benchmark_online(
            "frozen_w100_logistic_regression",
            "CAN classifier",
            100,
            model_function,
            model_samples,
            args.warmup,
            args.iterations,
            int(x[0:1].nbytes),
            (
                "predict_proba on one engineered w100 row; "
                "raw parsing/feature extraction excluded"
            ),
        )
    )
    print("[2/4] Benchmarking w20 temporal evidence rule ...")
    runtime_rows.append(
        benchmark_online(
            "w20_temporal_micro_gate",
            "CAN temporal rule",
            20,
            micro_function,
            micro_samples,
            args.warmup,
            args.iterations,
            16,
            (
                "threshold, consecutive-memory and trust computation on "
                "recorded deviation; feature extraction excluded"
            ),
        )
    )
    print("[3/4] Benchmarking graded multi-source policy core ...")
    runtime_rows.append(
        benchmark_online(
            "graded_multisource_policy_core",
            "Zero Trust policy decision",
            100,
            graded_policy_core,
            policy_samples,
            args.warmup,
            args.iterations,
            max(
                1,
                int(inputs["policy_memory"] / max(len(policy_samples), 1)),
            ),
            (
                "graded local/cooperative/telemetry actions from recorded "
                "source evidence"
            ),
        )
    )
    print("[4/4] Benchmarking combined model-plus-policy path ...")
    runtime_rows.append(
        benchmark_online(
            "w100_model_plus_graded_policy_runtime_path",
            "CAN classifier and Zero Trust policy",
            100,
            combined_function,
            combined_samples,
            args.warmup,
            args.iterations,
            int(x[0:1].nbytes)
            + max(
                1,
                int(inputs["policy_memory"] / max(len(policy_samples), 1)),
            ),
            (
                "runtime-only pairing of real feature/evidence rows; "
                "not an efficacy endpoint"
            ),
        )
    )
    runtime_rows.extend(
        benchmark_model_batches(
            model, x, args.batch_repetitions, DEFAULT_BATCH_SIZES
        )
    )

    cold_rows: list[dict[str, Any]] = []
    if not args.skip_cold_start:
        print("Measuring fresh-process cold start ...")
        cold_rows = cold_start_benchmark(
            out,
            inputs["paths"]["model"],
            x[0],
            args.cold_repetitions,
        )
        cold_wall = [
            float(row["fresh_process_total_ms"]) for row in cold_rows
        ]
        cold_rss = [
            float(row["child_process_rss_mb"])
            for row in cold_rows
            if row["child_process_rss_mb"] is not None
        ]
        runtime_rows.append(
            {
                "method": "frozen_w100_fresh_process_startup",
                "component": (
                    "Python import, model load and first CAN inference"
                ),
                "window_size_frames": 100,
                "execution_mode": "fresh_process_cold_start",
                "batch_size": 1,
                "warmup_iterations": 0,
                "measurement_repetitions": len(cold_rows),
                "measured_items": len(cold_rows),
                "wall_latency_median_ms": statistics.median(cold_wall),
                "wall_latency_p95_ms": percentile(cold_wall, 95),
                "wall_latency_p99_ms": percentile(cold_wall, 99),
                "cpu_latency_median_ms": math.nan,
                "throughput_items_per_second": math.nan,
                "process_cpu_core_equivalent_percent": math.nan,
                "python_allocation_peak_mb": math.nan,
                "process_rss_start_mb": math.nan,
                "process_rss_peak_mb": (
                    max(cold_rss) if cold_rss else math.nan
                ),
                "process_rss_increment_mb": math.nan,
                "input_buffer_bytes": int(x[0:1].nbytes),
                "measurement_scope": (
                    "new Python process; operating-system filesystem cache "
                    "not flushed"
                ),
            }
        )

    hashes_after = {
        name: sha256_file(path)
        for name, path in inputs["paths"].items()
    }
    changed = [
        name
        for name in inputs["hashes_before"]
        if inputs["hashes_before"][name] != hashes_after[name]
    ]
    if changed:
        raise RuntimeError(
            f"Input artifacts changed during benchmark: {changed}"
        )

    hardware = hardware_manifest()
    resource_rows = [
        {
            "resource_type": "serialized_model",
            "resource_name": "group_disjoint_logistic_regression",
            "size_bytes": inputs["paths"]["model"].stat().st_size,
            "size_mb": (
                inputs["paths"]["model"].stat().st_size / (1024**2)
            ),
            "rows": 1,
            "columns": len(inputs["feature_names"]),
            "window_size_frames": 100,
            "notes": "Frozen joblib model file",
        },
        {
            "resource_type": "evidence_buffer",
            "resource_name": "w100_engineered_feature_sample",
            "size_bytes": inputs["features_memory"],
            "size_mb": inputs["features_memory"] / (1024**2),
            "rows": len(x),
            "columns": x.shape[1],
            "window_size_frames": 100,
            "notes": "Pandas deep-memory size of benchmark sample",
        },
        {
            "resource_type": "evidence_buffer",
            "resource_name": "w20_temporal_evidence_sample",
            "size_bytes": inputs["micro_memory"],
            "size_mb": inputs["micro_memory"] / (1024**2),
            "rows": len(micro_samples),
            "columns": 2,
            "window_size_frames": 20,
            "notes": "Recorded session ID and deviation evidence",
        },
        {
            "resource_type": "evidence_buffer",
            "resource_name": "graded_policy_context_sample",
            "size_bytes": inputs["policy_memory"],
            "size_mb": inputs["policy_memory"] / (1024**2),
            "rows": len(policy_samples),
            "columns": len(POLICY_COLUMNS),
            "window_size_frames": 100,
            "notes": "Recorded multi-source policy evidence",
        },
        {
            "resource_type": "research_hardware",
            "resource_name": "executing_computer",
            "size_bytes": math.nan,
            "size_mb": math.nan,
            "rows": math.nan,
            "columns": math.nan,
            "window_size_frames": math.nan,
            "notes": json.dumps(hardware, sort_keys=True),
        },
    ]

    runtime = pd.DataFrame(runtime_rows)
    resources = pd.DataFrame(resource_rows)
    runtime.to_csv(
        out / "publication_runtime_metrics.csv", index=False
    )
    resources.to_csv(
        out / "publication_resource_summary.csv", index=False
    )
    if cold_rows:
        pd.DataFrame(cold_rows).to_csv(
            out / "publication_cold_start_metrics.csv", index=False
        )
    runtime[
        runtime["execution_mode"] == "batch_inference"
    ].to_csv(out / "publication_batch_throughput.csv", index=False)
    plot_summary(
        runtime, out / "publication_efficiency_summary.png"
    )

    manifest = {
        "stage": "30D",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "existing_project_artifacts_changed": 0,
        "input_paths": {
            name: str(path) for name, path in inputs["paths"].items()
        },
        "input_sha256": hashes_after,
        "hardware": hardware,
        "settings": {
            "sample_rows_requested": args.sample_rows,
            "w100_rows_loaded": len(x),
            "w20_rows_loaded": len(micro_samples),
            "policy_rows_loaded": len(policy_samples),
            "warmup": args.warmup,
            "online_iterations": args.iterations,
            "batch_repetitions": args.batch_repetitions,
            "cold_repetitions": (
                0 if args.skip_cold_start else args.cold_repetitions
            ),
            "batch_sizes": list(DEFAULT_BATCH_SIZES),
        },
        "scope": {
            "included": (
                "inference and policy decision over engineered/recorded "
                "evidence"
            ),
            "excluded": [
                "raw CAN parsing",
                "CAN feature extraction",
                "sensor acquisition",
                "network transport",
                "SUMO simulation runtime",
                "embedded ECU deployment",
            ],
            "cold_start": (
                "fresh Python process with operating-system filesystem "
                "cache unflushed"
            ),
            "combined_path": (
                "runtime-only pairing; not an efficacy endpoint"
            ),
        },
        "security_readiness": {
            "step30c_h4_supported": False,
            "step30c_h5_supported": False,
            "step31_permitted": False,
            "reason": (
                "Efficiency evidence cannot repair failed "
                "source-robustness hypotheses."
            ),
        },
    }
    (out / "publication_efficiency_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (out / "README.md").write_text(
        "# Step 30D publication efficiency benchmark\n\n"
        "These measurements apply to inference over already-engineered "
        "feature/evidence rows, not raw CAN ingestion or embedded automotive "
        "deployment. The combined path is a timing exercise only and is not "
        "an accuracy endpoint. Inputs were hashed before and after execution "
        "and were unchanged. Step 30C robustness failures remain in force, "
        "so Step 31 is still blocked.\n",
        encoding="utf-8",
    )
    archive_base = (
        out.parent / f"publication_efficiency_benchmark_{run_id}"
    )
    archive_path = Path(
        shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=out.parent,
            base_dir=out.name,
        )
    )

    online = runtime[
        runtime["execution_mode"] == "online_single_item"
    ]
    print("\n" + "=" * 78)
    print(
        "Step 30D publication efficiency benchmark completed successfully."
    )
    print("Existing project artifacts changed: 0")
    print(f"Research hardware: {hardware['processor']}")
    print(f"Online measurements per method: {args.iterations:,}")
    print("\nWarm online median / p95 / p99 latency (ms per item):")
    for row in online.itertuples(index=False):
        print(
            f"  {row.method}: {row.wall_latency_median_ms:.6f} / "
            f"{row.wall_latency_p95_ms:.6f} / "
            f"{row.wall_latency_p99_ms:.6f}"
        )
    print(f"\nResults directory: {out}")
    print(f"Results archive: {archive_path}")
    print("Step 30C H4/H5 failures remain unchanged.")
    print("Do not run Step 31 yet.")


if __name__ == "__main__":
    main()
