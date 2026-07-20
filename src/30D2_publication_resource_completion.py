#!/usr/bin/env python3
r"""Complete Step 30D with long-run CPU and native process-memory metrics.

This additive audit preserves the original Step 30D run. It reuses the frozen
model/evidence inputs, extends each workload to a stable measurement interval,
and records process working-set memory without requiring psutil on Windows.

Run from D:\ztav_project after Step 30D:

    .\.venv\Scripts\python.exe .\src\30D2_publication_resource_completion.py

No detector is trained or recalibrated. Raw CAN parsing, feature engineering,
sensor/network I/O, SUMO runtime, and embedded-ECU execution remain excluded.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
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
from types import ModuleType
from typing import Any, Callable, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KEY_COLUMNS = ["method", "execution_mode", "batch_size"]
DEFAULT_BATCH_SIZES = (1, 32, 256, 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete Step 30D CPU and process-memory evidence."
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--sample-rows", type=int, default=10_000)
    parser.add_argument("--probe-seconds", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--cold-repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.sample_rows < 1:
        parser.error("--sample-rows must be at least 1")
    if args.probe_seconds < 1.0:
        parser.error("--probe-seconds must be at least 1.0")
    if args.warmup < 1:
        parser.error("--warmup must be at least 1")
    if args.cold_repetitions < 1:
        parser.error("--cold-repetitions must be at least 1")
    return args


def discover_root(explicit: Path | None) -> Path:
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
        "Cannot find project root. Run from D:\\ztav_project or use --project-root."
    )


def locate_step30d(root: Path) -> Path:
    for path in (
        root / "src" / "30D_publication_efficiency_benchmark.py",
        root / "30D_publication_efficiency_benchmark.py",
    ):
        if path.is_file():
            return path
    raise FileNotFoundError("Cannot find 30D_publication_efficiency_benchmark.py")


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("ztav_step30d", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Step 30D: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def latest_step30d_run(root: Path) -> Path:
    base = root / "results" / "publication_efficiency_benchmark"
    candidates = sorted(
        path
        for path in base.glob("run_*")
        if (path / "publication_runtime_metrics.csv").is_file()
        and (path / "publication_resource_summary.csv").is_file()
        and (path / "publication_efficiency_manifest.json").is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            "No completed Step 30D run found under "
            "results/publication_efficiency_benchmark/"
        )
    return candidates[-1]


class WindowsMemoryReader:
    """Read current/private process memory through the Windows PSAPI."""

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    def __init__(self) -> None:
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        self.handle = self.kernel32.GetCurrentProcess()
        self.psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self.Counters),
            ctypes.c_ulong,
        ]
        self.psapi.GetProcessMemoryInfo.restype = ctypes.c_int

    def read(self) -> tuple[int, int, int]:
        counters = self.Counters()
        counters.cb = ctypes.sizeof(counters)
        ok = self.psapi.GetProcessMemoryInfo(
            self.handle, ctypes.byref(counters), counters.cb
        )
        if not ok:
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return (
            int(counters.WorkingSetSize),
            int(counters.PrivateUsage),
            int(counters.PeakWorkingSetSize),
        )


class PsutilMemoryReader:
    def __init__(self, psutil_module: Any) -> None:
        self.process = psutil_module.Process()

    def read(self) -> tuple[int, int, int]:
        info = self.process.memory_info()
        rss = int(info.rss)
        private = int(getattr(info, "private", rss))
        peak = int(getattr(info, "peak_wset", rss))
        return rss, private, peak


def total_system_memory_gb(step30d: ModuleType) -> float:
    if platform.system() == "Windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError(
                ctypes.get_last_error(), "GlobalMemoryStatusEx failed"
            )
        return float(status.ullTotalPhys) / (1024**3)
    if getattr(step30d, "psutil", None) is not None:
        return float(step30d.psutil.virtual_memory().total) / (1024**3)
    return math.nan


class ResourceMonitor:
    def __init__(self, step30d: ModuleType) -> None:
        if platform.system() == "Windows":
            self.reader: Any = WindowsMemoryReader()
            self.backend = "Windows PSAPI GetProcessMemoryInfo"
        elif getattr(step30d, "psutil", None) is not None:
            self.reader = PsutilMemoryReader(step30d.psutil)
            self.backend = "psutil Process.memory_info"
        else:
            self.reader = None
            self.backend = "Python tracemalloc only"
        self.rss_start = self.rss_peak = math.nan
        self.private_start = self.private_peak = math.nan
        self.lifetime_peak = math.nan
        self.python_peak = math.nan
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.wait(0.002):
            try:
                rss, private, lifetime_peak = self.reader.read()
                self.rss_peak = max(self.rss_peak, float(rss))
                self.private_peak = max(self.private_peak, float(private))
                self.lifetime_peak = max(
                    self.lifetime_peak, float(lifetime_peak)
                )
            except Exception:
                return

    def __enter__(self) -> "ResourceMonitor":
        if self.reader is not None:
            rss, private, lifetime_peak = self.reader.read()
            self.rss_start = self.rss_peak = float(rss)
            self.private_start = self.private_peak = float(private)
            self.lifetime_peak = float(lifetime_peak)
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
        if self.reader is not None:
            try:
                rss, private, lifetime_peak = self.reader.read()
                self.rss_peak = max(self.rss_peak, float(rss))
                self.private_peak = max(self.private_peak, float(private))
                self.lifetime_peak = max(
                    self.lifetime_peak, float(lifetime_peak)
                )
            except Exception:
                pass


def long_probe(
    method: str,
    execution_mode: str,
    batch_size: int,
    function: Callable[[Any], Any],
    samples: Sequence[Any],
    target_seconds: float,
    warmup: int,
    step30d: ModuleType,
    scope: str,
) -> dict[str, Any]:
    if not samples:
        raise ValueError(f"No samples for {method}")
    for index in range(warmup):
        function(samples[index % len(samples)])
    monitor = ResourceMonitor(step30d)
    calls = 0
    block = 100
    with monitor:
        wall_start = time.perf_counter_ns()
        cpu_start = time.process_time_ns()
        while (time.perf_counter_ns() - wall_start) / 1e9 < target_seconds:
            for _ in range(block):
                function(samples[calls % len(samples)])
                calls += 1
        cpu_seconds = (time.process_time_ns() - cpu_start) / 1e9
        wall_seconds = (time.perf_counter_ns() - wall_start) / 1e9
    items = calls * batch_size
    mb = 1024**2
    return {
        "method": method,
        "execution_mode": execution_mode,
        "batch_size": batch_size,
        "long_run_calls": calls,
        "long_run_items": items,
        "long_run_wall_seconds": wall_seconds,
        "long_run_cpu_seconds": cpu_seconds,
        "long_run_wall_time_per_item_ms": 1000.0
        * wall_seconds
        / max(items, 1),
        "long_run_cpu_time_per_item_ms": 1000.0
        * cpu_seconds
        / max(items, 1),
        "long_run_throughput_items_per_second": items
        / max(wall_seconds, 1e-12),
        "long_run_process_cpu_core_equivalent_percent": 100.0
        * cpu_seconds
        / max(wall_seconds, 1e-12),
        "long_run_process_rss_start_mb": monitor.rss_start / mb,
        "long_run_process_rss_peak_mb": monitor.rss_peak / mb,
        "long_run_process_rss_increment_mb": (
            monitor.rss_peak - monitor.rss_start
        )
        / mb,
        "long_run_private_memory_start_mb": monitor.private_start / mb,
        "long_run_private_memory_peak_mb": monitor.private_peak / mb,
        "long_run_python_allocation_peak_mb": monitor.python_peak / mb,
        "memory_measurement_backend": monitor.backend,
        "long_run_measurement_scope": scope,
    }


CHILD_RUNNER = r'''import ctypes, json, platform, sys, time
start = time.perf_counter_ns()
cpu_start = time.process_time_ns()
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
rss = private = peak = None
if platform.system() == "Windows":
    class C(ctypes.Structure):
        _fields_=[("cb",ctypes.c_ulong),("PageFaultCount",ctypes.c_ulong),
        ("PeakWorkingSetSize",ctypes.c_size_t),("WorkingSetSize",ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage",ctypes.c_size_t),("QuotaPagedPoolUsage",ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage",ctypes.c_size_t),("QuotaNonPagedPoolUsage",ctypes.c_size_t),
        ("PagefileUsage",ctypes.c_size_t),("PeakPagefileUsage",ctypes.c_size_t),
        ("PrivateUsage",ctypes.c_size_t)]
    k=ctypes.WinDLL("kernel32",use_last_error=True)
    p=ctypes.WinDLL("psapi",use_last_error=True)
    k.GetCurrentProcess.restype=ctypes.c_void_p
    p.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(C),ctypes.c_ulong]
    c=C(); c.cb=ctypes.sizeof(c)
    if p.GetProcessMemoryInfo(k.GetCurrentProcess(),ctypes.byref(c),c.cb):
        rss=c.WorkingSetSize; private=c.PrivateUsage; peak=c.PeakWorkingSetSize
print(json.dumps({
    "child_import_ms":(import_done-start)/1e6,
    "child_model_load_ms":(load_done-load_start)/1e6,
    "child_first_inference_ms":(end-infer_start)/1e6,
    "child_internal_total_ms":(end-start)/1e6,
    "child_cpu_seconds":(time.process_time_ns()-cpu_start)/1e9,
    "child_rss_mb":None if rss is None else rss/(1024**2),
    "child_private_mb":None if private is None else private/(1024**2),
    "child_peak_working_set_mb":None if peak is None else peak/(1024**2),
}))
'''


def cold_completion(
    out: Path,
    model_path: Path,
    sample: np.ndarray,
    repetitions: int,
) -> list[dict[str, Any]]:
    runner = out / "cold_resource_runner.py"
    sample_path = out / "cold_resource_sample.npy"
    runner.write_text(CHILD_RUNNER, encoding="utf-8")
    np.save(sample_path, np.ascontiguousarray(sample.reshape(1, -1)))
    rows: list[dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        wall_start = time.perf_counter_ns()
        result = subprocess.run(
            [sys.executable, str(runner), str(model_path), str(sample_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        parent_wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
        values = json.loads(result.stdout.strip().splitlines()[-1])
        rows.append(
            {
                "repetition": repetition,
                "fresh_process_total_ms": parent_wall_ms,
                **values,
                "cache_state": (
                    "new Python process; operating-system filesystem cache "
                    "not flushed"
                ),
            }
        )
    return rows


def plot_completion(metrics: pd.DataFrame, output: Path) -> None:
    labels = [
        (
            f"{row.method}\n(batch {int(row.batch_size)})"
            if row.execution_mode == "batch_inference"
            else row.method
        )
        for row in metrics.itertuples(index=False)
    ]
    x = np.arange(len(metrics))
    figure, axes = plt.subplots(
        1, 2, figsize=(17, 6.5), constrained_layout=True
    )
    axes[0].bar(
        x,
        metrics["long_run_process_cpu_core_equivalent_percent"],
        color="#4C78A8",
    )
    axes[0].axhline(100.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Core-equivalent CPU (%)")
    axes[0].set_title("Long-run process CPU")
    axes[1].bar(
        x - 0.18,
        metrics["long_run_process_rss_start_mb"],
        width=0.36,
        label="Start RSS",
        color="#F58518",
    )
    axes[1].bar(
        x + 0.18,
        metrics["long_run_process_rss_peak_mb"],
        width=0.36,
        label="Peak RSS",
        color="#54A24B",
    )
    axes[1].set_ylabel("Process working set (MB)")
    axes[1].set_title("Native process-memory measurement")
    axes[1].legend()
    for axis in axes:
        axis.set_xticks(x, labels, rotation=24, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Step 30D2 publication resource completion")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    root = discover_root(args.project_root)
    step30d_path = locate_step30d(root)
    step30d = load_module(step30d_path)
    source_run = latest_step30d_run(root)

    run_id = datetime.now(timezone.utc).strftime(
        "run_%Y%m%dT%H%M%S_%fZ"
    )
    out = (
        root
        / "results"
        / "publication_efficiency_resource_completion"
        / run_id
    )
    out.mkdir(parents=True, exist_ok=False)

    source_runtime_path = source_run / "publication_runtime_metrics.csv"
    source_resource_path = source_run / "publication_resource_summary.csv"
    source_manifest_path = source_run / "publication_efficiency_manifest.json"
    source_hashes = {
        "runtime": step30d.sha256_file(source_runtime_path),
        "resources": step30d.sha256_file(source_resource_path),
        "manifest": step30d.sha256_file(source_manifest_path),
    }
    source_runtime = pd.read_csv(source_runtime_path)
    source_resources = pd.read_csv(source_resource_path)
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )

    print("Loading the frozen components used by Step 30D ...")
    inputs = step30d.load_inputs(root, args.sample_rows)
    model = inputs["model"]
    x: np.ndarray = inputs["x"]
    micro_samples = inputs["micro_samples"]
    policy_samples = inputs["policy_samples"]
    model_samples = [x[index : index + 1] for index in range(len(x))]
    combined_samples = [
        (
            model_samples[index % len(model_samples)],
            policy_samples[index % len(policy_samples)],
        )
        for index in range(min(len(model_samples), len(policy_samples)))
    ]

    model_function = lambda sample: model.predict_proba(sample)
    micro_function = lambda sample: step30d.micro_gate_core(
        sample, inputs["micro_threshold"]
    )

    def combined_function(sample: Any) -> Any:
        feature_row, policy_row = sample
        probability = float(model.predict_proba(feature_row)[0, 1])
        model_alarm = int(probability >= inputs["model_threshold"])
        instant, persistent, warning, sources = policy_row
        return step30d.graded_policy_core(
            (
                int(bool(instant) or bool(model_alarm)),
                int(bool(persistent) or bool(model_alarm)),
                warning,
                sources,
            )
        )

    probes: list[dict[str, Any]] = []
    online_specs = [
        (
            "frozen_w100_logistic_regression",
            model_function,
            model_samples,
            (
                "long-run online predict_proba; engineered features only"
            ),
        ),
        (
            "w20_temporal_micro_gate",
            micro_function,
            micro_samples,
            (
                "long-run threshold, persistence and trust rule; "
                "feature extraction excluded"
            ),
        ),
        (
            "graded_multisource_policy_core",
            step30d.graded_policy_core,
            policy_samples,
            "long-run graded policy decision over recorded evidence",
        ),
        (
            "w100_model_plus_graded_policy_runtime_path",
            combined_function,
            combined_samples,
            (
                "long-run timing-only paired path; not an efficacy endpoint"
            ),
        ),
    ]
    for index, (name, function, samples, scope) in enumerate(
        online_specs, start=1
    ):
        print(f"[{index}/8] Long-run resource probe: {name}")
        probes.append(
            long_probe(
                name,
                "online_single_item",
                1,
                function,
                samples,
                args.probe_seconds,
                args.warmup,
                step30d,
                scope,
            )
        )

    for offset, requested_batch in enumerate(
        DEFAULT_BATCH_SIZES, start=len(online_specs) + 1
    ):
        batch_size = min(requested_batch, len(x))
        batch = np.ascontiguousarray(x[:batch_size])
        print(
            f"[{offset}/8] Long-run resource probe: "
            f"w100 batch {batch_size}"
        )
        probes.append(
            long_probe(
                "frozen_w100_logistic_regression",
                "batch_inference",
                batch_size,
                lambda value: model.predict_proba(value),
                [batch],
                args.probe_seconds,
                max(1, min(args.warmup, 20)),
                step30d,
                (
                    "long-run batch predict_proba; engineered features only"
                ),
            )
        )

    print("Completing fresh-process native memory measurements ...")
    cold_rows = cold_completion(
        out,
        inputs["paths"]["model"],
        x[0],
        args.cold_repetitions,
    )
    probes_frame = pd.DataFrame(probes)

    completed = source_runtime.rename(
        columns={
            "process_cpu_core_equivalent_percent": (
                "short_probe_cpu_percent_superseded"
            ),
            "process_rss_start_mb": "short_probe_rss_start_mb_superseded",
            "process_rss_peak_mb": "short_probe_rss_peak_mb_superseded",
            "process_rss_increment_mb": (
                "short_probe_rss_increment_mb_superseded"
            ),
        }
    ).merge(probes_frame, on=KEY_COLUMNS, how="left", validate="one_to_one")

    completed["process_cpu_core_equivalent_percent"] = completed[
        "long_run_process_cpu_core_equivalent_percent"
    ]
    completed["process_rss_start_mb"] = completed[
        "long_run_process_rss_start_mb"
    ]
    completed["process_rss_peak_mb"] = completed[
        "long_run_process_rss_peak_mb"
    ]
    completed["process_rss_increment_mb"] = completed[
        "long_run_process_rss_increment_mb"
    ]
    cold_mask = (
        completed["execution_mode"] == "fresh_process_cold_start"
    )
    cold_peak_values = [
        float(row["child_peak_working_set_mb"])
        for row in cold_rows
        if row["child_peak_working_set_mb"] is not None
    ]
    cold_rss_values = [
        float(row["child_rss_mb"])
        for row in cold_rows
        if row["child_rss_mb"] is not None
    ]
    cold_cpu_values = [
        100.0
        * float(row["child_cpu_seconds"])
        / max(float(row["child_internal_total_ms"]) / 1000.0, 1e-12)
        for row in cold_rows
    ]
    if any(cold_mask):
        completed.loc[
            cold_mask, "process_cpu_core_equivalent_percent"
        ] = statistics.median(cold_cpu_values)
        if cold_peak_values:
            completed.loc[
                cold_mask, "process_rss_peak_mb"
            ] = max(cold_peak_values)
        if cold_rss_values:
            completed.loc[
                cold_mask, "process_rss_start_mb"
            ] = statistics.median(cold_rss_values)
        completed.loc[
            cold_mask, "memory_measurement_backend"
        ] = "Windows PSAPI in fresh child process"

    resources = source_resources.copy()
    total_memory_gb = total_system_memory_gb(step30d)
    resources = pd.concat(
        [
            resources,
            pd.DataFrame(
                [
                    {
                        "resource_type": "measurement_completion",
                        "resource_name": (
                            "long_run_cpu_and_native_process_memory"
                        ),
                        "size_bytes": math.nan,
                        "size_mb": math.nan,
                        "rows": len(probes_frame),
                        "columns": len(probes_frame.columns),
                        "window_size_frames": math.nan,
                        "notes": json.dumps(
                            {
                                "source_step30d_run": str(source_run),
                                "probe_seconds_per_workload": (
                                    args.probe_seconds
                                ),
                                "process_cpu_clock_resolution_seconds": (
                                    time.get_clock_info(
                                        "process_time"
                                    ).resolution
                                ),
                                "memory_backend": sorted(
                                    probes_frame[
                                        "memory_measurement_backend"
                                    ].unique()
                                ),
                                "original_psutil_available": (
                                    source_manifest["hardware"].get(
                                        "psutil_available"
                                    )
                                ),
                                "native_total_system_memory_gb": (
                                    total_memory_gb
                                ),
                            },
                            sort_keys=True,
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    input_hashes_after = {
        name: step30d.sha256_file(path)
        for name, path in inputs["paths"].items()
    }
    changed_inputs = [
        name
        for name, before in inputs["hashes_before"].items()
        if input_hashes_after[name] != before
    ]
    source_hashes_after = {
        "runtime": step30d.sha256_file(source_runtime_path),
        "resources": step30d.sha256_file(source_resource_path),
        "manifest": step30d.sha256_file(source_manifest_path),
    }
    if changed_inputs or source_hashes_after != source_hashes:
        raise RuntimeError(
            "A prior input changed during Step 30D2: "
            f"inputs={changed_inputs}, source_run_changed="
            f"{source_hashes_after != source_hashes}"
        )

    completed.to_csv(out / "publication_runtime_metrics.csv", index=False)
    resources.to_csv(
        out / "publication_resource_summary.csv", index=False
    )
    probes_frame.to_csv(
        out / "publication_long_run_resource_metrics.csv", index=False
    )
    pd.DataFrame(cold_rows).to_csv(
        out / "publication_cold_start_resource_metrics.csv", index=False
    )
    plot_completion(
        probes_frame, out / "publication_resource_completion.png"
    )

    manifest = {
        "stage": "30D2",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "source_step30d_run": str(source_run),
        "source_step30d_sha256": source_hashes,
        "existing_project_artifacts_changed": 0,
        "frozen_input_sha256": input_hashes_after,
        "settings": {
            "probe_seconds_per_workload": args.probe_seconds,
            "warmup": args.warmup,
            "sample_rows_requested": args.sample_rows,
            "w100_rows_loaded": len(x),
            "w20_rows_loaded": len(micro_samples),
            "policy_rows_loaded": len(policy_samples),
            "cold_repetitions": args.cold_repetitions,
            "process_cpu_clock_resolution_seconds": time.get_clock_info(
                "process_time"
            ).resolution,
            "native_total_system_memory_gb": total_memory_gb,
        },
        "measurement_correction": {
            "reason": (
                "Step 30D short probes could not measure native RSS without "
                "psutil and produced unstable CPU percentages for very short "
                "workloads."
            ),
            "authoritative_cpu_fields": (
                "long_run_process_cpu_core_equivalent_percent"
            ),
            "authoritative_memory_fields": [
                "long_run_process_rss_start_mb",
                "long_run_process_rss_peak_mb",
                "long_run_private_memory_peak_mb",
            ],
            "superseded_fields_retained": True,
            "memory_backends": sorted(
                probes_frame["memory_measurement_backend"].unique()
            ),
        },
        "scope": source_manifest["scope"],
        "security_readiness": {
            "step30c_h4_supported": False,
            "step30c_h5_supported": False,
            "step31_permitted": False,
            "reason": (
                "Resource completion does not repair failed source-robustness "
                "hypotheses."
            ),
        },
    }
    (out / "publication_resource_completion_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (out / "README.md").write_text(
        "# Step 30D2 publication resource completion\n\n"
        "This additive run preserves Step 30D and supersedes only its unstable "
        "short-probe CPU/RSS fields. Long-run process CPU and native process "
        "working-set values are the authoritative resource measurements. "
        "Inference scope and exclusions are unchanged. Step 31 remains "
        "blocked by Step 30C H4/H5 failures.\n",
        encoding="utf-8",
    )
    archive_base = (
        out.parent / f"publication_resource_completion_{run_id}"
    )
    archive = Path(
        shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=out.parent,
            base_dir=out.name,
        )
    )

    print("\n" + "=" * 78)
    print("Step 30D2 publication resource completion succeeded.")
    print("Existing project artifacts changed: 0")
    print(
        "Original unstable short-probe CPU/RSS fields retained and "
        "explicitly superseded."
    )
    print("\nAuthoritative long-run CPU and peak process RSS:")
    for row in probes_frame.itertuples(index=False):
        label = (
            f"{row.method} batch={row.batch_size}"
            if row.execution_mode == "batch_inference"
            else row.method
        )
        print(
            f"  {label}: CPU={row.long_run_process_cpu_core_equivalent_percent:.2f}%, "
            f"peak RSS={row.long_run_process_rss_peak_mb:.2f} MB"
        )
    print(f"\nResults directory: {out}")
    print(f"Results archive: {archive}")
    print("Step 30C H4/H5 failures remain unchanged.")
    print("Do not run Step 31 yet.")


if __name__ == "__main__":
    main()
