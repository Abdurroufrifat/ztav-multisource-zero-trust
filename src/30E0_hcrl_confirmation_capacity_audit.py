#!/usr/bin/env python3
"""Step 30E0: non-consuming HCRL confirmation-capacity audit.

This additive preflight answers one question before Step 30E is allowed to
consume the final confirmation partition: do enough HCRL parent windows remain
that were never used by the frozen Step 24/25 development replay?

It does not run SUMO, create ``CONFIRMATION_LOCK.json``, train a model, change a
threshold, or overwrite a historical result.  The output is timestamped and is
safe to rerun.  A failed capacity verdict is evidence about the dataset design;
it must not be bypassed with replacement sampling and then described as an
untouched confirmation.

Run from ``D:\\ztav_project``:

    .\\.venv\\Scripts\\python.exe .\\src\\30E0_hcrl_confirmation_capacity_audit.py

Send the terminal result and ``hcrl_leakage_safe_capacity.csv`` before running
Step 30E or Step 31.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def infer_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name.lower() == "src" else here


ROOT = infer_root()
RESULTS = ROOT / "results"

PARENTS_RELATIVE = Path(
    "results/multiscale_sparse_can_gate/multiscale_parent_predictions.csv"
)
DEVELOPMENT_RELATIVE = Path(
    "results/graded_zero_trust_policy/graded_policy_decisions.csv"
)

PARENT_COLUMNS = {
    "source_file",
    "parent_window_index",
    "w100_binary_target",
    "w100_attack_frame_count",
}
DEVELOPMENT_COLUMNS = {
    "source_file",
    "parent_window_index",
    "density_scenario",
}

CONFIRMATION_SEEDS = 5
DENSITIES = 4
BENIGN_ASSIGNMENTS_PER_REPLAY = 560
ATTACK_ASSIGNMENTS_PER_REPLAY = 160

# A crossed-panel design would deliberately replay the same held-out HCRL
# panel against five new SUMO contexts.  It must still contain enough distinct
# attack parents to avoid a scientifically uninformative tiny sparse panel.
# Fifty per sparse source/band gives at least 200 unique low- and 200 unique
# medium-density attack parents across the four HCRL captures.  High-density
# needs two disjoint 160-parent panels: high_21_100 and representative_all.
CROSSED_PANEL_SPARSE_MINIMUM = 50
CROSSED_PANEL_ATTACK_TARGET = ATTACK_ASSIGNMENTS_PER_REPLAY
CROSSED_PANEL_HIGH_REQUIRED = 2 * ATTACK_ASSIGNMENTS_PER_REPLAY
CROSSED_PANEL_BENIGN_REQUIRED = DENSITIES * BENIGN_ASSIGNMENTS_PER_REPLAY

BANDS = {
    "low_1_5": (1, 5),
    "medium_6_20": (6, 20),
    "high_21_100": (21, 100),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit untouched HCRL capacity without consuming Step 30E."
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(columns: Iterable[str], required: set[str], label: str) -> None:
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def pool_mask(frame: pd.DataFrame, pool: str) -> pd.Series:
    if pool == "benign":
        return frame["w100_binary_target"].astype(int).eq(0)
    if pool == "all_attack":
        return frame["w100_binary_target"].astype(int).eq(1)
    lower, upper = BANDS[pool]
    return frame["w100_binary_target"].astype(int).eq(1) & frame[
        "w100_attack_frame_count"
    ].astype(int).between(lower, upper)


def unique_ids(frame: pd.DataFrame, mask: pd.Series) -> set[int]:
    return set(frame.loc[mask, "parent_window_index"].astype(int).unique())


def requirement(pool: str) -> tuple[int, int, str]:
    """Return original requirement, conservative crossed minimum, rationale."""
    if pool == "benign":
        return (
            CONFIRMATION_SEEDS
            * DENSITIES
            * BENIGN_ASSIGNMENTS_PER_REPLAY,
            CROSSED_PANEL_BENIGN_REQUIRED,
            "four disjoint 560-parent benign panels; panels may cross five new SUMO seeds",
        )
    if pool in {"low_1_5", "medium_6_20"}:
        return (
            CONFIRMATION_SEEDS * ATTACK_ASSIGNMENTS_PER_REPLAY,
            CROSSED_PANEL_SPARSE_MINIMUM,
            "minimum distinct sparse parents per source; repeated rows are not independent units",
        )
    if pool == "high_21_100":
        return (
            CONFIRMATION_SEEDS * ATTACK_ASSIGNMENTS_PER_REPLAY,
            CROSSED_PANEL_HIGH_REQUIRED,
            "two disjoint 160-parent panels for high-density and representative conditions",
        )
    if pool == "all_attack":
        target = DENSITIES * ATTACK_ASSIGNMENTS_PER_REPLAY
        minimum = (
            2 * CROSSED_PANEL_SPARSE_MINIMUM
            + CROSSED_PANEL_HIGH_REQUIRED
        )
        return (
            CONFIRMATION_SEEDS * DENSITIES * ATTACK_ASSIGNMENTS_PER_REPLAY,
            minimum,
            "aggregate check for four mutually disjoint attack panels",
        )
    raise KeyError(pool)


def audit_rows(
    parents: pd.DataFrame, development: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources = sorted(
        str(value)
        for value in parents["source_file"].dropna().unique()
        if str(value) != "normal_run_data.txt"
    )
    rows: list[dict[str, Any]] = []
    verdicts: list[dict[str, Any]] = []
    pools = ("benign", "low_1_5", "medium_6_20", "high_21_100", "all_attack")

    for source in sources:
        source_parents = parents[parents["source_file"].astype(str) == source].copy()
        source_development = development[
            development["source_file"].astype(str) == source
        ]
        development_ids = set(
            source_development["parent_window_index"].astype(int).unique()
        )
        unseen_ids = set(source_parents["parent_window_index"].astype(int)) - development_ids

        source_passes: dict[str, bool] = {}
        for pool in pools:
            mask = pool_mask(source_parents, pool)
            total = unique_ids(source_parents, mask)
            used = total & development_ids
            unseen = total & unseen_ids
            original_required, crossed_required, rationale = requirement(pool)
            crossed_passed = len(unseen) >= crossed_required
            source_passes[pool] = crossed_passed
            rows.append(
                {
                    "source_file": source,
                    "pool": pool,
                    "total_unique_parents": len(total),
                    "development_used_unique_parents": len(used),
                    "unseen_unique_parents": len(unseen),
                    "development_exhaustion_rate": (
                        len(used) / len(total) if total else 0.0
                    ),
                    "original_unique_every_replay_required": original_required,
                    "original_unique_every_replay_feasible": len(unseen)
                    >= original_required,
                    "conservative_crossed_panel_required": crossed_required,
                    "conservative_crossed_panel_feasible": crossed_passed,
                    "requirement_rationale": rationale,
                }
            )

        crossed_feasible = all(source_passes.values())
        verdicts.append(
            {
                "source_file": source,
                "crossed_panel_feasible": crossed_feasible,
                "failed_pools": ";".join(
                    pool for pool in pools if not source_passes[pool]
                ),
            }
        )
    return rows, verdicts


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    parents_path = root / PARENTS_RELATIVE
    development_path = root / DEVELOPMENT_RELATIVE
    for path in (parents_path, development_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    parents = pd.read_csv(parents_path)
    development = pd.read_csv(development_path)
    require_columns(parents.columns, PARENT_COLUMNS, "Step 21 parent predictions")
    require_columns(
        development.columns, DEVELOPMENT_COLUMNS, "Step 25 policy decisions"
    )
    parents["parent_window_index"] = parents["parent_window_index"].astype(int)
    development["parent_window_index"] = development[
        "parent_window_index"
    ].astype(int)
    if parents.duplicated(["source_file", "parent_window_index"]).any():
        raise ValueError("Step 21 has duplicate source/parent identifiers")

    rows, verdicts = audit_rows(parents, development)
    feasible = bool(verdicts) and all(
        bool(row["crossed_panel_feasible"]) for row in verdicts
    )

    out = (
        root
        / "results"
        / "publication_untouched_capacity_preflight"
        / run_id()
    )
    out.mkdir(parents=True, exist_ok=False)
    capacity_path = out / "hcrl_leakage_safe_capacity.csv"
    verdict_path = out / "hcrl_capacity_verdict.csv"
    pd.DataFrame(rows).to_csv(capacity_path, index=False)
    pd.DataFrame(verdicts).to_csv(verdict_path, index=False)

    summary = {
        "experiment": "Step 30E0 non-consuming HCRL confirmation-capacity audit",
        "completed_utc": utc_now(),
        "confirmation_consumed": False,
        "sumo_executed": False,
        "confirmation_lock_created": False,
        "existing_project_artifacts_changed": 0,
        "parents_input": str(parents_path.relative_to(root)),
        "parents_sha256": sha256(parents_path),
        "development_input": str(development_path.relative_to(root)),
        "development_sha256": sha256(development_path),
        "strict_original_design_feasible": all(
            bool(row["original_unique_every_replay_feasible"]) for row in rows
        ),
        "conservative_crossed_panel_design_feasible": feasible,
        "crossed_panel_statistical_unit": (
            "new SUMO seed after macro-averaging repeated HCRL panels; pure CAN "
            "comparisons use unique source panels"
        ),
        "decision": (
            "eligible_for_predeclared_crossed-panel Step 30E design"
            if feasible
            else "not enough untouched HCRL capacity; acquire new capture data or "
            "report Step 30E as unavailable without claiming confirmation"
        ),
        "step31_permitted": False,
    }
    (out / "hcrl_capacity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 88)
    print("Step 30E0 HCRL confirmation-capacity audit completed successfully.")
    print("Confirmation consumed: False")
    print("SUMO executed: False")
    print("Confirmation lock created: False")
    print("Existing project artifacts changed: 0")
    print("\nUnseen unique parents after all Step 25 development use:")
    table = pd.DataFrame(rows)
    display = table.pivot(
        index="source_file", columns="pool", values="unseen_unique_parents"
    ).reindex(columns=["benign", "low_1_5", "medium_6_20", "high_21_100", "all_attack"])
    print(display.to_string())
    print(
        "\nConservative crossed-panel design feasible: "
        f"{feasible}"
    )
    if not feasible:
        failed = [
            f"{row['source_file']}:{row['failed_pools']}"
            for row in verdicts
            if not row["crossed_panel_feasible"]
        ]
        print("Failed capacity checks: " + ", ".join(failed))
        print("Do not run Step 30E or Step 31.")
        print("Replacement sampling cannot convert reused parents into untouched evidence.")
    else:
        print("Do not run Step 30E yet; first freeze the crossed-panel protocol in its code.")
    print(f"\nResults directory: {out}")
    print(f"Capacity table: {capacity_path}")
    print(f"Verdict table: {verdict_path}")
    print("\nNext: send the terminal result and hcrl_leakage_safe_capacity.csv.")


if __name__ == "__main__":
    main()
