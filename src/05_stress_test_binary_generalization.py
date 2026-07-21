#!/usr/bin/env python3
"""Stress-test CICIoV2024 binary IDS generalization.

This stage performs three checks:
1. Exact engineered-window feature overlap across train/validation/test.
2. Leave-one-attack-type-out (LOAO) detection of completely unseen attacks.
3. A shuffled-label sanity test that should fall near chance.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # type: ignore[assignment]


RANDOM_STATE = 42
WINDOW_SIZE = 100
BENIGN_CLASS = "BENIGN"
NON_MODEL_COLUMNS = {
    "source_file",
    "window_index",
    "start_row",
    "end_row",
    "split",
    "binary_target",
    "multiclass_target",
}


def find_project_root() -> Path:
    script = Path(__file__).resolve()
    for candidate in (script.parent.parent, script.parent, Path.cwd()):
        expected = candidate / "data" / "processed" / "ciciov2024_windows_w100_all.csv"
        if expected.is_file():
            return candidate
    print("ERROR: Could not find data/processed/ciciov2024_windows_w100_all.csv")
    sys.exit(1)


def choose_f1_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if not len(thresholds):
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1_values))])


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    prediction = (probability >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "samples": len(y_true),
        "benign_samples": int((y_true == 0).sum()),
        "attack_samples": int((y_true == 1).sum()),
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "mcc": matthews_corrcoef(y_true, prediction),
        "false_positive_rate": safe_rate(int(fp), int(fp + tn)),
        "false_negative_rate": safe_rate(int(fn), int(fn + tp)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def model_builders(scale_pos_weight: float) -> dict[str, Callable[[], Any]]:
    builders: dict[str, Callable[[], Any]] = {
        "Logistic Regression": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }
    if XGBClassifier is not None:
        builders["XGBoost"] = lambda: XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    return builders


def audit_feature_overlap(
    data: pd.DataFrame,
    feature_names: list[str],
    results_dir: Path,
) -> None:
    print("[1/3] Auditing exact engineered-window overlap ...")
    signature = pd.util.hash_pandas_object(
        data[feature_names], index=False, categorize=True
    ).astype("uint64")
    data["_window_signature"] = signature.to_numpy()

    split_counts = data.groupby("_window_signature", observed=True)["split"].nunique()
    label_counts = data.groupby("_window_signature", observed=True)["binary_target"].nunique()
    cross_split_ids = split_counts[split_counts > 1].index
    conflicting_ids = label_counts[label_counts > 1].index

    audit = pd.DataFrame(
        [
            {
                "total_windows": len(data),
                "unique_engineered_windows": data["_window_signature"].nunique(),
                "repeated_window_percentage": (
                    1 - data["_window_signature"].nunique() / len(data)
                )
                * 100,
                "signatures_crossing_partitions": len(cross_split_ids),
                "rows_in_cross_partition_signatures": int(
                    data["_window_signature"].isin(cross_split_ids).sum()
                ),
                "binary_label_conflicting_signatures": len(conflicting_ids),
                "rows_in_label_conflicting_signatures": int(
                    data["_window_signature"].isin(conflicting_ids).sum()
                ),
            }
        ]
    )
    audit.to_csv(results_dir / "window_feature_overlap_audit.csv", index=False)

    class_rows: list[dict[str, Any]] = []
    for class_name, subset in data.groupby("multiclass_target", observed=True):
        subset_cross = subset["_window_signature"].isin(cross_split_ids)
        class_rows.append(
            {
                "multiclass_target": class_name,
                "windows": len(subset),
                "unique_engineered_windows": subset["_window_signature"].nunique(),
                "rows_in_cross_partition_signatures": int(subset_cross.sum()),
                "cross_partition_row_percentage": float(subset_cross.mean() * 100),
            }
        )
    pd.DataFrame(class_rows).to_csv(
        results_dir / "window_feature_overlap_by_class.csv", index=False
    )

    print(audit.to_string(index=False))


def leave_one_attack_out(
    data: pd.DataFrame,
    feature_names: list[str],
    results_dir: Path,
) -> pd.DataFrame:
    print("\n[2/3] Running leave-one-attack-type-out evaluation ...")
    attacks = sorted(
        class_name
        for class_name in data["multiclass_target"].unique()
        if class_name != BENIGN_CLASS
    )
    records: list[dict[str, Any]] = []

    for attack_index, held_out in enumerate(attacks, start=1):
        train = data[
            (data["split"] == "train") & (data["multiclass_target"] != held_out)
        ]
        validation = data[
            (data["split"] == "validation")
            & (data["multiclass_target"] != held_out)
        ]
        test = data[
            (data["split"] == "test")
            & (data["multiclass_target"].isin([BENIGN_CLASS, held_out]))
        ]

        x_train = train[feature_names].to_numpy(dtype=np.float64)
        y_train = train["binary_target"].to_numpy(dtype=np.uint8)
        x_validation = validation[feature_names].to_numpy(dtype=np.float64)
        y_validation = validation["binary_target"].to_numpy(dtype=np.uint8)
        x_test = test[feature_names].to_numpy(dtype=np.float64)
        y_test = test["binary_target"].to_numpy(dtype=np.uint8)

        negative = int((y_train == 0).sum())
        positive = int((y_train == 1).sum())
        builders = model_builders(negative / positive)
        known_attacks = sorted(
            set(train.loc[train["binary_target"] == 1, "multiclass_target"])
        )

        print(
            f"  [{attack_index}/{len(attacks)}] Holding out {held_out}; "
            f"unseen test windows={int((y_test == 1).sum())}"
        )
        for model_name, builder in builders.items():
            model = builder()
            start = time.perf_counter()
            model.fit(x_train, y_train)
            train_seconds = time.perf_counter() - start
            validation_probability = model.predict_proba(x_validation)[:, 1]
            test_probability = model.predict_proba(x_test)[:, 1]
            tuned_threshold = choose_f1_threshold(
                y_validation, validation_probability
            )

            for threshold_mode, threshold in (
                ("validation_f1", tuned_threshold),
                ("fixed_0.5", 0.5),
            ):
                row = evaluate(y_test, test_probability, threshold)
                row.update(
                    {
                        "model": model_name,
                        "held_out_attack": held_out,
                        "threshold_mode": threshold_mode,
                        "known_attack_types": ";".join(known_attacks),
                        "train_samples": len(train),
                        "validation_samples": len(validation),
                        "train_seconds": train_seconds,
                    }
                )
                records.append(row)

            tuned_row = records[-2]
            print(
                f"      {model_name:<20} threshold={tuned_threshold:.6f} "
                f"unseen recall={tuned_row['recall']:.4f} "
                f"FPR={tuned_row['false_positive_rate']:.4f}"
            )

    metrics = pd.DataFrame(records)
    metrics.to_csv(results_dir / "leave_one_attack_out_metrics.csv", index=False)
    return metrics


def shuffled_label_sanity(
    data: pd.DataFrame,
    feature_names: list[str],
    results_dir: Path,
) -> pd.DataFrame:
    print("\n[3/3] Running shuffled-label sanity test ...")
    train = data[data["split"] == "train"]
    test = data[data["split"] == "test"]
    x_train = train[feature_names].to_numpy(dtype=np.float64)
    original_y = train["binary_target"].to_numpy(dtype=np.uint8)
    shuffled_y = np.random.default_rng(RANDOM_STATE).permutation(original_y)
    x_test = test[feature_names].to_numpy(dtype=np.float64)
    y_test = test["binary_target"].to_numpy(dtype=np.uint8)

    negative = int((shuffled_y == 0).sum())
    positive = int((shuffled_y == 1).sum())
    rows: list[dict[str, Any]] = []
    for model_name, builder in model_builders(negative / positive).items():
        model = builder()
        model.fit(x_train, shuffled_y)
        probability = model.predict_proba(x_test)[:, 1]
        row = evaluate(y_test, probability, 0.5)
        row["model"] = model_name
        rows.append(row)
        print(
            f"    {model_name:<20} balanced accuracy={row['balanced_accuracy']:.4f}, "
            f"PR-AUC={row['pr_auc']:.4f}"
        )

    result = pd.DataFrame(rows)
    result.to_csv(results_dir / "shuffled_label_sanity_metrics.csv", index=False)
    return result


def plot_loao_heatmap(metrics: pd.DataFrame, output: Path) -> None:
    selected = metrics[metrics["threshold_mode"] == "validation_f1"]
    pivot = selected.pivot(
        index="model", columns="held_out_attack", values="recall"
    )
    figure, axis = plt.subplots(
        figsize=(max(8, 1.5 * len(pivot.columns)), 1.2 * len(pivot.index) + 2.5)
    )
    image = axis.imshow(pivot.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1)
    axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    axis.set_title("Recall for Completely Unseen Attack Types")
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            value = pivot.iloc[row, column]
            axis.text(column, row, f"{value:.3f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Unseen-attack recall")
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    root = find_project_root()
    input_path = (
        root
        / "data"
        / "processed"
        / f"ciciov2024_windows_w{WINDOW_SIZE}_all.csv"
    )
    results_dir = root / "results" / "stress_tests_w100"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path} ...")
    data = pd.read_csv(input_path)
    feature_names = [
        column for column in data.columns if column not in NON_MODEL_COLUMNS
    ]
    if not feature_names:
        raise ValueError("No engineered feature columns found")
    if data[feature_names].isna().any().any():
        raise ValueError("Engineered features contain missing values")

    print(
        f"Windows={len(data):,}, features={len(feature_names)}, "
        f"classes={sorted(data['multiclass_target'].unique())}\n"
    )
    audit_feature_overlap(data, feature_names, results_dir)
    loao_metrics = leave_one_attack_out(data, feature_names, results_dir)
    shuffled_label_sanity(data, feature_names, results_dir)
    plot_loao_heatmap(
        loao_metrics, results_dir / "leave_one_attack_out_recall_heatmap.png"
    )

    tuned = loao_metrics[loao_metrics["threshold_mode"] == "validation_f1"]
    print("\n" + "=" * 96)
    print("Unseen-attack recall using validation-selected thresholds:")
    print(
        tuned.pivot(
            index="held_out_attack", columns="model", values="recall"
        ).to_string()
    )
    print(f"\nResults directory: {results_dir}")
    print("Stress testing completed successfully.")


if __name__ == "__main__":
    main()
