#!/usr/bin/env python3
"""Train binary CICIoV2024 window-based intrusion-detection baselines.

Models:
    1. Majority-class dummy baseline
    2. Class-weighted Logistic Regression
    3. Class-weighted Random Forest
    4. XGBoost with scale_pos_weight

Thresholds are selected using validation data only. The chronological test set
is evaluated once with the selected thresholds.

Place in D:\\ztav_project\\src and run from D:\\ztav_project:
    python src/04_train_binary_baselines.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any

# Keep matplotlib cache inside the project on restrictive systems.
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # type: ignore[assignment]


RANDOM_STATE = 42
WINDOW_SIZE = 100

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
    script_path = Path(__file__).resolve()
    for candidate in (script_path.parent.parent, script_path.parent, Path.cwd()):
        if (candidate / "data" / "processed").is_dir():
            return candidate
    print("ERROR: Could not find data/processed.")
    sys.exit(1)


def select_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Select the highest-F1 threshold using validation data only."""

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5, f1_score(y_true, probabilities >= 0.5, zero_division=0)

    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def metric_row(
    model_name: str,
    partition: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    train_seconds: float,
    prediction_seconds: float,
) -> tuple[dict[str, Any], np.ndarray]:
    predictions = (probabilities >= threshold).astype(np.uint8)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    row: dict[str, Any] = {
        "model": model_name,
        "partition": partition,
        "threshold": threshold,
        "samples": len(y_true),
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "mcc": matthews_corrcoef(y_true, predictions),
        "false_positive_rate": safe_rate(int(fp), int(fp + tn)),
        "false_negative_rate": safe_rate(int(fn), int(fn + tp)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "train_seconds": train_seconds,
        "prediction_seconds": prediction_seconds,
        "prediction_ms_per_window": prediction_seconds / max(len(y_true), 1) * 1000,
    }
    return row, matrix


def flatten_report(model_name: str, report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, values in report.items():
        if not isinstance(values, dict):
            continue
        rows.append(
            {
                "model": model_name,
                "class": label,
                "precision": values.get("precision"),
                "recall": values.get("recall"),
                "f1_score": values.get("f1-score"),
                "support": values.get("support"),
            }
        )
    return rows


def plot_confusion_matrices(
    matrices: dict[str, np.ndarray], output: Path
) -> None:
    columns = 2
    rows = int(np.ceil(len(matrices) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(10, 4.4 * rows))
    axes_array = np.atleast_1d(axes).ravel()

    for axis, (name, matrix) in zip(axes_array, matrices.items()):
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(name)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks([0, 1], ["Benign", "Attack"])
        axis.set_yticks([0, 1], ["Benign", "Attack"])
        for row in range(2):
            for column in range(2):
                axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    for axis in axes_array[len(matrices) :]:
        axis.axis("off")
    figure.suptitle("Binary IDS Confusion Matrices — Chronological Test Set", fontsize=14)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_roc_curves(
    y_true: np.ndarray, probabilities: dict[str, np.ndarray], output: Path
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    for name, scores in probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        axis.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.4f})")
    axis.plot([0, 1], [0, 1], "--", color="grey", label="Random")
    axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="Binary IDS ROC Curves")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_pr_curves(
    y_true: np.ndarray, probabilities: dict[str, np.ndarray], output: Path
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    for name, scores in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        axis.plot(recall, precision, linewidth=2, label=f"{name} (AP={ap:.4f})")
    prevalence = float(np.mean(y_true))
    axis.axhline(prevalence, linestyle="--", color="grey", label=f"Prevalence={prevalence:.3f}")
    axis.set(xlabel="Recall", ylabel="Precision", title="Binary IDS Precision–Recall Curves")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_feature_importance(
    model_name: str, model: Any, feature_names: list[str], results_dir: Path
) -> None:
    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1]

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        return

    importance = pd.DataFrame({"feature": feature_names, "importance": values})
    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
    path = results_dir / f"binary_feature_importance_{model_name.lower().replace(' ', '_')}.csv"
    importance.to_csv(path, index=False)


def main() -> None:
    root = find_project_root()
    processed_dir = root / "data" / "processed"
    results_dir = root / "results" / "binary_baseline_w100"
    models_dir = root / "models" / "binary_baseline_w100"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        partition: processed_dir / f"ciciov2024_windows_w{WINDOW_SIZE}_{partition}.csv"
        for partition in ("train", "validation", "test")
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        print("ERROR: Missing processed window files:")
        for path in missing:
            print(f"  {path}")
        print("Run src/03_build_window_dataset.py first.")
        sys.exit(1)

    print("Loading chronological window datasets ...")
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    feature_names = [column for column in frames["train"].columns if column not in NON_MODEL_COLUMNS]
    if not feature_names:
        raise ValueError("No model feature columns were found")

    for name, frame in frames.items():
        if frame[feature_names].isna().any().any():
            raise ValueError(f"{name} contains missing model features")
        if set(frame["binary_target"].unique()) != {0, 1}:
            raise ValueError(f"{name} must contain both binary classes")

    x_train = frames["train"][feature_names].to_numpy(dtype=np.float64)
    y_train = frames["train"]["binary_target"].to_numpy(dtype=np.uint8)
    x_validation = frames["validation"][feature_names].to_numpy(dtype=np.float64)
    y_validation = frames["validation"]["binary_target"].to_numpy(dtype=np.uint8)
    x_test = frames["test"][feature_names].to_numpy(dtype=np.float64)
    y_test = frames["test"]["binary_target"].to_numpy(dtype=np.uint8)

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / positive_count

    models: dict[str, Any] = {
        "Dummy Majority": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": Pipeline(
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
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
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
    else:
        print("WARNING: xgboost is not installed; XGBoost will be skipped.")
        print("Install it with: pip install xgboost")

    print(
        f"Features={len(feature_names)}, train={len(y_train):,}, "
        f"validation={len(y_validation):,}, test={len(y_test):,}, "
        f"scale_pos_weight={scale_pos_weight:.4f}\n"
    )

    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    test_probabilities: dict[str, np.ndarray] = {}
    test_matrices: dict[str, np.ndarray] = {}
    chosen_thresholds: dict[str, float] = {}

    for index, (name, model) in enumerate(models.items(), start=1):
        print(f"[{index}/{len(models)}] Training {name} ...")
        start = time.perf_counter()
        model.fit(x_train, y_train)
        train_seconds = time.perf_counter() - start

        validation_start = time.perf_counter()
        validation_probability = model.predict_proba(x_validation)[:, 1]
        validation_prediction_seconds = time.perf_counter() - validation_start

        if name == "Dummy Majority":
            threshold = 0.5
        else:
            threshold, _ = select_f1_threshold(y_validation, validation_probability)
        chosen_thresholds[name] = threshold

        validation_row, _ = metric_row(
            name,
            "validation",
            y_validation,
            validation_probability,
            threshold,
            train_seconds,
            validation_prediction_seconds,
        )
        validation_rows.append(validation_row)

        test_start = time.perf_counter()
        test_probability = model.predict_proba(x_test)[:, 1]
        test_prediction_seconds = time.perf_counter() - test_start
        test_row, test_matrix = metric_row(
            name,
            "test",
            y_test,
            test_probability,
            threshold,
            train_seconds,
            test_prediction_seconds,
        )
        test_rows.append(test_row)
        test_probabilities[name] = test_probability
        test_matrices[name] = test_matrix

        test_predictions = (test_probability >= threshold).astype(np.uint8)
        report = classification_report(
            y_test,
            test_predictions,
            labels=[0, 1],
            target_names=["BENIGN", "ATTACK"],
            output_dict=True,
            zero_division=0,
        )
        report_rows.extend(flatten_report(name, report))

        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, models_dir / f"{safe_name}.joblib")
        save_feature_importance(name, model, feature_names, results_dir)
        print(
            f"    threshold={threshold:.6f}, validation F1={validation_row['f1']:.4f}, "
            f"test F1={test_row['f1']:.4f}, test PR-AUC={test_row['pr_auc']:.4f}"
        )

    validation_metrics = pd.DataFrame(validation_rows).sort_values("f1", ascending=False)
    test_metrics = pd.DataFrame(test_rows).sort_values("f1", ascending=False)
    validation_metrics.to_csv(results_dir / "binary_validation_metrics.csv", index=False)
    test_metrics.to_csv(results_dir / "binary_test_metrics.csv", index=False)
    pd.DataFrame(report_rows).to_csv(results_dir / "binary_test_classification_reports.csv", index=False)

    with (results_dir / "binary_selected_thresholds.json").open("w", encoding="utf-8") as handle:
        json.dump(chosen_thresholds, handle, indent=2)
    with (results_dir / "binary_feature_columns.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_names, handle, indent=2)

    plot_confusion_matrices(test_matrices, results_dir / "binary_confusion_matrices.png")
    plot_roc_curves(y_test, test_probabilities, results_dir / "binary_roc_curves.png")
    plot_pr_curves(y_test, test_probabilities, results_dir / "binary_pr_curves.png")

    best_validation_model = str(validation_metrics.iloc[0]["model"])
    print("\n" + "=" * 92)
    print("Validation metrics (used for model/threshold selection):")
    print(
        validation_metrics[
            ["model", "threshold", "precision", "recall", "f1", "pr_auc", "roc_auc", "false_positive_rate"]
        ].to_string(index=False)
    )
    print("\nChronological test metrics:")
    print(
        test_metrics[
            ["model", "precision", "recall", "f1", "pr_auc", "roc_auc", "false_positive_rate", "false_negative_rate"]
        ].to_string(index=False)
    )
    print(f"\nValidation-selected candidate: {best_validation_model}")
    print(f"Results directory: {results_dir}")
    print(f"Models directory:  {models_dir}")
    print("\nBinary baseline training completed successfully.")


if __name__ == "__main__":
    main()
