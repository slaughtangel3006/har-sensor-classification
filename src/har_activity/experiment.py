"""Exécution, comparaison et enregistrement des expériences."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from har_activity.data import ACTIVITIES, HARData
from har_activity.models import build_classical_models
from har_activity.neural import train_and_predict

METRIC_FIELDS = ("accuracy", "precision_macro", "recall_macro", "f1_macro")
CNN_NAME = "CNN 1D (signaux bruts)"
JOURNAL_FIELDS = (
    "timestamp_utc",
    "modele",
    "epochs_max",
    "epochs_retenues",
    *METRIC_FIELDS,
    "seed",
)


def _save_confusion_matrix(matrix: np.ndarray, model_name: str, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(ACTIVITIES)),
        yticks=np.arange(len(ACTIVITIES)),
        xticklabels=ACTIVITIES,
        yticklabels=ACTIVITIES,
        xlabel="Activité prédite",
        ylabel="Activité réelle",
        title=f"Matrice de confusion — {model_name}",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    safe_name = "".join(character.lower() if character.isalnum() else "_" for character in model_name)
    figure.savefig(output_dir / f"confusion_{safe_name}.png", dpi=150)
    plt.close(figure)


def _record_journal(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as journal:
            writer = csv.DictWriter(journal, fieldnames=JOURNAL_FIELDS)
            writer.writeheader()
            writer.writerows(records)
        return

    with path.open("r", newline="", encoding="utf-8") as journal:
        reader = csv.DictReader(journal)
        previous_records = list(reader)
        needs_migration = not set(JOURNAL_FIELDS).issubset(reader.fieldnames or [])

    if needs_migration:
        with path.open("w", newline="", encoding="utf-8") as journal:
            writer = csv.DictWriter(journal, fieldnames=JOURNAL_FIELDS)
            writer.writeheader()
            writer.writerows(
                {
                    field: row.get(field, "")
                    for field in JOURNAL_FIELDS
                }
                for row in previous_records
            )

    with path.open("a", newline="", encoding="utf-8") as journal:
        csv.DictWriter(journal, fieldnames=JOURNAL_FIELDS).writerows(records)


def _save_cnn_curves(
    history: dict[str, list[float]], best_epoch: int, seed: int, output_dir: Path
) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="Loss entraînement")
    axes[1].plot(epochs, history["validation_accuracy"], label="Accuracy validation")
    for axis in axes:
        axis.axvline(best_epoch, color="tab:red", linestyle="--", label=f"Époque retenue : {best_epoch}")
        axis.set_xlabel("Époque")
        axis.legend()
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Accuracy")
    figure.suptitle(f"Apprentissage du CNN — graine {seed}")
    figure.tight_layout()
    figure.savefig(output_dir / "courbes_cnn.png", dpi=150)
    plt.close(figure)


def _mean_and_std(values: list[float]) -> tuple[float, float]:
    deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return float(np.mean(values)), deviation


def _compute_metrics(truth: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    report = classification_report(
        truth,
        predictions,
        labels=np.arange(len(ACTIVITIES)),
        target_names=ACTIVITIES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
    }


def _save_metrics(
    path: Path, scores: dict[str, dict[str, list[float]]]
) -> list[dict[str, float | str]]:
    metric_rows = []
    for model_name, model_scores in scores.items():
        accuracy_mean, accuracy_std = _mean_and_std(model_scores["accuracy"])
        f1_mean, f1_std = _mean_and_std(model_scores["f1_macro"])
        metric_rows.append(
            {
                "modele": model_name,
                "accuracy_mean": accuracy_mean,
                "accuracy_std": accuracy_std,
                "f1_macro_mean": f1_mean,
                "f1_macro_std": f1_std,
            }
        )

    with path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=(
                "modele",
                "accuracy_mean",
                "accuracy_std",
                "f1_macro_mean",
                "f1_macro_std",
            ),
        )
        writer.writeheader()
        writer.writerows(metric_rows)
    return metric_rows


def run_experiment(
    data: HARData,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    patience: int,
    seeds: Sequence[int],
) -> None:
    if not seeds:
        raise ValueError("Au moins une graine est nécessaire pour lancer les expériences.")
    output_dir.mkdir(parents=True, exist_ok=True)
    journal_records = []
    scores = {
        "SVM": {field: [] for field in METRIC_FIELDS},
        "Forêt aléatoire": {field: [] for field in METRIC_FIELDS},
        CNN_NAME: {field: [] for field in METRIC_FIELDS},
    }

    first_cnn_history = None
    first_cnn_epoch = 0
    for seed_index, seed in enumerate(seeds):
        print(f"\nGraine {seed} ({seed_index + 1}/{len(seeds)})")
        models = build_classical_models(seed)
        predictions_by_model = {}
        for model_name, model in models.items():
            model.fit(data.train_features, data.train_labels)
            predictions_by_model[model_name] = (model.predict(data.test_features), None, None)

        cnn_predictions, best_epoch, history = train_and_predict(
            data.train_raw,
            data.train_labels,
            data.train_subjects,
            data.test_raw,
            epochs,
            batch_size,
            patience,
            seed,
        )
        predictions_by_model[CNN_NAME] = (
            cnn_predictions,
            best_epoch,
            history,
        )
        if seed_index == 0:
            first_cnn_history = history
            first_cnn_epoch = best_epoch

        for model_name, (predictions, retained_epochs, _) in predictions_by_model.items():
            metrics = _compute_metrics(data.test_labels, predictions)
            for field, value in metrics.items():
                scores[model_name][field].append(float(value))
            print(
                f"  {model_name}: accuracy={metrics['accuracy']:.4f}, "
                f"F1 macro={metrics['f1_macro']:.4f}"
            )
            if seed_index == 0:
                matrix = confusion_matrix(
                    data.test_labels, predictions, labels=np.arange(len(ACTIVITIES))
                )
                _save_confusion_matrix(matrix, model_name, output_dir)
            journal_records.append(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "modele": model_name,
                    "seed": seed,
                    "epochs_max": epochs if retained_epochs is not None else "",
                    "epochs_retenues": retained_epochs if retained_epochs is not None else "",
                    **metrics,
                }
            )

    metric_rows = _save_metrics(output_dir / "metrics.csv", scores)
    for row in metric_rows:
        print(
            f"\n{row['modele']}: accuracy={row['accuracy_mean']:.4f} ± "
            f"{row['accuracy_std']:.4f}, F1 macro={row['f1_macro_mean']:.4f} ± "
            f"{row['f1_macro_std']:.4f}"
        )
    _record_journal(output_dir / "journal_experiences.csv", journal_records)
    if first_cnn_history is not None:
        _save_cnn_curves(first_cnn_history, first_cnn_epoch, seeds[0], output_dir)
