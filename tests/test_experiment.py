import csv

import numpy as np

from har_activity.data import HARData
from har_activity import experiment


class FixedModel:
    def __init__(self, prediction):
        self.prediction = prediction

    def fit(self, features, labels):
        return self

    def predict(self, features):
        return np.full(len(features), self.prediction)


def test_run_experiment_records_seed_scores_and_first_seed_images(tmp_path, monkeypatch):
    monkeypatch.setattr(
        experiment,
        "build_classical_models",
        lambda seed: {
            "SVM": FixedModel(seed % 2),
            "Forêt aléatoire": FixedModel((seed + 1) % 2),
        },
    )
    monkeypatch.setattr(
        experiment,
        "train_and_predict",
        lambda train, labels, subjects, test, epochs, batch_size, patience, seed: (
            np.full(len(test), seed % 6),
            2,
            {"train_loss": [1.0, 0.5], "validation_accuracy": [0.5, 0.75]},
        ),
    )
    data = HARData(
        train_features=np.zeros((6, 3)),
        train_labels=np.arange(6),
        train_subjects=np.arange(1, 7),
        train_raw=np.zeros((6, 9, 128)),
        test_features=np.zeros((6, 3)),
        test_labels=np.arange(6),
        test_subjects=np.arange(7, 13),
        test_raw=np.zeros((6, 9, 128)),
    )
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    journal_path = output_dir / "journal_experiences.csv"
    old_fields = (
        "timestamp_utc",
        "modele",
        "epochs_max",
        "epochs_retenues",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    )
    with journal_path.open("w", newline="", encoding="utf-8") as journal:
        writer = csv.DictWriter(journal, fieldnames=old_fields)
        writer.writeheader()
        writer.writerow({field: "ancienne valeur" for field in old_fields})

    experiment.run_experiment(data, output_dir, epochs=2, batch_size=8, patience=2, seeds=[42, 43])

    with (output_dir / "metrics.csv").open(newline="", encoding="utf-8") as metrics_file:
        metrics = list(csv.DictReader(metrics_file))
    with journal_path.open(newline="", encoding="utf-8") as journal:
        records = list(csv.DictReader(journal))

    assert len(metrics) == 3
    assert {"accuracy_mean", "accuracy_std", "f1_macro_mean", "f1_macro_std"} <= set(metrics[0])
    assert len(records) == 7
    assert records[0]["seed"] == ""
    assert [record["seed"] for record in records[1:]] == ["42", "42", "42", "43", "43", "43"]
    assert len(list(output_dir.glob("confusion_*.png"))) == 3
    assert (output_dir / "courbes_cnn.png").is_file()


def test_record_journal_migrates_header_without_rows(tmp_path):
    journal_path = tmp_path / "journal_experiences.csv"
    old_fields = ("timestamp_utc", "modele", "accuracy")
    with journal_path.open("w", newline="", encoding="utf-8") as journal:
        csv.writer(journal).writerow(old_fields)

    experiment._record_journal(journal_path, [])

    with journal_path.open(newline="", encoding="utf-8") as journal:
        reader = csv.DictReader(journal)
        assert reader.fieldnames == list(experiment.JOURNAL_FIELDS)
        assert list(reader) == []
