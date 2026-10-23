"""Téléchargement et chargement de UCI HAR."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import numpy as np

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)
NESTED_ARCHIVE_NAME = "UCI HAR Dataset.zip"
ACTIVITIES = (
    "Marcher",
    "Monter les escaliers",
    "Descendre les escaliers",
    "S’asseoir",
    "Se tenir debout",
    "S’allonger",
)
SIGNALS = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)
WINDOW_LENGTH = 128
TRAIN_FEATURES_PATH = "train/X_train.txt"
REQUIRED_DATASET_FILES = (
    TRAIN_FEATURES_PATH,
    "train/y_train.txt",
    "train/subject_train.txt",
    "test/X_test.txt",
    "test/y_test.txt",
    "test/subject_test.txt",
    *(
        f"{split}/Inertial Signals/{signal}_{split}.txt"
        for split in ("train", "test")
        for signal in SIGNALS
    ),
)


@dataclass(frozen=True)
class HARData:
    """Partitions officielles du jeu de données et signaux bruts."""

    train_features: np.ndarray
    train_labels: np.ndarray
    train_subjects: np.ndarray
    train_raw: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray
    test_subjects: np.ndarray
    test_raw: np.ndarray


def download_dataset(archive_path: Path) -> Path:
    """Télécharge l'archive UCI HAR si elle n'est pas déjà présente."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.is_file():
        return archive_path

    temporary_path = archive_path.with_suffix(archive_path.suffix + ".download")
    try:
        with urlopen(DATASET_URL, timeout=60) as response, temporary_path.open("wb") as target:
            target.write(response.read())
        with ZipFile(temporary_path) as archive:
            with _dataset_archive(archive):
                pass
        temporary_path.replace(archive_path)
    except (OSError, BadZipFile, ValueError):
        temporary_path.unlink(missing_ok=True)
        raise

    return archive_path


@contextmanager
def _dataset_archive(archive: ZipFile) -> Iterator[ZipFile]:
    """Ouvre les données UCI, directement ou depuis le ZIP imbriqué officiel."""
    try:
        _dataset_prefix(archive)
    except ValueError as outer_error:
        nested_name = next(
            (
                name
                for name in archive.namelist()
                if Path(name).name.lower() == NESTED_ARCHIVE_NAME.lower()
            ),
            None,
        )
        if nested_name is None:
            raise outer_error
        try:
            with archive.open(nested_name) as nested_file, ZipFile(nested_file) as nested_archive:
                _dataset_prefix(nested_archive)
                yield nested_archive
        except BadZipFile as error:
            raise ValueError("L’archive UCI HAR imbriquée est invalide ou incomplète.") from error
        return

    yield archive


def _dataset_prefix(archive: ZipFile) -> str:
    """Repère les fichiers UCI HAR, qu'ils soient à la racine ou dans un dossier."""
    names = set(archive.namelist())
    prefixes = (
        name[: -len(TRAIN_FEATURES_PATH)]
        for name in names
        if name.endswith(TRAIN_FEATURES_PATH)
    )
    for prefix in prefixes:
        if all(f"{prefix}{path}" in names for path in REQUIRED_DATASET_FILES):
            return prefix
    raise ValueError(
        "L’archive ne contient pas les fichiers UCI HAR attendus "
        "(train/test et signaux inertiels). Vérifiez le téléchargement."
    )


def _read_array(archive: ZipFile, prefix: str, relative_path: str, dtype: type) -> np.ndarray:
    path = f"{prefix}{relative_path}"
    try:
        with archive.open(path) as file:
            array = np.loadtxt(file, dtype=dtype, ndmin=2)
    except KeyError as error:
        raise ValueError(f"Fichier manquant dans l’archive UCI HAR : {path}") from error
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"Le fichier UCI HAR est vide ou contient des valeurs non finies : {path}")
    return array


def _load_raw_signals(
    archive: ZipFile, prefix: str, split: str, expected_rows: int
) -> np.ndarray:
    channels = []
    for signal_name in SIGNALS:
        relative_path = f"{split}/Inertial Signals/{signal_name}_{split}.txt"
        values = _read_array(archive, prefix, relative_path, np.float32)
        if values.shape != (expected_rows, WINDOW_LENGTH):
            raise ValueError(
                f"Forme inattendue pour {relative_path}: {values.shape}; "
                f"attendu ({expected_rows}, {WINDOW_LENGTH})."
            )
        channels.append(values)
    return np.stack(channels, axis=1)


def load_dataset(archive_path: Path) -> HARData:
    """Lit caractéristiques et signaux bruts depuis l'archive ZIP UCI HAR."""
    try:
        archive = ZipFile(archive_path)
    except (OSError, BadZipFile) as error:
        raise ValueError(f"Impossible de lire l’archive UCI HAR : {archive_path}") from error

    with archive, _dataset_archive(archive) as dataset_archive:
        prefix = _dataset_prefix(dataset_archive)
        partitions = {}
        for split in ("train", "test"):
            features = _read_array(dataset_archive, prefix, f"{split}/X_{split}.txt", np.float32)
            labels = _read_array(dataset_archive, prefix, f"{split}/y_{split}.txt", np.int64).ravel()
            subjects = _read_array(
                dataset_archive, prefix, f"{split}/subject_{split}.txt", np.int64
            ).ravel()
            if features.shape[0] != labels.shape[0]:
                raise ValueError(f"Les caractéristiques et les labels de {split} n’ont pas la même taille.")
            if features.shape[0] != subjects.shape[0]:
                raise ValueError(f"Les caractéristiques et les sujets de {split} n’ont pas la même taille.")
            if not np.isin(labels, np.arange(1, len(ACTIVITIES) + 1)).all():
                raise ValueError(f"Labels inattendus dans la partition {split}.")
            raw = _load_raw_signals(dataset_archive, prefix, split, features.shape[0])
            partitions[split] = (features, labels - 1, subjects, raw)

    train_features, train_labels, train_subjects, train_raw = partitions["train"]
    test_features, test_labels, test_subjects, test_raw = partitions["test"]
    return HARData(
        train_features,
        train_labels,
        train_subjects,
        train_raw,
        test_features,
        test_labels,
        test_subjects,
        test_raw,
    )
