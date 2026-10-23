from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pytest

from har_activity.data import ACTIVITIES, SIGNALS, load_dataset


def _matrix(values: np.ndarray) -> str:
    return "\n".join(" ".join(str(value) for value in row) for row in values) + "\n"


def _write_dataset(archive: ZipFile, prefix: str) -> None:
    for split, labels in (("train", [1, 2]), ("test", [3])):
        features = np.arange(len(labels) * 3, dtype=np.float32).reshape(len(labels), 3)
        root = f"{prefix}{split}"
        archive.writestr(f"{root}/X_{split}.txt", _matrix(features))
        archive.writestr(f"{root}/y_{split}.txt", _matrix(np.asarray(labels).reshape(-1, 1)))
        subjects = np.arange(1, len(labels) + 1, dtype=np.int64).reshape(-1, 1)
        archive.writestr(f"{root}/subject_{split}.txt", _matrix(subjects))
        for channel_index, signal in enumerate(SIGNALS):
            raw = np.full((len(labels), 128), channel_index, dtype=np.float32)
            archive.writestr(
                f"{root}/Inertial Signals/{signal}_{split}.txt",
                _matrix(raw),
            )


@pytest.mark.parametrize(
    ("prefix", "nested"),
    (("", False), ("UCI HAR Dataset/", False), ("UCI HAR Dataset/", True)),
)
def test_load_dataset_reads_official_splits_and_raw_channels(tmp_path, prefix, nested):
    archive_path = tmp_path / "mini-har.zip"
    if nested:
        nested_content = BytesIO()
        with ZipFile(nested_content, "w") as dataset_archive:
            _write_dataset(dataset_archive, prefix)
        with ZipFile(archive_path, "w") as archive:
            archive.writestr("UCI HAR Dataset.zip", nested_content.getvalue())
    else:
        with ZipFile(archive_path, "w") as archive:
            _write_dataset(archive, prefix)

    data = load_dataset(archive_path)

    assert data.train_features.shape == (2, 3)
    assert data.train_labels.tolist() == [0, 1]
    assert data.train_subjects.tolist() == [1, 2]
    assert data.test_labels.tolist() == [2]
    assert data.test_subjects.tolist() == [1]
    assert data.train_raw.shape == (2, len(SIGNALS), 128)
    assert data.test_raw[0, :, 0].tolist() == list(range(len(SIGNALS)))
    assert len(ACTIVITIES) == 6
