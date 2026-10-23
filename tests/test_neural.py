import numpy as np
import torch

from har_activity.neural import (
    HARConvNet,
    _split_by_subject,
    _standardize,
    train_and_predict,
)


def test_standardize_uses_training_statistics_and_handles_constant_channel():
    train = np.array(
        [
            [[1.0, 3.0], [10.0, 10.0]],
            [[5.0, 7.0], [10.0, 10.0]],
        ],
        dtype=np.float32,
    )
    other = np.array([[[5.0, 5.0], [12.0, 12.0]]], dtype=np.float32)

    normalized_train, normalized_other = _standardize(train, other)

    np.testing.assert_allclose(normalized_train.mean(axis=(0, 2)), [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(normalized_train.std(axis=(0, 2)), [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(
        normalized_other[0, :, 0],
        [1.0 / np.sqrt(5.0), 2.0],
        atol=1e-6,
    )


def test_har_convnet_returns_one_score_per_activity():
    model = HARConvNet(channel_count=9, class_count=6)

    predictions = model(torch.randn(4, 9, 128))

    assert predictions.shape == (4, 6)


def test_validation_split_keeps_subjects_separate():
    signals = np.zeros((24, 9, 128), dtype=np.float32)
    labels = np.tile(np.arange(6), 4)
    subjects = np.repeat(np.arange(4), 6)

    train_indices, validation_indices = _split_by_subject(signals, labels, subjects, seed=42)

    assert set(subjects[train_indices]).isdisjoint(set(subjects[validation_indices]))


def test_train_and_predict_runs_two_epochs_on_synthetic_data():
    rng = np.random.default_rng(7)
    labels = np.tile(np.arange(6), 4)
    subjects = np.repeat(np.arange(4), 6)
    signals = rng.normal(size=(24, 9, 128)).astype(np.float32)
    test_signals = rng.normal(size=(6, 9, 128)).astype(np.float32)

    predictions, best_epoch, history = train_and_predict(
        signals,
        labels,
        subjects,
        test_signals,
        epochs=2,
        batch_size=32,
        patience=2,
        seed=42,
    )

    assert predictions.shape == (6,)
    assert np.issubdtype(predictions.dtype, np.integer)
    assert 1 <= best_epoch <= 2
    assert len(history["train_loss"]) == len(history["validation_accuracy"])
    assert len(history["train_loss"]) <= 2
