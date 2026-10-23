"""CNN 1D entraîné directement sur les signaux bruts."""

from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class HARConvNet(nn.Module):
    def __init__(self, channel_count: int, class_count: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channel_count, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(128, class_count))

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(signals))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=(0, 2), keepdims=True)
    scale = train.std(axis=(0, 2), keepdims=True)
    scale[scale == 0] = 1.0
    return (train - mean) / scale, (other - mean) / scale


def _split_by_subject(
    signals: np.ndarray, labels: np.ndarray, subjects: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(subjects) != len(labels):
        raise ValueError("Chaque fenêtre d’entraînement doit avoir un sujet associé.")
    if np.unique(subjects).size < 2:
        raise ValueError("La validation par sujet nécessite au moins deux sujets dans le train.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    return next(splitter.split(signals, labels, groups=subjects))


def _loader(
    signals: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(signals, dtype=np.float32)),
        torch.from_numpy(np.asarray(labels, dtype=np.int64)),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    sample_count = 0
    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(signals), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        sample_count += labels.size(0)
    return total_loss / sample_count


def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for signals, labels in loader:
            predictions = model(signals.to(device)).argmax(dim=1).cpu()
            correct += int((predictions == labels).sum())
            total += labels.numel()
    return correct / total


def _fit_fixed_epochs(
    train_signals: np.ndarray,
    train_labels: np.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> HARConvNet:
    _set_seed(seed)
    model = HARConvNet(train_signals.shape[1], int(train_labels.max()) + 1).to(device)
    loader = _loader(train_signals, train_labels, batch_size, True, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        _train_epoch(model, loader, optimizer, device)
    return model


def train_and_predict(
    train_signals: np.ndarray,
    train_labels: np.ndarray,
    train_subjects: np.ndarray,
    test_signals: np.ndarray,
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
) -> tuple[np.ndarray, int, dict[str, list[float]]]:
    """Choisit les époques sur des sujets distincts, puis réentraîne sur tout le train."""
    _set_seed(seed)
    train_indices, validation_indices = _split_by_subject(
        train_signals, train_labels, train_subjects, seed
    )
    fit_signals, validation_signals = _standardize(
        train_signals[train_indices], train_signals[validation_indices]
    )
    fit_labels = train_labels[train_indices]
    validation_loader = _loader(
        validation_signals, train_labels[validation_indices], batch_size, False, seed
    )
    fit_loader = _loader(fit_signals, fit_labels, batch_size, True, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HARConvNet(train_signals.shape[1], int(train_labels.max()) + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_accuracy = -1.0
    best_epoch = 1
    epochs_without_improvement = 0
    history = {"train_loss": [], "validation_accuracy": []}
    for epoch in range(1, epochs + 1):
        history["train_loss"].append(_train_epoch(model, fit_loader, optimizer, device))
        validation_accuracy = _accuracy(model, validation_loader, device)
        history["validation_accuracy"].append(validation_accuracy)
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    full_train, normalized_test = _standardize(train_signals, test_signals)
    final_model = _fit_fixed_epochs(
        full_train, train_labels, best_epoch, batch_size, seed, device
    )
    test_loader = _loader(normalized_test, np.zeros(len(normalized_test)), batch_size, False, seed)
    final_model.eval()
    predictions = []
    with torch.no_grad():
        for signals, _ in test_loader:
            predictions.extend(final_model(signals.to(device)).argmax(dim=1).cpu().tolist())
    return np.asarray(predictions, dtype=np.int64), best_epoch, history
