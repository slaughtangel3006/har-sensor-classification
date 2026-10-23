"""Modèles classiques sur les caractéristiques déjà extraites par UCI HAR."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_classical_models(seed: int) -> dict:
    """Crée les deux modèles de référence avec prétraitement adapté."""
    return {
        "SVM": make_pipeline(StandardScaler(), SVC(C=10, kernel="rbf", gamma="scale")),
        "Forêt aléatoire": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }
