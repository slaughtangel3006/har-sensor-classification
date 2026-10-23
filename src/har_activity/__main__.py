"""Point d'entrée de la comparaison UCI HAR."""

from __future__ import annotations

import argparse
from pathlib import Path

from har_activity.data import download_dataset, load_dataset
from har_activity.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare des modèles classiques et un CNN sur UCI HAR."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Dossier de l’archive UCI HAR.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Dossier des résultats.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Nombre maximal d’époques pour le CNN.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Taille des lots du CNN.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=6,
        help="Époques sans progrès avant l’arrêt anticipé.",
    )
    seed_options = parser.add_mutually_exclusive_group()
    seed_options.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Graines à comparer (par défaut : 42 43 44 45 46).",
    )
    seed_options.add_argument(
        "--seed",
        type=int,
        help="Une seule graine, conservée pour compatibilité avec les anciennes commandes.",
    )
    args = parser.parse_args()
    if args.seeds is not None:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [42, 43, 44, 45, 46]

    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("--epochs, --batch-size et --patience doivent être supérieurs à zéro.")
    if len(set(seeds)) != len(seeds):
        parser.error("Chaque graine doit être unique.")

    archive_path = download_dataset(args.data_dir / "UCI_HAR.zip")
    dataset = load_dataset(archive_path)
    print("Données prêtes. Lancement de la comparaison, courage aux petits calculs ! 🔬")
    print(f"Graines utilisées : {', '.join(map(str, seeds))}\n")
    run_experiment(dataset, args.output_dir, args.epochs, args.batch_size, args.patience, seeds)
    print(f"\nExpérience terminée ✨ Résultats et journal : {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
