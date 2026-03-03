#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from modelisation_macro.identification import executer_pipeline_univariee


def construire_parseur() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline univariée: chargement, modélisation, Monte Carlo, synthèse Plotly."
    )
    parser.add_argument("--input-csv", required=True, help="Chemin du CSV d'entrée")
    parser.add_argument("--date-column", default="Date", help="Nom de la colonne date")
    parser.add_argument(
        "--value-column",
        required=True,
        help="Nom de la colonne de niveau à transformer en log-return",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/task2_bourse",
        help="Dossier de sortie de la pipeline",
    )
    parser.add_argument("--n-paths", type=int, default=1000, help="Nombre de tirages Monte Carlo")
    parser.add_argument("--seed", type=int, default=42, help="Seed aléatoire")
    return parser


def main() -> None:
    args = construire_parseur().parse_args()
    resultats, meilleur_modele, figure_path = executer_pipeline_univariee(
        chemin_csv=args.input_csv,
        colonne_date=args.date_column,
        colonne_niveau=args.value_column,
        dossier_sortie=args.output_dir,
        n_paths=args.n_paths,
        seed=args.seed,
    )
    print("Pipeline terminée.")
    print(resultats.to_string(index=False))
    print(f"Modèle recommandé: {meilleur_modele}")
    print(f"Graphique de synthèse: {figure_path}")


if __name__ == "__main__":
    main()
