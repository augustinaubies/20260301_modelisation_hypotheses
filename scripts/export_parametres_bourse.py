#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from modelisation_macro.identification.univariee import (  # noqa: E402
    ModeleSkewTIID,
    charger_et_preparer_serie,
    detecter_meilleure_date_depart,
)


def construire_parseur() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibre la loi skew-t retenue et exporte un fichier de paramètres bourse.")
    parser.add_argument("--input-csv", default="data/raw/s_and_p_500.csv")
    parser.add_argument("--date-column", default="Date")
    parser.add_argument("--value-column", default="SP500")
    parser.add_argument("--output-py", default="src/modelisation_macro/bourse/parametres_bourse.py")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = construire_parseur().parse_args()
    serie = charger_et_preparer_serie(
        chemin_csv=args.input_csv,
        colonne_date=args.date_column,
        colonne_niveau=args.value_column,
    )
    _, meilleure_date = detecter_meilleure_date_depart(
        serie_historique=serie,
        modele_a_tester="skew_t_asymetrique_iid",
        fenetre_min_mois=60,
        pas_mois=6,
        n_paths=120,
        seed=args.seed,
    )

    # meilleure_date = pd.to_datetime("1960-01-01")

    fenetre = serie.loc[meilleure_date:]
    modele = ModeleSkewTIID.calibrer(fenetre)

    contenu = f'''from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParametresSkewTBourse:
    """Paramètres calibrés pour la génération des variations mensuelles boursières."""

    a: float
    b: float
    mu: float
    sigma: float
    date_debut_calibration: str
    date_fin_calibration: str
    source: str


# Paramètres V1 issus de `scripts/export_parametres_bourse.py`.
PARAMETRES_BOURSE_V1 = ParametresSkewTBourse(
    a={modele.a!r},
    b={modele.b!r},
    mu={modele.mu!r},
    sigma={modele.sigma!r},
    date_debut_calibration="{meilleure_date.strftime('%Y-%m-%d')}",
    date_fin_calibration="{fenetre.index[-1].strftime('%Y-%m-%d')}",
    source="{args.input_csv} ({args.date_column}/{args.value_column})",
)
'''

    output = Path(args.output_py)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(contenu, encoding="utf-8")
    print(f"Paramètres exportés: {output}")


if __name__ == "__main__":
    main()
