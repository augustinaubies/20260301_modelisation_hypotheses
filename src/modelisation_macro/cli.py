from __future__ import annotations

import argparse
from pathlib import Path

from .generation import generer_trajectoire_statique_corrigee, generer_trajectoire_var1
from .io_yaml import charger_parametres
from .types import ParametresModeleStatique, ParametresModeleVAR1



def construire_parseur() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(description="Génération de trajectoires macro (V1).")
    parseur.add_argument("--config", required=True, help="Chemin YAML des paramètres modèle")
    parseur.add_argument("--horizon-mois", type=int, required=True, help="Horizon mensuel")
    parseur.add_argument(
        "--sortie",
        default="outputs/trajectoire.csv",
        help="Chemin de sortie CSV pour debug/validation",
    )
    parseur.add_argument("--seed", type=int, default=None, help="Seed aléatoire")
    parseur.add_argument("--date-depart", default="2025-01", help="Date de départ (YYYY-MM)")
    return parseur



def main() -> None:
    args = construire_parseur().parse_args()
    parametres = charger_parametres(args.config)

    if isinstance(parametres, ParametresModeleStatique):
        trajectoire = generer_trajectoire_statique_corrigee(
            parametres=parametres,
            horizon_mois=args.horizon_mois,
            seed=args.seed,
            date_depart=args.date_depart,
        )
    elif isinstance(parametres, ParametresModeleVAR1):
        trajectoire = generer_trajectoire_var1(
            parametres=parametres,
            horizon_mois=args.horizon_mois,
            seed=args.seed,
            date_depart=args.date_depart,
        )
    else:
        raise TypeError("Type de paramètres non supporté.")

    sortie = Path(args.sortie)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    trajectoire.sauvegarder_csv(str(sortie))
    print(f"Trajectoire générée : {sortie}")
    print(trajectoire.donnees.head(3).to_string())


if __name__ == "__main__":
    main()
