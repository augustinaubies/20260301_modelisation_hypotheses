"""Squelette V1 pour calibration VAR(1) depuis historique.

TODO principaux:
- Charger des séries historiques nettoyées (source locale, pas de web en V1).
- Estimer c, A et Sigma (moindres carrés + diagnostics).
- Exporter un YAML `type_modele: var1` versionné.
"""

from __future__ import annotations

from pathlib import Path



def calibrer_var1_depuis_historique(chemin_donnees: Path) -> None:
    """Point d'entrée placeholder pour la future calibration VAR(1)."""

    raise NotImplementedError(
        "Calibration historique non implémentée en V1. "
        "Voir TODO dans ce module pour la V2."
    )
