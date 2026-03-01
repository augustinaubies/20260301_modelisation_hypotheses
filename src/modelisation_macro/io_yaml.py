from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

from .types import ParametresModeleStatique, ParametresModeleVAR1

TypeParametres = Union[ParametresModeleStatique, ParametresModeleVAR1]



def charger_parametres(path: str | Path) -> TypeParametres:
    """Charge un fichier YAML de paramètres modèle (schema versionné)."""

    chemin = Path(path)
    contenu = yaml.safe_load(chemin.read_text(encoding="utf-8"))

    if not isinstance(contenu, dict):
        raise ValueError("Le YAML de paramètres doit contenir un objet racine.")

    version_schema = contenu.get("version_schema")
    if version_schema != 1:
        raise ValueError(f"Version de schéma non supportée: {version_schema!r}")

    type_modele = contenu.get("type_modele")
    if type_modele == "statique":
        return ParametresModeleStatique.model_validate(contenu)
    if type_modele == "var1":
        return ParametresModeleVAR1.model_validate(contenu)

    raise ValueError(f"Type de modèle non reconnu: {type_modele!r}")



def sauvegarder_parametres(path: str | Path, parametres: TypeParametres) -> None:
    """Sauvegarde des paramètres modèle au format YAML stable (V1)."""

    chemin = Path(path)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    contenu = parametres.model_dump(mode="python")
    contenu["version_schema"] = 1

    entete = (
        "# Paramètres de modèle macro (V1)\n"
        "# version_schema: 1\n"
        "# Toutes les valeurs sont des taux mensuels.\n"
    )
    yaml_dump = yaml.safe_dump(contenu, sort_keys=False, allow_unicode=True)
    chemin.write_text(entete + yaml_dump, encoding="utf-8")
