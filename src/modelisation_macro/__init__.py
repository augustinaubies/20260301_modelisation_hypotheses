"""Bibliothèque de modélisation macro (V1)."""

from .bourse import generer_trajectoires_bourse
from .generation import generer_trajectoire_statique_corrigee, generer_trajectoire_var1
from .types import (
    EtatModeleVAR1,
    ParametresModeleStatique,
    ParametresModeleVAR1,
    TrajectoireMacro,
)

__all__ = [
    "ParametresModeleStatique",
    "ParametresModeleVAR1",
    "EtatModeleVAR1",
    "TrajectoireMacro",
    "generer_trajectoires_bourse",
    "generer_trajectoire_statique_corrigee",
    "generer_trajectoire_var1",
]
