from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .variables import VARIABLES_CANONIQUES


class ParametresModeleStatique(BaseModel):
    """Paramètres du modèle statique corrélé gaussien."""

    model_config = ConfigDict(extra="forbid")

    version_schema: int = 1
    type_modele: str = "statique"
    variables: list[str] = Field(default_factory=lambda: list(VARIABLES_CANONIQUES))
    pas_temps: str = "mensuel"
    moyennes: list[float]
    volatilites: list[float]
    correlation: list[list[float]]

    @field_validator("variables")
    @classmethod
    def verifier_variables(cls, variables: list[str]) -> list[str]:
        if len(variables) == 0:
            raise ValueError("La liste de variables ne peut pas être vide.")
        return variables

    @model_validator(mode="after")
    def verifier_dimensions(self) -> "ParametresModeleStatique":
        n = len(self.variables)
        if len(self.moyennes) != n:
            raise ValueError("`moyennes` doit avoir la même dimension que `variables`.")
        if len(self.volatilites) != n:
            raise ValueError("`volatilites` doit avoir la même dimension que `variables`.")
        if len(self.correlation) != n or any(len(ligne) != n for ligne in self.correlation):
            raise ValueError("`correlation` doit être une matrice carrée de taille n x n.")
        return self


class EtatModeleVAR1(BaseModel):
    """État initial d'un VAR(1)."""

    model_config = ConfigDict(extra="forbid")

    x0: list[float]


class ParametresModeleVAR1(BaseModel):
    """Paramètres du modèle VAR(1) : X_t = c + A X_(t-1) + eps_t."""

    model_config = ConfigDict(extra="forbid")

    version_schema: int = 1
    type_modele: str = "var1"
    variables: list[str] = Field(default_factory=lambda: list(VARIABLES_CANONIQUES))
    pas_temps: str = "mensuel"
    c: list[float]
    A: list[list[float]]
    Sigma: list[list[float]]
    etat_initial: EtatModeleVAR1

    @model_validator(mode="after")
    def verifier_dimensions(self) -> "ParametresModeleVAR1":
        n = len(self.variables)
        if len(self.c) != n:
            raise ValueError("`c` doit être de taille n.")
        if len(self.A) != n or any(len(ligne) != n for ligne in self.A):
            raise ValueError("`A` doit être une matrice n x n.")
        if len(self.Sigma) != n or any(len(ligne) != n for ligne in self.Sigma):
            raise ValueError("`Sigma` doit être une matrice n x n.")
        if len(self.etat_initial.x0) != n:
            raise ValueError("`etat_initial.x0` doit être de taille n.")
        return self


@dataclass(slots=True)
class TrajectoireMacro:
    """Wrapper léger autour d'une trajectoire macro en DataFrame."""

    donnees: pd.DataFrame

    def vers_dataframe(self) -> pd.DataFrame:
        return self.donnees.copy()

    def sauvegarder_csv(self, chemin: str) -> None:
        self.donnees.to_csv(chemin, index=True)
