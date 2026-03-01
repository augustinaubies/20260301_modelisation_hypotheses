from __future__ import annotations

from enum import StrEnum


class VariableMacro(StrEnum):
    """Variables macro canoniques pour la V1."""

    INFLATION = "inflation"
    CROISSANCE_SALAIRE = "croissance_salaire"
    INDEXATION_LOYERS = "indexation_loyers"
    REVALORISATION_IMMOBILIERE = "revalorisation_immobiliere"
    RENDEMENT_BOURSE = "rendement_bourse"
    TAUX_CREDIT = "taux_credit"


VARIABLES_CANONIQUES: tuple[str, ...] = tuple(var.value for var in VariableMacro)


def normaliser_rendement_bourse_en_log_return(rendement_simple: float) -> float:
    """Convertit un rendement simple mensuel en log-return (approximation simple V1)."""

    # TODO(V2): gérer précisément les cas extrêmes et conventions annualisées.
    if rendement_simple <= -1.0:
        raise ValueError("Un rendement simple <= -100% n'est pas convertible en log-return.")
    import math

    return math.log1p(rendement_simple)


def denormaliser_log_return_en_rendement_simple(log_return: float) -> float:
    """Convertit un log-return mensuel en rendement simple."""

    import math

    return math.expm1(log_return)
