from __future__ import annotations

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
    a=1.8080581014532573,
    b=3.8114291994202674,
    mu=0.03574967377387041,
    sigma=0.01979835060109976,
    date_debut_calibration="2009-08-01",
    date_fin_calibration="2026-02-01",
    source="data/raw/s_and_p_500.csv (Date/SP500)",
)
