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
    a=1.9532048993697146,
    b=3.089966721743309,
    mu=0.024609101445355497,
    sigma=0.02486533927106159,
    date_debut_calibration="1960-01-01",
    date_fin_calibration="2026-02-01",
    source="data/raw/s_and_p_500.csv (Date/SP500)",
)
