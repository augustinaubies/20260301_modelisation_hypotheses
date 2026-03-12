from __future__ import annotations

from datetime import date
from typing import Final

import numpy as np
import pandas as pd
from scipy import stats

from .parametres_bourse import PARAMETRES_BOURSE_V1, ParametresSkewTBourse

FREQUENCE_MENSUELLE: Final[str] = "MS"


def _construire_index_mensuel(date_debut: str, date_fin: str) -> pd.DatetimeIndex:
    debut = pd.Timestamp(date_debut)
    fin = pd.Timestamp(date_fin)
    if fin < debut:
        raise ValueError("date_fin doit être postérieure ou égale à date_debut.")
    return pd.date_range(start=debut, end=fin, freq=FREQUENCE_MENSUELLE)


def generer_trajectoires_bourse(
    date_debut: str | date,
    date_fin: str | date,
    n_monte_carlo: int,
    seed: int | None = None,
    parametres: ParametresSkewTBourse = PARAMETRES_BOURSE_V1,
) -> np.ndarray:
    """Génère une matrice (n_mois, n_mc) de variations multiplicatives mensuelles.

    Exemple de conversion:
    -2% => 0.98
    +0.3% => 1.003
    """
    if n_monte_carlo <= 0:
        raise ValueError("n_monte_carlo doit être strictement positif.")

    index_mensuel = _construire_index_mensuel(str(date_debut), str(date_fin))
    n_mois = len(index_mensuel)
    if n_mois == 0:
        return np.empty((0, n_monte_carlo), dtype=float)

    rng = np.random.default_rng(seed)
    retours_log = stats.jf_skew_t.rvs(
        parametres.a,
        parametres.b,
        loc=parametres.mu,
        scale=max(parametres.sigma, 1e-12),
        size=(n_monte_carlo, n_mois),
        random_state=rng,
    )
    facteurs_multiplicatifs = np.exp(retours_log)
    return np.asarray(facteurs_multiplicatifs.T, dtype=float)


__all__ = ["generer_trajectoires_bourse", "PARAMETRES_BOURSE_V1", "ParametresSkewTBourse"]
