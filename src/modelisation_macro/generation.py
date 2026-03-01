from __future__ import annotations

import numpy as np
import pandas as pd

from .types import ParametresModeleStatique, ParametresModeleVAR1, TrajectoireMacro



def _index_mensuel(horizon_mois: int, date_depart: str = "2025-01") -> pd.DatetimeIndex:
    return pd.date_range(start=f"{date_depart}-01", periods=horizon_mois, freq="MS")



def _verifier_psd(matrice: np.ndarray, tolerance: float = 1e-10) -> None:
    valeurs_propres = np.linalg.eigvalsh(matrice)
    if np.min(valeurs_propres) < -tolerance:
        raise ValueError("La matrice fournie n'est pas semi-définie positive (PSD).")



def generer_trajectoire_statique_corrigee(
    parametres: ParametresModeleStatique,
    horizon_mois: int,
    seed: int | None = None,
    date_depart: str = "2025-01",
) -> TrajectoireMacro:
    """Génère une trajectoire mensuelle par gaussienne corrélée (i.i.d. dans le temps)."""

    generateur = np.random.default_rng(seed)
    moyennes = np.array(parametres.moyennes, dtype=float)
    volatilites = np.array(parametres.volatilites, dtype=float)
    correlation = np.array(parametres.correlation, dtype=float)

    _verifier_psd(correlation)

    matrice_vol = np.diag(volatilites)
    covariance = matrice_vol @ correlation @ matrice_vol
    _verifier_psd(covariance)

    echantillons = generateur.multivariate_normal(
        mean=moyennes,
        cov=covariance,
        size=horizon_mois,
        check_valid="raise",
    )

    df = pd.DataFrame(
        echantillons,
        index=_index_mensuel(horizon_mois=horizon_mois, date_depart=date_depart),
        columns=parametres.variables,
    )
    return TrajectoireMacro(donnees=df)



def generer_trajectoire_var1(
    parametres: ParametresModeleVAR1,
    horizon_mois: int,
    seed: int | None = None,
    date_depart: str = "2025-01",
) -> TrajectoireMacro:
    """Génère une trajectoire mensuelle via VAR(1)."""

    generateur = np.random.default_rng(seed)
    c = np.array(parametres.c, dtype=float)
    A = np.array(parametres.A, dtype=float)
    sigma = np.array(parametres.Sigma, dtype=float)
    x_t = np.array(parametres.etat_initial.x0, dtype=float)

    _verifier_psd(sigma)

    trajectoire = np.zeros((horizon_mois, len(parametres.variables)), dtype=float)

    for t in range(horizon_mois):
        epsilon_t = generateur.multivariate_normal(
            mean=np.zeros(len(c), dtype=float),
            cov=sigma,
            check_valid="raise",
        )
        x_t = c + A @ x_t + epsilon_t
        trajectoire[t, :] = x_t

    df = pd.DataFrame(
        trajectoire,
        index=_index_mensuel(horizon_mois=horizon_mois, date_depart=date_depart),
        columns=parametres.variables,
    )
    return TrajectoireMacro(donnees=df)
