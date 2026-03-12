from __future__ import annotations

import numpy as np

from modelisation_macro.bourse import generer_trajectoires_bourse


def test_generer_trajectoires_bourse_retourne_matrice_mois_x_mc() -> None:
    sorties = generer_trajectoires_bourse(
        date_debut="2020-01-01",
        date_fin="2020-06-01",
        n_monte_carlo=50,
        seed=123,
    )
    assert sorties.shape == (6, 50)
    assert (sorties > 0).all()


def test_generer_trajectoires_bourse_seed_reproductible() -> None:
    a = generer_trajectoires_bourse("2021-01-01", "2021-03-01", 10, seed=42)
    b = generer_trajectoires_bourse("2021-01-01", "2021-03-01", 10, seed=42)
    assert np.allclose(a, b)
