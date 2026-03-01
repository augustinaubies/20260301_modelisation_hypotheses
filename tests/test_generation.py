import numpy as np

from modelisation_macro.generation import (
    generer_trajectoire_statique_corrigee,
    generer_trajectoire_var1,
)
from modelisation_macro.io_yaml import charger_parametres
from modelisation_macro.types import ParametresModeleStatique, ParametresModeleVAR1


def test_generation_statique_dimensions_et_seed() -> None:
    params = charger_parametres("config/exemple_statique.yaml")
    assert isinstance(params, ParametresModeleStatique)

    traj1 = generer_trajectoire_statique_corrigee(params, horizon_mois=24, seed=123).vers_dataframe()
    traj2 = generer_trajectoire_statique_corrigee(params, horizon_mois=24, seed=123).vers_dataframe()

    assert traj1.shape == (24, 6)
    assert np.allclose(traj1.values, traj2.values)


def test_generation_var1_dimensions() -> None:
    params = charger_parametres("config/exemple_var1.yaml")
    assert isinstance(params, ParametresModeleVAR1)

    traj = generer_trajectoire_var1(params, horizon_mois=18, seed=7).vers_dataframe()
    assert traj.shape == (18, 6)


def test_sigma_var1_psd() -> None:
    params = charger_parametres("config/exemple_var1.yaml")
    assert isinstance(params, ParametresModeleVAR1)

    sigma = np.array(params.Sigma, dtype=float)
    min_eig = np.min(np.linalg.eigvalsh(sigma))
    assert min_eig >= -1e-10
