from __future__ import annotations

import pandas as pd

from modelisation_macro.identification.univariee import calculer_metriques, comparer_strategies


def test_calculer_metriques_retourne_cles_attendues() -> None:
    serie = [0.01, -0.02, 0.03, 0.01, -0.01]
    metriques = calculer_metriques(serie)
    assert set(metriques.keys()) == {"mean", "std", "acf1", "skew", "kurtosis"}


def test_comparer_strategies_retourne_modele_existant() -> None:
    serie = pd.Series([0.01, 0.0, -0.01, 0.02, 0.01, -0.02, 0.0, 0.01])
    resultats, meilleur_modele = comparer_strategies(serie_historique=serie, n_paths=20, seed=123)
    modeles = set(resultats["modele"].tolist())
    assert modeles == {"gaussien_iid", "ar1_bruit_colore"}
    assert meilleur_modele in modeles
