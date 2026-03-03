from __future__ import annotations

import pandas as pd

from modelisation_macro.identification.univariee import (
    calculer_metriques,
    comparer_strategies,
    detecter_meilleure_date_depart,
    construire_figure_rejeu,
    construire_html_rapport,
)


def test_calculer_metriques_retourne_cles_attendues() -> None:
    serie = [0.01, -0.02, 0.03, 0.01, -0.01]
    metriques = calculer_metriques(serie)
    assert set(metriques.keys()) == {"mean", "std", "acf1", "skew", "kurtosis"}


def test_comparer_strategies_retourne_modele_existant() -> None:
    serie = pd.Series([0.01, 0.0, -0.01, 0.02, 0.01, -0.02, 0.0, 0.01, 0.015, -0.005, 0.008, -0.012])
    resultats, meilleur_modele, simulations = comparer_strategies(serie_historique=serie, n_paths=20, seed=123)
    modeles = set(resultats["modele"].tolist())
    assert modeles == {
        "gaussien_iid",
        "ar1_bruit_colore",
        "student_t_iid",
        "volatilite_ewma",
        "markov_switching_2_regimes",
    }
    assert meilleur_modele in modeles
    assert set(simulations.keys()) == modeles


def test_construire_figure_rejeu_cree_2_sous_graphes_attendus() -> None:
    index = pd.date_range("2020-01-01", periods=6, freq="MS")
    serie = pd.Series([0.01, -0.02, 0.01, 0.0, 0.015, -0.01], index=index)
    simulations = {"gaussien_iid": [[0.0, 0.01, -0.01, 0.0, 0.0, 0.01]]}

    fig = construire_figure_rejeu(serie_historique=serie, simulations_par_modele=simulations)

    assert "intégrale des variations log" in fig.layout.annotations[0].text
    assert "couloir 95%" in fig.layout.annotations[1].text


def test_construire_html_rapport_separe_bien_texte_et_graphiques() -> None:
    index = pd.date_range("2020-01-01", periods=4, freq="MS")
    serie = pd.Series([0.01, -0.01, 0.02, -0.005], index=index)
    simulations = {"gaussien_iid": [[0.0, 0.01, -0.01, 0.0]]}
    fig = construire_figure_rejeu(serie_historique=serie, simulations_par_modele=simulations)
    resultats = pd.DataFrame([{"modele": "gaussien_iid", "score_fidelite": 0.1}])
    resultats_dates = pd.DataFrame(
        [
            {
                "date_depart": pd.Timestamp("2020-01-01"),
                "score_fidelite": 0.05,
                "n_observations": 48,
            }
        ]
    )

    rapport = construire_html_rapport(
        fig=fig,
        resultats=resultats,
        meilleur_modele="gaussien_iid",
        resultats_dates=resultats_dates,
        meilleure_date=pd.Timestamp("2020-01-01"),
        modele_date="volatilite_ewma",
    )

    assert "<section>" in rapport
    assert "class=\"plot-container\"" in rapport
    assert "Meilleur modèle selon score global" in rapport
    assert "Meilleure date de départ" in rapport


def test_detecter_meilleure_date_depart_retourne_une_date_existante() -> None:
    index = pd.date_range("2010-01-01", periods=72, freq="MS")
    serie = pd.Series([0.002 * ((i % 6) - 2) for i in range(72)], index=index)

    resultats, meilleure_date = detecter_meilleure_date_depart(
        serie_historique=serie,
        modele_a_tester="gaussien_iid",
        fenetre_min_mois=60,
        n_paths=20,
        seed=123,
    )

    assert not resultats.empty
    assert meilleure_date in serie.index
