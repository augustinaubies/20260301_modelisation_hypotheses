from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px


@dataclass(slots=True)
class ModeleGaussienIID:
    mu: float
    sigma: float

    @classmethod
    def calibrer(cls, serie: pd.Series) -> "ModeleGaussienIID":
        return cls(mu=float(serie.mean()), sigma=float(serie.std(ddof=1)))

    def simuler(self, n_periodes: int, n_paths: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(loc=self.mu, scale=self.sigma, size=(n_paths, n_periodes))


@dataclass(slots=True)
class ModeleAR1:
    alpha: float
    beta: float
    sigma_epsilon: float

    @classmethod
    def calibrer(cls, serie: pd.Series) -> "ModeleAR1":
        y = serie.to_numpy(dtype=float)
        x_tm1 = y[:-1]
        y_t = y[1:]
        x_design = np.column_stack([np.ones_like(x_tm1), x_tm1])
        alpha, beta = np.linalg.lstsq(x_design, y_t, rcond=None)[0]
        residus = y_t - (alpha + beta * x_tm1)
        sigma_epsilon = float(np.std(residus, ddof=1))
        return cls(alpha=float(alpha), beta=float(beta), sigma_epsilon=sigma_epsilon)

    def simuler(
        self,
        n_periodes: int,
        n_paths: int,
        x0: float,
        seed: int | None = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        simulations = np.zeros((n_paths, n_periodes), dtype=float)
        simulations[:, 0] = x0

        for t in range(1, n_periodes):
            bruits = rng.normal(loc=0.0, scale=self.sigma_epsilon, size=n_paths)
            simulations[:, t] = self.alpha + self.beta * simulations[:, t - 1] + bruits

        return simulations


def charger_et_preparer_serie(
    chemin_csv: str,
    colonne_date: str,
    colonne_niveau: str,
    frequence: str = "MS",
) -> pd.Series:
    df = pd.read_csv(chemin_csv)
    if colonne_date not in df.columns or colonne_niveau not in df.columns:
        raise ValueError("Colonnes introuvables dans le fichier d'entrée.")

    df = df[[colonne_date, colonne_niveau]].copy()
    df[colonne_date] = pd.to_datetime(df[colonne_date], errors="coerce")
    df[colonne_niveau] = pd.to_numeric(df[colonne_niveau], errors="coerce")
    df = df.dropna().sort_values(colonne_date)

    if (df[colonne_niveau] <= 0).any():
        raise ValueError("La série de niveau doit être strictement positive pour le log-return.")

    df = df.set_index(colonne_date).asfreq(frequence)
    df[colonne_niveau] = df[colonne_niveau].interpolate(method="time")

    serie_retours = np.log(df[colonne_niveau]).diff().dropna()
    serie_retours.name = "retour_log"
    return serie_retours


def calculer_metriques(serie: np.ndarray) -> dict[str, float]:
    s = pd.Series(serie)
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)),
        "acf1": float(s.autocorr(lag=1)),
        "skew": float(s.skew()),
        "kurtosis": float(s.kurtosis()),
    }


def comparer_strategies(
    serie_historique: pd.Series,
    n_paths: int,
    seed: int | None = None,
) -> tuple[pd.DataFrame, str]:
    n_periodes = len(serie_historique)
    metriques_hist = calculer_metriques(serie_historique.to_numpy())

    modele_iid = ModeleGaussienIID.calibrer(serie_historique)
    sims_iid = modele_iid.simuler(n_periodes=n_periodes, n_paths=n_paths, seed=seed)

    modele_ar1 = ModeleAR1.calibrer(serie_historique)
    sims_ar1 = modele_ar1.simuler(
        n_periodes=n_periodes,
        n_paths=n_paths,
        x0=float(serie_historique.iloc[0]),
        seed=None if seed is None else seed + 1,
    )

    lignes: list[dict[str, float | str]] = []
    for nom_modele, sims in {"gaussien_iid": sims_iid, "ar1_bruit_colore": sims_ar1}.items():
        metriques_sims = pd.DataFrame([calculer_metriques(path) for path in sims])
        ecarts = {
            metrique: float(abs(metriques_sims[metrique].mean() - valeur_hist))
            for metrique, valeur_hist in metriques_hist.items()
        }
        score = float(np.mean(list(ecarts.values())))
        lignes.append({"modele": nom_modele, "score_fidelite": score, **ecarts})

    resultats = pd.DataFrame(lignes).sort_values("score_fidelite", ascending=True)
    meilleur_modele = str(resultats.iloc[0]["modele"])
    return resultats, meilleur_modele


def executer_pipeline_univariee(
    chemin_csv: str,
    colonne_date: str,
    colonne_niveau: str,
    dossier_sortie: str,
    n_paths: int = 1000,
    seed: int | None = 42,
) -> tuple[pd.DataFrame, str, Path]:
    serie = charger_et_preparer_serie(
        chemin_csv=chemin_csv,
        colonne_date=colonne_date,
        colonne_niveau=colonne_niveau,
    )
    resultats, meilleur_modele = comparer_strategies(serie_historique=serie, n_paths=n_paths, seed=seed)

    sortie = Path(dossier_sortie)
    sortie.mkdir(parents=True, exist_ok=True)

    serie.to_csv(sortie / "serie_pretraitee_retours_log.csv", header=True)
    resultats.to_csv(sortie / "scores_modeles.csv", index=False)

    fig = px.bar(
        resultats,
        x="modele",
        y="score_fidelite",
        title="Comparaison de fidélité des stratégies de modélisation",
        text_auto=".4f",
    )
    fig.update_layout(yaxis_title="Score (plus faible = meilleur rejeu)")
    figure_path = sortie / "comparaison_fidelite.html"
    fig.write_html(figure_path)

    (sortie / "conclusion.txt").write_text(
        "\n".join(
            [
                "Pipeline d'identification univariée terminée.",
                f"Variable modélisée: {colonne_niveau}",
                f"Modèle retenu (score minimal): {meilleur_modele}",
            ]
        ),
        encoding="utf-8",
    )

    return resultats, meilleur_modele, figure_path
