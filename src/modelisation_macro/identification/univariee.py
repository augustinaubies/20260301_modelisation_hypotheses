from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


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


@dataclass(slots=True)
class ModeleStudentTIID:
    nu: float
    mu: float
    sigma: float

    @classmethod
    def calibrer(cls, serie: pd.Series) -> "ModeleStudentTIID":
        nu, mu, sigma = stats.t.fit(serie.to_numpy(dtype=float))
        return cls(nu=float(nu), mu=float(mu), sigma=float(sigma))

    def simuler(self, n_periodes: int, n_paths: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return stats.t.rvs(
            df=self.nu,
            loc=self.mu,
            scale=self.sigma,
            size=(n_paths, n_periodes),
            random_state=rng,
        )


@dataclass(slots=True)
class ModeleVolatiliteEWMA:
    mu: float
    lambda_vol: float
    sigma0: float

    @classmethod
    def calibrer(cls, serie: pd.Series, lambda_vol: float = 0.94) -> "ModeleVolatiliteEWMA":
        return cls(
            mu=float(serie.mean()),
            lambda_vol=float(lambda_vol),
            sigma0=float(serie.std(ddof=1)),
        )

    def simuler(self, n_periodes: int, n_paths: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        simulations = np.zeros((n_paths, n_periodes), dtype=float)
        sigma2 = np.full(n_paths, self.sigma0**2, dtype=float)

        for t in range(n_periodes):
            chocs = rng.normal(loc=0.0, scale=np.sqrt(sigma2), size=n_paths)
            simulations[:, t] = self.mu + chocs
            sigma2 = self.lambda_vol * sigma2 + (1.0 - self.lambda_vol) * (chocs**2)

        return simulations


@dataclass(slots=True)
class ModeleMarkovSwitching:
    transition: np.ndarray
    moyennes: np.ndarray
    ecarts_types: np.ndarray
    proba_initiale: np.ndarray

    @classmethod
    def calibrer(cls, serie: pd.Series) -> "ModeleMarkovSwitching":
        try:
            modele = MarkovRegression(serie.to_numpy(dtype=float), k_regimes=2, trend="c", switching_variance=True)
            fit = modele.fit(disp=False)

            p_00 = float(np.clip(fit.params[0], 1e-6, 1 - 1e-6))
            p_10 = float(np.clip(fit.params[1], 1e-6, 1 - 1e-6))
            transition = np.array([[p_00, 1.0 - p_00], [p_10, 1.0 - p_10]], dtype=float)
            moyennes = np.array([fit.params[2], fit.params[3]], dtype=float)
            ecarts_types = np.sqrt(np.maximum(np.array([fit.params[4], fit.params[5]], dtype=float), 1e-12))

            proba = np.asarray(fit.smoothed_marginal_probabilities)
            if proba.ndim == 2:
                proba_initiale = proba[0].astype(float)
            else:
                proba_initiale = np.array([0.5, 0.5], dtype=float)
        except Exception:
            seuil = float(serie.median())
            regime0 = serie[serie <= seuil]
            regime1 = serie[serie > seuil]
            moyennes = np.array([regime0.mean(), regime1.mean()], dtype=float)
            ecarts_types = np.array([
                max(float(regime0.std(ddof=1)), 1e-6),
                max(float(regime1.std(ddof=1)), 1e-6),
            ])
            transition = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=float)
            proba_initiale = np.array([0.5, 0.5], dtype=float)

        proba_initiale = proba_initiale / proba_initiale.sum()
        return cls(
            transition=transition,
            moyennes=moyennes,
            ecarts_types=ecarts_types,
            proba_initiale=proba_initiale,
        )

    def simuler(self, n_periodes: int, n_paths: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        simulations = np.zeros((n_paths, n_periodes), dtype=float)

        for path_idx in range(n_paths):
            etat = int(rng.choice([0, 1], p=self.proba_initiale))
            for t in range(n_periodes):
                simulations[path_idx, t] = rng.normal(self.moyennes[etat], self.ecarts_types[etat])
                etat = int(rng.choice([0, 1], p=self.transition[etat]))

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
) -> tuple[pd.DataFrame, str, dict[str, np.ndarray]]:
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

    modele_student = ModeleStudentTIID.calibrer(serie_historique)
    sims_student = modele_student.simuler(
        n_periodes=n_periodes,
        n_paths=n_paths,
        seed=None if seed is None else seed + 2,
    )

    modele_vol = ModeleVolatiliteEWMA.calibrer(serie_historique)
    sims_vol = modele_vol.simuler(
        n_periodes=n_periodes,
        n_paths=n_paths,
        seed=None if seed is None else seed + 3,
    )

    modele_markov = ModeleMarkovSwitching.calibrer(serie_historique)
    sims_markov = modele_markov.simuler(
        n_periodes=n_periodes,
        n_paths=n_paths,
        seed=None if seed is None else seed + 4,
    )

    simulations_par_modele = {
        "gaussien_iid": sims_iid,
        "ar1_bruit_colore": sims_ar1,
        "student_t_iid": sims_student,
        "volatilite_ewma": sims_vol,
        "markov_switching_2_regimes": sims_markov,
    }

    lignes: list[dict[str, float | str]] = []
    for nom_modele, sims in simulations_par_modele.items():
        metriques_sims = pd.DataFrame([calculer_metriques(path) for path in sims])
        ecarts = {
            metrique: float(abs(metriques_sims[metrique].mean() - valeur_hist))
            for metrique, valeur_hist in metriques_hist.items()
        }
        score = float(np.mean(list(ecarts.values())))
        lignes.append({"modele": nom_modele, "score_fidelite": score, **ecarts})

    resultats = pd.DataFrame(lignes).sort_values("score_fidelite", ascending=True)
    meilleur_modele = str(resultats.iloc[0]["modele"])
    return resultats, meilleur_modele, simulations_par_modele


def construire_figure_rejeu(
    serie_historique: pd.Series,
    simulations_par_modele: dict[str, np.ndarray],
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Rejeu Monte Carlo des log-returns", "Moyennes glissantes (12 mois)"),
    )

    x = serie_historique.index
    fig.add_trace(
        go.Scatter(
            x=x,
            y=serie_historique.to_numpy(),
            mode="lines",
            name="historique",
            line=dict(color="black", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=serie_historique.rolling(12, min_periods=1).mean().to_numpy(),
            mode="lines",
            name="historique_rolling_12m",
            line=dict(color="black", dash="dash"),
        ),
        row=2,
        col=1,
    )

    palette = px.colors.qualitative.Plotly
    for idx, (nom_modele, simulations) in enumerate(simulations_par_modele.items()):
        couleur = palette[idx % len(palette)]
        for path in simulations:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=path,
                    mode="lines",
                    line=dict(color=couleur, width=1),
                    opacity=0.08,
                    name=nom_modele,
                    legendgroup=nom_modele,
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=simulations.mean(axis=0),
                mode="lines",
                line=dict(color=couleur, width=2),
                name=f"{nom_modele}_moyenne",
                legendgroup=nom_modele,
            ),
            row=1,
            col=1,
        )
        rolling_sims = pd.DataFrame(simulations.T, index=x).rolling(12, min_periods=1).mean().mean(axis=1)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=rolling_sims.to_numpy(),
                mode="lines",
                line=dict(color=couleur, width=2),
                name=f"{nom_modele}_rolling_12m",
                legendgroup=nom_modele,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title="Comparaison des rejeux Monte Carlo par stratégie",
        legend_title_text="Cliquer pour activer/désactiver un modèle",
        height=850,
    )
    return fig


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
    resultats, meilleur_modele, simulations = comparer_strategies(
        serie_historique=serie,
        n_paths=n_paths,
        seed=seed,
    )

    sortie = Path(dossier_sortie)
    sortie.mkdir(parents=True, exist_ok=True)

    fig = construire_figure_rejeu(serie_historique=serie, simulations_par_modele=simulations)
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0,
        y=-0.16,
        showarrow=False,
        align="left",
        text=(
            "<b>Lecture critique des stratégies</b><br>"
            "• Gaussien i.i.d.: baseline simple, sous-estime les queues épaisses.<br>"
            "• AR(1): capture une partie de la persistance moyenne, mais pas la volatilité conditionnelle.<br>"
            "• Student-t i.i.d.: améliore la modélisation des extrêmes (queues épaisses).<br>"
            "• Volatilité EWMA (proxy GARCH): approximation légère et robuste sans dépendance externe.<br>"
            "• Markov-switching 2 régimes: pertinent si alternance de régimes drift/vol détectable.<br><br>"
            f"<b>Meilleur modèle selon score global:</b> {meilleur_modele}<br>"
            f"<b>Tableau de synthèse:</b><br>{resultats.to_html(index=False, float_format='%.6f')}"
        ),
    )
    figure_path = sortie / "comparaison_fidelite.html"
    fig.write_html(figure_path, include_plotlyjs="cdn")

    return resultats, meilleur_modele, figure_path
