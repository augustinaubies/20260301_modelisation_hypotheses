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


def _retours_vers_niveaux(retours: np.ndarray, base_indice: float = 100.0) -> np.ndarray:
    return base_indice * np.exp(np.cumsum(retours, axis=-1))


def _calculer_score_metriques_standardisees(
    simulations: np.ndarray,
    metriques_hist: dict[str, float],
) -> tuple[float, dict[str, float]]:
    metriques_sims = pd.DataFrame([calculer_metriques(path) for path in simulations])
    ecarts_standardises: dict[str, float] = {}
    for metrique, valeur_hist in metriques_hist.items():
        denominateur = max(abs(float(valeur_hist)), 1e-6)
        ecarts_standardises[metrique] = float(abs(metriques_sims[metrique].mean() - valeur_hist) / denominateur)

    score = float(np.mean(list(ecarts_standardises.values())))
    return score, ecarts_standardises


def _calculer_score_rejeu_niveaux(
    serie_historique: pd.Series,
    simulations: np.ndarray,
) -> tuple[float, float]:
    historique_niveaux = _retours_vers_niveaux(serie_historique.to_numpy())
    simulations_niveaux = _retours_vers_niveaux(simulations)
    moyenne = simulations_niveaux.mean(axis=0)
    q_bas = np.quantile(simulations_niveaux, 0.025, axis=0)
    q_haut = np.quantile(simulations_niveaux, 0.975, axis=0)

    rmse_relatif = float(
        np.sqrt(np.mean((historique_niveaux - moyenne) ** 2)) / max(np.mean(historique_niveaux), 1e-6)
    )
    couverture_95 = float(np.mean((historique_niveaux >= q_bas) & (historique_niveaux <= q_haut)))
    penalite_couverture = float(abs(couverture_95 - 0.95))
    return rmse_relatif, penalite_couverture


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
        score_metriques, ecarts = _calculer_score_metriques_standardisees(
            simulations=sims,
            metriques_hist=metriques_hist,
        )
        rmse_relatif_niveaux, penalite_couverture = _calculer_score_rejeu_niveaux(
            serie_historique=serie_historique,
            simulations=sims,
        )
        score = float(0.4 * score_metriques + 0.4 * rmse_relatif_niveaux + 0.2 * penalite_couverture)
        lignes.append(
            {
                "modele": nom_modele,
                "score_fidelite": score,
                "score_metriques_standardisees": score_metriques,
                "rmse_relatif_niveaux": rmse_relatif_niveaux,
                "penalite_couverture_95": penalite_couverture,
                **ecarts,
            }
        )

    resultats = pd.DataFrame(lignes).sort_values("score_fidelite", ascending=True)
    meilleur_modele = str(resultats.iloc[0]["modele"])
    return resultats, meilleur_modele, simulations_par_modele


def _score_fidelite_depuis_simulations(simulations: np.ndarray, metriques_hist: dict[str, float]) -> float:
    score_metriques, _ = _calculer_score_metriques_standardisees(simulations=simulations, metriques_hist=metriques_hist)
    return score_metriques


def _simuler_pour_modele(
    nom_modele: str,
    serie_historique: pd.Series,
    n_periodes: int,
    n_paths: int,
    seed: int | None,
) -> np.ndarray:
    if nom_modele == "gaussien_iid":
        return ModeleGaussienIID.calibrer(serie_historique).simuler(n_periodes=n_periodes, n_paths=n_paths, seed=seed)

    if nom_modele == "ar1_bruit_colore":
        return ModeleAR1.calibrer(serie_historique).simuler(
            n_periodes=n_periodes,
            n_paths=n_paths,
            x0=float(serie_historique.iloc[0]),
            seed=seed,
        )

    if nom_modele == "student_t_iid":
        return ModeleStudentTIID.calibrer(serie_historique).simuler(n_periodes=n_periodes, n_paths=n_paths, seed=seed)

    if nom_modele == "volatilite_ewma":
        return ModeleVolatiliteEWMA.calibrer(serie_historique).simuler(n_periodes=n_periodes, n_paths=n_paths, seed=seed)

    if nom_modele == "markov_switching_2_regimes":
        return ModeleMarkovSwitching.calibrer(serie_historique).simuler(n_periodes=n_periodes, n_paths=n_paths, seed=seed)

    raise ValueError(f"Modèle inconnu pour simulation: {nom_modele}")


def detecter_meilleure_date_depart(
    serie_historique: pd.Series,
    modele_a_tester: str = "volatilite_ewma",
    fenetre_min_mois: int = 60,
    pas_mois: int = 6,
    n_paths: int = 120,
    seed: int | None = 42,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    if len(serie_historique) < fenetre_min_mois:
        raise ValueError("Série trop courte pour la détection de date de départ.")

    candidats: list[dict[str, float | pd.Timestamp | int]] = []
    for start_idx in range(0, len(serie_historique) - fenetre_min_mois + 1, pas_mois):
        fenetre = serie_historique.iloc[start_idx:]
        n_periodes = len(fenetre)
        metriques_hist = calculer_metriques(fenetre.to_numpy())
        simulations = _simuler_pour_modele(
            nom_modele=modele_a_tester,
            serie_historique=fenetre,
            n_periodes=n_periodes,
            n_paths=n_paths,
            seed=None if seed is None else seed + start_idx,
        )
        score = _score_fidelite_depuis_simulations(simulations=simulations, metriques_hist=metriques_hist)
        candidats.append(
            {
                "date_depart": fenetre.index[0],
                "score_fidelite": score,
                "n_observations": n_periodes,
            }
        )

    resultats = pd.DataFrame(candidats).sort_values("score_fidelite", ascending=True).reset_index(drop=True)
    meilleure_date = pd.Timestamp(resultats.loc[0, "date_depart"])
    return resultats, meilleure_date




def _hex_vers_rgba(couleur_hex: str, alpha: float) -> str | None:
    if not couleur_hex.startswith("#") or len(couleur_hex) != 7:
        return None
    r = int(couleur_hex[1:3], 16)
    g = int(couleur_hex[3:5], 16)
    b = int(couleur_hex[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def construire_figure_rejeu(
    serie_historique: pd.Series,
    simulations_par_modele: dict[str, np.ndarray],
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "Courbes temporelles (intégrale des variations log)",
            "Distribution temporelle des stratégies (moyenne et couloir 95%)",
        ),
    )

    x = serie_historique.index
    base_indice = 100.0
    historique_niveaux = base_indice * np.exp(np.cumsum(serie_historique.to_numpy()))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=historique_niveaux,
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
            y=historique_niveaux,
            mode="lines",
            name="historique",
            line=dict(color="black", width=2),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    palette = px.colors.qualitative.Plotly
    for idx, (nom_modele, simulations) in enumerate(simulations_par_modele.items()):
        couleur = palette[idx % len(palette)]
        simulations_niveaux = base_indice * np.exp(np.cumsum(simulations, axis=1))
        for path in simulations_niveaux:
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

        moyenne = simulations_niveaux.mean(axis=0)
        q_bas = np.quantile(simulations_niveaux, 0.025, axis=0)
        q_haut = np.quantile(simulations_niveaux, 0.975, axis=0)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=moyenne,
                mode="lines",
                line=dict(color=couleur, width=2),
                name=f"{nom_modele} - moyenne",
                legendgroup=nom_modele,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=q_bas,
                mode="lines",
                line=dict(width=0),
                name=f"{nom_modele} - q2.5",
                legendgroup=nom_modele,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=q_haut,
                mode="lines",
                fill="tonexty",
                fillcolor=_hex_vers_rgba(couleur, 0.18),
                line=dict(width=0),
                name=f"{nom_modele} - couloir 95%",
                legendgroup=nom_modele,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=moyenne,
                mode="lines",
                line=dict(color=couleur, width=2),
                name=f"{nom_modele} - moyenne",
                legendgroup=nom_modele,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title="Comparaison des rejeux Monte Carlo par stratégie",
        legend_title_text="Cliquer pour activer/désactiver un modèle",
        height=900,
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#111827",
    )
    fig.update_yaxes(title_text="Indice base 100", row=1, col=1)
    fig.update_yaxes(title_text="Indice base 100", row=2, col=1)
    return fig


def construire_html_rapport(
    fig: go.Figure,
    resultats: pd.DataFrame,
    meilleur_modele: str,
    resultats_dates: pd.DataFrame,
    meilleure_date: pd.Timestamp,
    modele_date: str,
) -> str:
    tableau_html = resultats.to_html(index=False, float_format="%.6f")
    tableau_dates_html = resultats_dates.head(10).to_html(index=False, float_format="%.6f")
    figure_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    return f"""<!DOCTYPE html>
<html lang=\"fr\">
  <head>
    <meta charset=\"utf-8\" />
    <title>Comparaison de fidélité des stratégies</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 24px 32px; line-height: 1.5; background: #0b1120; color: #e5e7eb; }}
      h1, h2 {{ margin-top: 0; }}
      section {{ margin-bottom: 28px; }}
      .plot-container {{ margin: 30px 0 36px 0; }}
      table {{ border-collapse: collapse; }}
      th, td {{ border: 1px solid #334155; padding: 6px 10px; }}
      th {{ background: #1e293b; }}
      td {{ background: #0f172a; }}
      ul {{ margin-top: 8px; }}
      a {{ color: #93c5fd; }}
    </style>
  </head>
  <body>
    <section>
      <h1>Comparaison des stratégies de modélisation</h1>
      <p><b>Meilleur modèle selon score global :</b> {meilleur_modele}</p>
    </section>

    <section>
      <h2>Lecture critique des stratégies</h2>
      <ul>
        <li><b>Gaussien i.i.d.</b> : baseline simple, sous-estime les queues épaisses.</li>
        <li><b>AR(1)</b> : capture une partie de la persistance moyenne, mais pas la volatilité conditionnelle.</li>
        <li><b>Student-t i.i.d.</b> : améliore la modélisation des extrêmes (queues épaisses).</li>
        <li><b>Volatilité EWMA</b> : proxy GARCH léger, robuste et sans dépendance lourde.</li>
        <li><b>Markov-switching 2 régimes</b> : pertinent si alternance de régimes drift/vol détectable.</li>
      </ul>
    </section>

    <section>
      <h2>Tableau de synthèse</h2>
      <p>Le score global combine 3 briques: fidélité des métriques de rendements (standardisées), RMSE relatif sur le rejeu en niveaux (base 100) et pénalité de mauvaise couverture 95%.</p>
      {tableau_html}
    </section>

    <section>
      <h2>Recherche de fenêtre historique plus fidèle</h2>
      <p><b>Méthode testée :</b> {modele_date}</p>
      <p><b>Meilleure date de départ :</b> {meilleure_date.strftime("%Y-%m-%d")}</p>
      <p>Top 10 des dates candidates (score plus bas = meilleure fidélité) :</p>
      {tableau_dates_html}
    </section>

    <section class=\"plot-container\">
      <h2>Visualisation Monte Carlo</h2>
      {figure_html}
    </section>
  </body>
</html>
"""


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
    modele_date = "volatilite_ewma"
    resultats_dates, meilleure_date = detecter_meilleure_date_depart(
        serie_historique=serie,
        modele_a_tester=modele_date,
        fenetre_min_mois=60,
        pas_mois=6,
        n_paths=min(n_paths, 120),
        seed=seed,
    )

    sortie = Path(dossier_sortie)
    sortie.mkdir(parents=True, exist_ok=True)

    fig = construire_figure_rejeu(serie_historique=serie, simulations_par_modele=simulations)
    rapport_html = construire_html_rapport(
        fig=fig,
        resultats=resultats,
        meilleur_modele=meilleur_modele,
        resultats_dates=resultats_dates,
        meilleure_date=meilleure_date,
        modele_date=modele_date,
    )
    figure_path = sortie / "comparaison_fidelite.html"
    figure_path.write_text(rapport_html, encoding="utf-8")

    return resultats, meilleur_modele, figure_path
