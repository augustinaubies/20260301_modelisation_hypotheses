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
    fenetres_calibration: dict[str, pd.Series] | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, str, dict[str, np.ndarray]]:
    n_periodes = len(serie_historique)
    metriques_hist = calculer_metriques(serie_historique.to_numpy())

    fenetres_calibration = fenetres_calibration or {}
    serie_gauss = fenetres_calibration.get("gaussien_iid", serie_historique)
    serie_ar1 = fenetres_calibration.get("ar1_bruit_colore", serie_historique)
    serie_student = fenetres_calibration.get("student_t_iid", serie_historique)
    serie_vol = fenetres_calibration.get("volatilite_ewma", serie_historique)
    serie_markov = fenetres_calibration.get("markov_switching_2_regimes", serie_historique)

    modele_iid = ModeleGaussienIID.calibrer(serie_gauss)
    sims_iid = modele_iid.simuler(n_periodes=n_periodes, n_paths=n_paths, seed=seed)

    modele_ar1 = ModeleAR1.calibrer(serie_ar1)
    sims_ar1 = modele_ar1.simuler(
        n_periodes=n_periodes,
        n_paths=n_paths,
        x0=float(serie_historique.iloc[0]),
        seed=None if seed is None else seed + 1,
    )

    modele_student = ModeleStudentTIID.calibrer(serie_student)
    sims_student = modele_student.simuler(
        n_periodes=n_periodes,
        n_paths=n_paths,
        seed=None if seed is None else seed + 2,
    )

    modele_vol = ModeleVolatiliteEWMA.calibrer(serie_vol)
    sims_vol = modele_vol.simuler(
        n_periodes=n_periodes,
        n_paths=n_paths,
        seed=None if seed is None else seed + 3,
    )

    modele_markov = ModeleMarkovSwitching.calibrer(serie_markov)
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


def detecter_date_stable_gaussienne(
    serie_historique: pd.Series,
    fenetre_min_mois: int = 60,
    pas_mois: int = 3,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    if len(serie_historique) < fenetre_min_mois:
        raise ValueError("Série trop courte pour détecter une stabilité de densité.")

    candidats: list[dict[str, float | pd.Timestamp | int]] = []
    for start_idx in range(0, len(serie_historique) - fenetre_min_mois + 1, pas_mois):
        fenetre = serie_historique.iloc[start_idx:]
        n = len(fenetre)
        demi = n // 2
        if demi < 24:
            continue

        premier = fenetre.iloc[:demi].to_numpy(dtype=float)
        second = fenetre.iloc[demi:].to_numpy(dtype=float)
        mouvement = float(stats.wasserstein_distance(premier, second))
        candidats.append(
            {
                "date_depart": fenetre.index[0],
                "mouvement_densite": mouvement,
                "n_observations": n,
            }
        )

    resultats = pd.DataFrame(candidats).sort_values("date_depart").reset_index(drop=True)
    seuil = float(resultats["mouvement_densite"].quantile(0.25))
    stables = resultats[resultats["mouvement_densite"] <= seuil]

    if stables.empty:
        meilleure_date = pd.Timestamp(resultats.sort_values("mouvement_densite").iloc[0]["date_depart"])
    else:
        meilleure_date = pd.Timestamp(stables.iloc[0]["date_depart"])

    return resultats.sort_values("mouvement_densite").reset_index(drop=True), meilleure_date




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
        mediane = np.median(simulations_niveaux, axis=0)
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
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mediane,
                mode="lines",
                line=dict(color=couleur, width=2, dash="dash"),
                name=f"{nom_modele} - médiane",
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


def _construire_grille_densite_commune(series: list[np.ndarray], n_points: int = 400) -> np.ndarray:
    donnees = [np.asarray(serie, dtype=float) for serie in series if np.asarray(serie).size > 0]
    if not donnees:
        return np.array([0.0])

    fusion = np.concatenate([serie[np.isfinite(serie)] for serie in donnees])
    if fusion.size == 0:
        return np.array([0.0])

    xmin, xmax = float(np.quantile(fusion, 0.001)), float(np.quantile(fusion, 0.999))
    if xmin == xmax:
        xmin -= 1e-6
        xmax += 1e-6
    return np.linspace(xmin, xmax, n_points)


def _calculer_kde(
    serie: np.ndarray,
    n_points: int = 300,
    x_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    serie = np.asarray(serie, dtype=float)
    serie = serie[np.isfinite(serie)]
    if serie.size < 2:
        x = np.array([0.0])
        y = np.array([0.0])
        return x, y

    if x_grid is None:
        xmin, xmax = float(np.quantile(serie, 0.005)), float(np.quantile(serie, 0.995))
        if xmin == xmax:
            xmin -= 1e-6
            xmax += 1e-6
        x = np.linspace(xmin, xmax, n_points)
    else:
        x = np.asarray(x_grid, dtype=float)
    try:
        kde = stats.gaussian_kde(serie)
        y = kde(x)
    except Exception:
        y = np.zeros_like(x)
    return x, y


def _calculer_densite_gaussienne_theorique(
    serie_historique: pd.Series,
    n_points: int = 300,
    x_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mu = float(serie_historique.mean())
    sigma = max(float(serie_historique.std(ddof=1)), 1e-12)
    if x_grid is None:
        borne_basse, borne_haute = np.quantile(serie_historique.to_numpy(), [0.005, 0.995])
        x = np.linspace(float(borne_basse), float(borne_haute), n_points)
    else:
        x = np.asarray(x_grid, dtype=float)
    y = stats.norm.pdf(x, loc=mu, scale=sigma)
    return x, y


def _estimer_mode_kde(serie: np.ndarray, x_grid: np.ndarray) -> float:
    x, y = _calculer_kde(serie, x_grid=x_grid)
    if x.size == 0 or y.size == 0:
        return float("nan")
    return float(x[np.argmax(y)])


def _evaluer_calibration_distributions(
    serie_historique: pd.Series,
    simulations_par_modele: dict[str, np.ndarray],
) -> pd.DataFrame:
    lignes: list[dict[str, float | str]] = []
    hist = serie_historique.to_numpy(dtype=float)
    grille_commune = _construire_grille_densite_commune([hist, *[np.asarray(sim, dtype=float).reshape(-1) for sim in simulations_par_modele.values()]])
    mean_hist = float(np.mean(hist))
    median_hist = float(np.median(hist))
    mode_hist = _estimer_mode_kde(hist, grille_commune)
    for nom_modele, simulations in simulations_par_modele.items():
        echantillon = np.asarray(simulations, dtype=float).reshape(-1)
        if echantillon.size == 0:
            continue
        ks_stat, ks_pvalue = stats.ks_2samp(hist, echantillon)
        lignes.append(
            {
                "modele": nom_modele,
                "mean_hist": mean_hist,
                "mean_modele": float(np.mean(echantillon)),
                "median_hist": median_hist,
                "median_modele": float(np.median(echantillon)),
                "mode_hist_kde": mode_hist,
                "mode_modele_kde": _estimer_mode_kde(echantillon, grille_commune),
                "std_hist": float(np.std(hist, ddof=1)),
                "std_modele": float(np.std(echantillon, ddof=1)),
                "ks_stat": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
            }
        )

    return pd.DataFrame(lignes).sort_values("ks_stat", ascending=True)


def construire_figure_distribution_variations(
    serie_historique: pd.Series,
    simulations_par_modele: dict[str, np.ndarray],
) -> go.Figure:
    palette = px.colors.qualitative.Plotly
    couleurs = {nom: palette[idx % len(palette)] for idx, nom in enumerate(simulations_par_modele.keys())}

    fig = go.Figure()

    variations_modeles = [np.asarray(simulations, dtype=float).reshape(-1) for simulations in simulations_par_modele.values()]
    grille_commune = _construire_grille_densite_commune([serie_historique.to_numpy(), *variations_modeles])

    x_hist, y_hist = _calculer_kde(serie_historique.to_numpy(), x_grid=grille_commune)
    fig.add_trace(
        go.Scatter(
            x=x_hist,
            y=y_hist,
            mode="lines",
            name="historique",
            line=dict(color="#f8fafc", width=3),
        )
    )

    x_gauss_theorique, y_gauss_theorique = _calculer_densite_gaussienne_theorique(
        serie_historique,
        x_grid=grille_commune,
    )
    fig.add_trace(
        go.Scatter(
            x=x_gauss_theorique,
            y=y_gauss_theorique,
            mode="lines",
            name="gaussienne théorique (fit historique)",
            line=dict(color="#f97316", width=2, dash="dot"),
        )
    )

    for nom_modele, simulations in simulations_par_modele.items():
        variations_modele = np.asarray(simulations, dtype=float).reshape(-1)
        x_modele, y_modele = _calculer_kde(variations_modele, x_grid=grille_commune)
        fig.add_trace(
            go.Scatter(
                x=x_modele,
                y=y_modele,
                mode="lines",
                name=nom_modele,
                line=dict(color=couleurs[nom_modele], width=2),
            )
        )

    fig.update_layout(
        title="Distribution des variations mensuelles (densité)",
        xaxis_title="Variation mensuelle (log-return)",
        yaxis_title="Densité de probabilité",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#111827",
        height=500,
        legend_title_text="Série",
    )
    return fig


def construire_html_rapport(
    fig: go.Figure,
    fig_distribution: go.Figure,
    resultats: pd.DataFrame,
    meilleur_modele: str,
    resultats_dates: pd.DataFrame,
    meilleure_date: pd.Timestamp,
    modele_date: str,
    resultats_dates_gauss: pd.DataFrame,
    meilleure_date_gauss: pd.Timestamp,
    date_fin: pd.Timestamp,
    diagnostic_calibration: pd.DataFrame,
) -> str:
    tableau_html = resultats.to_html(index=False, float_format="%.6f")
    tableau_dates_html = resultats_dates.head(10).to_html(index=False, float_format="%.6f")
    tableau_dates_gauss_html = resultats_dates_gauss.head(10).to_html(index=False, float_format="%.6f")
    figure_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    figure_distribution_html = fig_distribution.to_html(full_html=False, include_plotlyjs=False)
    tableau_diagnostic_html = diagnostic_calibration.to_html(index=False, float_format="%.6f")
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

    <section>
      <h2>Stabilité de densité pour la gaussienne i.i.d.</h2>
      <p><b>Date de calibration retenue pour la gaussienne :</b> {meilleure_date_gauss.strftime("%Y-%m-%d")}</p>
      <p>Critère : distance de Wasserstein entre densités empiriques (première moitié vs seconde moitié de la fenêtre). La première date "stable" est retenue pour limiter les mouvements de distribution.</p>
      {tableau_dates_gauss_html}
    </section>
    
    <section>
      <h2>Diagnostic de calibration des lois</h2>
      <p>Comparaison directe distribution historique vs tirages simulés (moyenne, volatilité et test KS à 2 échantillons). Un <i>ks_stat</i> faible indique une densité plus proche de l'historique.</p>
      {tableau_diagnostic_html}
    </section>

    <section class=\"plot-container\">
      <h2>Visualisation Monte Carlo</h2>
      <p>Le rejeu ci-dessous est recalé sur la fenêtre optimale détectée, de <b>{meilleure_date.strftime("%Y-%m-%d")}</b> à <b>{date_fin.strftime("%Y-%m-%d")}</b>.</p>
      {figure_html}
    </section>

    <section class=\"plot-container\">
      <h2>Distribution des variations mensuelles</h2>
      <p>Cette courbe compare la densité des variations mensuelles historiques avec l'ensemble des tirages Monte Carlo de chaque stratégie.</p>
      {figure_distribution_html}
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
    resultats, meilleur_modele, _ = comparer_strategies(
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
    # meilleure_date = pd.Timestamp('1980-01-01')
    
    resultats_dates_gauss, meilleure_date_gauss = detecter_date_stable_gaussienne(
        serie_historique=serie,
        fenetre_min_mois=60,
        pas_mois=3,
    )

    serie_optimale = serie.loc[meilleure_date:]
    fenetres_calibration = {"gaussien_iid": serie.loc[meilleure_date_gauss:]}
    _, _, simulations_optimales = comparer_strategies(
        serie_historique=serie_optimale,
        n_paths=n_paths,
        fenetres_calibration=fenetres_calibration,
        seed=seed,
    )

    sortie = Path(dossier_sortie)
    sortie.mkdir(parents=True, exist_ok=True)

    fig = construire_figure_rejeu(serie_historique=serie_optimale, simulations_par_modele=simulations_optimales)
    fig_distribution = construire_figure_distribution_variations(
        serie_historique=serie_optimale,
        simulations_par_modele=simulations_optimales,
    )
    diagnostic_calibration = _evaluer_calibration_distributions(
        serie_historique=serie_optimale,
        simulations_par_modele=simulations_optimales,
    )
    rapport_html = construire_html_rapport(
        fig=fig,
        fig_distribution=fig_distribution,
        resultats=resultats,
        meilleur_modele=meilleur_modele,
        resultats_dates=resultats_dates,
        meilleure_date=meilleure_date,
        modele_date=modele_date,
        resultats_dates_gauss=resultats_dates_gauss,
        meilleure_date_gauss=meilleure_date_gauss,
        date_fin=pd.Timestamp(serie_optimale.index[-1]),
        diagnostic_calibration=diagnostic_calibration,
    )
    figure_path = sortie / "comparaison_fidelite.html"
    figure_path.write_text(rapport_html, encoding="utf-8")

    return resultats, meilleur_modele, figure_path
