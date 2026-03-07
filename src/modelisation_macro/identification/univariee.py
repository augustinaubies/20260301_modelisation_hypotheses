from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
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
class ModeleSkewTIID:
    a: float
    b: float
    mu: float
    sigma: float

    @classmethod
    def calibrer(cls, serie: pd.Series) -> "ModeleSkewTIID":
        a, b, mu, sigma = stats.jf_skew_t.fit(serie.to_numpy(dtype=float))
        return cls(a=float(a), b=float(b), mu=float(mu), sigma=float(sigma))

    def simuler(self, n_periodes: int, n_paths: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return stats.jf_skew_t.rvs(
            self.a,
            self.b,
            loc=self.mu,
            scale=max(self.sigma, 1e-12),
            size=(n_paths, n_periodes),
            random_state=rng,
        )


@dataclass(slots=True)
class ModeleSkewNormNuInfIID:
    a: float
    mu: float
    sigma: float

    @classmethod
    def calibrer(cls, serie: pd.Series) -> "ModeleSkewNormNuInfIID":
        a, mu, sigma = stats.skewnorm.fit(serie.to_numpy(dtype=float))
        return cls(a=float(a), mu=float(mu), sigma=float(sigma))

    def simuler(self, n_periodes: int, n_paths: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return stats.skewnorm.rvs(
            self.a,
            loc=self.mu,
            scale=max(self.sigma, 1e-12),
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


@dataclass(slots=True)
class ModeleMarkovSwitchingSkewT:
    transition: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    a: np.ndarray
    b: np.ndarray
    proba_initiale: np.ndarray
    log_likelihood: float
    aic: float
    bic: float
    filtered_probabilities: np.ndarray
    smoothed_probabilities: np.ndarray

    @staticmethod
    def _logistic(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    @classmethod
    def _construire_parametres(
        cls,
        theta: np.ndarray,
        reduced: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p00 = np.clip(cls._logistic(float(theta[0])), 1e-6, 1 - 1e-6)
        p11 = np.clip(cls._logistic(float(theta[1])), 1e-6, 1 - 1e-6)
        transition = np.array([[p00, 1.0 - p00], [1.0 - p11, p11]], dtype=float)

        mu = np.array([theta[2], theta[3]], dtype=float)
        sigma = np.exp(np.array([theta[4], theta[5]], dtype=float))

        if reduced:
            a_shared = float(np.clip(np.exp(theta[6]), 1e-3, 50.0))
            b_shared = float(np.clip(np.exp(theta[7]), 1e-3, 50.0))
            a = np.array([a_shared, a_shared], dtype=float)
            b = np.array([b_shared, b_shared], dtype=float)
        else:
            a = np.clip(np.exp(np.array([theta[6], theta[7]], dtype=float)), 1e-3, 50.0)
            b = np.clip(np.exp(np.array([theta[8], theta[9]], dtype=float)), 1e-3, 50.0)

        delta = max(1.0 - p00 - p11, 1e-6)
        proba_initiale = np.array([(1.0 - p11) / delta, (1.0 - p00) / delta], dtype=float)
        proba_initiale = np.clip(proba_initiale, 1e-6, 1.0)
        proba_initiale = proba_initiale / proba_initiale.sum()
        return transition, mu, sigma, a, b, proba_initiale

    @staticmethod
    def _log_emissions(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        log_emissions = np.zeros((2, y.size), dtype=float)
        for k in range(2):
            log_emissions[k] = stats.jf_skew_t.logpdf(y, a[k], b[k], loc=mu[k], scale=max(sigma[k], 1e-12))
        return np.nan_to_num(log_emissions, neginf=-1e12, posinf=-1e12)

    @classmethod
    def _hamilton_filtre_lisse(
        cls,
        y: np.ndarray,
        transition: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        proba_initiale: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        n = y.size
        if n == 0:
            return -np.inf, np.zeros((0, 2)), np.zeros((0, 2))

        emissions = np.exp(cls._log_emissions(y=y, mu=mu, sigma=sigma, a=a, b=b))
        filtered = np.zeros((n, 2), dtype=float)
        predicted = np.zeros((n, 2), dtype=float)
        c = np.zeros(n, dtype=float)

        predicted[0] = proba_initiale
        numerateur0 = predicted[0] * emissions[:, 0]
        c[0] = max(numerateur0.sum(), 1e-300)
        filtered[0] = numerateur0 / c[0]

        for t in range(1, n):
            predicted[t] = filtered[t - 1] @ transition
            numerateur = predicted[t] * emissions[:, t]
            c[t] = max(numerateur.sum(), 1e-300)
            filtered[t] = numerateur / c[t]

        ll = float(np.sum(np.log(c)))

        smoothed = np.zeros_like(filtered)
        smoothed[-1] = filtered[-1]
        for t in range(n - 2, -1, -1):
            ratio = smoothed[t + 1] / np.clip(predicted[t + 1], 1e-300, None)
            smoothed[t] = filtered[t] * (transition @ ratio)
            denom = max(smoothed[t].sum(), 1e-300)
            smoothed[t] /= denom

        return ll, filtered, smoothed

    @classmethod
    def calibrer(
        cls,
        serie: pd.Series,
        reduced: bool = True,
        n_starts: int = 4,
        maxiter: int = 150,
        seed: int | None = 42,
    ) -> "ModeleMarkovSwitchingSkewT":
        y = serie.to_numpy(dtype=float)
        if y.size < 10:
            seuil = float(np.median(y)) if y.size else 0.0
            regime0 = y[y <= seuil] if y.size else np.array([0.0])
            regime1 = y[y > seuil] if y.size else np.array([0.0])
            mu = np.array([float(np.mean(regime0)), float(np.mean(regime1))], dtype=float)
            sigma = np.array([
                max(float(np.std(regime0, ddof=1)) if regime0.size > 1 else 1e-3, 1e-6),
                max(float(np.std(regime1, ddof=1)) if regime1.size > 1 else 1e-3, 1e-6),
            ])
            a = np.array([2.0, 2.0], dtype=float)
            b = np.array([2.0, 2.0], dtype=float)
            transition = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=float)
            proba_initiale = np.array([0.5, 0.5], dtype=float)
            ll, filtered, smoothed = cls._hamilton_filtre_lisse(y, transition, mu, sigma, a, b, proba_initiale)
            k = 8 if reduced else 10
            return cls(transition, mu, sigma, a, b, proba_initiale, ll, 2 * k - 2 * ll, k * np.log(max(y.size, 1)) - 2 * ll, filtered, smoothed)

        rng = np.random.default_rng(seed)
        mu0, sigma0 = float(np.mean(y)), max(float(np.std(y, ddof=1)), 1e-4)
        a0, b0, _, _ = stats.jf_skew_t.fit(y)

        best: tuple[float, np.ndarray] | None = None
        nb_params = 8 if reduced else 10

        def objective(theta: np.ndarray) -> float:
            transition, mu, sigma, a, b, proba_initiale = cls._construire_parametres(theta=theta, reduced=reduced)
            ll, _, _ = cls._hamilton_filtre_lisse(y, transition, mu, sigma, a, b, proba_initiale)
            if not np.isfinite(ll):
                return 1e12
            return -ll

        for i in range(max(n_starts, 1)):
            perturb = rng.normal(0.0, 0.35, size=nb_params)
            init = np.zeros(nb_params, dtype=float)
            init[0:2] = np.array([2.2, 2.0]) + perturb[0:2]
            init[2:4] = np.array([mu0 - 0.5 * sigma0, mu0 + 0.5 * sigma0]) + sigma0 * perturb[2:4]
            init[4:6] = np.log(np.array([0.7 * sigma0, 1.3 * sigma0])) + perturb[4:6]
            if reduced:
                init[6:8] = np.log(np.array([max(float(a0), 1e-3), max(float(b0), 1e-3)])) + 0.2 * perturb[6:8]
            else:
                init[6:10] = np.log(np.array([max(float(a0), 1e-3), max(float(a0), 1e-3), max(float(b0), 1e-3), max(float(b0), 1e-3)])) + 0.2 * perturb[6:10]

            opt = minimize(objective, init, method="L-BFGS-B", options={"maxiter": maxiter})
            val = float(opt.fun) if np.isfinite(opt.fun) else 1e12
            if best is None or val < best[0]:
                best = (val, np.asarray(opt.x, dtype=float))

        if best is None:
            raise RuntimeError("Echec de calibration MS(2)-Skew-t")

        transition, mu, sigma, a, b, proba_initiale = cls._construire_parametres(theta=best[1], reduced=reduced)
        ll, filtered, smoothed = cls._hamilton_filtre_lisse(y, transition, mu, sigma, a, b, proba_initiale)
        k = nb_params
        aic = float(2 * k - 2 * ll)
        bic = float(k * np.log(y.size) - 2 * ll)
        return cls(transition, mu, sigma, a, b, proba_initiale, ll, aic, bic, filtered, smoothed)

    def simuler(self, n_periodes: int, n_paths: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        simulations = np.zeros((n_paths, n_periodes), dtype=float)
        for path_idx in range(n_paths):
            etat = int(rng.choice([0, 1], p=self.proba_initiale))
            for t in range(n_periodes):
                simulations[path_idx, t] = stats.jf_skew_t.rvs(
                    self.a[etat],
                    self.b[etat],
                    loc=self.mu[etat],
                    scale=max(self.sigma[etat], 1e-12),
                    random_state=rng,
                )
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
    inclure_markov_skew_t: bool = True,
    seed: int | None = None,
) -> tuple[pd.DataFrame, str, dict[str, np.ndarray]]:
    n_periodes = len(serie_historique)
    metriques_hist = calculer_metriques(serie_historique.to_numpy())

    fenetres_calibration = fenetres_calibration or {}
    serie_gauss = fenetres_calibration.get("gaussien_iid", serie_historique)
    serie_ar1 = fenetres_calibration.get("ar1_bruit_colore", serie_historique)
    serie_student = fenetres_calibration.get("student_t_iid", serie_historique)
    serie_skew_t = fenetres_calibration.get("skew_t_asymetrique_iid", serie_historique)
    serie_skew_nu_inf = fenetres_calibration.get("skew_t_asymetrique_nu_inf", serie_historique)
    serie_vol = fenetres_calibration.get("volatilite_ewma", serie_historique)
    serie_markov = fenetres_calibration.get("markov_switching_2_regimes", serie_historique)
    serie_markov_skew_t = fenetres_calibration.get("markov_switching_2_regimes_skew_t", serie_historique)

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

    modele_skew_t = ModeleSkewTIID.calibrer(serie_skew_t)
    sims_skew_t = modele_skew_t.simuler(
        n_periodes=n_periodes,
        n_paths=n_paths,
        seed=None if seed is None else seed + 5,
    )

    modele_skew_nu_inf = ModeleSkewNormNuInfIID.calibrer(serie_skew_nu_inf)
    sims_skew_nu_inf = modele_skew_nu_inf.simuler(
        n_periodes=n_periodes,
        n_paths=n_paths,
        seed=None if seed is None else seed + 6,
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
        "skew_t_asymetrique_iid": sims_skew_t,
        "skew_t_asymetrique_nu_inf": sims_skew_nu_inf,
        "volatilite_ewma": sims_vol,
        "markov_switching_2_regimes": sims_markov,
    }

    if inclure_markov_skew_t:
        modele_markov_skew_t = ModeleMarkovSwitchingSkewT.calibrer(serie_markov_skew_t)
        sims_markov_skew_t = modele_markov_skew_t.simuler(
            n_periodes=n_periodes,
            n_paths=n_paths,
            seed=None if seed is None else seed + 7,
        )
        simulations_par_modele["markov_switching_2_regimes_skew_t"] = sims_markov_skew_t

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

    if nom_modele == "skew_t_asymetrique_iid":
        return ModeleSkewTIID.calibrer(serie_historique).simuler(n_periodes=n_periodes, n_paths=n_paths, seed=seed)

    if nom_modele == "skew_t_asymetrique_nu_inf":
        return ModeleSkewNormNuInfIID.calibrer(serie_historique).simuler(n_periodes=n_periodes, n_paths=n_paths, seed=seed)

    if nom_modele == "markov_switching_2_regimes":
        return ModeleMarkovSwitching.calibrer(serie_historique).simuler(n_periodes=n_periodes, n_paths=n_paths, seed=seed)

    if nom_modele == "markov_switching_2_regimes_skew_t":
        return ModeleMarkovSwitchingSkewT.calibrer(serie_historique).simuler(n_periodes=n_periodes, n_paths=n_paths, seed=seed)

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
    mu: float | None = None,
    sigma: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mu = float(serie_historique.mean()) if mu is None else float(mu)
    sigma_hist = float(serie_historique.std(ddof=1))
    sigma = sigma_hist if sigma is None else float(sigma)
    sigma = max(sigma, 1e-12)
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
    skew_hist = float(stats.skew(hist, bias=False)) if hist.size > 2 else 0.0
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
                "skew_hist": skew_hist,
                "skew_modele": float(stats.skew(echantillon, bias=False)) if echantillon.size > 2 else 0.0,
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
    diagnostic_calibration: pd.DataFrame | None = None,
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

    mu_gauss = None
    sigma_gauss = None
    if diagnostic_calibration is not None and not diagnostic_calibration.empty:
        ligne_gauss = diagnostic_calibration.loc[diagnostic_calibration["modele"] == "gaussien_iid"]
        if not ligne_gauss.empty:
            mu_gauss = float(ligne_gauss.iloc[0]["mean_modele"])
            sigma_gauss = float(ligne_gauss.iloc[0]["std_modele"])

    x_gauss_theorique, y_gauss_theorique = _calculer_densite_gaussienne_theorique(
        serie_historique,
        x_grid=grille_commune,
        mu=mu_gauss,
        sigma=sigma_gauss,
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
    markov_skew_t_resume_html: str = "",
) -> str:
    tableau_html = resultats.to_html(index=False, float_format="%.6f")
    tableau_dates_html = resultats_dates.head(10).to_html(index=False, float_format="%.6f")
    tableau_dates_gauss_html = resultats_dates_gauss.head(10).to_html(index=False, float_format="%.6f")
    figure_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    figure_distribution_html = fig_distribution.to_html(full_html=False, include_plotlyjs=False)
    tableau_diagnostic_html = diagnostic_calibration.to_html(index=False, float_format="%.6f")

    resume_densite = ""
    if not diagnostic_calibration.empty:
        ligne_ref = diagnostic_calibration.iloc[0]
        interpretation = ""
        if float(ligne_ref["skew_hist"]) < -0.1 and float(ligne_ref["mode_hist_kde"]) > float(ligne_ref["mean_hist"]):
            interpretation = (
                "La distribution historique est asymétrique à gauche : quelques chocs baissiers extrêmes "
                "tirent la moyenne vers le bas, tandis que le centre visuel (mode KDE) reste plus à droite."
            )
        texte_interpretation = f"<p>{interpretation}</p>" if interpretation else ""
        resume_densite = f"""
      <p><b>Repères historiques (fenêtre affichée)</b> — mean: {float(ligne_ref['mean_hist']):.6f}, median: {float(ligne_ref['median_hist']):.6f}, mode KDE: {float(ligne_ref['mode_hist_kde']):.6f}, skewness: {float(ligne_ref['skew_hist']):.6f}.</p>
      {texte_interpretation}
"""
    section_markov_skew_t = markov_skew_t_resume_html if markov_skew_t_resume_html else ""
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
        <li><b>Skew-t asymétrique</b> : capte simultanément asymétrie et queues épaisses observées.</li>
        <li><b>Skew-t ν→∞ (proxy skew-normal)</b> : test de version sans queues épaisses pour comparer la nécessité des extrêmes.</li>
        <li><b>Volatilité EWMA</b> : proxy GARCH léger, robuste et sans dépendance lourde.</li>
        <li><b>Markov-switching 2 régimes</b> : pertinent si alternance de régimes drift/vol détectable.</li>
        <li><b>Markov-switching 2 régimes + skew-t</b> : combine changements de régimes et asymétrie/queues épaisses par régime.</li>
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
    
    {section_markov_skew_t}

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
      {resume_densite}
      {figure_distribution_html}
    </section>
  </body>
</html>
"""


def _construire_section_markov_skew_t(modele: ModeleMarkovSwitchingSkewT) -> str:
    p00 = float(modele.transition[0, 0])
    p11 = float(modele.transition[1, 1])
    duree0 = 1.0 / max(1.0 - p00, 1e-6)
    duree1 = 1.0 / max(1.0 - p11, 1e-6)
    return f"""
    <section>
      <h2>MS(2)-Skew-t : paramètres estimés</h2>
      <p><b>Log-likelihood:</b> {modele.log_likelihood:.3f} | <b>AIC:</b> {modele.aic:.3f} | <b>BIC:</b> {modele.bic:.3f}</p>
      <p><b>Transition:</b> [[{modele.transition[0,0]:.4f}, {modele.transition[0,1]:.4f}], [{modele.transition[1,0]:.4f}, {modele.transition[1,1]:.4f}]]</p>
      <p><b>Durées moyennes (mois):</b> régime 0 = {duree0:.2f}, régime 1 = {duree1:.2f}</p>
      <p><b>Paramètres émission</b> — μ: [{modele.mu[0]:.6f}, {modele.mu[1]:.6f}], σ: [{modele.sigma[0]:.6f}, {modele.sigma[1]:.6f}], a: [{modele.a[0]:.4f}, {modele.a[1]:.4f}], b: [{modele.b[0]:.4f}, {modele.b[1]:.4f}]</p>
    </section>
"""


def executer_pipeline_univariee(
    chemin_csv: str,
    colonne_date: str,
    colonne_niveau: str,
    dossier_sortie: str,
    n_paths: int = 1000,
    inclure_markov_skew_t: bool = False,
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
        inclure_markov_skew_t=inclure_markov_skew_t,
        seed=seed,
    )
    modele_date = "skew_t_asymetrique_iid"
    resultats_dates, meilleure_date = detecter_meilleure_date_depart(
       serie_historique=serie,
       modele_a_tester=modele_date,
       fenetre_min_mois=60,
       pas_mois=6,
       n_paths=min(n_paths, 120),
       seed=seed,
   )
    #meilleure_date = pd.Timestamp('1980-01-01')
    
    resultats_dates_gauss, meilleure_date_gauss = detecter_date_stable_gaussienne(
        serie_historique=serie,
        fenetre_min_mois=60,
        pas_mois=3,
    )

    serie_optimale = serie.loc[meilleure_date:]
    # Toutes les lois sont calibrées sur la même fenêtre de rejeu (date optimale commune)
    # afin d'éviter des comparaisons de densité biaisées par des horizons historiques différents.
    fenetres_calibration = {
        "gaussien_iid": serie_optimale,
        "ar1_bruit_colore": serie_optimale,
        "student_t_iid": serie_optimale,
        "skew_t_asymetrique_iid": serie_optimale,
        "skew_t_asymetrique_nu_inf": serie_optimale,
        "volatilite_ewma": serie_optimale,
        "markov_switching_2_regimes": serie_optimale,
        "markov_switching_2_regimes_skew_t": serie_optimale,
    }
    _, _, simulations_optimales = comparer_strategies(
        serie_historique=serie_optimale,
        n_paths=n_paths,
        fenetres_calibration=fenetres_calibration,
        inclure_markov_skew_t=inclure_markov_skew_t,
        seed=seed,
    )

    sortie = Path(dossier_sortie)
    sortie.mkdir(parents=True, exist_ok=True)

    diagnostic_calibration = _evaluer_calibration_distributions(
        serie_historique=serie_optimale,
        simulations_par_modele=simulations_optimales,
    )
    fig = construire_figure_rejeu(serie_historique=serie_optimale, simulations_par_modele=simulations_optimales)
    fig_distribution = construire_figure_distribution_variations(
        serie_historique=serie_optimale,
        simulations_par_modele=simulations_optimales,
        diagnostic_calibration=diagnostic_calibration,
    )
    section_markov = ""
    if inclure_markov_skew_t:
        modele_markov_skew_t_resume = ModeleMarkovSwitchingSkewT.calibrer(serie_optimale)
        section_markov = _construire_section_markov_skew_t(modele_markov_skew_t_resume)
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
        markov_skew_t_resume_html=section_markov,
    )
    figure_path = sortie / "comparaison_fidelite.html"
    figure_path.write_text(rapport_html, encoding="utf-8")

    return resultats, meilleur_modele, figure_path
