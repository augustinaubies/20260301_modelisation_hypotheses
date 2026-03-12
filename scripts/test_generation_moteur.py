#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from modelisation_macro.bourse import generer_trajectoires_bourse  # noqa: E402


def _build_registry() -> dict[str, callable]:
    return {
        "bourse": generer_trajectoires_bourse,
    }


def _construire_figure_bourse(index: pd.DatetimeIndex, facteurs: np.ndarray, max_paths: int = 200) -> go.Figure:
    # facteurs: (n_mois, n_mc)
    niveaux = np.cumprod(facteurs, axis=0)
    n_mois, n_mc = niveaux.shape
    n_visu = min(n_mc, max_paths)

    fig = go.Figure()
    for i in range(n_visu):
        fig.add_trace(
            go.Scatter(
                x=index,
                y=niveaux[:, i],
                mode="lines",
                line=dict(color="rgba(120, 120, 120, 0.15)", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    couleur_ref = "#22d3ee"
    fig.add_trace(
        go.Scatter(
            x=index,
            y=niveaux.mean(axis=1),
            mode="lines",
            name="Moyenne",
            line=dict(color=couleur_ref, width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=index,
            y=np.median(niveaux, axis=1),
            mode="lines",
            name="Médiane",
            line=dict(color=couleur_ref, width=3, dash="dot"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title=f"Bourse — intégrale des variations ({n_mois} mois, {n_mc} trajectoires)",
        xaxis_title="Date",
        yaxis_title="Indice base 1",
        hovermode="x unified",
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Script de test modulaire de génération pour le moteur de simulation.")
    parser.add_argument("--date-debut", required=True)
    parser.add_argument("--date-fin", required=True)
    parser.add_argument("--n-monte-carlo", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variables", nargs="+", default=["bourse"], help="Variables à simuler, ex: bourse")
    parser.add_argument("--output-html", default="outputs/test_generation_moteur.html")
    args = parser.parse_args()

    index = pd.date_range(start=args.date_debut, end=args.date_fin, freq="MS")
    if len(index) == 0:
        raise ValueError("Horizon vide: vérifier date-debut/date-fin.")

    registry = _build_registry()
    figures_html: list[str] = []
    for variable in args.variables:
        if variable not in registry:
            raise ValueError(f"Variable inconnue: {variable}. Variables supportées: {sorted(registry)}")

        facteurs = registry[variable](
            date_debut=args.date_debut,
            date_fin=args.date_fin,
            n_monte_carlo=args.n_monte_carlo,
            seed=args.seed,
        )
        if variable == "bourse":
            fig = _construire_figure_bourse(index=index, facteurs=facteurs)
            figures_html.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    html = f"""
<!doctype html>
<html lang=\"fr\">
  <head>
    <meta charset=\"utf-8\" />
    <title>Test génération moteur</title>
    <style>
      body {{ background:#0f172a; color:#e2e8f0; font-family:Arial, sans-serif; margin:0; padding:24px; }}
      h1 {{ margin-top:0; }}
      .bloc {{ margin-bottom:24px; padding:16px; background:#111827; border-radius:8px; }}
    </style>
  </head>
  <body>
    <h1>Simulation de test — moteur (mode modulaire)</h1>
    <p>Variables simulées: {', '.join(args.variables)} | Horizon: {args.date_debut} → {args.date_fin} | Monte Carlo: {args.n_monte_carlo}</p>
    {''.join(f'<div class="bloc">{bloc}</div>' for bloc in figures_html)}
  </body>
</html>
"""

    output = Path(args.output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"Rapport HTML généré: {output}")


if __name__ == "__main__":
    main()
