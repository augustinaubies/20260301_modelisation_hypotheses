# Modélisation macro — génération de trajectoires temporelles

Ce sous-projet Python est dédié à la **modélisation macro simplifiée** et à la **génération de trajectoires mensuelles** pour :

- `inflation`
- `croissance_salaire`
- `indexation_loyers`
- `revalorisation_immobiliere`
- `rendement_bourse`
- `taux_credit`

> Convention : toutes les valeurs représentent des **taux mensuels** (ex. `0.002` = `0,2%` par mois).

## Installation

Pré-requis : Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Optionnel (dev + notebook + tests) :

```bash
pip install -e .[dev]
```

## Structure

```text
.
├── AGENT.md
├── TODO_LIST.md
├── config/
│   ├── exemple_statique.yaml
│   └── exemple_var1.yaml
├── notebooks/
│   └── 01_exploration_generation.ipynb
├── pyproject.toml
└── src/
    └── modelisation_macro/
        ├── __init__.py
        ├── cli.py
        ├── generation.py
        ├── io_yaml.py
        ├── types.py
        ├── variables.py
        └── calibration/
            ├── __init__.py
            └── calibrer_var1_depuis_historique.py
```

## Notebook

Les notebooks utilisent exclusivement le code de `src/`.


## Pipeline scripts (itération tâche 2)

Un script Python générique est disponible pour l'identification univariée (ex: indice actions):

```bash
python scripts/pipeline_identification_univariee.py \
  --input-csv data/raw/s_and_p_500.csv \
  --value-column SP500 \
  --output-dir outputs/task2_bourse
```

Sorties générées:
- série prétraitée en log-retours,
- tableau des scores de fidélité Monte Carlo,
- graphique Plotly de comparaison des modèles,
- conclusion textuelle sur le modèle recommandé.

## Génération standalone — bourse (sortie moteur)

Export des paramètres calibrés (à relancer après nouvelle identification):

```bash
python scripts/export_parametres_bourse.py
```

Génération des variations multiplicatives mensuelles pour la bourse:

```python
from modelisation_macro.bourse import generer_trajectoires_bourse

matrice = generer_trajectoires_bourse(
    date_debut="2026-01-01",
    date_fin="2026-12-01",
    n_monte_carlo=5000,
    seed=123,
)
# matrice.shape == (12, 5000)
```

Script de test modulaire (format proche d'un appel moteur + rapport HTML Plotly dark):

```bash
python scripts/test_generation_moteur.py --date-debut 2026-01-01 --date-fin 2030-12-01 --n-monte-carlo 1000 --variables bourse --output-html outputs/test_generation_moteur.html
```
