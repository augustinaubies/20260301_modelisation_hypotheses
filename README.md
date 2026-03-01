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
