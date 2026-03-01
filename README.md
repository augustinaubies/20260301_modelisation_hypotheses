# Modélisation macro — génération de trajectoires temporelles (V1)

Ce sous-projet Python est dédié à la **modélisation macro simplifiée** et à la **génération de trajectoires mensuelles** pour :

- `inflation`
- `croissance_salaire`
- `indexation_loyers`
- `revalorisation_immobiliere`
- `rendement_bourse`
- `taux_credit`

L’objectif de cette V1 est de poser une base propre, importable et versionnable, avec :

- un modèle **statique corrélé gaussien**,
- un modèle **VAR(1)** simple,
- un contrat d’export des **paramètres** via YAML (`version_schema: 1`),
- une CLI minimale,
- un notebook d’exploration.

> Convention V1 : toutes les valeurs représentent des **taux mensuels** (ex. `0.002` = `0,2%` par mois).

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

## Quickstart

### 1) Générer une trajectoire statique corrélée

```bash
python -m modelisation_macro.cli \
  --config config/exemple_statique.yaml \
  --horizon-mois 120 \
  --sortie outputs/traj_statique.csv \
  --seed 42
```

### 2) Générer une trajectoire VAR(1)

```bash
python -m modelisation_macro.cli \
  --config config/exemple_var1.yaml \
  --horizon-mois 360 \
  --sortie outputs/traj_var1.csv \
  --seed 42
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

Le notebook `notebooks/01_exploration_generation.ipynb` :

- charge un YAML d’exemple,
- génère une trajectoire mensuelle sur 30 ans,
- trace des variables clés (`inflation`, `rendement_bourse`, `taux_credit`),
- affiche la corrélation empirique.

Le notebook utilise exclusivement le code de `src/`.

## TODO connus (V2+)

- calibration depuis historique réel,
- dépendances non gaussiennes (copules),
- modèles multi-régimes,
- diagnostics avancés et validation statistique.
