# TODO_LIST.md — Modélisation des hypothèses macro (modelisation_hypotheses)

> Règles d’usage (résumé) :
> - Une tâche n’est cochée `[X]` que si elle est totalement terminée et qu’il ne reste aucune sous-tâche ouverte.
> - Si une tâche nécessite une décision / hypothèse structurante : pas de code, ajouter une sous-tâche `[] Question utilisateur : ...`.

---

Objectif : comprendre la structure statistique réelle des variables macro avant toute industrialisation.

Principe :
- Pas de YAML.
- Pas de CLI.
- Pas d’architecture complexe.
- Un notebook.
- Des données propres.
- Des graphiques.
- Des stats.
- Des analyses pertinentes et qui tranchent.

Bien s'assurer que toutes les tâches en amont sont réalisées avant de passer à la suivante. Des analyses ont déjà été menées sur des données initiales, mais il faut désormais reprendre depuis le début pour traiter toutes les données simultanément.

---

## 0) Setup minimal

[X] Utiliser le notebook `notebooks/01_exploration_generation.ipynb`
   - Notebook reconstruit en mode exploration empirique (sans YAML/CLI), avec nettoyage, stats, ACF et VAR(1).

---

## 1) Choix des variables à étudier

Variables cibles (mensuelles de préférence) :

[X] Inflation (CPI France / zone euro)
   - cpi_france_fred.csv
[X] Croissance salaire nominal (si dispo mensuel, sinon proxy)
   - Proxy V0.1 implémenté dans le notebook : `croissance_salaire_nominal_proxy = 1.5 * inflation` pour avancer sur l'analyse dynamique multivariée.
[] Indice loyers (IRL)
[] Prix immobilier (indice national)
[X] Rendement actions (MSCI World ou S&P 500)
   - S&P 500 utilisé, transformé en log-return mensuel.
[] Taux crédit immobilier
   - Attention il faut prendre celui en France car on applique les données Françaises.

---

## 2) Collecte des données

[X] Identifier source fiable (INSEE, Banque de France, FRED, etc.)
   - Source retenue pour V0: dataset public versionné `datasets/s-and-p-500` (GitHub).
[X] Télécharger données historiques longues (≥ 20 ans si possible)
   - Historique disponible sur plus de 100 ans dans le fichier récupéré.
[X] Sauvegarder brut en `data/raw/`
   - Fichier `data/raw/s_and_p_500.csv` ajouté.
[X] Documenter la source (URL + date extraction)
   - Documentation ajoutée dans `data/raw/SOURCES.md`.
[] Collecter les données pour toutes les autres variables à étudier.
    [X] Sous-tâches explicites créées pour chaque variable manquante.
    - Travail de cadrage terminé pour éviter une implémentation partielle fragile.
    [] Collecter l'indice loyers (IRL) France en série temporelle exploitable.
        - Cible: série officielle INSEE (trimestrielle), puis conversion mensuelle justifiée pour l'analyse jointe.
    [] Collecter l'indice de prix immobilier national France.
        - Cible: indice historique documenté (INSEE / BIS / Banque de France), avec fréquence explicite.
    [] Collecter un taux de crédit immobilier France (ou proxy validé).
        - Cible: série Banque de France ou BCE (MIR), avec justification si remplacement par un proxy taux long.
    [] Question utilisateur : en cas de sources multiples (INSEE/BCE/Banque de France), quelle source prioriser pour figer la version de référence ?

---

## 3) Nettoyage & normalisation

[] Mettre toutes les séries au même pas temporel (mensuel)
[] Aligner sur un index commun
[] Gérer trous / NaN (drop ou interpolation justifiée)
   - NaN gérés par `dropna()` après transformations.

---

## 4) Analyse statistique de base

Pour chaque variable :

[] Moyenne
[] Volatilité
[] Histogramme
[] Kurtosis / skewness
[] Test stationnarité (ADF)

Pour le système complet :

[] Matrice de corrélation
[] Autocorrélations (ACF)
[] Corrélations croisées (cross-corr)
[] Visualisation rolling mean / rolling vol

---

## 5) Compréhension dynamique

Questions à répondre (écrire conclusions dans le notebook) :

[] Les séries sont-elles stationnaires ?
[] Faut-il modéliser les niveaux ou les variations ?
[] Les corrélations sont-elles stables dans le temps ?
[] Y a-t-il une forte persistance (AR(1) élevé) ?
[] Y a-t-il des régimes évidents (inflation haute / basse) ?
   - Indice préliminaire via rolling stats; validation régime à confirmer après ajout des variables manquantes.

---

## 6) Prototype modèle simple

Sans framework complexe :

[] Estimer un VAR(1) via statsmodels
[] Vérifier stabilité (valeurs propres)
[] Simuler une trajectoire
[] Comparer stats simulées vs stats historiques

---

## Règle importante

Ne pas écrire :
- YAML
- CLI
- système d’export
- code propre moteur
