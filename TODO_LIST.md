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

---

## 0) Setup minimal

[] Utiliser le notebook `notebooks/01_exploration_generation.ipynb`

---

## 1) Choix des variables à étudier

Variables cibles (mensuelles de préférence) :

[] Inflation (CPI France / zone euro)
[] Croissance salaire nominal (si dispo mensuel, sinon proxy)
[] Indice loyers (IRL)
[] Prix immobilier (indice national)
[] Rendement actions (MSCI World ou S&P 500)
[] Taux crédit immobilier

---

## 2) Collecte des données

[] Identifier source fiable (INSEE, Banque de France, FRED, etc.)
[] Télécharger données historiques longues (≥ 20 ans si possible)
[] Sauvegarder brut en `data/raw/`
[] Documenter la source (URL + date extraction)

---

## 3) Nettoyage & normalisation

[] Mettre toutes les séries au même pas temporel (mensuel)
[] Aligner sur un index commun
[] Gérer trous / NaN (drop ou interpolation justifiée)
[] Transformer en variables modélisables :
   - Inflation → taux mensuel
   - Prix immo → log-return
   - Bourse → log-return
   - Taux crédit → niveau ou variation ?

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

---

## 6) Prototype modèle simple

Sans framework complexe :

[] Estimer un VAR(1) via statsmodels
[] Vérifier stabilité (valeurs propres)
[] Simuler une trajectoire
[] Comparer stats simulées vs stats historiques

---

## 7) Décision structurante (à la fin seulement)

Après exploration :

[] Décider si :
   - Modèle statique corrélé suffit
   - VAR(1) suffit
   - Modèle à régimes nécessaire
   - Copule nécessaire
[] Décider quelles variables garder / retirer
[] Décider transformation finale canonique (log-return ou croissance simple)

---

## Règle importante

Ne pas écrire :
- YAML
- CLI
- système d’export
- code propre moteur

tant que les points 1 → 6 ne sont pas validés.

---

But final de cette phase :
Avoir une compréhension empirique solide avant toute architecture.
[] Rédiger une “note d’intégration” (pas de code) pour l'intégration à la simulation en aval
    [] Stratégie Monte Carlo aval :
        - tirer paramètres (statique/VAR) puis générer trajectoire
        - ou tirer directement la trajectoire
    [] Question utilisateur : modèle de taux de crédit aval ?
    - Taux fixé à la date de signature du prêt (recommandé) vs taux variable pendant le prêt (plus complexe). Réponse : taux fixe

---