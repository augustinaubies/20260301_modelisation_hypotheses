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

[X] Utiliser le notebook `notebooks/01_exploration_generation.ipynb`
   - Notebook reconstruit en mode exploration empirique (sans YAML/CLI), avec nettoyage, stats, ACF et VAR(1).

---

## 1) Choix des variables à étudier

Variables cibles (mensuelles de préférence) :

[X] Inflation (CPI France / zone euro)
   - Implémenté en V0 avec le CPI de la source macro S&P (proxy US) pour lancer l'analyse statistique.
[] Croissance salaire nominal (si dispo mensuel, sinon proxy)
   - On pourrait peut être partir sur un facteur multiplicateur de l'inflation, disons 1.5.
[] Indice loyers (IRL)
[] Prix immobilier (indice national)
[X] Rendement actions (MSCI World ou S&P 500)
   - S&P 500 utilisé, transformé en log-return mensuel.
[] Taux crédit immobilier
   - Proxy initial via `Long Interest Rate` mensuel (donnée historique longue).
   - Attention il faut prendre celui en France car on applique les données Françaises.
[] Ajouter une source complémentaire pour couvrir les variables manquantes (salaire nominal, IRL, prix immobilier) en fréquence mensuelle

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

---

## 3) Nettoyage & normalisation

[X] Mettre toutes les séries au même pas temporel (mensuel)
[X] Aligner sur un index commun
[X] Gérer trous / NaN (drop ou interpolation justifiée)
   - NaN gérés par `dropna()` après transformations.
[X] Transformer en variables modélisables :
   - Inflation → taux mensuel
   - Prix immo → log-return
   - Bourse → log-return
   - Taux crédit → niveau ou variation ?

---

## 4) Analyse statistique de base

Pour chaque variable :

[X] Moyenne
[X] Volatilité
[X] Histogramme
[X] Kurtosis / skewness
[X] Test stationnarité (ADF)

Pour le système complet :

[X] Matrice de corrélation
[X] Autocorrélations (ACF)
[X] Corrélations croisées (cross-corr)
[X] Visualisation rolling mean / rolling vol

---

## 5) Compréhension dynamique

Questions à répondre (écrire conclusions dans le notebook) :

[X] Les séries sont-elles stationnaires ?
[X] Faut-il modéliser les niveaux ou les variations ?
[X] Les corrélations sont-elles stables dans le temps ?
[X] Y a-t-il une forte persistance (AR(1) élevé) ?
[X] Y a-t-il des régimes évidents (inflation haute / basse) ?
   - Indice préliminaire via rolling stats; validation régime à confirmer après ajout des variables manquantes.

---

## 6) Prototype modèle simple

Sans framework complexe :

[X] Estimer un VAR(1) via statsmodels
[X] Vérifier stabilité (valeurs propres)
[X] Simuler une trajectoire
[X] Comparer stats simulées vs stats historiques

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