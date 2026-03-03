# TODO_LIST.md — Modélisation des hypothèses macro (modelisation_hypotheses)

> Règles d’usage (résumé) :
> - Une tâche n’est cochée `[X]` que si elle est totalement terminée et qu’il ne reste aucune sous-tâche ouverte.
> - Si une tâche nécessite une décision / hypothèse structurante : pas de code, ajouter une sous-tâche `[] Question utilisateur : ...`.

---

Objectif : comprendre la structure statistique réelle des variables macro avant toute industrialisation.

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
[X] Indice loyers (IRL)
[X] Prix immobilier (indice national)
[X] Rendement actions (MSCI World ou S&P 500)
   - S&P 500 utilisé, transformé en log-return mensuel.
[X] Taux crédit immobilier
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
[X] Collecter les données pour toutes les autres variables à étudier.
    [X] Sous-tâches explicites créées pour chaque variable manquante.
    - Travail de cadrage terminé pour éviter une implémentation partielle fragile.
    [X] Collecter l'indice loyers (IRL) France en série temporelle exploitable.
        - Cible: série officielle INSEE (trimestrielle), puis conversion mensuelle justifiée pour l'analyse jointe.
    [X] Collecter l'indice de prix immobilier national France : IPL.
        - Cible: indice historique documenté (INSEE / BIS / Banque de France), avec fréquence explicite.
    [X] Collecter un taux de crédit immobilier France (ou proxy validé).
        - Cible: série Banque de France ou BCE (MIR), avec justification si remplacement par un proxy taux long.
        - taux_credit_habitation.csv

---

## 3) Nettoyage & normalisation

[X] Mettre toutes les séries au même pas temporel (mensuel)
   - IRL/IPL trimestriels interpolés linéairement au pas mensuel; autres séries converties et harmonisées en fin de mois.
[X] Aligner sur un index commun
   - Alignement inner-join sur la fenêtre commune 2010-10 à 2024-02.
[X] Gérer trous / NaN (drop ou interpolation justifiée)
   - Interpolation justifiée sur niveaux IRL/IPL puis `dropna()` final sur le jeu aligné.

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
   - Vérification faite dans le notebook via ADF, autocorr lag-1 et découpage régime inflation par médiane.

---

## 6) Prototype modèle simple

Sans framework complexe :

[X] Estimer un VAR(1) via statsmodels
[X] Vérifier stabilité (valeurs propres)
[X] Simuler une trajectoire
[X] Comparer stats simulées vs stats historiques

---

Les tâches ci-dessus ont toutes été réalisées.
Les conclusions principales de cette première partie sont les suivantes : 
- Sur la fenêtre commune 2010-10 → 2024-02 (161 mois), les corrélations contemporaines montrent actions ≈ indépendantes de l’inflation (ρ ≈ -0.007) et une relation faible inflation↔taux (ρ ≈ -0.035).
- L’indexation des loyers (IRL) est modérément corrélée à l’inflation (ρ ≈ 0.335) et quasi nulle avec la revalorisation immo (ρ ≈ -0.008). 
- Le couple le plus marqué est revalorisation immobilière ↔ taux de crédit avec une corrélation forte et négative (ρ ≈ -0.636). 
- Les séries montrent une forte persistance pour IRL, immo et surtout taux (autocorr lag1 ≈ 0.89, 0.94, 0.997), alors que les actions ont une autocorr faible (≈ 0.06). 
- Le VAR(1) est mathématiquement stable (racines > 1) et la corrélation des chocs (résidus) reste faible (|ρ| ≤ ~0.18), suggérant que l’essentiel du couplage vient des niveaux/dynamiques plutôt que de chocs communs.

Le prochain travail de modélisation se découpera alors en 4 volets :

1 ) Mise en place préalable des scripts et des dossiers de travail.
2 ) Modélisation indépendante de la bourse. 
3 ) Modélisation Inflation + indexation loyers (bloc corrélé)
4 ) Modélisation Revalorisation immobilière + taux de crédit (bloc corrélé)

TACHE 1 : Mise en place préalable des scripts et des dossiers de travail.
   [] Il faut que pour toutes les tâches suivantes, la pipeline de traitement des données soit similaire : 
      - chargement / prétraitement des données.
      - Tests de différentes stratégies de modélisation possibles (gaussienne, bruit coloré, VAR, ...)
      - Comparaison de la fidélité des différentes stratégies via un test de rejeu des données sur des tirages Monte Carlo.
      - Tracé d'un graphique résumant les résultats via plotly.
      - Conclusion sur la modélisation la plus adaptée à (aux) la (les) variable(s) actuellement étudiée(s).
      - Toutes cette pipeline sera effectuée exclusivement via des scripts Python, pour que Codex puisse les faire tourner dans son environnement, observer les sorties et faire une analyse critique des résultats.
   [] Pour l'instant ne fais cette tâche que pour préparer la tâche 2. Une fois la tâche 2 finie, nous répercuterons l'architecture terminée sur les autres tâches, pour éviter de devoir recopier les modifications à chaque itération. 

TACHE 2 : Modélisation indépendante de la bourse. 
   [] Il faut que ce script puisse accepter d'autres indices que le S&P500, donc il ne doit pas y avoir de noms dépendant de la série étudiée, et que les scripts soient robustes à un changement de données d'entrées (supposées tout de même au même format).
   [] Implémenter la pipeline d'identification sur cette partie...

TACHE 3 : 

TACHE 4 :