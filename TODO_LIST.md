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
[X] Croissance salaire nominal (si dispo mensuel, sinon proxy)
   - Proxy V0.1 implémenté dans le notebook : `croissance_salaire_nominal_proxy = 1.5 * inflation` pour avancer sur l'analyse dynamique multivariée.
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

[] Corriger transformation inflation
    [] Vérifier si `Consumer Price Index` est un niveau ou déjà une variation
    [] Si déjà variation : supprimer `pct_change()`
    [] Sinon : remplacer par vraie série CPI niveau
    [] Recalculer statistiques inflation (moyenne, vol, skew, kurtosis, ADF)

[] Remplacer source inflation par CPI niveau réel (France ou US cohérent avec S&P)
    [] Identifier source officielle fiable (INSEE / FRED)
    [] Télécharger série longue ≥ 30 ans
    [] Documenter source dans `data/raw/SOURCES.md`
    [] Mettre à jour notebook avec nouvelle série

[] Corriger modélisation taux_credit
    [] Tester stationnarité du niveau
    [] Calculer `delta_taux_credit = diff(taux_credit)`
    [] Tester stationnarité de la variation
    [] Comparer VAR niveau vs VAR en différences
    [] Décider variable canonique (niveau ou variation)

[] Refaire estimation VAR(1) avec données corrigées
    [] Vérifier stabilité (racines)
    [] Vérifier significativité coefficients
    [] Comparer stats simulées vs historiques

[] Tester normalité résidus VAR
    [] Jarque-Bera par équation
    [] Inspecter kurtosis résidus
    [] Décider si hypothèse gaussienne acceptable

[] Tester cointégration inflation / taux_credit
    [] Test de Johansen
    [] Décider VAR vs VECM

[] Analyser stabilité temporelle des corrélations
    [] Rolling corr 60 mois
    [] Identifier ruptures visibles (ex : post 2000, post 2008)

[] Vérifier présence d’hétéroscédasticité
    [] Rolling volatilité
    [] Test ARCH sur résidus actions

[] Décider transformations finales canoniques
    [] Inflation : variation simple ou log ?
    [] Actions : log-return confirmé ?
    [] Taux crédit : niveau ou variation ?
    [] Documenter choix dans notebook

[] Ajouter variable manquantes
    [] Croissance salaire nominal (source mensuelle ou proxy)
    [] Indice loyers (IRL)
    [] Prix immobilier national
    [] Harmoniser fréquence mensuelle

[] Refaire analyse complète avec système élargi (≥ 5 variables)
    [] Stats descriptives
    [] Corrélations
    [] ACF
    [] VAR(1)
    [] Diagnostic résidus

[] Décision modèle V1
    [] Statique corrélé suffisant ?
    [] VAR(1) retenu ?
    [] Régimes nécessaires ?
    [] Copule nécessaire ?

---

## Règle importante

Ne pas écrire :
- YAML
- CLI
- système d’export
- code propre moteur

