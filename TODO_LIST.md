# TODO_LIST — Modélisation macro

## Modélisation
- [ ] Ajouter un modèle multi-régime (switching simple)
- [ ] Ajouter un modèle de dépendance non-gaussienne (copule)
- [ ] Documenter précisément les conventions annualisées vs mensuelles

## Données
- [ ] Définir un schéma de données historique standardisé
- [ ] Ajouter la validation des fréquences / trous / outliers
- [ ] Préparer un dossier `data/` avec exemples synthétiques

## Calibration
- [ ] Implémenter calibration VAR(1) depuis historique réel
- [ ] Ajouter estimation robuste de covariance (shrinkage)
- [ ] Ajouter export de diagnostics de calibration

## Intégration
- [ ] Spécifier le contrat d’échange avec le moteur de simulation aval
- [ ] Ajouter une API d’export versionnée des paramètres
- [ ] Ajouter une CI minimale (lint + tests + exécution CLI)
