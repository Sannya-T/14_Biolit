# ML - Qualité d'image

Objectif : filtrer les images trop floues ou mal exposées.

## Entrées

- Images du jour : `data/raw/images/`
- Métadonnées : `data/raw/observations.csv` (ou dump API)

## Sorties attendues

- Images validées : `data/staging/images/`
- Scores/labels : `data/exports/quality_predictions.csv`

## Routage

- Si qualité insuffisante → arrêt du flux.
