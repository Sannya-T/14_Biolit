# ML - Classification hiérarchique

Objectif : prédire la taxonomie (règne → espèce) à partir des crops.

## Entrées

- Images : `data/crops/images/`
- Métadonnées : `data/raw/observations.csv`

## Sorties attendues

- Prédictions : `data/exports/classification_predictions.csv`
- Export final : `data/exports/annotations.csv`

## Routage

- **Certitude faible** → Label Studio (pré-annotation + probas)
- **Certitude forte** → export direct
