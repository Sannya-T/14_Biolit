# Pipelines

Ce dossier regroupe les scripts d'orchestration et documente le **flux quotidien**.
Les pipelines utilisent le dossier `data/` (non versionné) comme espace de travail.

## Flux quotidien (API → ML → Label Studio)

1. **Récupération quotidienne** depuis l'API (à venir) ou CSV local.
2. **Qualité** : si l'image est mauvaise → stop.
3. **YOLOv8** : détection + crop.
   - si aucune détection → **Label Studio (CROP)**
4. **Classification** : prédiction + probabilité.
   - certitude faible → **Label Studio (pré-annotations + probas)**
   - certitude forte → export direct
5. **Export CSV** : `data/exports/annotations.csv`
6. **Dataviz** : `data/dataviz/observations.csv` (Metabase)
