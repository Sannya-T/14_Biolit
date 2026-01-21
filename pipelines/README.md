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

## Scripts disponibles (MVP)

### 1) Ingestion CSV

Entrée attendue : `data/raw/observations.csv`

```bash
uv run pipelines/run.py ingest-csv --input-path data/raw/observations.csv
```

Résultat : `data/export_biolit.csv` (fichier utilisé par `biolit.observations.format_observations`).

### 2) Export INPN + Dataviz

```bash
uv run pipelines/run.py export-inpn
```

Sorties principales :
- `data/biolit_valid_observations.parquet`
- `data/observations_missing_taxref.csv`
- `data/biolit_observation_missing_nom.csv`
- `data/biolit_observation_validated_non_identifiable.csv`
- `data/distribution_images.html`
