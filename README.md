# Biolit — workflow de classification d'espèces (images)

Ce dépôt fournit un **template infra + data** pour lancer le pipeline de
classification et guider les bénévoles sur les 3 tâches ML :
1. **Qualité d'image** (filtrage)
2. **YOLOv8** (détection + crop)
3. **Classification hiérarchique** (règne → espèce)

Les pipelines existants d'export sont conservés et intégrés.

## Architecture (résumé)

1. **Entrée** : API quotidienne (à venir) ou CSV (`data/raw/observations.csv`).
2. **Filtrage qualité** (ML) → on garde les images utilisables.
3. **Détection YOLOv8** (ML) → bboxes + crops.
4. **Classification hiérarchique** (ML) → taxonomie.
5. **Label Studio** : boucle d'annotation/correction si besoin.
6. **Dataviz** : CSV compatible Metabase (puis dashboard).
7. **Exports** : CSV d'annotations (base de données plus tard).

## Structure du repo

```
biolit/                # Lib Python (taxref, observations, dataviz)
cmd/                   # Script export existant (export INPN)
pipelines/             # Orchestration (ingestion CSV + export)
ml/                    # Dossiers des 3 tâches ML
dataviz/               # Docs dataviz
infra/                 # Docker Compose (Label Studio)
data/                  # Workspace local (non versionné)
```

### Dossiers data (proposés)

- `data/raw/` : CSV brut + images du jour (dump API)
- `data/staging/` : images filtrées qualité + métadonnées
- `data/crops/` : crops issus de YOLOv8
- `data/label-studio/files/` : images à annoter
- `data/exports/` : sorties CSV (annotations, qualité, etc.)
- `data/dataviz/` : CSV pour Metabase

## Installation

Ce projet utilise [uv](https://docs.astral.sh/uv/) pour la gestion des dépendances.

```bash
uv sync
```

Si besoin :

```bash
source .venv/bin/activate
```

## Flux quotidien (API → ML → Label Studio)

1. **Récupération quotidienne** depuis l'API (à venir) ou CSV local.
2. **Qualité** : si l'image est mauvaise → stop.
3. **YOLOv8** : détection + crop.
   - si aucune détection → **Label Studio (CROP)**
   - si crop manuel → retour vers **annotation**
4. **Classification** : prédiction + probabilité.
   - certitude faible → **Label Studio (pré-annotations + probas)**
   - certitude forte → export direct
5. **Export CSV** : `data/exports/annotations.csv`
6. **Dataviz** : `data/dataviz/observations.csv` (Metabase)

## Pipelines (CSV → export + dataviz)

### 1) Ingestion CSV

Placez votre CSV dans `data/raw/observations.csv`, puis :

```bash
uv run pipelines/run.py ingest-csv --input-path data/raw/observations.csv
```

Résultat : `data/export_biolit.csv` (utilisé par `biolit.observations`).

### 2) Export INPN + dataviz

```bash
uv run pipelines/run.py export-inpn
```

Sorties principales :
- `data/biolit_valid_observations.parquet`
- `data/observations_missing_taxref.csv`
- `data/biolit_observation_missing_nom.csv`
- `data/biolit_observation_validated_non_identifiable.csv`
- `data/distribution_images.html`

## Label Studio (annotation)

```bash
docker compose -f infra/docker-compose.yml up
```

UI : http://localhost:8080

Les images à annoter sont montées depuis `data/label-studio/files`.

## Déploiement local

Il est possible de lancer l'ensemble en local pour les premiers tests.
L'objectif est d'étudier les sorties de chaque modèle avant d'automatiser
le workflow complet.


## Contribution

### Pre-commit

```bash
pre-commit run --all-files
```

### Tests

```bash
tox -vv
```
