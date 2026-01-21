# Dataviz

Objectif : produire un **CSV de base** pour Metabase, puis brancher un dashboard
quand l'API sera disponible.

## CSV Metabase (cible)

- Fichier : `data/dataviz/observations.csv`
- Source : API quotidienne (à venir) ou CSV local

## Dataviz existante (Sankey)

La dataviz actuelle est générée via `biolit/visualisation/species_distribution.py`.
Elle produit un Sankey HTML à partir des observations valides.

```bash
uv run pipelines/run.py export-inpn
```

Sortie : `data/distribution_images.html`.
