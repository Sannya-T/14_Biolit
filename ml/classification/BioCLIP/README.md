# 🌊 Identification automatique des espèces — Système hybride BioCLIP + enrichissement taxoonomique GBIF


**83.6% accuracy · 100 espèces · 88% de couverture**

---

## Approche

Le système repose sur [BioCLIP](https://huggingface.co/imageomics/bioclip), un modèle CLIP entraîné sur 10 millions d'images d'organismes vivants (TreeOfLife-10M). BioCLIP est utilisé comme **feature extractor gelé** (vecteurs 512d) — seule la tête de classification est adaptée aux données BioLit.

L'architecture finale est un **système hybride à deux composantes** :

- Un **classifier MLP** entraîné sur les 50 espèces les plus représentées (>50 images)
- Un **réseau prototypique** pour les 50 espèces rares (10–50 images), avec fusion visuel/textuel Proto-CLIP

Le routing entre les deux est déterminé par des seuils de confiance calibrés sur le jeu de validation.

Cette approche a été retenue car elle résout le déséquilibre fort des données BioLit : le classifier gère les espèces communes avec haute précision (89.7%), tandis que les prototypes permettent une classification few-shot des espèces rares sans surapprentissage (48.2%).

**Références scientifiques :**
- Prototypes pondérés par similarité : [arXiv:2110.11553](https://arxiv.org/abs/2110.11553)
- Température apprise par descente de gradient : [arXiv:2108.00340](https://arxiv.org/abs/2108.00340)
- Proto-CLIP (fusion visuel + textuel) : [arXiv:2307.03073](https://arxiv.org/abs/2307.03073)

---

## Résultats

| | Espèces communes (top 50) | Espèces rares (51–100) | Global |
|---|---|---|---|
| Accuracy | 89.7% | 48.2% | 83.6% |
| Couverture | — | — | 88.0% |

Le taux de rejet (12%) correspond aux images sous le seuil de confiance minimal — le modèle préfère ne pas répondre plutôt que de prédire avec une faible certitude.

---

## Modèles pré-entraînés

Télécharger les modèles pré-entrainés

| Fichier | Description | Lien |
|---|---|---|
| `best_model_top50.pth` | Classifier MLP — 50 espèces communes 
| `prototypes_v3.pt` | Prototypes Proto-CLIP — 100 espèces 

---

## Installation

```bash
pip install open-clip-torch torch torchvision pandas tqdm pillow
```

---

## Inférence sur nouvelles images

```bash
python scripts/inference/infer_local_v3.py \
  --images     /chemin/vers/dossier/ \
  --prototypes prototypes_v3.pt \
  --classifier best_model_top50.pth \
  --output     resultats.csv \
  --ext        webp        # ou jpg, png
```

**Colonnes du CSV de sortie :**

| Colonne | Description |
|---|---|
| `espece_pred` | Espèce prédite (`?` si rejetée) |
| `confiance` | Score de confiance [0–1] |
| `methode` | `classifier_top50` / `classifier_top50_low` / `prototypical_rare` / `rejected` |
| `top1_common` … `top3_proto` | Top 3 alternatives pour chaque composante |

---

## Structure des fichiers ajoutés

```
scripts/
└── inference/
    └── infer_local_v3.py   # Inférence hybride v3 (CPU/GPU)
```

---

## Espèces couvertes

Les 100 espèces correspondent aux espèces les plus observées dans la base BioLit au moment de l'entraînement (février 2026). La liste complète est disponible dans [`scripts/inference/infer_local_v3.py`](scripts/inference/infer_local_v3.py) (dictionnaire `SPECIES_DESCRIPTIONS`).
