# Pipeline de Filtrage Qualité Images Marines - BioLit ML1

Pipeline automatisé pour filtrer les images de qualité insuffisante avant détection d'espèces marines.

## Contexte

Ce projet fait partie du système BioLit de classification d'espèces marines littorales. Le module ML1 filtre les images de mauvaise qualité avant qu'elles ne passent à la détection d'objets (ML2 - YOLOv8).

**Philosophie** : "Mieux vaut laisser passer des images inutiles que supprimer des images utiles"

## Architecture

```
Images brutes
    |
ML1 (Filtrage qualité)
    |-- Rejetées (5-15%)
    |-- Acceptées (85-95%)
            |
        ML2 (YOLOv8)
```

## Approche

### Pipeline Conservateur

Le pipeline combine deux outils complémentaires :

1. **BRISQUE** - Qualité technique
   - Évalue flou, compression, bruit
   - Rapide, fonctionne sur CPU
   - Seuil conservateur : score >= 65

2. **OpenCLIP** - Qualité perceptuelle  
   - Évalue netteté visuelle, exposition
   - Utilise des prompts focus qualité
   - Haute confiance requise : >= 0.85

### Résultats

- 5.1% d'images rejetées (1,044 sur 20,387)
- 100% respect des annotations humaines
- Pas de training ML requis (zero-shot)

## Installation

```bash
# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer dépendances
pip install -r requirements.txt
```

## Structure des Données

### Input Requis

```
data/
└── sorted_by_quality/
    └── quality_sorting_summary.csv
```

Le fichier CSV doit contenir les colonnes :
- `image_id` : Identifiant unique
- `filepath` : Chemin vers l'image
- `category` : Catégorie (identifiable, non_identifiable, ne_sais_pas, non_annote)

### Output Généré

```
data/
├── pipeline_conservateur_stage1_brisque/
│   ├── brisque_conservateur_results.csv    # Tous les résultats BRISQUE
│   ├── rejected_extreme.csv                # Images rejetées (score >= 65)
│   ├── accepted_excellent.csv              # Images acceptées (score <= 30)
│   └── uncertain.csv                       # Images incertaines (30-65)
│
├── pipeline_conservateur_stage2_clip/
│   ├── clip_conservateur_results.csv       # Tous les résultats OpenCLIP
│   ├── rejected_very_confident.csv         # Rejetées haute conf (>= 0.85)
│   ├── accepted_very_confident.csv         # Acceptées haute conf
│   └── pass_to_ml2.csv                     # Confiance insuffisante
│
└── pipeline_conservateur_stage3_combined/
    ├── final_training_data_conservateur.csv  # Dataset complet
    ├── ml1_training_data.csv                 # Seulement labels 0 et 1
    └── ml2_images.csv                        # Images label -1 pour YOLOv8
```

## Utilisation

### Workflow Standard

```bash
# Étape 1 : Filtrage BRISQUE
python step1_brisque_conservateur.py

# Étape 2 : Filtrage OpenCLIP (sur incertains BRISQUE)
python step2_openclip_conservateur.py

# Étape 3 : Combinaison résultats
python step3_combine_conservateur.py
```

### Configuration des Chemins

Avant de lancer, vérifier les chemins dans chaque script :

**step1_brisque_conservateur.py** :
```python
INPUT_CSV = "../data/sorted_by_quality/quality_sorting_summary.csv"
OUTPUT_DIR = "../data/pipeline_conservateur_stage1_brisque"
```

**step2_openclip_conservateur.py** :
```python
INPUT_CSV = "../data/pipeline_conservateur_stage1_brisque/uncertain.csv"
OUTPUT_DIR = "../data/pipeline_conservateur_stage2_clip"
```

**step3_combine_conservateur.py** :
```python
BRISQUE_DIR = "../data/pipeline_conservateur_stage1_brisque"
CLIP_DIR = "../data/pipeline_conservateur_stage2_clip"
OUTPUT_DIR = "../data/pipeline_conservateur_stage3_combined"
```

## Configuration

### Ajuster les Seuils

**step1_brisque_conservateur.py** :
```python
BRISQUE_THRESHOLD_EXTREME = 65   # Score >= 65 : rejet
BRISQUE_THRESHOLD_EXCELLENT = 30 # Score <= 30 : accepté
# Entre 30-65 : passe à OpenCLIP
```

**step2_openclip_conservateur.py** :
```python
CONFIDENCE_THRESHOLD_REJECT = 0.85  # Confiance >= 0.85 requise pour rejet
```

### Impact des Seuils

- **Rejeter plus (10-15%)** : Baisser à 60-65 (BRISQUE) et 0.75-0.80 (CLIP)
- **Rejeter moins (3-5%)** : Augmenter à 70-75 (BRISQUE) et 0.90 (CLIP)

## Résultats du Pipeline

### Statistiques Globales

```
Total dataset : 20,387 images

Rejetées ML1 : 1,044 (5.1%)
├── Annotations humaines : 91
├── BRISQUE aberrations : 68
└── OpenCLIP haute conf : 885

Acceptées ML1 : 15,527 (76.2%)
├── Annotations humaines : 2,205
├── BRISQUE excellentes : 9,429
└── OpenCLIP haute conf : 3,893

Passées ML2/YOLO : 3,816 (18.7%)
```

### Métriques de Performance

- **Précision** : 100% respect annotations humaines
- **Temps BRISQUE** : ~0.1s/image (CPU)
- **Temps OpenCLIP** : ~0.5s/image (CPU acceptable, GPU recommandé)
- **Reproductibilité** : Déterministe
- **Pas de training requis** : Zero-shot

## Fichiers de Sortie

### ml1_training_data.csv

Contient toutes les images avec leur label final :
- `image_id` : ID image
- `filepath` : Chemin
- `label` : 0 (non-identifiable), 1 (identifiable), -1 (passe ML2)
- `source` : Source de la décision (humain, BRISQUE, OpenCLIP)
- `brisque_score` : Score BRISQUE

### ml2_images.csv

Liste des images (label = -1) à traiter par YOLOv8 :
- Qualité moyenne selon BRISQUE (35-65)
- Confiance OpenCLIP insuffisante (< 0.85)

## Approche Technique

### BRISQUE vs OpenCLIP

- **BRISQUE** : Qualité technique (compression, artefacts, bruit)
- **OpenCLIP** : Qualité perceptuelle (netteté visuelle, flou, exposition)
- **Complémentaires** : Une image peut avoir bon score BRISQUE mais être rejetée par CLIP (ex: techniquement OK mais visuellement floue)

### Pourquoi Pas de Fine-tuning ?

Plusieurs approches deep learning ont été testées :
- Weighted CrossEntropy (poids 15x, 50x)
- Focal Loss (alpha=5, gamma=2)
- Sous-échantillonnage classe majoritaire

**Résultat** : Échec systématique (rappel classe 0 = 35% au lieu de 75% requis)

**Cause** : Dataset trop déséquilibré (1,044 non-ID vs 15,527 ID)

**Conclusion** : Le pipeline simple BRISQUE + OpenCLIP (zero-shot) est plus robuste et efficace que le deep learning pour cette tâche spécifique.

## Dépendances Principales

- **piq** : Calcul score BRISQUE
- **open_clip_torch** : Modèle vision-langage OpenCLIP
- **torch** : Backend PyTorch
- **pandas** : Manipulation données
- **Pillow** : Traitement images

Voir `requirements.txt` pour la liste complète et les versions.

## Limitations

- **BRISQUE** : Détecte qualité technique, pas l'identifiabilité biologique
- **OpenCLIP** : Modèle générique, pas spécialisé biodiversité marine
- **Dataset BioLit** : Qualité globalement bonne (0.4% aberrations BRISQUE seulement)
- **Prompts** : En anglais (limitation modèle OpenCLIP)

## Améliorations Futures

1. **Court terme** : Ajuster seuils selon retours utilisateurs terrain
2. **Moyen terme** : Tester modèles spécialisés biodiversité (iNaturalist, EMODnet)
3. **Long terme** : Annoter 2,000-3,000 images supplémentaires classe minoritaire pour rendre viable le fine-tuning

## Notes d'Implémentation

### Gestion des Annotations Humaines

Le script `step3_combine_conservateur.py` donne **priorité absolue** aux annotations humaines :
- Si annotateur dit "identifiable" → label = 1 (même si BRISQUE/CLIP en désaccord)
- Si annotateur dit "non-identifiable" → label = 0 (même si BRISQUE/CLIP en désaccord)

### Ordre de Priorité

```
1. Annotations humaines (toujours respectées)
2. BRISQUE rejets extrêmes (>= 65)
3. BRISQUE acceptations excellentes (<= 30)
4. OpenCLIP haute confiance (>= 0.85)
5. Par défaut → passe à ML2
```

## Auteur

Sania (@Sannya-t) - Développé dans le cadre du projet BioLit - Filtrage qualité images marines littorales

## Licence

Usage interne projet BioLit
