Visualisations - Pipeline ML1
Ce dossier contient les visualisations générées par le pipeline de filtrage qualité.
Structure
Images/
├── visualisation_rejets_5pct/
│   ├── index.html                           # Navigation interactive
│   ├── 1_humain_page_*.png                  # Annotations humaines (91)
│   ├── 2_brisque_page_*.png                 # Aberrations BRISQUE (68)
│   ├── 3a_clip_good_brisque_page_*.png      # OpenCLIP + bonne qualité
│   ├── 3b_clip_medium_brisque_page_*.png    # OpenCLIP + qualité moyenne
│   └── 4_sample_random_page_*.png           # Échantillon aléatoire global
│
└── verification_non_annote_accepte/
    ├── index.html                           # Navigation interactive
    ├── 1_brisque_excellent_page_*.png       # BRISQUE excellentes
    ├── 2_clip_identifiable_page_*.png       # OpenCLIP identifiables
    ├── 3_global_random_page_*.png           # Échantillon aléatoire
    ├── 4a_brisque_0_15_page_*.png          # Très bonne qualité
    ├── 4b_brisque_15_25_page_*.png         # Bonne qualité
    ├── 4c_brisque_25_35_page_*.png         # Qualité moyenne
    └── 4d_brisque_35_75_clip_page_*.png    # Validées par OpenCLIP
Fichiers HTML Interactifs
Ouvrez les fichiers index.html dans un navigateur pour :

Navigation facile entre les grilles d'images
Statistiques globales
Organisation par catégorie

visualisation_rejets_5pct/index.html
Visualisation complète des 1,044 images rejetées (5.1%) organisées par source :
1. Annotations Humaines (91 images)

Images marquées "non-identifiable" par les observateurs
Respectées à 100% par le pipeline

2. Aberrations BRISQUE (68 images)

Score BRISQUE >= 65
Qualité technique très mauvaise (flou, compression)

3. Aberrations OpenCLIP (885 images)

3a. Avec bonne qualité BRISQUE (score <= 50)

Techniquement bonnes mais visuellement non-identifiables
Problèmes : angle, camouflage, distance


3b. Avec qualité moyenne BRISQUE (score > 50)

Problèmes combinés technique + perceptuel



4. Échantillon Aléatoire (100 images)

Vue d'ensemble représentative des rejets

verification_non_annote_accepte/index.html
Visualisation des images non-annotées acceptées qui passeront à YOLOv8 :
1. BRISQUE Excellentes (100 échantillons)

Score BRISQUE <= 30
Qualité technique excellente

2. OpenCLIP Identifiables (100 échantillons)

BRISQUE incertain (35-75) mais OpenCLIP haute confiance
Qualité perceptuelle validée

3. Échantillon Aléatoire Global (200 images)

Vue d'ensemble représentative

4. Distribution par Qualité BRISQUE

4a. Très bonnes (0-15) : 40 échantillons
4b. Bonnes (15-25) : 40 échantillons
4c. Moyennes (25-35) : 40 échantillons
4d. Validées OpenCLIP (35-75) : 40 échantillons

Grilles d'Images
Chaque grille PNG contient :

20 images par page (format 4x5)
ID de l'image
Score BRISQUE
Source de la décision
Prédiction et confiance OpenCLIP (si applicable)

Utilisation
bash# Ouvrir visualisation des rejets
firefox Images/visualisation_rejets_5pct/index.html

# Ouvrir visualisation des acceptés
firefox Images/verification_non_annote_accepte/index.html

# Ou sur Mac
open Images/visualisation_rejets_5pct/index.html
open Images/verification_non_annote_accepte/index.html

# Ou double-cliquer directement sur les fichiers index.html
Statistiques Globales
Images Rejetées (5.1%)
Total : 1,044 images
├── Humains : 91 (8.7%)
├── BRISQUE : 68 (6.5%)
└── OpenCLIP : 885 (84.8%)
Images Acceptées (76.2%)
Total : 15,527 images
├── Humains : 2,205 (14.2%)
├── BRISQUE : 9,429 (60.7%)
└── OpenCLIP : 3,893 (25.1%)
Images ML2 (18.7%)
Total : 3,816 images
└── Qualité moyenne à évaluer par YOLOv8

Les visualisations permettent de vérifier que :

Les annotations humaines sont bien respectées
BRISQUE détecte correctement les aberrations techniques
OpenCLIP identifie les problèmes perceptuels (même sur images techniquement bonnes)

Images Acceptées
Les visualisations montrent que :

La majorité a une excellente qualité technique (BRISQUE <= 30)
OpenCLIP valide la qualité perceptuelle pour les cas incertains
Les images sont adaptées pour la détection d'espèces par YOLOv8

Notes Techniques

Échantillonnage aléatoire : seed=42 pour reproductibilité
Grilles représentatives : Chaque catégorie est échantillonnée proportionnellement
Fichiers HTML standalone : Pas de dépendances externes, fonctionne offline
Format PNG : Haute résolution (150 DPI) pour vérification détaillée
