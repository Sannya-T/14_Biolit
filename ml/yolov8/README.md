# ML - YOLOv8 (détection + crop)

Objectif : détecter l'espèce (végétal/animal) et générer un crop centré sur l'objet.

## Entrées

- Images : `data/staging/images/`
- Métadonnées : `data/raw/observations.csv`

## Sorties attendues

- Bboxes + classes : `data/exports/yolov8_detections.csv`
- Images crops : `data/crops/images/`

## Routage

- Si aucune détection → **Label Studio (CROP)**.
