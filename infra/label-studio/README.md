# Label Studio (Biolit)

## Lancer en local

```bash
docker compose -f infra/docker-compose.yml up
```

- UI: http://localhost:8080
- Les fichiers locaux sont accessibles via `/label-studio/files`.

## Configs suggerees

### 1) Qualite image (classification)

- Labels: `good`, `bad`, `blurry`, `overexposed`, `underexposed`.

### 2) Detection YOLOv8 + crop

- Outil: Bounding Box
- Classes: `vegetal`, `animal`

### 3) Classification hierarchique

- Champs libres pour `regne > phylum > classe > ordre > famille > genre > espece`.

## Pré-annotations (inférence ML)

Label Studio peut afficher les **prédictions ML** directement dans l'interface
(bounding boxes, classes, scores). On peut :

- pousser des prédictions via l'API Label Studio,
- ou brancher un ML backend (Label Studio ML).

C'est ce qui permettra d'avoir le visuel des inférences directement dans l'outil.
