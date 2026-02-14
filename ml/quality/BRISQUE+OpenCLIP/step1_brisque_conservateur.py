"""
ETAPE 1: Filtrage BRISQUE - MODE CONSERVATEUR
Rejette UNIQUEMENT les aberrations (10-15% du dataset)

Philosophie: Mieux vaut laisser passer des inutiles que supprimer des utiles
Objectif: Filtrer les cas extremes uniquement
"""

import os
import torch
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from piq import brisque
    BRISQUE_AVAILABLE = True
except ImportError:
    print("ATTENTION: piq non installe. Installation requise:")
    print("pip install piq")
    BRISQUE_AVAILABLE = False
    exit(1)


# Configuration MODE CONSERVATEUR
INPUT_CSV = "../data/sorted_by_quality/quality_sorting_summary.csv"
OUTPUT_DIR = "../data/pipeline_conservateur_stage1_brisque"

# SEUILS AJUSTÉS (rejette 10-15%)
BRISQUE_THRESHOLD_EXTREME = 65  # A varié si besoin
BRISQUE_THRESHOLD_EXCELLENT = 30  # A varié si besoin

os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_brisque_score(img_path):
    """
    Calcule le score BRISQUE d'une image
    """
    try:
        img = Image.open(img_path).convert("L")
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        score = brisque(tensor, data_range=1.0, reduction='none').item()
        return score
    except Exception as e:
        print(f"Erreur BRISQUE pour {img_path}: {e}")
        return 100.0


def process_images_conservateur(csv_file):
    """
    Mode conservateur: rejette seulement aberrations
    """
    print("="*70)
    print("BRISQUE - MODE CONSERVATEUR (ABERRATIONS UNIQUEMENT)")
    print("="*70)
    print(f"Seuil rejet: >= {BRISQUE_THRESHOLD_EXTREME} (tres strict)")
    print(f"Objectif: Rejeter 10-15% du dataset\n")
    
    df = pd.read_csv(csv_file)
    print(f"Total images: {len(df)}")
    
    results = []
    stats = {
        'rejected_extreme': 0,  # >= 75 : Aberrations
        'accepted_excellent': 0,  # <= 25 : Excellent
        'uncertain': 0  # Entre 25-75 : A passer à ML2/YOLO
    }
    
    print("\nCalcul scores BRISQUE...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row['filepath']
        category = row.get('category', 'unknown')
        image_id = row['image_id']
        
        brisque_score = calculate_brisque_score(img_path)
        
        # Decision TRÈS conservative
        if brisque_score >= BRISQUE_THRESHOLD_EXTREME:
            decision = 'rejected_extreme'
            final_label = 0  # Non-identifiable
            stats['rejected_extreme'] += 1
            notes = 'Aberration BRISQUE'
        elif brisque_score <= BRISQUE_THRESHOLD_EXCELLENT:
            decision = 'accepted_excellent'
            final_label = 1  # Identifiable
            stats['accepted_excellent'] += 1
            notes = 'Excellente qualite'
        else:
            decision = 'uncertain'
            final_label = -1  # Passe à OpenCLIP
            stats['uncertain'] += 1
            notes = 'Qualite moyenne, passe a OpenCLIP'
        
        results.append({
            'image_id': image_id,
            'filepath': img_path,
            'original_category': category,
            'brisque_score': brisque_score,
            'brisque_decision': decision,
            'final_label': final_label,
            'notes': notes
        })
    
    results_df = pd.DataFrame(results)
    
    # Sauvegarder
    output_csv = f"{OUTPUT_DIR}/brisque_conservateur_results.csv"
    results_df.to_csv(output_csv, index=False)
    
    # Fichiers separes
    for decision in ['rejected_extreme', 'accepted_excellent', 'uncertain']:
        subset = results_df[results_df['brisque_decision'] == decision]
        subset.to_csv(f"{OUTPUT_DIR}/{decision}.csv", index=False)
    
    # Statistiques
    print("\n" + "="*70)
    print("RESULTATS MODE CONSERVATEUR")
    print("="*70)
    total = len(df)
    
    print(f"\nABERRATIONS rejetees (>= {BRISQUE_THRESHOLD_EXTREME}): {stats['rejected_extreme']} ({stats['rejected_extreme']/total*100:.1f}%)")
    print(f"EXCELLENTES acceptees (<= {BRISQUE_THRESHOLD_EXCELLENT}): {stats['accepted_excellent']} ({stats['accepted_excellent']/total*100:.1f}%)")
    print(f"INCERTAINES (pour OpenCLIP): {stats['uncertain']} ({stats['uncertain']/total*100:.1f}%)")
    
    # Verification objectif 10-15%
    rejection_rate = stats['rejected_extreme'] / total * 100
    print("\n" + "="*70)
    print("VERIFICATION OBJECTIF")
    print("="*70)
    print(f"Taux rejet: {rejection_rate:.1f}%")
    
    if rejection_rate < 10:
        print("Trop peu de rejets (<10%). Considerez baisser le seuil a 70.")
    elif rejection_rate > 15:
        print("Trop de rejets (>15%). Considerez augmenter le seuil a 80.")
    else:
        print("Taux de rejet dans l'objectif (10-15%)")
    
    # Analyse par categorie originale
    print("\n" + "="*70)
    print("ANALYSE PAR CATEGORIE ORIGINALE")
    print("="*70)
    
    for cat in results_df['original_category'].unique():
        subset = results_df[results_df['original_category'] == cat]
        rejected = len(subset[subset['brisque_decision'] == 'rejected_extreme'])
        print(f"\n{cat}: {len(subset)} images")
        print(f"  Rejetes: {rejected} ({rejected/len(subset)*100:.1f}%)")
    
    print(f"\nFichiers crees:")
    print(f"  - Resultats: {output_csv}")
    print(f"  - Aberrations: {OUTPUT_DIR}/rejected_extreme.csv")
    print(f"  - Incertaines: {OUTPUT_DIR}/uncertain.csv")
    
    print("\n" + "="*70)
    print("PROCHAINE ETAPE")
    print("="*70)
    print(f"Traiter les {stats['uncertain']} incertaines avec OpenCLIP:")
    print("python step2_openclip_conservateur.py")
    
    return results_df


if __name__ == "__main__":
    if not BRISQUE_AVAILABLE:
        print("Installation requise: pip install piq")
        exit(1)
    
    process_images_conservateur(INPUT_CSV)
