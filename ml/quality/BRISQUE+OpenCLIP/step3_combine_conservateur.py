"""
ETAPE 3: Combinaison Mode Conservateur
Rejette seulement 10-15% (aberrations)
Respecte 100% annotations humaines
"""

import os
import pandas as pd

# Configuration
BRISQUE_DIR = "../data/pipeline_conservateur_stage1_brisque"
CLIP_DIR = "../data/pipeline_conservateur_stage2_clip"
OUTPUT_DIR = "../data/pipeline_conservateur_stage3_combined"
FINETUNING_CSV = f"{OUTPUT_DIR}/final_training_data_conservateur.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def combine_conservateur():
    """
    Combine resultats en mode conservateur
    """
    print("="*70)
    print("COMBINAISON MODE CONSERVATEUR")
    print("="*70)
    
    # Charger BRISQUE
    brisque_file = f"{BRISQUE_DIR}/brisque_conservateur_results.csv"
    df_brisque = pd.read_csv(brisque_file)
    
    # Charger OpenCLIP si disponible
    clip_file = f"{CLIP_DIR}/clip_conservateur_results.csv"
    df_clip = None
    if os.path.exists(clip_file):
        df_clip = pd.read_csv(clip_file)
        print(f"OpenCLIP disponible: {len(df_clip)} images")
    
    # Créer index OpenCLIP
    clip_dict = {}
    if df_clip is not None:
        for _, row in df_clip.iterrows():
            clip_dict[row['image_id']] = row
    
    all_data = []
    stats = {
        'humain_id': 0,
        'humain_non_id': 0,
        'brisque_reject': 0,
        'brisque_accept': 0,
        'clip_reject': 0,
        'clip_accept': 0,
        'pass_to_ml2': 0
    }
    
    print("\nTraitement...")
    
    for _, row in df_brisque.iterrows():
        image_id = row['image_id']
        filepath = row['filepath']
        category = row['original_category']
        brisque_score = row['brisque_score']
        
        # PRIORITÉ 1: Annotations humaines
        if category == 'identifiable':
            label = 1
            source = 'humain_identifiable'
            stats['humain_id'] += 1
        elif category == 'non_identifiable':
            label = 0
            source = 'humain_non_identifiable'
            stats['humain_non_id'] += 1
        
        # PRIORITÉ 2: BRISQUE rejets extrêmes
        elif row['brisque_decision'] == 'rejected_extreme':
            label = 0
            source = 'brisque_aberration'
            stats['brisque_reject'] += 1
        
        # PRIORITÉ 3: BRISQUE acceptations excellentes
        elif row['brisque_decision'] == 'accepted_excellent':
            label = 1
            source = 'brisque_excellent'
            stats['brisque_accept'] += 1
        
        # PRIORITÉ 4: OpenCLIP haute confiance
        elif image_id in clip_dict:
            clip_info = clip_dict[image_id]
            
            if clip_info['clip_decision'] == 'rejected_very_confident':
                label = 0
                source = 'clip_aberration'
                stats['clip_reject'] += 1
            elif clip_info['clip_decision'] == 'accepted_very_confident':
                label = 1
                source = 'clip_excellent'
                stats['clip_accept'] += 1
            else:
                # Confiance insuffisante → passe à ML2
                label = -1
                source = 'pass_to_ml2'
                stats['pass_to_ml2'] += 1
        
        # Par défaut: passe à ML2
        else:
            label = -1
            source = 'pass_to_ml2'
            stats['pass_to_ml2'] += 1
        
        all_data.append({
            'image_id': image_id,
            'filepath': filepath,
            'label': label,
            'source': source,
            'brisque_score': brisque_score
        })
    
    final_df = pd.DataFrame(all_data)
    
    # Sauvegarder tout
    final_df.to_csv(FINETUNING_CSV, index=False)
    
    # Dataset pour ML1 (seulement labels 0 et 1)
    ml1_dataset = final_df[final_df['label'].isin([0, 1])]
    ml1_csv = f"{OUTPUT_DIR}/ml1_training_data.csv"
    ml1_dataset.to_csv(ml1_csv, index=False)
    
    # Images passées à ML2
    ml2_images = final_df[final_df['label'] == -1]
    ml2_csv = f"{OUTPUT_DIR}/ml2_images.csv"
    ml2_images.to_csv(ml2_csv, index=False)
    
    # Statistiques
    print("\n" + "="*70)
    print("RESULTATS FINAUX")
    print("="*70)
    
    total = len(final_df)
    n_rejected = len(final_df[final_df['label'] == 0])
    n_accepted = len(final_df[final_df['label'] == 1])
    n_ml2 = len(final_df[final_df['label'] == -1])
    
    print(f"\nTotal images: {total}")
    print(f"\nREJETÉES (aberrations): {n_rejected} ({n_rejected/total*100:.1f}%)")
    print(f"  - Humains: {stats['humain_non_id']}")
    print(f"  - BRISQUE: {stats['brisque_reject']}")
    print(f"  - OpenCLIP: {stats['clip_reject']}")
    
    print(f"\nACCEPTÉES (excellentes): {n_accepted} ({n_accepted/total*100:.1f}%)")
    print(f"  - Humains: {stats['humain_id']}")
    print(f"  - BRISQUE: {stats['brisque_accept']}")
    print(f"  - OpenCLIP: {stats['clip_accept']}")
    
    print(f"\nPASSÉES À ML2/YOLO: {n_ml2} ({n_ml2/total*100:.1f}%)")
    
    # Vérification objectif
    print("\n" + "="*70)
    print("VERIFICATION OBJECTIF 10-15%")
    print("="*70)
    rejection_rate = n_rejected / total * 100
    print(f"Taux rejet: {rejection_rate:.1f}%")
    
    if rejection_rate < 10:
        print("Trop peu (<10%). Baisser seuils.")
    elif rejection_rate > 15:
        print("Trop de rejets (>15%). Augmenter seuils.")
    else:
        print("Dans l'objectif (10-15%)")
    
    # Dataset ML1
    print("\n" + "="*70)
    print("DATASET POUR ML1 (FINE-TUNING)")
    print("="*70)
    print(f"Total: {len(ml1_dataset)}")
    print(f"  Non-identifiables: {n_rejected}")
    print(f"  Identifiables: {n_accepted}")
    print(f"  Ratio: {n_accepted/n_rejected:.1f}:1" if n_rejected > 0 else "")
    
    print(f"\nFichiers créés:")
    print(f"  - Dataset ML1: {ml1_csv}")
    print(f"  - Images ML2: {ml2_csv}")

    
    return final_df


if __name__ == "__main__":
    combine_conservateur()
