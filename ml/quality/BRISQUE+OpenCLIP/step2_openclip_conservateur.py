"""
ETAPE 2: Filtrage OpenCLIP - MODE CONSERVATEUR
Rejette UNIQUEMENT avec très haute confiance (>=0.85)

Remarque : Mieux vaut laisser passer des inutiles que supprimer des utiles
"""

import os
import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np

# token hugging face
#os.environ['HF_TOKEN'] = "hf_token_ici" # Optionnel

try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    print("ATTENTION: open_clip_torch non installe")
    print("pip install open_clip_torch")
    CLIP_AVAILABLE = False
    exit(1)


# Configuration MODE CONSERVATEUR
INPUT_CSV = "../data/pipeline_conservateur_stage1_brisque/uncertain.csv"
OUTPUT_DIR = "../data/pipeline_conservateur_stage2_clip"
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# SEUIL TRÈS STRICT (rejette seulement si TRÈS sûr)
CONFIDENCE_THRESHOLD_REJECT = 0.85  # Au lieu de 0.65

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prompts focus qualité
PROMPTS = {
    'identifiable': [
        "a clear sharp photograph",
        "a high quality well-focused image",
        "a sharp detailed photo with good lighting",
        "a crisp well-exposed photograph"
    ],
    'non_identifiable': [
        "a completely blurry unusable photo",
        "an extremely dark unreadable image",
        "a severely overexposed washed out picture",
        "a totally unfocused fuzzy mess"
    ]
}


def load_clip_model(device):
    """
    Charge OpenCLIP
    """
    print(f"Chargement OpenCLIP: {CLIP_MODEL}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, 
        pretrained=CLIP_PRETRAINED,
        device=device
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    return model, preprocess, tokenizer


def classify_image_clip(image_path, model, preprocess, tokenizer, device):
    """
    Classification OpenCLIP
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        all_prompts = PROMPTS['identifiable'] + PROMPTS['non_identifiable']
        text_inputs = tokenizer(all_prompts).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            logit_scale = model.logit_scale.exp()
            logits = (logit_scale * image_features @ text_features.T)
            similarity = logits.softmax(dim=-1)
            
            n_identifiable = len(PROMPTS['identifiable'])
            prob_identifiable = similarity[0, :n_identifiable].sum().item()
            prob_non_identifiable = similarity[0, n_identifiable:].sum().item()
        
        if prob_identifiable > prob_non_identifiable:
            prediction = 'identifiable'
            confidence = prob_identifiable
        else:
            prediction = 'non_identifiable'
            confidence = prob_non_identifiable
        
        return prediction, confidence, {
            'identifiable': prob_identifiable,
            'non_identifiable': prob_non_identifiable
        }
        
    except Exception as e:
        print(f"Erreur OpenCLIP: {e}")
        return 'error', 0.0, {}


def process_images_conservateur(csv_file, device):
    """
    Mode conservateur: rejette seulement si TRÈS confiant
    """
    print("="*70)
    print("OPENCLIP - MODE CONSERVATEUR (HAUTE CONFIANCE UNIQUEMENT)")
    print("="*70)
    print(f"Seuil rejet: >= {CONFIDENCE_THRESHOLD_REJECT} (tres strict)")
    print(f"Objectif: Rejeter seulement aberrations evidentes\n")
    
    model, preprocess, tokenizer = load_clip_model(device)
    
    df = pd.read_csv(csv_file)
    print(f"Images incertaines BRISQUE: {len(df)}")
    
    if len(df) == 0:
        print("Aucune image incertaine")
        return
    
    results = []
    stats = {
        'rejected_very_confident': 0,  # >= 0.85 non-ID
        'accepted_very_confident': 0,  # >= 0.85 ID
        'pass_to_ml2': 0  # < 0.85 : Laisse passer à ML2/YOLO
    }
    
    print("\nClassification OpenCLIP...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row['filepath']
        image_id = row['image_id']
        brisque_score = row['brisque_score']
        
        prediction, confidence, probs = classify_image_clip(
            img_path, model, preprocess, tokenizer, device
        )
        
        # Decision TRÈS conservative
        if prediction == 'non_identifiable' and confidence >= CONFIDENCE_THRESHOLD_REJECT:
            decision = 'rejected_very_confident'
            final_label = 0
            stats['rejected_very_confident'] += 1
            notes = f'OpenCLIP tres confiant non-ID (conf={confidence:.2f})'
        elif prediction == 'identifiable' and confidence >= CONFIDENCE_THRESHOLD_REJECT:
            decision = 'accepted_very_confident'
            final_label = 1
            stats['accepted_very_confident'] += 1
            notes = f'OpenCLIP tres confiant ID (conf={confidence:.2f})'
        else:
            decision = 'pass_to_ml2'
            final_label = -1  # Laisse passer
            stats['pass_to_ml2'] += 1
            notes = f'Confiance insuffisante (conf={confidence:.2f}), passe a ML2'
        
        results.append({
            'image_id': image_id,
            'filepath': img_path,
            'brisque_score': brisque_score,
            'clip_prediction': prediction,
            'clip_confidence': confidence,
            'clip_decision': decision,
            'final_label': final_label,
            'notes': notes
        })
    
    results_df = pd.DataFrame(results)
    
    # Sauvegarder
    output_csv = f"{OUTPUT_DIR}/clip_conservateur_results.csv"
    results_df.to_csv(output_csv, index=False)
    
    # Fichiers separes
    for decision in ['rejected_very_confident', 'accepted_very_confident', 'pass_to_ml2']:
        subset = results_df[results_df['clip_decision'] == decision]
        subset.to_csv(f"{OUTPUT_DIR}/{decision}.csv", index=False)
    
    # Statistiques
    print("\n" + "="*70)
    print("RESULTATS MODE CONSERVATEUR")
    print("="*70)
    total = len(df)
    
    print(f"\nREJETES haute confiance (>= {CONFIDENCE_THRESHOLD_REJECT}): {stats['rejected_very_confident']} ({stats['rejected_very_confident']/total*100:.1f}%)")
    print(f"ACCEPTES haute confiance (>= {CONFIDENCE_THRESHOLD_REJECT}): {stats['accepted_very_confident']} ({stats['accepted_very_confident']/total*100:.1f}%)")
    print(f"PASSES à ML2/YOLO: {stats['pass_to_ml2']} ({stats['pass_to_ml2']/total*100:.1f}%)")
    
    print(f"\nFichiers crees:")
    print(f"  - Resultats: {output_csv}")
    print(f"  - Rejetes: {OUTPUT_DIR}/rejected_very_confident.csv")
    print(f"  - Passes ML2: {OUTPUT_DIR}/pass_to_ml2.csv")
    
    print("\n" + "="*70)
    print("PROCHAINE ETAPE :")
    print("="*70)
    print("Combiner resultats BRISQUE + OpenCLIP:")
    print("python step3_combine_conservateur.py")
    
    return results_df


if __name__ == "__main__":
    if not CLIP_AVAILABLE:
        print("Installation: pip install open_clip_torch")
        exit(1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    process_images_conservateur(INPUT_CSV, device)
