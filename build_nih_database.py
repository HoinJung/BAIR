import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModel

# Configuration
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw" / "iuchest"
MEDSIGLIP_MODEL_ID = "google/medsiglip-448"
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
OUTPUT_JSON = BASE_DIR / "data" / "generated" / "iuchest_nih_retrieval_dataset.json"
BATCH_SIZE = 32

def main():
    print("Loading metadata...")
    projections_df = pd.read_csv(DATA_DIR / "indiana_projections.csv")
    reports_df = pd.read_csv(DATA_DIR / "indiana_reports.csv")
    
    projections_df['uid'] = projections_df['uid'].astype(str)
    reports_df['uid'] = reports_df['uid'].astype(str)
    
    # Get frontal images only
    frontal_images = projections_df[projections_df['projection'] == 'Frontal']
    uid_to_image = dict(zip(frontal_images['uid'], frontal_images['filename']))
    
    reports_df.fillna("", inplace=True)
    uid_to_report = {}
    uid_to_problems = {}
    for _, row in reports_df.iterrows():
        uid = str(row['uid'])
        uid_to_report[uid] = f"Findings: {row['findings']}\n\nImpression: {row['impression']}"
        uid_to_problems[uid] = str(row['Problems']) if pd.notna(row['Problems']) else ""
        
    valid_uids = [uid for uid in uid_to_report.keys() if uid in uid_to_image]
    print(f"Found {len(valid_uids)} valid cases.")

    print(f"Loading {MEDSIGLIP_MODEL_ID}...")
    model = AutoModel.from_pretrained(MEDSIGLIP_MODEL_ID).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(MEDSIGLIP_MODEL_ID)
    
    all_embeddings = []
    print("Extracting image embeddings...")
    with torch.no_grad():
        for i in tqdm(range(0, len(valid_uids), BATCH_SIZE)):
            batch_uids = valid_uids[i:i+BATCH_SIZE]
            batch_imgs = [Image.open(DATA_DIR / "images_normalized" / uid_to_image[uid]).convert("RGB") for uid in batch_uids]
            
            inputs = processor(images=batch_imgs, return_tensors="pt").to(DEVICE)
            outputs = model.get_image_features(**inputs)
            all_embeddings.append(F.normalize(outputs, p=2, dim=1).cpu())
            
    embeddings_tensor = torch.cat(all_embeddings, dim=0) # [N, D]
    
    # Pre-compute similarities (cosine sim = dot product for L2 normalized vectors)
    print("Computing similarity matrix...")
    similarity_matrix = torch.mm(embeddings_tensor, embeddings_tensor.T) # [N, N]
    
    dataset = []
    print("Building Needle-in-a-Haystack contexts...")
    for target_idx, uid in enumerate(tqdm(valid_uids)):
        sims = similarity_matrix[target_idx].clone()
        # Prevent retrieving itself as a distractor
        sims[target_idx] = -2.0 
        
        # Get Top 4 nearest neighbors
        top_4_indices = torch.topk(sims, 4).indices.tolist()
        
        distractor_reports = [uid_to_report[valid_uids[idx]] for idx in top_4_indices]
        gt_report = uid_to_report[uid]
        
        # Assemble Context: Doc 1, Doc 2, GT (Doc 3), Doc 4, Doc 5
        context_parts = [
            f"--- Document 1 ---\n{distractor_reports[0]}",
            f"--- Document 2 ---\n{distractor_reports[1]}",
            f"--- Document 3 ---\n{gt_report}",
            f"--- Document 4 ---\n{distractor_reports[2]}",
            f"--- Document 5 ---\n{distractor_reports[3]}"
        ]
        nih_context = "\n\n".join(context_parts)
        
        dataset.append({
            "uid": uid,
            "image_filename": uid_to_image[uid],
            "ground_truth_report": gt_report,
            "ground_truth_problems": uid_to_problems[uid],
            "nih_context": nih_context
        })
        
    with open(OUTPUT_JSON, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved {len(dataset)} items to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
