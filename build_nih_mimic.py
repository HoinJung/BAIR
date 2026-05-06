import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModel

# Configuration
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw" / "mimic-cxr"
IMAGE_DIR = DATA_DIR / "images"
METADATA_PATH = DATA_DIR / "metadata.json"
MEDSIGLIP_MODEL_ID = "google/medsiglip-448"
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"

# CHANGED: New output filename to preserve your old database
OUTPUT_JSON = BASE_DIR / "data" / "generated" / "mimic_nih_retrieval_dataset_findings_only.json"
BATCH_SIZE = 64 

def resolve_image_path(image_name: str, image_dir: Path, data_dir: Path):
    """Safely resolves image extensions and subdirectories for MIMIC."""
    candidate = image_dir / image_name
    if candidate.exists(): return candidate
    candidate = data_dir / image_name
    if candidate.exists(): return candidate
    stem = Path(image_name).stem
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists(): return candidate
        candidate = data_dir / f"{stem}{ext}"
        if candidate.exists(): return candidate
    return None

def main():
    print("Loading MIMIC-CXR metadata...")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing {METADATA_PATH}")
        
    with open(METADATA_PATH, "r") as f:
        raw_data = json.load(f)
        
    valid_items = []
    
    # CHANGED: Separate dictionaries for Findings vs Full Report
    uid_to_findings_only = {}
    uid_to_full_report = {}
    uid_to_image_path = {}
    
    print("Resolving image paths and splitting reports...")
    for row in tqdm(raw_data):
        image_name = str(row.get("image", "")).strip()
        if not image_name: continue
        
        resolved_path = resolve_image_path(image_name, IMAGE_DIR, DATA_DIR)
        if not resolved_path: continue
        
        findings = str(row.get("findings", "") or "").strip()
        impression = str(row.get("impression", "") or "").strip()
        full_report = f"Findings: {findings}\n\nImpression: {impression}".strip()
        
        # Fallback if a MIMIC report is completely missing the findings section
        if findings:
            findings_text = f"Findings: {findings}"
        else:
            findings_text = "Findings: No distinct findings section provided in historical record."
        
        uid_to_findings_only[image_name] = findings_text
        uid_to_full_report[image_name] = full_report
        uid_to_image_path[image_name] = resolved_path
        valid_items.append(image_name)
        
    print(f"Found {len(valid_items)} valid cases.")
    
    print(f"Loading {MEDSIGLIP_MODEL_ID}...")
    model = AutoModel.from_pretrained(MEDSIGLIP_MODEL_ID).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(MEDSIGLIP_MODEL_ID)
    
    all_embeddings = []
    print("Extracting image embeddings...")
    with torch.no_grad():
        for i in tqdm(range(0, len(valid_items), BATCH_SIZE)):
            batch_uids = valid_items[i:i+BATCH_SIZE]
            batch_imgs = [Image.open(uid_to_image_path[uid]).convert("RGB") for uid in batch_uids]
            
            inputs = processor(images=batch_imgs, return_tensors="pt").to(DEVICE)
            outputs = model.get_image_features(**inputs)
            all_embeddings.append(F.normalize(outputs, p=2, dim=1).cpu())
            
    embeddings_tensor = torch.cat(all_embeddings, dim=0) # [N, D]
    N = embeddings_tensor.size(0)
    
    dataset = []
    print("Computing top-4 similarities (Chunked) and building 'Findings-Only' Haystack contexts...")
    
    sim_batch_size = 1000
    for i in tqdm(range(0, N, sim_batch_size)):
        batch_emb = embeddings_tensor[i:i+sim_batch_size] # [B, D]
        sims = torch.mm(batch_emb, embeddings_tensor.T) # [B, N]
        
        for j in range(sims.size(0)):
            sims[j, i+j] = -2.0 
            
        top_4_indices = torch.topk(sims, 4, dim=1).indices.tolist()
        
        for j, top4 in enumerate(top_4_indices):
            target_uid = valid_items[i+j]
            gt_full_report = uid_to_full_report[target_uid]
            gt_findings_only = uid_to_findings_only[target_uid]
            
            # CHANGED: Fetch strictly the findings for the distractors
            distractor_findings = [uid_to_findings_only[valid_items[idx]] for idx in top4]
            
            # Assemble Context using ONLY Findings
            context_parts = [
                f"--- Document 1 ---\n{distractor_findings[0]}",
                f"--- Document 2 ---\n{distractor_findings[1]}",
                f"--- Document 3 ---\n{gt_findings_only}",
                f"--- Document 4 ---\n{distractor_findings[2]}",
                f"--- Document 5 ---\n{distractor_findings[3]}"
            ]
            nih_context = "\n\n".join(context_parts)
            
            dataset.append({
                "uid": target_uid,
                "image_path": str(uid_to_image_path[target_uid]), 
                "ground_truth_report": gt_full_report, # Retain the full truth for reference
                "ground_truth_problems": "", 
                "nih_context": nih_context # The context is now purely historical observations
            })
            
    with open(OUTPUT_JSON, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved {len(dataset)} items to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
