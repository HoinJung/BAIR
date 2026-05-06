#!/usr/bin/env python3
"""
Unified extraction script for dual-mode ground-truth pseudo-labels.
Supports IU-Chest CSV inputs.
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_PATH = "StanfordAIMI/SRR-BERT-Upper" 
TOKENIZER_PATH = "microsoft/BiomedVLP-CXR-BERT-general"
MAX_LENGTH = 128
NO_FINDINGS_LABEL = "No Finding"

def make_report(findings: str, impression: str) -> str:
    findings = str(findings if pd.notna(findings) else "").strip()
    impression = str(impression if pd.notna(impression) else "").strip()
    if findings or impression:
        return f"Findings: {findings}\n\nImpression: {impression}".strip()
    return ""

def load_model_and_tokenizer(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device).eval()
    idx2label = model.config.id2label 
    return tokenizer, model, idx2label, device

def _predict_single_pass(text: str, tokenizer, model, idx2label, device, apply_regex: bool) -> list:
    if not text or not str(text).strip():
        return []
    
    lowered = str(text).strip().lower()
    
    # if apply_regex:
    #     negation_pattern = r"\b(no|none|negative|without|absence of|clear of)\b"
    #     if re.search(negation_pattern, lowered):
    #         return [NO_FINDINGS_LABEL.lower()]

    inputs = tokenizer(
        str(text).strip(),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        preds = (torch.sigmoid(logits)[0].cpu().numpy() > 0.5).astype(int)
    
    predicted_labels = [idx2label[i].lower() for i, flag in enumerate(preds) if flag]
    return predicted_labels if predicted_labels else [NO_FINDINGS_LABEL.lower()]

def extract_labels(report_text: str, tokenizer, model, idx2label, device, eval_mode: str) -> set:
    if not report_text:
        return set()
    
    report_text = str(report_text).strip()
    union = set()
    
    if eval_mode == "sentence":
        import nltk
        try:
            sentences = nltk.sent_tokenize(report_text)
        except LookupError:
            nltk.download("punkt", quiet=True)
            sentences = nltk.sent_tokenize(report_text)
            
        for sent in sentences:
            preds = _predict_single_pass(sent, tokenizer, model, idx2label, device, apply_regex=True)
            union.update(preds)
    else:
        preds = _predict_single_pass(report_text, tokenizer, model, idx2label, device, apply_regex=False)
        union.update(preds)

    if len(union) > 1 and NO_FINDINGS_LABEL.lower() in union:
        union.discard(NO_FINDINGS_LABEL.lower())
        
    return union

def main():
    parser = argparse.ArgumentParser(description="Extract dual-mode SRR-BERT pseudo-labels.")
    parser.add_argument("--dataset", type=str, choices=["iu-chest"], required=True, 
                        help="Target dataset format to parse.")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to indiana_reports.csv or the directory containing it.")
    parser.add_argument("--output", type=str, required=True, 
                        help="Path to save the generated CSV with pseudo-labels.")
    args = parser.parse_args()

    print(f"Loading {args.dataset.upper()} data from {args.input} ...")
    input_path = Path(args.input)
    
    if args.dataset == "iu-chest":
        # Target the reports file directly to prevent image-level duplication
        if input_path.is_file():
            reports_csv = input_path
        else:
            reports_csv = input_path / "indiana_reports.csv"
            
        if not reports_csv.exists():
            raise FileNotFoundError(f"Missing required CSV: {reports_csv}")
            
        df = pd.read_csv(reports_csv)
        
        if "uid" not in df.columns:
            raise ValueError(f"{reports_csv.name} is missing the 'uid' column.")
            
        df["uid"] = df["uid"].astype(str).str.strip().str.replace(".0", "", regex=False)

    print("Constructing textual reports from findings + impression ...")
    df["report"] = df.apply(lambda row: make_report(row.get("findings"), row.get("impression")), axis=1)

    print("Loading SRR-BERT tokenizer and model...")
    tokenizer, model, idx2label, device = load_model_and_tokenizer()

    print(f"Computing dual-mode pseudo-labels for {len(df)} reports...")
    
    full_labels_list = []
    sentence_labels_list = []
    
    for idx, row in df.iterrows():
        full_set = extract_labels(row["report"], tokenizer, model, idx2label, device, "full")
        full_labels_list.append(";".join(full_set) if full_set else "")
        
        sentence_set = extract_labels(row["report"], tokenizer, model, idx2label, device, "sentence")
        sentence_labels_list.append(";".join(sentence_set) if sentence_set else "")
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(df)}")

    df["pseudo_labels_full"] = full_labels_list
    df["pseudo_labels_sentence"] = sentence_labels_list

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows with dual pseudo-labels to {out_path}")

if __name__ == "__main__":
    main()
