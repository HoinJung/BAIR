#!/usr/bin/env python3
"""
Collect strictly cured samples for gtonly setting.

Strictly cured definition follows eval_medical.py:
  is_recorrupted = score_nr > score_ori
  strictly_cured if is_recorrupted and score_intv >= score_nr

Where:
  score_nr   : baseline no_retrieval_answer score vs GT
  score_ori  : baseline oracle_answer score vs GT
  score_intv : intervention oracle_with_intervention score vs GT
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, FrozenSet, List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = "StanfordAIMI/SRR-BERT-Upper"
TOKENIZER_PATH = "microsoft/BiomedVLP-CXR-BERT-general"
MAX_LENGTH = 128
NO_FINDINGS_LABEL = "no finding"

DATASET_PATHS = {
    "mimic": BASE_DIR / "mimic_reports_with_pseudo_labels_dual.csv",
    "iu-chest": BASE_DIR / "indiana_reports_with_pseudo_labels_dual.csv",
}

REPO_LABEL_TO_ID = {
    "Pleural Effusion": 0,
    "Upper abdominal finding": 1,
    "Widened cardiac silhouette": 2,
    "Lung Finding": 3,
    "No Finding": 4,
    "Widened aortic contour": 5,
    "Pleural Thickening": 6,
    "Vascular finding": 7,
    "Consolidation": 8,
    "Pneumothorax": 9,
    "Subdiaphragmatic gas": 10,
    "Masslike opacity": 11,
    "Chest wall finding": 12,
    "Focal air space opacity": 13,
    "Segmental collapse": 14,
    "Fracture": 15,
    "Mediastinal mass": 16,
    "Solitary masslike opacity": 17,
    "Support Devices": 18,
    "Mediastinal finding": 19,
    "Pleural finding": 20,
    "Air space opacity": 21,
    "Diffuse air space opacity": 22,
    "Multiple masslike opacities": 23,
    "Musculoskeletal finding": 24,
}
ID_TO_STRING = {v: k.lower() for k, v in REPO_LABEL_TO_ID.items()}


def parse_labels(s) -> FrozenSet[str]:
    if not s or (isinstance(s, float) and pd.isna(s)):
        return frozenset()
    out = set()
    for lab in str(s).split(";"):
        lab = lab.strip().lower()
        if not lab:
            continue
        if lab.startswith("label_"):
            try:
                idx = int(lab.split("_")[1])
                if idx in ID_TO_STRING:
                    out.add(ID_TO_STRING[idx])
                    continue
            except ValueError:
                pass
        out.add(lab)
    return frozenset(out)


def _clean_uid(val) -> str:
    return str(val).strip().replace(".0", "")


def get_text_hash(text: str) -> str:
    # Keep full-mode prefix for cache compatibility with eval_medical.py.
    return hashlib.md5(f"full_{text}".encode("utf-8")).hexdigest()


def load_model_and_tokenizer(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device).eval()
    return tokenizer, model, device


def _predict_single_pass(text: str, tokenizer, model, device) -> List[str]:
    if not text or not str(text).strip():
        return []
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
    predicted_labels = [ID_TO_STRING[i] for i, flag in enumerate(preds) if flag and i in ID_TO_STRING]
    return predicted_labels if predicted_labels else [NO_FINDINGS_LABEL]


def predict_labels_full(text: str, tokenizer, model, device) -> FrozenSet[str]:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return frozenset()
    pred = set(_predict_single_pass(str(text).strip(), tokenizer, model, device))
    if len(pred) > 1 and NO_FINDINGS_LABEL in pred:
        pred.discard(NO_FINDINGS_LABEL)
    return frozenset(pred)


def compute_f1(pred: FrozenSet[str], gt: FrozenSet[str]) -> float:
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    tp = len(pred & gt)
    precision = tp / len(pred)
    recall = tp / len(gt)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def compute_accuracy(pred: FrozenSet[str], gt: FrozenSet[str]) -> float:
    return 1.0 if pred == gt else 0.0


def load_gt(dataset: str) -> Dict[str, FrozenSet[str]]:
    csv_path = DATASET_PATHS[dataset]
    df = pd.read_csv(csv_path)
    df["uid"] = df["uid"].astype(str).str.strip().str.replace(".0", "", regex=False)
    if "pseudo_labels_full" in df.columns:
        return df.set_index("uid")["pseudo_labels_full"].map(parse_labels).to_dict()
    if "pseudo_labels" in df.columns:
        return df.set_index("uid")["pseudo_labels"].map(parse_labels).to_dict()
    raise ValueError(f"Unsupported GT CSV schema: {csv_path}")


def load_baseline(path: Path) -> Dict[str, dict]:
    with open(path, "r") as f:
        data = json.load(f)
    out = {}
    for e in data:
        uid = _clean_uid(e.get("uid") or e.get("image", ""))
        if not uid:
            continue
        out[uid] = {
            "no_retrieval_answer": e.get("no_retrieval_answer", ""),
            "oracle_answer": e.get("oracle_answer", ""),
            "gt_report": e.get("gt_report", ""),
            "gt_problems": e.get("gt_problems", ""),
            "image_path": e.get("image_path", ""),
        }
    return out


def load_intervention(path: Path) -> Dict[str, str]:
    with open(path, "r") as f:
        data = json.load(f)
    out = {}
    for e in data:
        uid = _clean_uid(e.get("uid") or e.get("image", ""))
        if not uid:
            continue
        out[uid] = e.get("oracle_with_intervention", "")
    return out


def main():
    parser = argparse.ArgumentParser(description="Collect strictly cured gtonly samples.")
    parser.add_argument(
        "--baseline-json",
        type=str,
        default=str(BASE_DIR / "generation_results_medgemma_gtonly" / "iuchest_medgemma_results_baselines_ctx_gt_only.json"),
    )
    parser.add_argument(
        "--intervention-json",
        type=str,
        default=str(BASE_DIR / "generation_results_medgemma_gtonly" / "iuchest_medgemma_results_new_bair_av1.0_at1.0_gs1.0_mid_ctx_gt_only.json"),
    )
    parser.add_argument("--dataset", type=str, choices=["mimic", "iu-chest"], default="iu-chest")
    parser.add_argument("--metric", type=str, choices=["f1", "accuracy"], default="f1")
    parser.add_argument(
        "--cache-file",
        type=str,
        default=str(BASE_DIR / "srr_bert_f1_eval_cache.json"),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output json path. Default: alongside intervention json.",
    )
    parser.add_argument(
        "--include-empty-gt",
        action="store_true",
        help="Include empty GT rows. Default matches eval_medical.py behavior (skip empty GT).",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline_json)
    intervention_path = Path(args.intervention_json)
    if args.output_json:
        out_json = Path(args.output_json)
    else:
        out_json = intervention_path.parent / f"{intervention_path.stem}_strictly_cured_samples.json"

    gt = load_gt(args.dataset)
    baseline = load_baseline(baseline_path)
    intervention = load_intervention(intervention_path)
    score_fn = compute_f1 if args.metric == "f1" else compute_accuracy

    cache_path = Path(args.cache_file)
    cache = {}
    if cache_path.exists():
        with open(cache_path, "r") as f:
            cache = json.load(f)

    tokenizer, model, device = None, None, None
    new_evals = 0

    strictly_cured = []
    total_valid = 0
    recorrupted_count = 0

    common_uids = sorted(set(gt.keys()) & set(baseline.keys()) & set(intervention.keys()))
    for uid in tqdm(common_uids, desc="Scoring strictly-cured"):
        gt_labels = gt.get(uid, frozenset())
        if (not args.include_empty_gt) and len(gt_labels) == 0:
            continue

        row = baseline[uid]
        no_text = row.get("no_retrieval_answer", "")
        or_text = row.get("oracle_answer", "")
        intv_text = intervention.get(uid, "")
        if not str(no_text).strip() or not str(or_text).strip() or not str(intv_text).strip():
            continue

        preds = {}
        for key, text in [("nr", no_text), ("ori", or_text), ("intv", intv_text)]:
            h = get_text_hash(text)
            if h in cache:
                preds[key] = parse_labels(";".join(cache[h]))
            else:
                if model is None:
                    tokenizer, model, device = load_model_and_tokenizer()
                p = predict_labels_full(text, tokenizer, model, device)
                cache[h] = list(p)
                preds[key] = p
                new_evals += 1

        score_nr = score_fn(preds["nr"], gt_labels)
        score_ori = score_fn(preds["ori"], gt_labels)
        score_intv = score_fn(preds["intv"], gt_labels)
        total_valid += 1

        is_recorrupted = score_nr > score_ori
        if is_recorrupted:
            recorrupted_count += 1
        is_strictly_cured = is_recorrupted and (score_intv >= score_nr)
        if is_strictly_cured:
            strictly_cured.append(
                {
                    "uid": uid,
                    "score_no_retrieval": score_nr,
                    "score_oracle": score_ori,
                    "score_intervention": score_intv,
                    "gt_labels": sorted(list(gt_labels)),
                    "no_retrieval_predicted_labels": sorted(list(preds["nr"])),
                    "oracle_predicted_labels": sorted(list(preds["ori"])),
                    "intervention_predicted_labels": sorted(list(preds["intv"])),
                    "score_gap_nr_minus_oracle": score_nr - score_ori,
                    "restored_gain_vs_oracle": score_intv - score_ori,
                    "no_retrieval_answer": no_text,
                    "oracle_answer": or_text,
                    "oracle_with_intervention": intv_text,
                    "gt_report": row.get("gt_report", ""),
                    "gt_problems": row.get("gt_problems", ""),
                    "image_path": row.get("image_path", ""),
                }
            )

    if new_evals > 0:
        with open(cache_path, "w") as f:
            json.dump(cache, f)

    payload = {
        "dataset": args.dataset,
        "metric": args.metric,
        "strictly_cured_definition": "score_nr > score_oracle AND score_intervention >= score_nr",
        "baseline_json": str(baseline_path),
        "intervention_json": str(intervention_path),
        "total_valid_compared_samples": total_valid,
        "recorrupted_count": recorrupted_count,
        "strictly_cured_count": len(strictly_cured),
        "strictly_cured_rate_over_recorrupted": (len(strictly_cured) / recorrupted_count) if recorrupted_count else 0.0,
        "samples": strictly_cured,
    }

    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved strictly cured JSON: {out_json}")
    print(f"Strictly cured: {len(strictly_cured)} / recorrupted {recorrupted_count}")


if __name__ == "__main__":
    main()
