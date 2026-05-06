#!/usr/bin/env python3
"""
Master Evaluation Script for Medical RAG Interventions (Dual-Mode Evaluation).
Evaluates both 'sentence' and 'full' modes simultaneously.
Supports selecting between 'f1' (relaxed partial credit) and 'accuracy' (exact set match).
Includes correct population denominators and MS-PoE (Text/Full) + Combo models.

GFR (Generation Failure Rate) uses the same minimal definition as
evaluate_facet: empty/whitespace, strip length < MIN_GEN_STRLEN,
or >= MIN_CONSECUTIVE_WORD_REPEATS consecutive identical whitespace-delimited tokens.
"""

from __future__ import annotations
import argparse
import json
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

# --- CONFIGURATION ---
REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = REPO_ROOT / "outputs" / "cache" / "srr_bert_f1_eval_cache.json"
MODEL_PATH = "StanfordAIMI/SRR-BERT-Upper" 
TOKENIZER_PATH = "microsoft/BiomedVLP-CXR-BERT-general"
MAX_LENGTH = 128
NO_FINDINGS_LABEL = "no finding"

DATASET_PATHS = {
    "iu-chest": str(REPO_ROOT / "data" / "generated" / "indiana_reports_with_pseudo_labels_dual.csv"),
}

# --- REPOSITORY LABEL MAPPING ---
REPO_LABEL_TO_ID = {
    "Pleural Effusion": 0, "Upper abdominal finding": 1, "Widened cardiac silhouette": 2,
    "Lung Finding": 3, "No Finding": 4, "Widened aortic contour": 5, "Pleural Thickening": 6,
    "Vascular finding": 7, "Consolidation": 8, "Pneumothorax": 9, "Subdiaphragmatic gas": 10,
    "Masslike opacity": 11, "Chest wall finding": 12, "Focal air space opacity": 13,
    "Segmental collapse": 14, "Fracture": 15, "Mediastinal mass": 16, "Solitary masslike opacity": 17,
    "Support Devices": 18, "Mediastinal finding": 19, "Pleural finding": 20, "Air space opacity": 21,
    "Diffuse air space opacity": 22, "Multiple masslike opacities": 23, "Musculoskeletal finding": 24
}
ID_TO_STRING = {v: k.lower() for k, v in REPO_LABEL_TO_ID.items()}

# --- UNIVERSAL DICTIONARY (Handles both 25 Upper and 55 Leaf labels) ---
UNIVERSAL_TO_MACRO = {
    "No Finding": "Normal",
    "Lung Finding": "Parenchymal Opacities & Infections",
    "Consolidation": "Parenchymal Opacities & Infections",
    "Air space opacity": "Parenchymal Opacities & Infections",
    "Focal air space opacity": "Parenchymal Opacities & Infections",
    "Diffuse air space opacity": "Parenchymal Opacities & Infections",
    "Segmental collapse": "Parenchymal Opacities & Infections",
    "Lung Lesion": "Parenchymal Opacities & Infections",
    "Pneumonia": "Parenchymal Opacities & Infections",
    "Atelectasis": "Parenchymal Opacities & Infections",
    "Aspiration": "Parenchymal Opacities & Infections",
    "Lung collapse": "Parenchymal Opacities & Infections",
    "Perihilar airspace opacity": "Parenchymal Opacities & Infections",
    "Air space opacity-multifocal": "Parenchymal Opacities & Infections",
    "Masslike opacity": "Nodules & Masses",
    "Solitary masslike opacity": "Nodules & Masses",
    "Multiple masslike opacities": "Nodules & Masses",
    "Mass/Solitary lung mass": "Nodules & Masses",
    "Nodule/Solitary lung nodule": "Nodules & Masses",
    "Cavitating mass with content": "Nodules & Masses",
    "Cavitating masses": "Nodules & Masses",
    "Emphysema": "Chronic / Airway Disease",
    "Fibrosis": "Chronic / Airway Disease",
    "Bronchiectasis": "Chronic / Airway Disease",
    "Widened cardiac silhouette": "Cardiac & Vascular",
    "Widened aortic contour": "Cardiac & Vascular",
    "Vascular finding": "Cardiac & Vascular",
    "Edema": "Cardiac & Vascular",
    "Pulmonary congestion": "Cardiac & Vascular",
    "Cardiomegaly": "Cardiac & Vascular",
    "Pericardial effusion": "Cardiac & Vascular",
    "Tortuous Aorta": "Cardiac & Vascular",
    "Calcification of the Aorta": "Cardiac & Vascular",
    "Enlarged pulmonary artery": "Cardiac & Vascular",
    "Pleural Effusion": "Pleural Abnormalities",
    "Pneumothorax": "Pleural Abnormalities",
    "Pleural Thickening": "Pleural Abnormalities",
    "Pleural finding": "Pleural Abnormalities",
    "Simple pneumothorax": "Pleural Abnormalities",
    "Loculated pneumothorax": "Pleural Abnormalities",
    "Tension pneumothorax": "Pleural Abnormalities",
    "Simple pleural effusion": "Pleural Abnormalities",
    "Loculated pleural effusion": "Pleural Abnormalities",
    "Pleural scarring": "Pleural Abnormalities",
    "Hydropneumothorax": "Pleural Abnormalities",
    "Pleural Other": "Pleural Abnormalities",
    "Mediastinal mass": "Mediastinal & Hilar",
    "Mediastinal finding": "Mediastinal & Hilar",
    "Hilar lymphadenopathy": "Mediastinal & Hilar",
    "Inferior mediastinal mass": "Mediastinal & Hilar",
    "Superior mediastinal mass": "Mediastinal & Hilar",
    "Pneumomediastinum": "Mediastinal & Hilar",
    "Tracheal deviation": "Mediastinal & Hilar",
    "Fracture": "Bones & Fractures",
    "Musculoskeletal finding": "Bones & Fractures",
    "Chest wall finding": "Bones & Fractures",
    "Acute humerus fracture": "Bones & Fractures",
    "Acute rib fracture": "Bones & Fractures",
    "Acute clavicle fracture": "Bones & Fractures",
    "Acute scapula fracture": "Bones & Fractures",
    "Compression fracture": "Bones & Fractures",
    "Shoulder dislocation": "Bones & Fractures",
    "Subdiaphragmatic gas": "Extrapulmonary / Other Soft Tissue",
    "Upper abdominal finding": "Extrapulmonary / Other Soft Tissue",
    "Hernia": "Extrapulmonary / Other Soft Tissue",
    "Subcutaneous Emphysema": "Extrapulmonary / Other Soft Tissue",
    "Pneumoperitoneum": "Extrapulmonary / Other Soft Tissue",
    "Support Devices": "Medical Devices & Hardware",
    "Suboptimal central line": "Medical Devices & Hardware",
    "Suboptimal endotracheal tube": "Medical Devices & Hardware",
    "Suboptimal nasogastric tube": "Medical Devices & Hardware",
    "Suboptimal pulmonary arterial catheter": "Medical Devices & Hardware",
    "Pleural tube": "Medical Devices & Hardware",
    "PICC line": "Medical Devices & Hardware",
    "Port catheter": "Medical Devices & Hardware",
    "Pacemaker": "Medical Devices & Hardware",
    "Implantable defibrillator": "Medical Devices & Hardware",
    "LVAD": "Medical Devices & Hardware",
    "Intraaortic balloon pump": "Medical Devices & Hardware"
}

NORMALIZED_MACRO_MAP = {}
for k, v in UNIVERSAL_TO_MACRO.items():
    norm_k = re.sub(r'[\u2013\u2014\-]', '-', k.lower()).strip()
    NORMALIZED_MACRO_MAP[norm_k] = v.lower()

# --- UTILS & MODEL LOADING ---
def get_text_hash(text: str, eval_mode: str) -> str:
    unique_string = f"{eval_mode}_{text}"
    return hashlib.md5(unique_string.encode('utf-8')).hexdigest()

def load_model_and_tokenizer(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device).eval()
    return tokenizer, model, device

def compute_f1(pred: set, gt: set) -> float:
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    tp = len(pred & gt)
    precision = tp / len(pred)
    recall = tp / len(gt)
    if precision + recall == 0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)

def compute_accuracy(pred: set, gt: set) -> float:
    """Exact set match accuracy."""
    return 1.0 if pred == gt else 0.0

# --- Minimal generation failure (keep in sync with evaluate_facet.py) ---
MIN_GEN_STRLEN = 5
MIN_CONSECUTIVE_WORD_REPEATS = 5


def is_minimal_generation_failure(text: str) -> bool:
    if not isinstance(text, str):
        return True
    t = text.strip()
    if len(t) < MIN_GEN_STRLEN:
        return True
    tokens = t.split()
    return _has_consecutive_identical_token_run(tokens, MIN_CONSECUTIVE_WORD_REPEATS)


def _has_consecutive_identical_token_run(tokens: List[str], min_run: int) -> bool:
    if len(tokens) < min_run:
        return False
    run = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            run += 1
            if run >= min_run:
                return True
        else:
            run = 1
    return False


def parse_labels(s) -> frozenset:
    if not s or (isinstance(s, float) and pd.isna(s)):
        return frozenset()
    
    out = set()
    for lab in str(s).split(";"):
        lab = lab.strip().lower()
        if not lab: continue
        
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

def to_macro_set(labels: frozenset) -> frozenset:
    out = set()
    for lab in labels:
        norm_lab = re.sub(r'[\u2013\u2014\-]', '-', lab).strip()
        out.add(NORMALIZED_MACRO_MAP.get(norm_lab, norm_lab))
    return frozenset(out)

def _clean_uid(val) -> str:
    return str(val).strip().replace(".0", "")

def _ingest_condition_text(texts, display_names, cond_key, disp_name, uid, text):
    if not text:
        return
    if cond_key not in texts:
        texts[cond_key] = {}
        display_names[cond_key] = disp_name
    texts[cond_key][uid] = text

# --- PREDICTION LOGIC ---
def _predict_single_pass(text: str, tokenizer, model, device) -> list:
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

def predict_labels(report_text: str, tokenizer, model, device, eval_mode: str) -> set:
    if report_text is None or (isinstance(report_text, float) and pd.isna(report_text)):
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
            preds = _predict_single_pass(sent, tokenizer, model, device)
            union.update(preds)
    else:
        preds = _predict_single_pass(report_text, tokenizer, model, device)
        union.update(preds)

    if len(union) > 1 and NO_FINDINGS_LABEL in union:
        union.discard(NO_FINDINGS_LABEL)
        
    return frozenset(union)
# --- METRICS COMPUTATION ---
def accumulate_medical_metrics_for_uids(
    uids: List[str],
    gt_sets,
    predicted_sets,
    display_names,
    metric: str = "f1",
    texts_by_cond: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Same counting rules as compute_metrics, but restricted to an explicit UID list
    (used for bootstrap resampling). UIDs should already satisfy overlap/predicate
    filters used in compute_metrics (non-empty GT, predictions present per condition).
    """
    all_conds = list(display_names.keys())
    conditions = [pk for pk in ["nr", "prompt", "ori_mid"] if pk in all_conds]
    other_conds = sorted([c for c in all_conds if c not in conditions], key=lambda x: display_names[x])
    conditions.extend(other_conds)

    score_fn = compute_f1 if metric == "f1" else compute_accuracy

    stats = {
        c: {
            "score_sum": 0.0,
            "corr_gain_sum": 0.0,
            "deg_loss_sum": 0.0,
            "recovery_gain_sum": 0.0,
            "strictly_cured_gain_sum": 0.0,
            "novel_recovery_gain_sum": 0.0,
        }
        for c in conditions
    }
    gfr_fail = {c: 0 for c in conditions}

    n_total = 0
    nr_imperfect_count = 0
    nr_has_correct_count = 0
    rag_imperfect_count = 0
    recorrupted_count = 0
    novel_cand_count = 0
    base_degraded_count = 0
    base_corrected_count = 0

    for uid in uids:
        gt = gt_sets[uid]
        if not gt:
            continue
        n_total += 1

        if texts_by_cond is not None:
            for c in conditions:
                raw = (texts_by_cond.get(c) or {}).get(uid, "")
                if is_minimal_generation_failure(raw):
                    gfr_fail[c] += 1

        score_nr = score_fn(predicted_sets.get("nr", {}).get(uid, set()), gt)
        score_ori = score_fn(predicted_sets.get("ori_mid", {}).get(uid, set()), gt)

        if score_nr < 1.0:
            nr_imperfect_count += 1
        if score_nr > 0.0:
            nr_has_correct_count += 1
        if score_ori < 1.0:
            rag_imperfect_count += 1

        is_recorrupted = score_nr > score_ori
        if is_recorrupted:
            recorrupted_count += 1
            base_degraded_count += 1

        is_novel_cand = (score_nr < 1.0) and (score_ori < 1.0)
        if is_novel_cand:
            novel_cand_count += 1

        if score_ori > score_nr:
            base_corrected_count += 1

        for c in conditions:
            score_c = score_fn(predicted_sets[c].get(uid, set()), gt)
            stats[c]["score_sum"] += score_c

            if score_c > score_nr:
                stats[c]["corr_gain_sum"] += score_c - score_nr
            if score_c < score_nr:
                stats[c]["deg_loss_sum"] += score_nr - score_c

            if score_c > score_ori:
                stats[c]["recovery_gain_sum"] += score_c - score_ori

            if is_recorrupted and score_c >= score_nr:
                stats[c]["strictly_cured_gain_sum"] += score_c - score_ori

            if is_novel_cand:
                best_baseline = max(score_nr, score_ori)
                if score_c > best_baseline:
                    stats[c]["novel_recovery_gain_sum"] += score_c - best_baseline

    return {
        "conditions": conditions,
        "stats": stats,
        "gfr_fail": gfr_fail,
        "n_total": n_total,
        "nr_imperfect_count": nr_imperfect_count,
        "nr_has_correct_count": nr_has_correct_count,
        "rag_imperfect_count": rag_imperfect_count,
        "recorrupted_count": recorrupted_count,
        "novel_cand_count": novel_cand_count,
        "base_degraded_count": base_degraded_count,
        "base_corrected_count": base_corrected_count,
        "metric": metric,
        "display_names": display_names,
    }


def compute_metrics(
    gt_sets,
    predicted_sets,
    display_names,
    mode_name,
    metric: str = "f1",
    texts_by_cond: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    all_conds = list(display_names.keys())
    # Sort ordering: Baselines first, then RAG baseline, then interventions
    conditions = [pk for pk in ["nr", "prompt", "ori_mid"] if pk in all_conds]
    other_conds = sorted([c for c in all_conds if c not in conditions], key=lambda x: display_names[x])
    conditions.extend(other_conds)

    common_uids = set(gt_sets.keys())
    for c in conditions:
        if c in predicted_sets:
            common_uids &= set(predicted_sets[c].keys())

    if not common_uids:
        raise ValueError(
            f"\n[{mode_name.upper()}] No overlapping UIDs found!\n"
            f"Ensure your baseline files and intervention JSONs share the same image IDs."
        )

    uids_master = sorted(u for u in common_uids if gt_sets[u])
    agg = accumulate_medical_metrics_for_uids(
        uids_master,
        gt_sets,
        predicted_sets,
        display_names,
        metric,
        texts_by_cond,
    )
    stats = agg["stats"]
    gfr_fail = agg["gfr_fail"]
    n_total = agg["n_total"]
    nr_imperfect_count = agg["nr_imperfect_count"]
    nr_has_correct_count = agg["nr_has_correct_count"]
    rag_imperfect_count = agg["rag_imperfect_count"]
    recorrupted_count = agg["recorrupted_count"]
    novel_cand_count = agg["novel_cand_count"]
    base_degraded_count = agg["base_degraded_count"]
    base_corrected_count = agg["base_corrected_count"]

    print(f"\n========== [{mode_name.upper()}] Dataset Statistics ==========")
    print(f"Evaluated samples (non-empty GT): {n_total}")
    
    # RAG 베이스라인의 전체 Correction / Degradation 요약
    avg_degradation = (stats["ori_mid"]["deg_loss_sum"] / nr_has_correct_count) if ("ori_mid" in stats and nr_has_correct_count) else 0
    avg_correction = (stats["ori_mid"]["corr_gain_sum"] / nr_imperfect_count) if ("ori_mid" in stats and nr_imperfect_count) else 0
    
    metric_str = metric.upper()
    print(f"Standard RAG Degradation (Recorrupted instances: {base_degraded_count}): -{avg_degradation*100:.2f}% {metric_str}")
    print(f"Standard RAG Correction (Improved instances: {base_corrected_count}): +{avg_correction*100:.2f}% {metric_str}")

    print("-" * 155)
    header_score = f"Avg {metric.capitalize()}"
    print(
        f"{'Condition':<35} | {header_score:<8} | {'GFR':<8} | {'Correction':<12} | "
        f"{'Degradation':<12} | {'Recovery':<12} | {'Strictly Cured':<15} | {'Novel Recovery'}"
    )
    print("-" * 155)

    if n_total == 0:
        raise ValueError(f"Overlap exists but all GT labels are empty in {mode_name}; cannot compute metrics.")

    for c in conditions:
        avg_score = stats[c]["score_sum"] / n_total
        
        # NR: no comparison metrics. ori_mid (Standard RAG): no Recovery/Strictly Cured/Novel Recovery (it's the RAG baseline).
        # prompt (Strong Prompt) and interventions: show all metrics including Recovery vs Standard RAG.
        if c == "nr":
            corr_str, deg_str, rec_str, sc_str, nov_str = "-", "-", "-", "-", "-"
        elif c == "ori_mid":
            corr_gain = (stats[c]["corr_gain_sum"] / nr_imperfect_count) if nr_imperfect_count else 0
            corr_str = f"+{corr_gain*100:5.2f}%"
            deg_loss = (stats[c]["deg_loss_sum"] / nr_has_correct_count) if nr_has_correct_count else 0
            deg_str = f"-{deg_loss*100:5.2f}%"
            rec_str, sc_str, nov_str = "-", "-", "-"
        else:
            # 5개 지표 계산 (정확히 매칭되는 독립 분모 사용)
            corr_gain = (stats[c]["corr_gain_sum"] / nr_imperfect_count) if nr_imperfect_count else 0
            corr_str = f"+{corr_gain*100:5.2f}%"
            
            deg_loss = (stats[c]["deg_loss_sum"] / nr_has_correct_count) if nr_has_correct_count else 0
            deg_str = f"-{deg_loss*100:5.2f}%"
            
            rec_gain = (stats[c]["recovery_gain_sum"] / rag_imperfect_count) if rag_imperfect_count else 0
            rec_str = f"+{rec_gain*100:5.2f}%"
            
            # <-- CHANGED: Calculate average F1 points recovered during strict cures
            sc_gain = (stats[c]["strictly_cured_gain_sum"] / recorrupted_count) if recorrupted_count else 0
            sc_str = f"+{sc_gain*100:5.2f}%"
            
            nov_gain = (stats[c]["novel_recovery_gain_sum"] / novel_cand_count) if novel_cand_count else 0
            nov_str = f"+{nov_gain*100:5.2f}%"

        disp = display_names[c][:34]
        if texts_by_cond is not None and n_total:
            gfr_str = f"{gfr_fail[c] * 100.0 / n_total:5.2f}%"
        else:
            gfr_str = "-"
        print(
            f"{disp:<35} | {avg_score*100:5.2f}% | {gfr_str:<8} | {corr_str:<12} | {deg_str:<12} | "
            f"{rec_str:<12} | {sc_str:<15} | {nov_str}"
        )
    print("=" * 155)
    if texts_by_cond is not None:
        print(
            "[GFR] Minimal generation failure rate over evaluated samples (same rules as "
            f"evaluate_facet: empty/whitespace, strip length < {MIN_GEN_STRLEN}, "
            f"or ≥{MIN_CONSECUTIVE_WORD_REPEATS} consecutive identical whitespace-delimited tokens)."
        )
# # --- METRICS COMPUTATION ---
# def compute_metrics(gt_sets, predicted_sets, display_names, mode_name, metric="f1"):
#     all_conds = list(display_names.keys())
#     # Sort ordering: Baselines first, then RAG baseline, then interventions
#     conditions = [pk for pk in ["nr", "prompt", "ori_mid"] if pk in all_conds]
#     other_conds = sorted([c for c in all_conds if c not in conditions], key=lambda x: display_names[x])
#     conditions.extend(other_conds)

#     common_uids = set(gt_sets.keys())
#     for c in conditions:
#         if c in predicted_sets:
#             common_uids &= set(predicted_sets[c].keys())

#     if not common_uids:
#         raise ValueError(
#             f"\n[{mode_name.upper()}] No overlapping UIDs found!\n"
#             f"Ensure your baseline files and intervention JSONs share the same image IDs."
#         )

#     score_fn = compute_f1 if metric == "f1" else compute_accuracy

#     stats = {c: {
#         "score_sum": 0.0, 
#         "corr_gain_sum": 0.0,
#         "deg_loss_sum": 0.0,
#         "recovery_gain_sum": 0.0,
#         "strictly_cured_count": 0,
#         "novel_recovery_gain_sum": 0.0,
#     } for c in conditions}
    
#     n_total = 0
    
#     # 5개의 독립된 분모(Population) 추적기
#     nr_imperfect_count = 0     # Correction 분모: Baseline < 1.0
#     nr_has_correct_count = 0   # Degradation 분모: Baseline > 0.0
#     rag_imperfect_count = 0    # Recovery 분모: StandardRAG < 1.0
#     recorrupted_count = 0      # Strictly Cured 분모: Baseline > StandardRAG
#     novel_cand_count = 0       # Novel Recovery 분모: Baseline < 1.0 AND StandardRAG < 1.0

#     base_degraded_count = 0
#     base_corrected_count = 0

#     for uid in common_uids:
#         gt = gt_sets[uid]
#         if not gt: continue
#         n_total += 1

#         score_nr = score_fn(predicted_sets.get("nr", {}).get(uid, set()), gt)
#         score_ori = score_fn(predicted_sets.get("ori_mid", {}).get(uid, set()), gt)
        
#         # 1. 분모(Denominators) 할당
#         if score_nr < 1.0: nr_imperfect_count += 1
#         if score_nr > 0.0: nr_has_correct_count += 1
#         if score_ori < 1.0: rag_imperfect_count += 1
        
#         is_recorrupted = score_nr > score_ori
#         if is_recorrupted: 
#             recorrupted_count += 1
#             base_degraded_count += 1
            
#         is_novel_cand = (score_nr < 1.0) and (score_ori < 1.0)
#         if is_novel_cand: 
#             novel_cand_count += 1
            
#         if score_ori > score_nr:
#             base_corrected_count += 1

#         # 2. 분자(Numerators) 할당
#         for c in conditions:
#             score_c = score_fn(predicted_sets[c].get(uid, set()), gt)
#             stats[c]["score_sum"] += score_c
            
#             # Correction & Degradation: Any vs Baseline
#             if score_c > score_nr: stats[c]["corr_gain_sum"] += (score_c - score_nr)
#             if score_c < score_nr: stats[c]["deg_loss_sum"] += (score_nr - score_c)
            
#             # Recovery: Any vs StandardRAG
#             if score_c > score_ori: stats[c]["recovery_gain_sum"] += (score_c - score_ori)
            
#             # Strictly Cured (C -> I -> C): Any vs (Baseline & RAG)
#             if is_recorrupted and score_c >= score_nr:
#                 stats[c]["strictly_cured_count"] += 1
                
#             # Novel Recovery (I -> I -> C): Any vs (Baseline & RAG)
#             if is_novel_cand:
#                 best_baseline = max(score_nr, score_ori)
#                 if score_c > best_baseline:
#                     stats[c]["novel_recovery_gain_sum"] += (score_c - best_baseline)

#     print(f"\n========== [{mode_name.upper()}] Dataset Statistics ==========")
#     print(f"Evaluated samples (non-empty GT): {n_total}")
    
#     # RAG 베이스라인의 전체 Correction / Degradation 요약
#     avg_degradation = (stats["ori_mid"]["deg_loss_sum"] / nr_has_correct_count) if ("ori_mid" in stats and nr_has_correct_count) else 0
#     avg_correction = (stats["ori_mid"]["corr_gain_sum"] / nr_imperfect_count) if ("ori_mid" in stats and nr_imperfect_count) else 0
    
#     metric_str = metric.upper()
#     print(f"Standard RAG Degradation (Recorrupted instances: {base_degraded_count}): -{avg_degradation*100:.2f}% {metric_str}")
#     print(f"Standard RAG Correction (Improved instances: {base_corrected_count}): +{avg_correction*100:.2f}% {metric_str}")

#     print("-" * 145)
#     header_score = f"Avg {metric.capitalize()}"
#     print(
#         f"{'Condition':<35} | {header_score:<8} | {'Correction':<12} | "
#         f"{'Degradation':<12} | {'Recovery':<12} | {'Strictly Cured':<15} | {'Novel Recovery'}"
#     )
#     print("-" * 145)

#     if n_total == 0:
#         raise ValueError(f"Overlap exists but all GT labels are empty in {mode_name}; cannot compute metrics.")

#     for c in conditions:
#         avg_score = stats[c]["score_sum"] / n_total
        
#         # Baselines (NR, Prompt, Standard RAG)는 Recovery 지표를 보여주지 않습니다
#         if c == "nr":
#             corr_str, deg_str, rec_str, sc_str, nov_str = "-", "-", "-", "-", "-"
#         elif c in ["prompt", "ori_mid"]:
#             corr_gain = (stats[c]["corr_gain_sum"] / nr_imperfect_count) if nr_imperfect_count else 0
#             corr_str = f"+{corr_gain*100:5.2f}%"
#             deg_loss = (stats[c]["deg_loss_sum"] / nr_has_correct_count) if nr_has_correct_count else 0
#             deg_str = f"-{deg_loss*100:5.2f}%"
#             rec_str, sc_str, nov_str = "-", "-", "-"
#         else:
#             # 5개 지표 계산 (정확히 매칭되는 독립 분모 사용)
#             corr_gain = (stats[c]["corr_gain_sum"] / nr_imperfect_count) if nr_imperfect_count else 0
#             corr_str = f"+{corr_gain*100:5.2f}%"
            
#             deg_loss = (stats[c]["deg_loss_sum"] / nr_has_correct_count) if nr_has_correct_count else 0
#             deg_str = f"-{deg_loss*100:5.2f}%"
            
#             rec_gain = (stats[c]["recovery_gain_sum"] / rag_imperfect_count) if rag_imperfect_count else 0
#             rec_str = f"+{rec_gain*100:5.2f}%"
            
#             sc_pct = (stats[c]["strictly_cured_count"] / recorrupted_count) if recorrupted_count else 0
#             sc_str = f"{sc_pct*100:5.2f}%"
            
#             nov_gain = (stats[c]["novel_recovery_gain_sum"] / novel_cand_count) if novel_cand_count else 0
#             nov_str = f"+{nov_gain*100:5.2f}%"

#         disp = display_names[c][:34]
#         print(
#             f"{disp:<35} | {avg_score*100:5.2f}% | {corr_str:<12} | {deg_str:<12} | "
#             f"{rec_str:<12} | {sc_str:<15} | {nov_str}"
#         )
#     print("=" * 145)

# --- FILE PARSING MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["iu-chest"], required=True)
    parser.add_argument("--intervention-json", type=str, nargs='+', required=True)
    parser.add_argument(
        "--baseline-json",
        type=str,
        default=None,
        help="Optional explicit baseline JSON path containing no_retrieval_answer and oracle_answer. "
             "If provided, this file is used first to fill baselines before any auto-search.",
    )
    parser.add_argument("--metric", type=str, choices=["f1", "accuracy"], default="f1", 
                        help="Choose evaluation metric: 'f1' (partial credit) or 'accuracy' (exact match).")
    args = parser.parse_args()
    
    print(f"Loading Ground Truth for {args.dataset.upper()}...")
    csv_path = DATASET_PATHS[args.dataset]
    df = pd.read_csv(csv_path)
    df["uid"] = df["uid"].astype(str).str.strip().str.replace(".0", "", regex=False)
    
    if "pseudo_labels_full" in df.columns and "pseudo_labels_sentence" in df.columns:
        gt_sets_fine = {
            "full": df.set_index("uid")["pseudo_labels_full"].map(parse_labels).to_dict(),
            "sentence": df.set_index("uid")["pseudo_labels_sentence"].map(parse_labels).to_dict()
        }
    elif "pseudo_labels" in df.columns:
        gt_single = df.set_index("uid")["pseudo_labels"].map(parse_labels).to_dict()
        gt_sets_fine = {"full": gt_single, "sentence": gt_single}
    else:
        raise ValueError(f"Unsupported GT CSV schema at {csv_path}.")

    gt_sets_macro = {
        "full": {uid: to_macro_set(labels) for uid, labels in gt_sets_fine["full"].items()},
        "sentence": {uid: to_macro_set(labels) for uid, labels in gt_sets_fine["sentence"].items()}
    }

    texts = {}
    display_names = {}
    input_paths = [Path(p) for p in args.intervention_json]
    for path in input_paths:
        with open(path, "r") as f:
            data = json.load(f)
            
        fname = path.stem
        file_id = fname.split('results_')[-1] if 'results_' in fname else fname
            
        # JSON 내부 키가 "combo_mid_answer"로 동일하더라도 파일명으로 Text/Full 모드를 똑똑하게 분리합니다.
        is_combo_full = "COMBO_FULL" in fname
        combo_disp_name = "BAIR + Ms-PoE Full" if is_combo_full else "BAIR + Ms-PoE Text"
            
        mapping = {
            "no_retrieval_answer": ("nr", "Baseline (No Retrieval)"),
            "prompt_baseline_answer": ("prompt", "Strong Prompt"),
            "oracle_answer": ("ori_mid", "Standard RAG"),
            "oracle_with_intervention": (f"intv_mid_{file_id}", "BAIR Intervention"),
            "longllmlingua_mid_answer": (f"lll_mid_{file_id}", "LongLLMLingua"),
            "longllmlingua_combo_mid_answer": (f"lll_combo_{file_id}", "BAIR + LongLLMLingua"),
            "mspoe_text_answer": (f"mspoe_text_{file_id}", "Ms-PoE Text"),
            "mspoe_full_answer": (f"mspoe_full_{file_id}", "Ms-PoE Full"),
            "combo_mid_answer": (f"combo_{file_id}", combo_disp_name),
            "combo_full_answer": (f"combo_full_{file_id}", "BAIR + Ms-PoE Full"),  # 안전장치로 추가
            "madrag_answer": (f"madrag_{file_id}", "MADRAG"),
            "madrag_combo_mid_answer": (f"madrag_combo_{file_id}", "MADRAG Combo (mid)"),
        }

        for e in data:
            uid = _clean_uid(e.get("uid") or e.get("image", ""))
            if not uid: continue
                
            for json_key, (cond_key, disp_name) in mapping.items():
                if e.get(json_key):
                    final_cond_key = cond_key if cond_key in ["nr", "prompt", "ori_mid"] else cond_key
                    _ingest_condition_text(texts, display_names, final_cond_key, disp_name, uid, e[json_key])

    required = {"nr": "no_retrieval_answer", "ori_mid": "oracle_answer"}
    missing = [k for k in required if k not in texts]

    # Prefer explicit baseline file when provided.
    if missing and args.baseline_json:
        baseline_path = Path(args.baseline_json)
        if not baseline_path.exists():
            raise FileNotFoundError(f"--baseline-json not found: {baseline_path}")

        with open(baseline_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"--baseline-json must contain a top-level JSON list, got {type(data)}")

        for e in data:
            uid = _clean_uid(e.get("uid") or e.get("image", ""))
            if not uid:
                continue
            if e.get("no_retrieval_answer"):
                _ingest_condition_text(texts, display_names, "nr", "Baseline (No Retrieval)", uid, e.get("no_retrieval_answer"))
            if e.get("oracle_answer"):
                _ingest_condition_text(texts, display_names, "ori_mid", "Standard RAG", uid, e.get("oracle_answer"))
            if e.get("prompt_baseline_answer"):
                _ingest_condition_text(texts, display_names, "prompt", "Strong Prompt", uid, e.get("prompt_baseline_answer"))

        missing = [k for k in required if k not in texts]

    if missing:
        candidate_paths = []
        for src in input_paths:
            parent = src.parent
            if not parent.exists(): continue
            for pat in ("*baselines*.json", "*baseline*.json", "*.json"):
                for cp in sorted(parent.glob(pat)):
                    if cp not in input_paths and cp not in candidate_paths:
                        candidate_paths.append(cp)

        intervention_uids = set()
        for cond_key, uid_to_text in texts.items():
            if cond_key not in {"nr", "prompt", "ori_mid"}:
                intervention_uids.update(uid_to_text.keys())
                
        best = None
        for cp in candidate_paths:
            with open(cp, "r") as f: data = json.load(f)
            nr_map, ori_map, prompt_map = {}, {}, {}
            for e in data:
                uid = _clean_uid(e.get("uid") or e.get("image", ""))
                if not uid: continue
                if e.get("no_retrieval_answer"): nr_map[uid] = e.get("no_retrieval_answer")
                if e.get("oracle_answer"): ori_map[uid] = e.get("oracle_answer")
                if e.get("prompt_baseline_answer"): prompt_map[uid] = e.get("prompt_baseline_answer")
            
            overlap_nr = len(intervention_uids & set(nr_map.keys())) if intervention_uids else len(nr_map)
            overlap_ori = len(intervention_uids & set(ori_map.keys())) if intervention_uids else len(ori_map)
            score = overlap_nr + overlap_ori
            
            if best is None or score > best["score"]:
                best = {"path": cp, "score": score, "nr_map": nr_map, "ori_map": ori_map, "prompt_map": prompt_map}

        if best and best["score"] > 0:
            for uid, txt in best["nr_map"].items():
                _ingest_condition_text(texts, display_names, "nr", "Baseline (No Retrieval)", uid, txt)
            for uid, txt in best["ori_map"].items():
                _ingest_condition_text(texts, display_names, "ori_mid", "Standard RAG", uid, txt)
            for uid, txt in best["prompt_map"].items():
                _ingest_condition_text(texts, display_names, "prompt", "Strong Prompt", uid, txt)

    missing = [k for k in required if k not in texts]
    if missing:
        raise ValueError(f"Baseline and Standard RAG required. Missing: {missing}.")

    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f: cache = json.load(f)

    tokenizer, model, device = None, None, None
    new_evals_count = 0
    
    predicted_sets_fine = {"full": {k: {} for k in texts.keys()}, "sentence": {k: {} for k in texts.keys()}}
    predicted_sets_macro = {"full": {k: {} for k in texts.keys()}, "sentence": {k: {} for k in texts.keys()}}
    
    base_keys = set()
    for cond in texts: base_keys.update(texts[cond].keys())
    
    print("Evaluating models and generating metrics...")
    for uid in tqdm(sorted(list(base_keys)), desc="Generating Predictions"):
        for condition in texts.keys():
            if uid not in texts[condition]: continue
                
            text = texts[condition][uid]
            if not text or "[Error]" in text:
                continue
            
            for mode in ["full", "sentence"]:
                text_hash = get_text_hash(text, mode)
                if text_hash in cache:
                    union_set = parse_labels(";".join(cache[text_hash]))
                else:
                    if model is None: tokenizer, model, device = load_model_and_tokenizer()
                    union_set = predict_labels(text, tokenizer, model, device, mode)
                    cache[text_hash] = list(union_set)
                    new_evals_count += 1
                
                predicted_sets_fine[mode][condition][uid] = union_set
                predicted_sets_macro[mode][condition][uid] = to_macro_set(union_set)

    if new_evals_count > 0:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f: json.dump(cache, f)

    compute_metrics(
        gt_sets_fine["full"],
        predicted_sets_fine["full"],
        display_names,
        "Full Mode - Original Labels",
        args.metric,
        texts,
    )
    # compute_metrics(
    #     gt_sets_macro["full"],
    #     predicted_sets_macro["full"],
    #     display_names,
    #     "Full Mode - Macro Labels",
    #     args.metric,
    #     texts,
    # )
    # compute_metrics(gt_sets_fine["sentence"], predicted_sets_fine["sentence"], display_names, "Sentence Mode - Original Labels", args.metric)
    # compute_metrics(gt_sets_macro["sentence"], predicted_sets_macro["sentence"], display_names, "Sentence Mode - Macro Labels", args.metric)

if __name__ == "__main__":
    main()
