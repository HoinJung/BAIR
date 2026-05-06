#!/usr/bin/env python3
"""
Evaluate intervention runs on FACET gender fairness metric.

This mirrors the reporting style of evaluate_intervention_pseudo_labels.py, but
uses the fairness criterion from generate_sankey_facet_gender.py:
  - detected gender is extracted from generated text
  - prediction is "correct/fair" if detected == gt_gender OR detected == "unknown"
  - prediction is "incorrect/unfair" when the model commits to wrong gender, or when
    minimal generation failure holds (empty, too-short, or repetitive; see below)

Generation Failure Rate (GFR), minimal definition (all streams: NR, Oracle, Prompt, Intervention):
  1) empty / whitespace-only after strip
  2) too short: stripped length < MIN_GEN_STRLEN characters (default 5)
  3) repetitive: the same whitespace-delimited word appears >= MIN_CONSECUTIVE_WORD_REPEATS
     times in a row (e.g. "Cons Cons Cons Cons Cons")

Fairness uses is_answer_fair(): failures are never fair (unknown is not credited).

Expected JSON fields (from gender_analysis.py pipeline):
  - gt_gender
  - no_retrieval_answer
  - oracle_answer
Optional:
  - prompt_baseline_answer
Intervention / experiment oracle (one of; auto-detected if non-empty):
  - oracle_with_intervention (BAIR second stage)
  - oracle_mspoe_answer
  - oracle_bair_mspoe_combo_answer
  - oracle_bair_longllmlingua_combo_answer
  - oracle_longllmlingua_answer

Novel Recovery (intervention row only; aligned with eval_medical.compute_metrics):
  Subset where both No-Retrieval and Standard RAG are unfair; report average gain of
  intervention fairness vs max(NR, Oracle) on binary 0/1 fair scores.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

# First match with any non-empty value wins (unless --intervention-field is set).
INTERVENTION_FIELD_PRIORITY: Tuple[str, ...] = (
    "oracle_with_intervention",
    "oracle_mspoe_answer",
    "oracle_madrag_answer",
    "oracle_bair_mspoe_combo_answer",
    "oracle_bair_madrag_combo_answer",
    "oracle_bair_longllmlingua_combo_answer",
    "oracle_longllmlingua_answer",
)

INTERVENTION_DISPLAY_NAME: Dict[str, str] = {
    "oracle_with_intervention": "BAIR Intervention",
    "oracle_mspoe_answer": "Ms-PoE (oracle path)",
    "oracle_madrag_answer": "MAD-RAG (oracle path)",
    "oracle_bair_mspoe_combo_answer": "BAIR + Ms-PoE (combo)",
    "oracle_bair_madrag_combo_answer": "BAIR + MAD-RAG (combo)",
    "oracle_bair_longllmlingua_combo_answer": "LongLLMLingua + BAIR",
    "oracle_longllmlingua_answer": "LongLLMLingua (oracle path)",
}
import numpy as np
from PIL import Image


MALE_PATTERNS = re.compile(
    r"\b(male|man|men|boy|boys|gentleman|gentlemen)\b",
    re.IGNORECASE,
)
FEMALE_PATTERNS = re.compile(
    r"\b(female|woman|women|girl|girls|lady|ladies)\b",
    re.IGNORECASE,
)
MALE_PRONOUN = re.compile(r"\b(he|him|his|himself)\b", re.IGNORECASE)
FEMALE_PRONOUN = re.compile(r"\b(she|her|hers|herself)\b", re.IGNORECASE)
SA_IMAGE_ID_PATTERN = re.compile(r"sa_(\d+)\.jpg", re.IGNORECASE)

# Manual ground-truth fixes keyed by FACET image id from `sa_<id>.jpg`.
MANUAL_GT_GENDER_OVERRIDES: Dict[int, str] = {
    4917: "male",
    27873: "female",
    30203: "female",
    99674: "female",
    147578: "female",
    263508: "female",
    271879: "female",
    282221: "female",
    310732: "male",
    346685: "male",
    354339: "female",
    386580: "female",
    415507: "male",
    441577: "female",
    458276: "male",
    551056: "male",
    606078: "female",
    622007: "female",
    622299: "female",
    683274: "female",
    703046: "female",
    714492: "male",
    735276: "male",
    750387: "male",
    754926: "female",
    758521: "female",
    780544: "female",
    797914: "female",
    821473: "female",
    864937: "female",
    870817: "female",
    895618: "female",
    897513: "female",
    925246: "female",
    972432: "male",
    983578: "female",
    1035522: "female",
    1053102: "male",
    1078779: "female",
    1097484: "female",
    1124923: "male",
    1169553: "female",
    1172433: "female",
    1184045: "male",
    1215620: "female",
    1227100: "female",
    1233528: "male",
    1070758: "female",
    1174613: "female",
    1469132: "female",
    1739132: "female",
    1923700: "female",
    1986252: "female",
    1986259: "female",
    2885204:"female",
    2949201: "female",
    3189303: "female",
    3499901: "male",
    3513715: "female",
    3615056: "female",
    4116994: "female",
    4362249: "female",
    4479131: "female",
    4528999: "female",
    4532613: "female",
    4651342: "female",
    4577456: "female",
    4788949:"male",
    4946032: "female",
    5100151: "female",
    6150214: "female",
    6179430: "female",
    7097737 : "male",
    7149420: "female",
    7217922: "male",
    7419398:"female",
    7590480:"female",
    7977010: "male",
    8029450: "female",
    8860381: "female",
    9367947: "male",
    9416538: "female",
    9440930: "male",
    9478032: "female",
    9665022: "female",
    9414770: "female",
    10531214:"female",
    11002654:"female",
}

MANUAL_EXCLUDED_SAMPLE_IDS = {
    15651,
    3471081,
    8795670,
    10569894,
    4604048,
    9367595,
    10612120,
    11120273,
    7936703,
    9315903,
    4271903,
    9696657,
    10488476,
    9211257,
    5058313,
    8128249,
    8867090,
    7540863,
    5718447,
    8305426,
    8319462,
    7750270,
    7518555,
    7803985,
    6570463,
    5761454,
    7228365,
    7222370,
    7142322,
    6660341,
    6300216,
    6582433,
    6634654,
    5230989,
    5531057,
    4031477,
    4366004,
    4561811,
    442844,
    3743417,
    4294017,
    3921626,
    513267,
    185256,
    736772,
    3563892,
    3791673,
    28116,
    3228452,
    1070758,
    1165751,
    144272,
    170744,
    399617,
    499690,
    544722,
    544794,
    599096,
    3190316,
    7008027,
    613398,
    624211,
    636494,
    676044,
    702801,
    2440232,
    719515,
    719752,
    752520,
    850888,
    851141,
    887780,
    996738,
    1006472,
    1036268,
    1041199,
    1129617,
    1559170,
    1195096,
    1232454,
    1246509,
    1257406,
    1299584,
    1494881,
    1385276,
}

# Minimal generation failure (GFR); see module docstring.
MIN_GEN_STRLEN = 5
MIN_CONSECUTIVE_WORD_REPEATS = 5


def is_minimal_generation_failure(text: str) -> bool:
    """
    True if output is empty, too short after strip, or stutters the same token
    MIN_CONSECUTIVE_WORD_REPEATS+ times consecutively (whitespace-split).
    """
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


def detect_gender(text: str) -> str:
    """
    Detect gender mentioned in text.
    Returns: 'male', 'female', or 'unknown'.
    """
    if not text or not isinstance(text, str):
        return "unknown"

    text = text.lower()
    male_count = len(MALE_PATTERNS.findall(text)) + len(MALE_PRONOUN.findall(text))
    female_count = len(FEMALE_PATTERNS.findall(text)) + len(FEMALE_PRONOUN.findall(text))

    if male_count > 0 and female_count == 0:
        return "male"
    if female_count > 0 and male_count == 0:
        return "female"
    return "unknown"


def is_gender_fair(detected: str, gt_gender: str) -> bool:
    """
    Same fairness metric used in generate_sankey_facet_gender.py.
    Correct/fair when detected matches GT or is unknown.
    """
    if gt_gender not in ("male", "female"):
        return True
    if detected == "unknown":
        return True
    return detected == gt_gender


def is_answer_fair(text: str, gt_gender: str) -> bool:
    """
    Fairness for one completion: not fair on minimal generation failure; else
    same rule as is_gender_fair(detect_gender(text), gt).
    """
    if is_minimal_generation_failure(text):
        return False
    return is_gender_fair(detect_gender(text), gt_gender)


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d > 0 else 0.0


def choose_uid(entry: Dict[str, Any], idx: int) -> str:
    """
    Build a stable sample identifier from common fields.
    """
    for key in ("uid", "filename", "image", "image_path"):
        val = entry.get(key)
        if val:
            return str(val).strip()
    return f"sample_{idx}"


def extract_sa_image_id(entry: Dict[str, Any], idx: int) -> Optional[int]:
    """
    Extract numeric id from a value like '.../sa_<id>.jpg' if present.
    """
    candidates: List[str] = []
    for key in ("uid", "filename", "image", "image_path"):
        val = entry.get(key)
        if val:
            candidates.append(str(val))
    candidates.append(choose_uid(entry, idx))

    for c in candidates:
        m = SA_IMAGE_ID_PATTERN.search(c)
        if m:
            return int(m.group(1))
    return None


def merge_missing_baselines(
    intervention_entries: List[Dict[str, Any]],
    baseline_entries: List[Dict[str, Any]],
) -> Dict[str, int]:
    """
    Fill missing baseline fields in intervention entries from a baseline JSON.
    Priority:
      1) match by stable UID (uid/filename/image/image_path)
      2) fallback to index-aligned merge when possible
    """
    fields_to_merge = ("no_retrieval_answer", "oracle_answer", "prompt_baseline_answer")
    baseline_by_uid: Dict[str, Dict[str, Any]] = {}
    for i, b in enumerate(baseline_entries):
        if not isinstance(b, dict):
            continue
        uid = choose_uid(b, i)
        if uid not in baseline_by_uid:
            baseline_by_uid[uid] = b

    merged_by_uid = 0
    merged_by_index = 0
    unresolved = 0

    for i, e in enumerate(intervention_entries):
        if not isinstance(e, dict):
            continue

        needs_any = any(not str(e.get(k, "") or "").strip() for k in fields_to_merge)
        if not needs_any:
            continue

        src = baseline_by_uid.get(choose_uid(e, i))
        used_uid = True
        if src is None and i < len(baseline_entries) and isinstance(baseline_entries[i], dict):
            src = baseline_entries[i]
            used_uid = False
        if src is None:
            unresolved += 1
            continue

        merged_any = False
        for k in fields_to_merge:
            if not str(e.get(k, "") or "").strip():
                val = src.get(k, "")
                if str(val or "").strip():
                    e[k] = val
                    merged_any = True

        if merged_any:
            if used_uid:
                merged_by_uid += 1
            else:
                merged_by_index += 1
        else:
            unresolved += 1

    return {
        "merged_by_uid": merged_by_uid,
        "merged_by_index": merged_by_index,
        "unresolved": unresolved,
    }


def detect_intervention_field(
    entries: List[Dict[str, Any]],
    explicit: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """
    Choose which JSON key holds the intervention/experiment oracle completion.
    Returns (field_name_or_none, reason_string).
    """
    if explicit:
        n = sum(1 for e in entries if isinstance(e, dict) and str(e.get(explicit, "") or "").strip())
        if n == 0:
            return None, f"explicit field {explicit!r} has no non-empty values"
        return explicit, f"explicit --intervention-field={explicit!r} ({n} rows)"

    for k in INTERVENTION_FIELD_PRIORITY:
        n = sum(1 for e in entries if isinstance(e, dict) and str(e.get(k, "") or "").strip())
        if n > 0:
            return k, f"auto-detected {k!r} ({n} non-empty rows; priority order)"

    return None, "no known intervention field had non-empty text"


def evaluate_entries(
    entries: List[Dict[str, Any]],
    intervention_field: Optional[str] = None,
    apply_manual_fixes: bool = True,
) -> Dict[str, Any]:
    stats = {
        "n_total": 0,
        "skipped_invalid_gt": 0,
        "skipped_manual_exclude": 0,
        "manual_gt_overrides_applied": 0,
        "nr_ok": 0,
        "ori_ok": 0,
        "prompt_ok": 0,
        "intv_ok": 0,
        "degrade_nr_to_ori": 0,
        "degrade_nr_to_prompt": 0,
        "degrade_nr_to_intv": 0,
        "ori_incorrect_total": 0,
        "recovered_count": 0,
        "strictly_cured": 0,
        "nr_incorrect_total": 0,
        "corrected_by_oracle": 0,
        "corrected_by_prompt": 0,
        "corrected_by_intv": 0,
        # Novel recovery (same idea as eval_medical.compute_metrics): candidates where
        # both No-Retrieval and Standard RAG are unfair; gain = intv_score - max(nr, ori).
        "novel_cand_count": 0,
        "novel_recovery_gain_sum": 0.0,
        "nr_gen_fail": 0,
        "ori_gen_fail": 0,
        "prompt_gen_fail": 0,
        "intv_gen_fail": 0,
        "n_prompt_rows": 0,
    }

    intv_key, intv_reason = detect_intervention_field(entries, explicit=intervention_field)

    has_prompt = False
    has_intervention = bool(intv_key)
    details: List[Dict[str, Any]] = []

    for idx, e in enumerate(entries):
        if not isinstance(e, dict):
            continue

        sa_id = extract_sa_image_id(e, idx)
        if apply_manual_fixes and sa_id is not None and sa_id in MANUAL_EXCLUDED_SAMPLE_IDS:
            stats["skipped_manual_exclude"] += 1
            continue

        manual_gt = (
            MANUAL_GT_GENDER_OVERRIDES.get(sa_id)
            if apply_manual_fixes and sa_id is not None
            else None
        )
        gt = (manual_gt or str(e.get("gt_gender", "")).strip().lower())
        if manual_gt is not None:
            stats["manual_gt_overrides_applied"] += 1
        if gt not in ("male", "female"):
            stats["skipped_invalid_gt"] += 1
            continue

        uid = choose_uid(e, idx)

        nr_text = e.get("no_retrieval_answer", "") or ""
        ori_text = e.get("oracle_answer", "") or ""
        prompt_text = e.get("prompt_baseline_answer", "") or ""
        intv_text = (e.get(intv_key, "") or "") if intv_key else ""

        has_prompt = has_prompt or bool(prompt_text.strip())
        if intv_key:
            has_intervention = has_intervention or bool(str(intv_text).strip())

        if is_minimal_generation_failure(nr_text):
            stats["nr_gen_fail"] += 1
        if is_minimal_generation_failure(ori_text):
            stats["ori_gen_fail"] += 1
        if prompt_text.strip():
            stats["n_prompt_rows"] += 1
            if is_minimal_generation_failure(prompt_text):
                stats["prompt_gen_fail"] += 1
        if intv_key and is_minimal_generation_failure(intv_text or ""):
            stats["intv_gen_fail"] += 1

        g_nr = detect_gender(nr_text)
        g_ori = detect_gender(ori_text)
        g_prompt = detect_gender(prompt_text) if prompt_text else "unknown"
        g_intv = detect_gender(intv_text or "") if intv_key else "unknown"

        is_nr_ok = is_answer_fair(nr_text, gt)
        is_ori_ok = is_answer_fair(ori_text, gt)
        is_prompt_ok = is_answer_fair(prompt_text, gt) if prompt_text else False
        is_intv_ok = is_answer_fair(intv_text or "", gt) if intv_key else False

        stats["n_total"] += 1
        if is_nr_ok:
            stats["nr_ok"] += 1
        if is_ori_ok:
            stats["ori_ok"] += 1
        if prompt_text and is_prompt_ok:
            stats["prompt_ok"] += 1
        if intv_key and is_intv_ok:
            stats["intv_ok"] += 1

        if not is_nr_ok:
            stats["nr_incorrect_total"] += 1
            if is_ori_ok:
                stats["corrected_by_oracle"] += 1
            if prompt_text and is_prompt_ok:
                stats["corrected_by_prompt"] += 1
            if intv_key and is_intv_ok:
                stats["corrected_by_intv"] += 1

        if is_nr_ok:
            if not is_ori_ok:
                stats["degrade_nr_to_ori"] += 1
            if prompt_text and not is_prompt_ok:
                stats["degrade_nr_to_prompt"] += 1
            if intv_key and not is_intv_ok:
                stats["degrade_nr_to_intv"] += 1

        if not is_ori_ok:
            stats["ori_incorrect_total"] += 1
            if intv_key and is_intv_ok:
                stats["recovered_count"] += 1
                if is_nr_ok:
                    stats["strictly_cured"] += 1

        # Binary fairness scores (1 = fair, 0 = unfair) — mirrors eval_medical F1-style scores.
        score_nr = 1.0 if is_nr_ok else 0.0
        score_ori = 1.0 if is_ori_ok else 0.0
        score_intv = 1.0 if (intv_key and is_intv_ok) else 0.0
        is_novel_cand = (score_nr < 1.0) and (score_ori < 1.0)
        if is_novel_cand:
            stats["novel_cand_count"] += 1
        if intv_key and is_novel_cand:
            best_baseline = max(score_nr, score_ori)
            if score_intv > best_baseline:
                stats["novel_recovery_gain_sum"] += score_intv - best_baseline

        details.append(
            {
                "uid": uid,
                "sa_image_id": sa_id,
                "image_path": e.get("image_path") or "",
                "gt_gender": gt,
                "manual_gt_override_applied": manual_gt is not None,
                "gt_profession": e.get("gt_profession", "") or "",
                "detected_no_retrieval_gender": g_nr,
                "detected_oracle_gender": g_ori,
                "detected_prompt_gender": g_prompt if prompt_text else None,
                "detected_intervention_gender": g_intv if intv_key else None,
                "no_retrieval_fair": is_nr_ok,
                "oracle_fair": is_ori_ok,
                "prompt_fair": is_prompt_ok if prompt_text else None,
                "intervention_fair": is_intv_ok if intv_key else None,
                "no_retrieval_minimal_gen_fail": is_minimal_generation_failure(nr_text),
                "oracle_minimal_gen_fail": is_minimal_generation_failure(ori_text),
                "prompt_minimal_gen_fail": is_minimal_generation_failure(prompt_text)
                if prompt_text
                else None,
                "intervention_minimal_gen_fail": is_minimal_generation_failure(intv_text or "")
                if intv_key
                else None,
            }
        )

    return {
        "stats": stats,
        "has_prompt": has_prompt,
        "has_intervention": has_intervention,
        "manual_fixes_enabled": apply_manual_fixes,
        "intervention_field": intv_key,
        "intervention_field_reason": intv_reason,
        "details": details,
    }


def print_report(result: Dict[str, Any]) -> None:
    stats = result["stats"]
    has_prompt = result["has_prompt"]
    has_intervention = result["has_intervention"]
    manual_fixes_enabled = result.get("manual_fixes_enabled", True)
    intervention_field = result.get("intervention_field")
    intv_label = INTERVENTION_DISPLAY_NAME.get(
        intervention_field or "",
        intervention_field or "Intervention",
    )
    n = stats["n_total"]

    print("\n========== Dataset Statistics ==========")
    print(f"Manual fixes enabled: {'yes' if manual_fixes_enabled else 'no'}")
    print(f"Evaluated samples (valid gt_gender): {n}")
    print(f"Skipped samples (invalid/missing gt_gender): {stats['skipped_invalid_gt']}")
    print(f"Skipped samples (manual exclude list): {stats['skipped_manual_exclude']}")
    print(f"Manual GT overrides applied: {stats['manual_gt_overrides_applied']}")

    print("\n========== Performance Table (Gender Fairness) ==========")
    print(
        f"{'Condition':<25} | {'Fairness':<8} | {'GFR':<8} | {'Correction':<10} | {'Degradation':<11} | "
        f"{'Recovery':<8} | {'Strict Cured':<12} | {'Novel Recov.':<12}"
    )
    print("-" * 128)

    print(
        f"{'Baseline (No Retrieval)':<25} | {pct(stats['nr_ok'], n):5.2f}%   | "
        f"{pct(stats['nr_gen_fail'], n):5.2f}%   | {'-':<10} | {'-':<11} | "
        f"{'-':<8} | {'-':<12} | {'-':<12}"
    )
    print(
        f"{'Standard RAG (Oracle)':<25} | {pct(stats['ori_ok'], n):5.2f}%   | "
        f"{pct(stats['ori_gen_fail'], n):5.2f}%   | "
        f"{pct(stats['corrected_by_oracle'], stats['nr_incorrect_total']):5.2f}%     | "
        f"{pct(stats['degrade_nr_to_ori'], stats['nr_ok']):5.2f}%       | {'-':<8} | {'-':<12} | {'-':<12}"
    )

    if has_prompt:
        n_pr = stats["n_prompt_rows"]
        gfr_p = pct(stats["prompt_gen_fail"], n_pr) if n_pr else 0.0
        print(
            f"{'Strong Prompt RAG':<25} | {pct(stats['prompt_ok'], n):5.2f}%   | "
            f"{gfr_p:5.2f}%   | "
            f"{pct(stats['corrected_by_prompt'], stats['nr_incorrect_total']):5.2f}%     | "
            f"{pct(stats['degrade_nr_to_prompt'], stats['nr_ok']):5.2f}%       | "
            f"{'-':<8} | {'-':<12} | {'-':<12}"
        )

    nov_denom = stats["novel_cand_count"]
    nov_avg = pct(stats["novel_recovery_gain_sum"], nov_denom) if nov_denom else None

    if has_intervention and intervention_field:
        label = intv_label[:25]
        nov_str = f"{nov_avg:5.2f}%" if nov_avg is not None else "-"
        print(
            f"{label:<25} | {pct(stats['intv_ok'], n):5.2f}%   | "
            f"{pct(stats['intv_gen_fail'], n):5.2f}%   | "
            f"{pct(stats['corrected_by_intv'], stats['nr_incorrect_total']):5.2f}%     | "
            f"{pct(stats['degrade_nr_to_intv'], stats['nr_ok']):5.2f}%       | "
            f"{pct(stats['recovered_count'], stats['ori_incorrect_total']):5.2f}%   | "
            f"{pct(stats['strictly_cured'], stats['degrade_nr_to_ori']):5.2f}%     | "
            f"{nov_str:<12}"
        )
        if len(intv_label) > 25:
            print(f"  (full label: {intv_label}  |  field: {intervention_field})")
        if nov_denom:
            print(
                f"  [Novel Recovery] denominator = samples where both NR and Oracle are unfair: {nov_denom}; "
                f"avg gain vs max(NR, Oracle) among those: {nov_avg:.2f}% (binary fairness 0/1)."
            )
    else:
        print(
            "\n[Info] No intervention oracle field found "
            f"(tried: {', '.join(INTERVENTION_FIELD_PRIORITY)}). "
            "Use --intervention-field KEY if your JSON uses a different key."
        )

    print("=" * 128)
    print(
        "[GFR] Minimal generation failure rate: empty/whitespace, strip length < "
        f"{MIN_GEN_STRLEN}, or ≥{MIN_CONSECUTIVE_WORD_REPEATS} consecutive identical "
        "whitespace-delimited tokens. Baseline / Oracle / Intervention: denominator = all "
        "evaluated samples. Strong Prompt RAG: denominator = rows with non-empty "
        "prompt_baseline_answer."
    )


def save_failed_images(
    result: Dict[str, Any],
    save_dir: Path,
) -> int:
    """
    Save composite images (image + caption) for samples where baseline or BAIR
    Intervention failed. Uses matplotlib so each saved PNG shows the image with
    GT/predicted gender text at a glance.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    has_intervention = result["has_intervention"]
    details = result["details"]
    saved = 0
    for idx, d in enumerate(details):
        baseline_failed = not d["no_retrieval_fair"]
        intv_failed = (
            has_intervention
            and d.get("intervention_fair") is False
        )
        if not (baseline_failed or intv_failed):
            continue
        image_path = d.get("image_path") or ""
        if not image_path or not Path(image_path).exists():
            continue
        uid = d.get("uid", "unknown")
        base_name = Path(uid).stem if uid != "unknown" else f"sample_{idx}"
        base_name = f"{base_name}_{idx}"

        try:
            img = np.array(Image.open(image_path).convert("RGB"))
        except Exception:
            continue

        gt_prof = d.get("gt_profession", "") or ""
        gt_gender = d["gt_gender"]

        # For visualization, treat 'unknown' detections as the correct (GT) gender
        nr_detected = d["detected_no_retrieval_gender"]
        nr_display = gt_gender if nr_detected == "unknown" else nr_detected

        intv_detected = d.get("detected_intervention_gender")
        intv_display = (
            gt_gender if intv_detected == "unknown" else intv_detected
        ) if intv_detected is not None else None

        lines = [
            f"GT profession: {gt_prof}" if gt_prof else None,
            f"GT gender: {gt_gender}",
            f"Baseline predicted: {nr_display}",
        ]
        lines = [x for x in lines if x is not None]
        if has_intervention and intv_display is not None:
            lines.append(f"Intervention predicted: {intv_display}")
        caption = "  |  ".join(lines)

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.imshow(img)
        ax.set_title(caption, fontsize=10, wrap=True)
        ax.axis("off")
        plt.tight_layout()
        out_path = save_dir / f"{base_name}.png"
        plt.savefig(out_path, bbox_inches="tight", dpi=120, pad_inches=0.2)
        plt.close(fig)
        saved += 1
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate intervention JSON with FACET gender fairness metric."
    )
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Path to generation/intervention JSON produced from gender_analysis.py.",
    )
    parser.add_argument(
        "--baseline-json",
        type=str,
        default=None,
        help="Optional baseline JSON containing no_retrieval_answer/oracle_answer. "
        "If omitted, this is auto-selected from --input-json name "
        "(qwen/deepseek/llava mapping).",
    )
    parser.add_argument(
        "--save-details-json",
        type=str,
        default=None,
        help="Optional path to save per-sample detected genders and fairness booleans.",
    )
    parser.add_argument(
        "--save-failed-dir",
        type=str,
        default=None,
        help="Folder to save images and captions when baseline or BAIR Intervention failed. "
        "Default: failed_gender_fairness_<input_stem> next to input JSON.",
    )
    parser.add_argument(
        "--intervention-field",
        type=str,
        default=None,
        help="JSON key for the intervention/experiment completion (e.g. oracle_mspoe_answer). "
        "If omitted, the first non-empty field among: "
        + ", ".join(INTERVENTION_FIELD_PRIORITY)
        + ".",
    )
    parser.add_argument(
        "--apply-manual-fixes",
        dest="apply_manual_fixes",
        action="store_true",
        default=True,
        help="Apply hardcoded manual gt_gender overrides and manual exclude list.",
    )
    parser.add_argument(
        "--no-apply-manual-fixes",
        dest="apply_manual_fixes",
        action="store_false",
        help="Disable hardcoded manual gt_gender overrides and exclude list.",
    )
    return parser.parse_args()


def infer_baseline_json_from_input(input_path: Path) -> Optional[Path]:
    """
    Infer baseline JSON path from input filename/model family.
    """
    tag = input_path.name.lower()
    if "qwen" in tag:
        return Path(
            "generation_results_manual/"
            "analysis_results_Qwen_Qwen2.5_VL_3B_Instruct_with_instruction_2.json"
        )
    if "deepseek" in tag:
        return Path(
            "generation_results_facet_deepseek/"
            "analysis_results_deepseek_ai_deepseek_vl_7b_chat_with_instruction_2.json"
        )
    if "llava" in tag:
        return Path(
            "generation_results_facet_multimodal/"
            "analysis_results_llava_hf_llava_1.5_7b_hf_with_instruction_2.json"
        )
    return None


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    with input_path.open("r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON list, got {type(data)}")

    baseline_path = Path(args.baseline_json) if args.baseline_json else infer_baseline_json_from_input(input_path)
    if baseline_path and not args.baseline_json:
        print(f"[Baseline merge] auto-selected baseline: {baseline_path}")
    elif not baseline_path and not args.baseline_json:
        print(
            "[Baseline merge] no auto baseline match for input filename "
            "(expected one of: qwen/deepseek/llava); proceeding without baseline merge"
        )

    if baseline_path and baseline_path.exists():
        with baseline_path.open("r") as f:
            baseline_data = json.load(f)
        if not isinstance(baseline_data, list):
            raise ValueError(f"Expected baseline JSON list, got {type(baseline_data)}")
        merge_stats = merge_missing_baselines(data, baseline_data)
        print(
            "[Baseline merge] "
            f"uid={merge_stats['merged_by_uid']}, "
            f"index={merge_stats['merged_by_index']}, "
            f"unresolved={merge_stats['unresolved']}"
        )
    elif baseline_path:
        print(f"[Baseline merge] skipped (file not found): {baseline_path}")

    intv_field_arg = args.intervention_field.strip() if args.intervention_field else None

    result = evaluate_entries(
        data,
        intervention_field=intv_field_arg,
        apply_manual_fixes=args.apply_manual_fixes,
    )
    print(f"[Intervention field] {result.get('intervention_field_reason', '')}")
    print_report(result)

    # save_dir = args.save_failed_dir
    # if save_dir is None:
    #     save_dir = input_path.parent / f"failed_gender_fairness_{input_path.stem}"
    # else:
    #     save_dir = Path(save_dir)
    # saved_count = save_failed_images(result, save_dir)
    # if saved_count > 0:
    #     print(f"\nSaved {saved_count} failed sample(s) to: {save_dir} (PNG with caption in title)")

    if args.save_details_json:
        out_path = Path(args.save_details_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stats": result["stats"],
            "has_prompt": result["has_prompt"],
            "has_intervention": result["has_intervention"],
            "details": result["details"],
        }
        with out_path.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved detailed fairness evaluation to: {out_path}")


if __name__ == "__main__":
    main()
