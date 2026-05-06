#!/usr/bin/env python3
"""
Keyword-based evaluation for NWPU remote-sensing experiments.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


MIN_GEN_STRLEN = 5
MIN_CONSECUTIVE_WORD_REPEATS = 5
MISSING_OUTPUT_MARKERS = {
    "n/a",
    "na",
    "none",
    "null",
    "not created yet",
    "not generated yet",
    "not available yet",
    "pending",
    "tbd",
}


def is_minimal_generation_failure(text: str) -> bool:
    if not isinstance(text, str):
        return True
    t = text.strip()
    if len(t) < MIN_GEN_STRLEN:
        return True
    tokens = t.split()
    if len(tokens) < MIN_CONSECUTIVE_WORD_REPEATS:
        return False
    run = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            run += 1
            if run >= MIN_CONSECUTIVE_WORD_REPEATS:
                return True
        else:
            run = 1
    return False


def is_missing_output(text: Any) -> bool:
    if text is None:
        return True
    t = str(text).strip().lower()
    if not t:
        return True
    if t in MISSING_OUTPUT_MARKERS:
        return True
    return ("not created" in t) or ("not generated" in t)


def compile_keywords(keyword_json: Dict[str, Any]) -> Dict[str, List[re.Pattern]]:
    raw = keyword_json.get("keywords", {})
    out: Dict[str, List[re.Pattern]] = {}

    def plural_variants(kw: str) -> List[str]:
        """
        Build common English plural forms for one keyword phrase.
        We pluralize only the last token in a multi-word phrase.
        """
        parts = kw.split()
        if not parts:
            return [kw]
        head = parts[:-1]
        last = parts[-1]

        variants = {kw}
        if len(last) >= 2:
            # city -> cities, battery -> batteries (but keep "key" style words as keys)
            if last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
                variants.add(" ".join(head + [last[:-1] + "ies"]))
            # class -> classes, box -> boxes, church -> churches, bush -> bushes
            if last.endswith(("s", "x", "z", "ch", "sh")):
                variants.add(" ".join(head + [last + "es"]))
            # leaf -> leaves, knife -> knives (best-effort)
            if last.endswith("f"):
                variants.add(" ".join(head + [last[:-1] + "ves"]))
            if last.endswith("fe"):
                variants.add(" ".join(head + [last[:-2] + "ves"]))
            # default: airport -> airports
            variants.add(" ".join(head + [last + "s"]))
        return list(variants)

    for label, kws in raw.items():
        patterns = []
        for kw in kws:
            kw = str(kw).strip().lower()
            if not kw:
                continue
            for v in plural_variants(kw):
                patterns.append(re.compile(rf"\b{re.escape(v)}\b", re.IGNORECASE))
        out[label] = patterns
    return out


def score_answer(answer: str, label: str, compiled_keywords: Dict[str, List[re.Pattern]]) -> int:
    if is_minimal_generation_failure(answer):
        return 0
    patterns = compiled_keywords.get(label, [])
    if not patterns:
        return 0
    for p in patterns:
        if p.search(answer or ""):
            return 1
    return 0


def safe_pct(n: float, d: int) -> float:
    return (100.0 * n / d) if d > 0 else 0.0


def choose_uid(entry: Dict[str, Any], idx: int) -> str:
    for key in ("uid", "filename", "image", "image_path", "image_relpath"):
        val = entry.get(key)
        if val:
            return str(val).strip()
    return f"sample_{idx}"


def merge_missing_baselines(
    intervention_entries: List[Dict[str, Any]],
    baseline_entries: List[Dict[str, Any]],
) -> Dict[str, int]:
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
                v = src.get(k, "")
                if str(v or "").strip():
                    e[k] = v
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


def infer_baseline_json_from_input(input_path: Path) -> Optional[Path]:
    name = input_path.name
    parent = input_path.parent
    quick_tokens = ("_bair_", "_mspoe_", "_madrag", "_combo_", "_longllmlingua")
    for tok in quick_tokens:
        if tok in name:
            prefix = name.split(tok, 1)[0]
            candidate = input_path.with_name(f"{prefix}_baselines.json")
            if candidate.exists():
                return candidate
    # Fallback: any sibling with "baselines" keyword.
    candidates = [p for p in sorted(parent.glob("*baselines*.json")) if p != input_path]
    return candidates[0] if candidates else None


def select_best_baseline_json(input_path: Path, entries: List[Dict[str, Any]]) -> Optional[Path]:
    """
    Pick baseline JSON in the same folder using UID overlap and baseline field coverage.
    """
    parent = input_path.parent
    candidates = [p for p in sorted(parent.glob("*baselines*.json")) if p != input_path]
    if not candidates:
        return None

    intervention_uids = {choose_uid(e, i) for i, e in enumerate(entries) if isinstance(e, dict)}
    best_path: Optional[Path] = None
    best_score = -1

    for cp in candidates:
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue

        nr_uids = set()
        ori_uids = set()
        for i, row in enumerate(data):
            if not isinstance(row, dict):
                continue
            uid = choose_uid(row, i)
            if str(row.get("no_retrieval_answer", "") or "").strip():
                nr_uids.add(uid)
            if str(row.get("oracle_answer", "") or "").strip():
                ori_uids.add(uid)

        score = len(intervention_uids & nr_uids) + len(intervention_uids & ori_uids)
        if score > best_score:
            best_score = score
            best_path = cp

    return best_path if best_score > 0 else None


def load_excluded_labels(exclude_json_path: Optional[str]) -> set[str]:
    if not exclude_json_path:
        return set()
    p = Path(exclude_json_path)
    if not p.exists():
        raise FileNotFoundError(f"Exclude JSON not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Exclude JSON must be an object, got {type(data)}")

    excluded: set[str] = set()
    for key, val in data.items():
        if key == "exclude":
            if isinstance(val, list):
                excluded.update(str(x).strip().lower() for x in val if str(x).strip())
            continue
        # Also accept additional top-level list entries as label sets to exclude.
        if isinstance(val, list):
            excluded.update(str(x).strip().lower() for x in val if str(x).strip())
    return excluded


def evaluate(
    entries: List[Dict[str, Any]],
    compiled_keywords: Dict[str, List[re.Pattern]],
    intervention_field: Optional[str],
    excluded_labels: set[str],
) -> Dict[str, Any]:
    stats = {
        "n": 0,
        "excluded_by_label": 0,
        "excluded_missing_intervention": 0,
        "nr_ok": 0,
        "ori_ok": 0,
        "prompt_ok": 0,
        "intv_ok": 0,
        "nr_fail": 0,
        "ori_fail": 0,
        "prompt_fail": 0,
        "intv_fail": 0,
        "nr_bad": 0,
        "ori_recover_vs_nr": 0,
        "prompt_recover_vs_nr": 0,
        "intv_recover_vs_nr": 0,
        "ori_degrade_vs_nr": 0,
        "prompt_degrade_vs_nr": 0,
        "prompt_recover_vs_ori": 0,
        "prompt_strict_cured": 0,
        "prompt_novel_gain_sum": 0.0,
        "intv_degrade_vs_nr": 0,
        "ori_bad": 0,
        "intv_recover_vs_ori": 0,
        "strict_cured": 0,
        "novel_denom": 0,
        "novel_gain_sum": 0.0,
    }
    details = []

    for row in entries:
        label = str(row.get("ground_truth_label", "")).strip().lower()
        if not label:
            continue
        if label in excluded_labels:
            stats["excluded_by_label"] += 1
            continue
        nr = row.get("no_retrieval_answer", "") or ""
        ori = row.get("oracle_answer", "") or ""
        prompt = row.get("prompt_baseline_answer", "") or ""
        intv = row.get(intervention_field, "") if intervention_field else ""
        intv = intv or ""
        if intervention_field and is_missing_output(intv):
            stats["excluded_missing_intervention"] += 1
            continue

        s_nr = score_answer(nr, label, compiled_keywords)
        s_ori = score_answer(ori, label, compiled_keywords)
        s_prompt = score_answer(prompt, label, compiled_keywords) if prompt.strip() else 0
        s_intv = score_answer(intv, label, compiled_keywords) if intervention_field else 0

        stats["n"] += 1
        stats["nr_ok"] += s_nr
        stats["ori_ok"] += s_ori
        if prompt.strip():
            stats["prompt_ok"] += s_prompt
        if intervention_field:
            stats["intv_ok"] += s_intv

        stats["nr_fail"] += 1 if is_minimal_generation_failure(nr) else 0
        stats["ori_fail"] += 1 if is_minimal_generation_failure(ori) else 0
        if prompt.strip():
            stats["prompt_fail"] += 1 if is_minimal_generation_failure(prompt) else 0
        if intervention_field:
            stats["intv_fail"] += 1 if is_minimal_generation_failure(intv) else 0

        if s_nr == 0:
            stats["nr_bad"] += 1
            if s_ori > s_nr:
                stats["ori_recover_vs_nr"] += 1
            if prompt.strip() and s_prompt > s_nr:
                stats["prompt_recover_vs_nr"] += 1
            if intervention_field and s_intv > s_nr:
                stats["intv_recover_vs_nr"] += 1
        if s_nr == 1:
            stats["ori_degrade_vs_nr"] += 1 if s_ori < s_nr else 0
            if prompt.strip():
                stats["prompt_degrade_vs_nr"] += 1 if s_prompt < s_nr else 0
            if intervention_field:
                stats["intv_degrade_vs_nr"] += 1 if s_intv < s_nr else 0

        if s_ori == 0:
            stats["ori_bad"] += 1
            if prompt.strip() and s_prompt > s_ori:
                stats["prompt_recover_vs_ori"] += 1
                if s_nr == 1:
                    stats["prompt_strict_cured"] += 1
            if intervention_field and s_intv > s_ori:
                stats["intv_recover_vs_ori"] += 1
                if s_nr == 1:
                    stats["strict_cured"] += 1

        if s_nr == 0 and s_ori == 0:
            stats["novel_denom"] += 1
            if prompt.strip():
                stats["prompt_novel_gain_sum"] += max(0.0, float(s_prompt - max(s_nr, s_ori)))
            if intervention_field:
                stats["novel_gain_sum"] += max(0.0, float(s_intv - max(s_nr, s_ori)))

        details.append(
            {
                "uid": row.get("uid"),
                "ground_truth_label": label,
                "nr_score": s_nr,
                "ori_score": s_ori,
                "prompt_score": s_prompt if prompt.strip() else None,
                "intv_score": s_intv if intervention_field else None,
            }
        )

    return {"stats": stats, "details": details}


def collect_intervention_regressions(
    entries: List[Dict[str, Any]],
    compiled_keywords: Dict[str, List[re.Pattern]],
    intervention_field: Optional[str],
    excluded_labels: set[str],
) -> List[Dict[str, Any]]:
    """
    Collect rows where intervention is wrong but NR or Oracle is right.
    Condition:
      - intv_score == 0
      - (nr_score == 1 or ori_score == 1)
    """
    if not intervention_field:
        return []

    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(entries):
        label = str(row.get("ground_truth_label", "")).strip().lower()
        if not label:
            continue
        if label in excluded_labels:
            continue

        nr = row.get("no_retrieval_answer", "") or ""
        ori = row.get("oracle_answer", "") or ""
        intv = row.get(intervention_field, "") or ""
        if is_missing_output(intv):
            continue

        s_nr = score_answer(nr, label, compiled_keywords)
        s_ori = score_answer(ori, label, compiled_keywords)
        s_intv = score_answer(intv, label, compiled_keywords)

        if s_intv == 0 and (s_nr == 1 or s_ori == 1):
            out.append(
                {
                    "uid": row.get("uid") or choose_uid(row, idx),
                    "image_path": row.get("image_path"),
                    "image_relpath": row.get("image_relpath"),
                    "ground_truth_label": label,
                    "intervention_field": intervention_field,
                    "nr_score": s_nr,
                    "ori_score": s_ori,
                    "intv_score": s_intv,
                    "baseline_or_oracle_right": [
                        cond
                        for cond, score in (("no_retrieval", s_nr), ("oracle", s_ori))
                        if score == 1
                    ],
                    "no_retrieval_answer": nr,
                    "oracle_answer": ori,
                    intervention_field: intv,
                }
            )

    return out


def print_report(stats: Dict[str, Any], intervention_field: Optional[str]) -> None:
    n = stats["n"]
    intv_name = intervention_field or "intervention"
    print("\n========== NWPU Keyword Evaluation ==========")
    print(f"Evaluated samples: {n}")
    if stats.get("excluded_by_label", 0):
        print(f"Excluded samples by label: {stats['excluded_by_label']}")
    if stats.get("excluded_missing_intervention", 0):
        print(
            f"Excluded samples (missing intervention output): {stats['excluded_missing_intervention']}"
        )
    print(
        f"{'Condition':<30} | {'Accuracy':<8} | {'GFR':<8} | {'Correction':<10} | {'Degradation':<11} | {'Recovery':<8} | {'Strict Cured':<12} | {'Novel Recov.'}"
    )
    print("-" * 150)
    print(
        f"{'Baseline (No Retrieval)':<30} | {safe_pct(stats['nr_ok'], n):5.2f}%   | {safe_pct(stats['nr_fail'], n):5.2f}%   | {'-':<10} | {'-':<11} | {'-':<8} | {'-':<12} | {'-'}"
    )
    print(
        f"{'Standard RAG (Oracle)':<30} | {safe_pct(stats['ori_ok'], n):5.2f}%   | {safe_pct(stats['ori_fail'], n):5.2f}%   | "
        f"{safe_pct(stats['ori_recover_vs_nr'], stats['nr_bad']):5.2f}%     | {safe_pct(stats['ori_degrade_vs_nr'], stats['nr_ok']):5.2f}%       | {'-':<8} | {'-':<12} | {'-'}"
    )
    prompt_novel_avg = safe_pct(stats["prompt_novel_gain_sum"], stats["novel_denom"]) if stats["novel_denom"] else 0.0
    print(
        f"{'Strong Prompt':<30} | {safe_pct(stats['prompt_ok'], n):5.2f}%   | {safe_pct(stats['prompt_fail'], n):5.2f}%   | "
        f"{safe_pct(stats['prompt_recover_vs_nr'], stats['nr_bad']):5.2f}%     | {safe_pct(stats['prompt_degrade_vs_nr'], stats['nr_ok']):5.2f}%       | "
        f"{safe_pct(stats['prompt_recover_vs_ori'], stats['ori_bad']):5.2f}%   | {safe_pct(stats['prompt_strict_cured'], stats['ori_degrade_vs_nr']):5.2f}%     | {prompt_novel_avg:5.2f}%"
    )
    if intervention_field:
        novel_avg = safe_pct(stats["novel_gain_sum"], stats["novel_denom"]) if stats["novel_denom"] else 0.0
        print(
            f"{intv_name[:30]:<30} | {safe_pct(stats['intv_ok'], n):5.2f}%   | {safe_pct(stats['intv_fail'], n):5.2f}%   | "
            f"{safe_pct(stats['intv_recover_vs_nr'], stats['nr_bad']):5.2f}%     | {safe_pct(stats['intv_degrade_vs_nr'], stats['nr_ok']):5.2f}%       | "
            f"{safe_pct(stats['intv_recover_vs_ori'], stats['ori_bad']):5.2f}%   | {safe_pct(stats['strict_cured'], stats['ori_degrade_vs_nr']):5.2f}%     | {novel_avg:5.2f}%"
        )
    print("=" * 150)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NWPU experiments via keyword matching.")
    parser.add_argument("--input-json", type=str, required=True)
    base_dir = Path(__file__).resolve().parent
    parser.add_argument("--keyword-json", type=str, default=str(base_dir / "data" / "metadata" / "nwpu_keyword_matching.json"))
    parser.add_argument(
        "--baseline-json",
        type=str,
        default=None,
        help="Optional baseline JSON containing no_retrieval_answer/oracle_answer. "
        "If omitted, tries to infer from BAIR filename pattern.",
    )
    parser.add_argument(
        "--exclude-json",
        type=str,
        default=str(base_dir / "data" / "metadata" / "nwpu_exclude.json"),
        help="Optional JSON describing labels to exclude from evaluation denominator.",
    )
    parser.add_argument("--intervention-field", type=str, default=None)
    parser.add_argument("--save-details-json", type=str, default=None)
    parser.add_argument(
        "--save-intv-wrong-baseline-or-oracle-right-json",
        type=str,
        default=None,
        help="Save rows where intervention is wrong (score=0) but no-retrieval or oracle is right (score=1).",
    )
    args = parser.parse_args()

    input_path = Path(args.input_json)
    entries = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"Expected top-level list in input JSON, got {type(entries)}")

    baseline_path = Path(args.baseline_json) if args.baseline_json else infer_baseline_json_from_input(input_path)
    if baseline_path is None and args.baseline_json is None:
        baseline_path = select_best_baseline_json(input_path, entries)
    if baseline_path and baseline_path.exists():
        baseline_entries = json.loads(baseline_path.read_text(encoding="utf-8"))
        if not isinstance(baseline_entries, list):
            raise ValueError(f"Expected top-level list in baseline JSON, got {type(baseline_entries)}")
        merge_stats = merge_missing_baselines(entries, baseline_entries)
        print(
            "[Baseline merge] "
            f"path={baseline_path}, uid={merge_stats['merged_by_uid']}, "
            f"index={merge_stats['merged_by_index']}, unresolved={merge_stats['unresolved']}"
        )
    elif args.baseline_json:
        raise FileNotFoundError(f"--baseline-json not found: {baseline_path}")

    has_nr = any(str(e.get("no_retrieval_answer", "") or "").strip() for e in entries if isinstance(e, dict))
    has_ori = any(str(e.get("oracle_answer", "") or "").strip() for e in entries if isinstance(e, dict))
    if not has_nr or not has_ori:
        raise ValueError(
            "Baseline fields are missing in input rows. Please pass --baseline-json "
            "or place a matching *baselines*.json in the same folder."
        )

    keyword_json = json.loads(Path(args.keyword_json).read_text(encoding="utf-8"))
    compiled = compile_keywords(keyword_json)
    excluded_labels = load_excluded_labels(args.exclude_json)
    if excluded_labels:
        print(f"[Exclude] loaded {len(excluded_labels)} labels from {args.exclude_json}")

    intervention_field = args.intervention_field
    if intervention_field is None:
        candidates = [
            "oracle_with_intervention_strong",
            "oracle_with_intervention",
            "oracle_mspoe_answer",
            "oracle_madrag_answer",
            "oracle_bair_mspoe_combo_answer",
            "oracle_bair_madrag_combo_answer",
            "oracle_longllmlingua_answer",
            "oracle_bair_longllmlingua_combo_answer",
        ]
        for key in candidates:
            if any(str(e.get(key, "")).strip() for e in entries):
                intervention_field = key
                break

    result = evaluate(
        entries,
        compiled,
        intervention_field=intervention_field,
        excluded_labels=excluded_labels,
    )
    print_report(result["stats"], intervention_field=intervention_field)

    if args.save_intv_wrong_baseline_or_oracle_right_json:
        rows = collect_intervention_regressions(
            entries,
            compiled,
            intervention_field=intervention_field,
            excluded_labels=excluded_labels,
        )
        out_path = Path(args.save_intv_wrong_baseline_or_oracle_right_json)
        payload = {
            "input_json": str(input_path),
            "exclude_json": args.exclude_json,
            "excluded_labels_count": len(excluded_labels),
            "intervention_field": intervention_field,
            "count": len(rows),
            "samples": rows,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved intervention-regression rows: {out_path} (count={len(rows)})")

    if args.save_details_json:
        Path(args.save_details_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved detailed output: {args.save_details_json}")


if __name__ == "__main__":
    main()
