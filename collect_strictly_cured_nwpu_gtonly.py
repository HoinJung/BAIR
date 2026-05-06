#!/usr/bin/env python3
"""
Collect NWPU gtonly samples that are 'strictly cured' under the same logical definition
as collect_strictly_cured_gtonly.py for medical (continuous scores there, binary keyword scores here):

  is_recorrupted = score_nr > score_oracle
  strictly_cured   = is_recorrupted and score_intervention >= score_nr

Keyword scores are 0/1 from eval_nwpu.score_answer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from eval_nwpu import choose_uid, compile_keywords, load_excluded_labels, score_answer

BASE_DIR = Path(__file__).resolve().parent


def load_entries(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    return data


def index_by_uid(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            continue
        uid = choose_uid(e, i)
        out[uid] = e
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly-cured samples for NWPU gtonly (keyword scores).")
    parser.add_argument("--baseline-json", type=str, required=True)
    parser.add_argument("--intervention-json", type=str, required=True)
    base_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--keyword-json",
        type=str,
        default=str(base_dir / "data" / "metadata" / "nwpu_keyword_matching.json"),
    )
    parser.add_argument(
        "--exclude-json",
        type=str,
        default=str(base_dir / "data" / "metadata" / "nwpu_exclude.json"),
        help="Optional; same schema as eval_nwpu. Omit to evaluate all labels.",
    )
    parser.add_argument("--intervention-field", type=str, default="oracle_with_intervention")
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Default: alongside intervention json, stem + _strictly_cured_samples.json",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline_json)
    intervention_path = Path(args.intervention_json)
    out_json = Path(args.output_json) if args.output_json else intervention_path.with_name(
        f"{intervention_path.stem}_strictly_cured_samples.json"
    )

    kw = json.loads(Path(args.keyword_json).read_text(encoding="utf-8"))
    compiled = compile_keywords(kw)
    excluded = load_excluded_labels(args.exclude_json)

    base_map = index_by_uid(load_entries(baseline_path))
    int_entries = load_entries(intervention_path)
    int_map = index_by_uid(int_entries)

    strictly_cured: List[Dict[str, Any]] = []
    recorrupted_count = 0
    total_compared = 0

    common = sorted(set(base_map.keys()) & set(int_map.keys()))
    for uid in common:
        b = base_map[uid]
        inv = int_map[uid]
        label = str(b.get("ground_truth_label", "")).strip().lower()
        if not label or label in excluded:
            continue

        nr = b.get("no_retrieval_answer", "") or ""
        ori = b.get("oracle_answer", "") or ""
        intv = inv.get(args.intervention_field, "") or ""

        if not str(nr).strip() or not str(ori).strip() or not str(intv).strip():
            continue

        s_nr = score_answer(nr, label, compiled)
        s_ori = score_answer(ori, label, compiled)
        s_intv = score_answer(intv, label, compiled)
        total_compared += 1

        is_recorrupted = s_nr > s_ori
        if is_recorrupted:
            recorrupted_count += 1
        if is_recorrupted and s_intv >= s_nr:
            strictly_cured.append(
                {
                    "uid": uid,
                    "ground_truth_label": label,
                    "score_no_retrieval": s_nr,
                    "score_oracle": s_ori,
                    "score_intervention": s_intv,
                    "no_retrieval_answer": nr,
                    "oracle_answer": ori,
                    args.intervention_field: intv,
                    "image_path": b.get("image_path"),
                    "image_relpath": b.get("image_relpath"),
                }
            )

    payload = {
        "strictly_cured_definition": "score_nr > score_oracle AND score_intervention >= score_nr (keyword 0/1)",
        "baseline_json": str(baseline_path),
        "intervention_json": str(intervention_path),
        "intervention_field": args.intervention_field,
        "keyword_json": str(args.keyword_json),
        "total_compared_samples": total_compared,
        "recorrupted_count": recorrupted_count,
        "strictly_cured_count": len(strictly_cured),
        "strictly_cured_rate_over_recorrupted": (len(strictly_cured) / recorrupted_count) if recorrupted_count else 0.0,
        "samples": strictly_cured,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Strictly cured: {len(strictly_cured)} / recorrupted {recorrupted_count} (denom compared {total_compared})")


if __name__ == "__main__":
    main()
