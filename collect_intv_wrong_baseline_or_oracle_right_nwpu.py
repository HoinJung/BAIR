#!/usr/bin/env python3
"""
Collect NWPU rows where intervention is wrong but baseline or oracle is right.

Condition:
  - intervention score == 0
  - (no_retrieval score == 1 OR oracle score == 1)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from eval_nwpu import (
    choose_uid,
    compile_keywords,
    load_excluded_labels,
    merge_missing_baselines,
    score_answer,
)


def load_entries(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list JSON in {path}, got {type(data)}")
    return data


def collect_rows(
    entries: List[Dict[str, Any]],
    compiled_keywords: Dict[str, Any],
    excluded_labels: set[str],
    intervention_field: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(entries):
        if not isinstance(row, dict):
            continue

        label = str(row.get("ground_truth_label", "")).strip().lower()
        if not label or label in excluded_labels:
            continue

        nr = row.get("no_retrieval_answer", "") or ""
        ori = row.get("oracle_answer", "") or ""
        intv = row.get(intervention_field, "") or ""
        if not str(nr).strip() or not str(ori).strip() or not str(intv).strip():
            continue

        s_nr = score_answer(nr, label, compiled_keywords)
        s_ori = score_answer(ori, label, compiled_keywords)
        s_intv = score_answer(intv, label, compiled_keywords)
        if s_intv == 0 and (s_nr == 1 or s_ori == 1):
            rows.append(
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
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect rows where baseline/oracle right but intervention wrong (NWPU keyword scoring)."
    )
    base_dir = Path(__file__).resolve().parent
    parser.add_argument("--intervention-json", type=str, required=True)
    parser.add_argument("--baseline-json", type=str, default=None)
    parser.add_argument(
        "--keyword-json",
        type=str,
        default=str(base_dir / "data" / "metadata" / "nwpu_keyword_matching.json"),
    )
    parser.add_argument(
        "--exclude-json",
        type=str,
        default=str(base_dir / "data" / "metadata" / "nwpu_exclude.json"),
    )
    parser.add_argument("--intervention-field", type=str, required=True)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    intervention_path = Path(args.intervention_json)
    entries = load_entries(intervention_path)

    if args.baseline_json:
        baseline_path = Path(args.baseline_json)
        baseline_entries = load_entries(baseline_path)
        merge_stats = merge_missing_baselines(entries, baseline_entries)
        print(
            "[Baseline merge] "
            f"path={baseline_path}, uid={merge_stats['merged_by_uid']}, "
            f"index={merge_stats['merged_by_index']}, unresolved={merge_stats['unresolved']}"
        )
    else:
        baseline_path = None

    kw = json.loads(Path(args.keyword_json).read_text(encoding="utf-8"))
    compiled = compile_keywords(kw)
    excluded = load_excluded_labels(args.exclude_json)

    rows = collect_rows(
        entries=entries,
        compiled_keywords=compiled,
        excluded_labels=excluded,
        intervention_field=args.intervention_field,
    )

    if args.output_json:
        out_path = Path(args.output_json)
    else:
        out_path = intervention_path.with_name(
            f"{intervention_path.stem}_intv_wrong_but_baseline_or_oracle_right_excluded.json"
        )

    payload = {
        "intervention_json": str(intervention_path),
        "baseline_json": str(baseline_path) if baseline_path else None,
        "keyword_json": str(args.keyword_json),
        "exclude_json": args.exclude_json,
        "excluded_labels_count": len(excluded),
        "intervention_field": args.intervention_field,
        "count": len(rows),
        "samples": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out_path} (count={len(rows)})")


if __name__ == "__main__":
    main()
