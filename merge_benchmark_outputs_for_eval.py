#!/usr/bin/env python3
"""Merge per-method computational-benchmark outputs into one JSON row list for eval scripts."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Keys expected by eval_medical.py mapping (iu-chest).
METHOD_EVAL_KEYS = {
    "bair": "oracle_with_intervention",
    "mspoe": "mspoe_full_answer",
    "longllmlingua": "longllmlingua_mid_answer",
    "madrag": "madrag_answer",
}

METHODS_ORDER = ["standard_rag", "bair", "mspoe", "longllmlingua", "madrag"]


def _clean_uid(v: Any) -> str:
    return str(v).strip().replace(".0", "")


def _load_json(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def _load_outputs_map(benchmark_dir: Path, method: str) -> Dict[str, dict]:
    p = benchmark_dir / method / "outputs.json"
    if not p.is_file():
        return {}
    out: Dict[str, dict] = {}
    for row in _load_json(p):
        uid = _clean_uid(row.get("uid"))
        if uid:
            out[uid] = row
    return out


def _uid_key_for_row(row: dict) -> Optional[str]:
    uid = row.get("uid")
    if uid is not None:
        return _clean_uid(uid)
    return None


def build_baseline_index(rows: List[dict]) -> Dict[str, dict]:
    return {k: copy.deepcopy(v) for k, v in ((_uid_key_for_row(r), r) for r in rows) if k}


def pick_reference_uids(
    maps: Dict[str, Dict[str, dict]],
    prefer: str = "bair",
) -> List[str]:
    """UID order from preferred method outputs; skip rows with errors."""
    ref = maps.get(prefer) or {}
    uids: List[str] = []
    for uid, row in ref.items():
        err = row.get("error")
        if err is None or err == "":
            uids.append(uid)
    if uids:
        return uids
    for m in METHODS_ORDER:
        ref = maps.get(m) or {}
        for uid, row in ref.items():
            err = row.get("error")
            if err is None or err == "":
                uids.append(uid)
        if uids:
            return uids
    return []


def merge_rows(
    baseline_by_uid: Dict[str, dict],
    benchmark_dir: Path,
    reference_uids: List[str],
) -> Tuple[List[dict], Dict[str, int]]:
    maps = {m: _load_outputs_map(benchmark_dir, m) for m in METHODS_ORDER}
    stats = {"rows": 0, "missing_baseline": 0, "skipped_method_errors": 0}

    merged: List[dict] = []
    for uid in reference_uids:
        base = baseline_by_uid.get(uid)
        if base is None:
            stats["missing_baseline"] += 1
            continue
        row = copy.deepcopy(base)
        for method, eval_key in METHOD_EVAL_KEYS.items():
            om = maps.get(method, {}).get(uid)
            if not om:
                continue
            if om.get("error"):
                stats["skipped_method_errors"] += 1
                continue
            ans = om.get("answer")
            if ans is not None and str(ans).strip():
                row[eval_key] = str(ans).strip()
        # Optional: refreshed standard RAG text from benchmark (diagnostic).
        sr = maps.get("standard_rag", {}).get(uid)
        if sr and not sr.get("error") and sr.get("answer"):
            row["_benchmark_standard_rag_answer"] = str(sr["answer"]).strip()
        merged.append(row)
        stats["rows"] += 1
    return merged, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--suite",
        choices=["medgemma_iuchest", "chexagent_iuchest", "facet_qwen", "nwpu_skysense"],
        required=True,
    )
    ap.add_argument(
        "--rows-json",
        type=Path,
        required=True,
        help="Full-row JSON (IU/FACET baselines or NWPU baselines with nr/oracle fields).",
    )
    ap.add_argument("--benchmark-dir", type=Path, required=True, help="e.g. ROOT/medgemma")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--reference-method",
        default="bair",
        help="Take UID list / ordering from this method's outputs when possible.",
    )
    args = ap.parse_args()

    rows = _load_json(args.rows_json)
    baseline_by_uid = build_baseline_index(rows)
    maps = {m: _load_outputs_map(args.benchmark_dir, m) for m in METHODS_ORDER}

    ref_uids = pick_reference_uids(maps, prefer=args.reference_method)
    if not ref_uids:
        raise SystemExit(f"No successful outputs under {args.benchmark_dir}")

    merged, stats = merge_rows(baseline_by_uid, args.benchmark_dir, ref_uids)
    if not merged:
        raise SystemExit("merge produced zero rows (check baseline uids vs benchmark outputs)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.output} ({len(merged)} rows, suite={args.suite}) stats={stats}")


if __name__ == "__main__":
    main()
