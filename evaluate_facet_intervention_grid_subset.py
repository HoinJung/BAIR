#!/usr/bin/env python3
"""
Evaluate BAIR intervention grid results on a subset and rank av/at/gs combos.

Requirement check:
  intervention fairness accuracy > baseline(no_retrieval) accuracy
  intervention fairness accuracy > standard RAG(oracle) accuracy

All comparisons are computed on the same subset (first N rows after baseline merge).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluate_intervention_gender_fairness import evaluate_entries, merge_missing_baselines


FNAME_RE = re.compile(
    r"_intervention_av(?P<av>-?\d+(?:\.\d+)?)_at(?P<at>-?\d+(?:\.\d+)?)_gs(?P<gs>-?\d+(?:\.\d+)?)\.json$"
)


def pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


def parse_hparams(path: Path) -> Dict[str, Optional[float]]:
    m = FNAME_RE.search(path.name)
    if not m:
        return {"av": None, "at": None, "gs": None}
    return {
        "av": float(m.group("av")),
        "at": float(m.group("at")),
        "gs": float(m.group("gs")),
    }


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list in {path}, got {type(data)}")
    return data


def eval_one(
    intervention_json: Path,
    baseline_data: List[Dict[str, Any]],
    max_samples: Optional[int],
    intervention_field: Optional[str],
) -> Dict[str, Any]:
    data = load_json_list(intervention_json)
    merge_missing_baselines(data, baseline_data)
    if max_samples and max_samples > 0:
        data = data[:max_samples]

    result = evaluate_entries(data, intervention_field=intervention_field)
    stats = result["stats"]
    n = stats["n_total"]

    nr_acc = pct(stats["nr_ok"], n)
    ori_acc = pct(stats["ori_ok"], n)
    intv_acc = pct(stats["intv_ok"], n)
    meets_requirement = (intv_acc > nr_acc) and (intv_acc > ori_acc)

    parsed = parse_hparams(intervention_json)
    return {
        "file": str(intervention_json),
        "av": parsed["av"],
        "at": parsed["at"],
        "gs": parsed["gs"],
        "n_eval": n,
        "baseline_nr_acc": nr_acc,
        "standard_rag_acc": ori_acc,
        "intervention_acc": intv_acc,
        "delta_vs_baseline": intv_acc - nr_acc,
        "delta_vs_standard_rag": intv_acc - ori_acc,
        "meets_requirement": meets_requirement,
        "intervention_field": result.get("intervention_field"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank av/at/gs intervention combos on subset fairness accuracy."
    )
    parser.add_argument("--baseline-json", type=str, required=True)
    parser.add_argument(
        "--intervention-glob",
        type=str,
        required=True,
        help="Glob for intervention JSON files, e.g. ..._intervention_av*_at*_gs*.json",
    )
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument(
        "--intervention-field",
        type=str,
        default="oracle_with_intervention",
        help="Field to evaluate as intervention completion.",
    )
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    baseline_path = Path(args.baseline_json)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline JSON not found: {baseline_path}")
    baseline_data = load_json_list(baseline_path)

    paths = sorted(Path(p) for p in glob(args.intervention_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched --intervention-glob: {args.intervention_glob}")

    rows: List[Dict[str, Any]] = []
    for p in paths:
        rows.append(
            eval_one(
                intervention_json=p,
                baseline_data=baseline_data,
                max_samples=args.max_samples,
                intervention_field=args.intervention_field,
            )
        )

    rows.sort(
        key=lambda r: (
            r["meets_requirement"],
            r["intervention_acc"],
            r["delta_vs_standard_rag"],
            r["delta_vs_baseline"],
        ),
        reverse=True,
    )

    print("\n=== Grid Search Summary (subset) ===")
    print(
        "av\tat\tgs\tn\tbaseline%\tstd_rag%\tintv%\tdelta_vs_base\tdelta_vs_std\tmeets"
    )
    for r in rows:
        print(
            f"{r['av']}\t{r['at']}\t{r['gs']}\t{r['n_eval']}\t"
            f"{r['baseline_nr_acc']:.2f}\t{r['standard_rag_acc']:.2f}\t"
            f"{r['intervention_acc']:.2f}\t{r['delta_vs_baseline']:+.2f}\t"
            f"{r['delta_vs_standard_rag']:+.2f}\t{r['meets_requirement']}"
        )

    winners = [r for r in rows if r["meets_requirement"]]
    print(f"\nCombinations meeting requirement: {len(winners)}/{len(rows)}")
    if winners:
        best = winners[0]
        print(
            "Best passing combo: "
            f"av={best['av']}, at={best['at']}, gs={best['gs']} "
            f"(intv={best['intervention_acc']:.2f}%, "
            f"baseline={best['baseline_nr_acc']:.2f}%, std_rag={best['standard_rag_acc']:.2f}%)"
        )

    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV: {out_csv}")

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w") as f:
            json.dump(rows, f, indent=2)
        print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
