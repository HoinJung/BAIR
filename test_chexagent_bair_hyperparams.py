#!/usr/bin/env python3
"""
Test CheXagent BAIR with different alpha_v values on specific UIDs.
Runs unified_chexagent.py for each combo, then prints oracle vs intervention comparison.
Usage:
  python test_chexagent_bair_hyperparams.py
  python test_chexagent_bair_hyperparams.py --uids 2353,182,857 --device_id 0
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BASELINE_JSON = BASE_DIR / "generation_results_chexagent" / "iuchest_chexagent_results_baselines.json"
OUTPUT_DIR = BASE_DIR / "generation_results_chexagent"

# alpha_t and gamma_s are fixed internally at 1.0; only av is exposed.
COMBOS = [0.05, 0.1, 0.5, 1.0]


def load_baseline_oracles(uids: list[str]) -> dict[str, str]:
    """Load oracle_answer from baseline for given uids."""
    if not BASELINE_JSON.exists():
        return {}
    with open(BASELINE_JSON) as f:
        data = json.load(f)
    want = {str(x).strip().replace(".0", "") for x in uids}
    out = {}
    for e in data:
        if not isinstance(e, dict):
            continue
        uid = str(e.get("uid", "")).strip().replace(".0", "")
        if uid in want:
            out[uid] = (e.get("oracle_answer") or "").strip()
    return out


def run_combo(av: float, uids: str, device_id: int) -> dict[str, str]:
    """Run unified_chexagent for one av value, return uid -> oracle_with_intervention."""
    out_suffix = f"new_bair_av{av}_mid"
    out_file = OUTPUT_DIR / f"iuchest_chexagent_results_{out_suffix}.json"
    cmd = [
        sys.executable,
        str(BASE_DIR / "unified_chexagent.py"),
        "--dataset", "iuchest",
        "--data_dir", str(BASE_DIR / "data" / "raw" / "iuchest"),
        "--device_id", str(device_id),
        "--use_intervention",
        "--alpha_v", str(av),
        "--gt_position", "mid",
        "--uids", uids,
    ]
    subprocess.run(cmd, check=True, cwd=str(BASE_DIR))
    if not out_file.exists():
        return {}
    with open(out_file) as f:
        data = json.load(f)
    out = {}
    for e in data:
        if not isinstance(e, dict):
            continue
        uid = str(e.get("uid", "")).strip().replace(".0", "")
        out[uid] = (e.get("oracle_with_intervention") or "").strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uids", type=str, default="2353,182,857")
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument("--avs", type=str, default=None,
                    help="Override av values, comma-separated: '0.05,0.1,0.5,1.0'")
    ap.add_argument("--quick", action="store_true",
                    help="Run only 3 combos for a fast sanity check")
    args = ap.parse_args()
    uids = [x.strip() for x in args.uids.split(",") if x.strip()]
    if not uids:
        print("No UIDs provided.")
        return 1

    if args.avs:
        combos = [float(x.strip()) for x in args.avs.split(",") if x.strip()]
        if not combos:
            combos = COMBOS
    elif args.quick:
        combos = COMBOS[:3]
    else:
        combos = COMBOS

    oracles = load_baseline_oracles(uids)
    print("=" * 100)
    print("CheXagent BAIR Hyperparameter Test")
    print(f"UIDs: {uids}")
    print(f"Combos: {combos}")
    print("=" * 100)

    results = {}
    for av in combos:
        print(f"\n>>> Running av={av} ...")
        results[av] = run_combo(av, args.uids, args.device_id)

    # Print comparison table
    print("\n" + "=" * 100)
    print("COMPARISON: oracle_answer (baseline) vs oracle_with_intervention (BAIR)")
    print("=" * 100)
    for uid in uids:
        oracle = oracles.get(uid, "(no baseline)")
        print(f"\n--- UID {uid} ---")
        print(f"  oracle_answer (baseline): {oracle[:200]}{'...' if len(str(oracle)) > 200 else ''}")
        for av, interv_map in results.items():
            txt = interv_map.get(uid, "(missing)")
            same = " [SAME]" if txt == oracle else ""
            print(f"  av={av}: {str(txt)[:200]}{'...' if len(str(txt)) > 200 else ''}{same}")
    print("\n" + "=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
