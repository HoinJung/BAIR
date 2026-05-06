#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def pct(num: float, den: float) -> float:
    return 0.0 if not den else 100.0 * float(num) / float(den)


def read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or "stats" not in obj:
        raise ValueError(f"{path} is not an evaluation details JSON with a stats object")
    return obj


def compare(path: Path) -> tuple[str, float, float, bool]:
    result = read(path)
    stats = result["stats"]
    n = stats.get("n_total", stats.get("n", 0))
    standard = pct(stats.get("ori_ok", 0), n)
    bair = pct(stats.get("intv_ok", 0), n)
    return path.name, standard, bair, bair >= standard


def main() -> int:
    ap = argparse.ArgumentParser(description="Check FACET/NWPU smoke eval JSONs for BAIR >= StandardRAG.")
    ap.add_argument("eval_json", nargs="+", type=Path)
    args = ap.parse_args()

    failed = False
    for path in args.eval_json:
        name, standard, bair, ok = compare(path)
        marker = "PASS" if ok else "CHECK"
        print(f"{marker} {name}: StandardRAG={standard:.2f}% BAIR={bair:.2f}%")
        failed = failed or not ok
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
