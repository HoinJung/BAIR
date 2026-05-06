#!/usr/bin/env python3
"""Concatenate per-suite efficiency_summary.csv files into one table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", type=Path, required=True, help="efficiency_summary.csv paths")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    frames = []
    for p in args.inputs:
        if p.is_file():
            frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit("No input CSVs found.")
    out = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(out)} rows)")


if __name__ == "__main__":
    main()
