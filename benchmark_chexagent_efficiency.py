#!/usr/bin/env python3
"""CheXagent IU-Chest computational cost (same method names as MedGemma benchmark)."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from benchmark_cost_lib import ResourceMonitor
import unified_chexagent as uc

METHODS = ["standard_rag", "bair", "mspoe", "longllmlingua", "madrag"]
METHOD_LABELS = {
    "standard_rag": "Standard RAG",
    "bair": "BAIR",
    "mspoe": "MS-PoE",
    "longllmlingua": "LongLLMLingua",
    "madrag": "MAD-RAG",
}


def load_entries(path: Path, num_samples: int, seed: int) -> List[dict]:
    with path.open() as f:
        data = json.load(f)
    valid = [
        e
        for e in data
        if e.get("uid")
        and e.get("image_path")
        and (e.get("context") or e.get("retrieved_context"))
        and Path(e["image_path"]).exists()
    ]
    rng = random.Random(seed)
    rng.shuffle(valid)
    return valid[:num_samples]


def _ctx(e: dict) -> str:
    return (e.get("context") or e.get("retrieved_context") or "").strip()


def target_context(entry: dict) -> str:
    base = _ctx(entry)
    return uc.reorder_nih_context(base, 2)


def generate_one(
    method: str,
    entry: dict,
    max_new_tokens: int,
    instruction: str,
    device: str,
    compressor_device: str,
    alpha_v: float,
) -> str:
    image_path = str(entry["image_path"])
    q = entry.get("question") or "Based on the visual evidence, what are the primary impressions for this chest radiograph?"
    ctx = target_context(entry)

    if method == "standard_rag":
        return uc.generate_standard_chexagent(image_path, q, ctx, max_new_tokens, instruction, device)
    if method == "bair":
        return uc.generate_with_bair_and_mspoe_chexagent(
            image_path, q, ctx, max_new_tokens, instruction, alpha_v, 1.0, 1.0, device, 1.0, False, use_madrag=False
        )
    if method == "mspoe":
        return uc.generate_with_bair_and_mspoe_chexagent(
            image_path, q, ctx, max_new_tokens, instruction, 0.0, 0.0, 1.0, device, 1.5, False, use_madrag=False
        )
    if method == "madrag":
        return uc.generate_with_bair_and_mspoe_chexagent(
            image_path, q, ctx, max_new_tokens, instruction, 0.0, 0.0, 1.0, device, 1.0, False, use_madrag=True
        )
    if method == "longllmlingua":
        comp = uc.compress_with_longllmlingua(ctx, q, instruction, compressor_device, 0.5)
        return uc.generate_standard_chexagent(image_path, q, comp, max_new_tokens, instruction, device)
    raise ValueError(method)


def _compressor_device(args: argparse.Namespace) -> str:
    if getattr(args, "longllmlingua_compressor_device", None):
        return str(args.longllmlingua_compressor_device)
    if args.compressor_gpu_id < 0:
        return "cpu"
    return f"cuda:{args.compressor_gpu_id}"


def run_method(args: argparse.Namespace) -> None:
    device = f"cuda:{args.main_gpu_id}" if torch.cuda.is_available() else "cpu"
    comp = _compressor_device(args)
    if args.method == "longllmlingua":
        print(f"LongLLMLingua compressor device: {comp!r}", flush=True)
        if torch.cuda.is_available() and comp.startswith("cuda:"):
            cidx = int(comp.split(":")[1])
            nvis = torch.cuda.device_count()
            if cidx >= nvis:
                raise SystemExit(
                    f"Compressor asks for {comp} but only {nvis} CUDA device(s) are visible. "
                    f"Use CUDA_VISIBLE_DEVICES=0,2 and --longllmlingua-compressor-device cuda:1 "
                    f"or --compressor-gpu-id 2 with physical GPU IDs."
                )
        if comp == "cpu":
            print(
                "Warning: LongLLMLingua on CPU — expect very slow compression.",
                file=sys.stderr,
                flush=True,
            )

    entries = load_entries(args.input_json, args.num_samples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    method_dir = args.output_dir / args.method
    method_dir.mkdir(parents=True, exist_ok=True)

    rows, outputs = [], []
    t0 = time.perf_counter()
    with ResourceMonitor(args.monitor_gpu_id, interval=args.monitor_interval) as mon:
        for entry in entries:
            start = time.perf_counter()
            err, text = None, ""
            try:
                text = generate_one(args.method, entry, args.max_new_tokens, args.instruction, device, comp, args.alpha_v)
            except Exception as exc:
                err = repr(exc)
            rows.append({"method": args.method, "uid": entry.get("uid"), "seconds": time.perf_counter() - start, "error": err})
            outputs.append({"uid": entry.get("uid"), "answer": text, "error": err})
    total = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    ok = df[df["error"].isna()]
    summary = {
        "suite": "chexagent_iuchest",
        "model": "StanfordAIMI/CheXagent-8b",
        "method": args.method,
        "method_label": METHOD_LABELS[args.method],
        "alpha_v": float(args.alpha_v) if args.method == "bair" else np.nan,
        "alpha_t": 1.0 if args.method == "bair" else np.nan,
        "num_requested": args.num_samples,
        "num_success": int(len(ok)),
        "num_error": int(df["error"].notna().sum()),
        "total_seconds": float(total),
        "mean_seconds_per_sample": float(ok["seconds"].mean()) if not ok.empty else np.nan,
        "median_seconds_per_sample": float(ok["seconds"].median()) if not ok.empty else np.nan,
        "std_seconds_per_sample": float(ok["seconds"].std(ddof=0)) if len(ok) > 1 else 0.0,
        **mon.summary(),
    }
    df.to_csv(method_dir / "per_sample.csv", index=False)
    (method_dir / "outputs.json").write_text(json.dumps(outputs, indent=2))
    (method_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def run_all(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for m in METHODS:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--method",
            m,
            "--input-json",
            str(args.input_json),
            "--output-dir",
            str(args.output_dir),
            "--num-samples",
            str(args.num_samples),
            "--seed",
            str(args.seed),
            "--main-gpu-id",
            str(args.main_gpu_id),
            "--monitor-gpu-id",
            str(args.monitor_gpu_id),
            "--compressor-gpu-id",
            str(args.compressor_gpu_id),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--monitor-interval",
            str(args.monitor_interval),
            "--instruction",
            args.instruction,
            "--alpha-v",
            str(args.alpha_v),
            "--expect-conda-env",
            "",
        ]
        if args.longllmlingua_compressor_device:
            cmd.extend(["--longllmlingua-compressor-device", args.longllmlingua_compressor_device])
        print(f"\n=== CheXagent {METHOD_LABELS[m]} ===", flush=True)
        subprocess.run(cmd, check=False)
        sp = args.output_dir / m / "summary.json"
        if sp.is_file():
            summaries.append(json.loads(sp.read_text()))
    if summaries:
        pd.DataFrame(summaries).to_csv(args.output_dir / "efficiency_summary.csv", index=False)
        (args.output_dir / "efficiency_summary.json").write_text(json.dumps(summaries, indent=2))
        print(f"Saved {args.output_dir / 'efficiency_summary.csv'}")


def main() -> None:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="CheXagent efficiency benchmark (IU-Chest).")
    p.add_argument("--method", choices=METHODS + ["all"], default="all")
    p.add_argument(
        "--input-json",
        type=Path,
        default=base / "generation_results_chexagent" / "iuchest_chexagent_results_baselines.json",
    )
    p.add_argument("--output-dir", type=Path, default=base / "outputs" / "computational_cost_runs" / "chexagent")
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--main-gpu-id", type=int, default=0)
    p.add_argument("--monitor-gpu-id", type=int, default=None)
    p.add_argument("--compressor-gpu-id", type=int, default=2)
    p.add_argument("--longllmlingua-compressor-device", default=None)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--monitor-interval", type=float, default=0.2)
    p.add_argument(
        "--instruction",
        default="You are a radiologist. When context is provided, refer to it to accurately describe the image. "
        "If no context is provided, describe the image based on your knowledge.",
    )
    p.add_argument("--alpha-v", type=float, default=0.5)
    p.add_argument("--expect-conda-env", default="lingua")
    args = p.parse_args()
    if args.monitor_gpu_id is None:
        args.monitor_gpu_id = args.main_gpu_id
    if args.expect_conda_env and os.environ.get("CONDA_DEFAULT_ENV") != args.expect_conda_env:
        print(
            f"Warning: CONDA_DEFAULT_ENV={os.environ.get('CONDA_DEFAULT_ENV')!r} expected {args.expect_conda_env!r}",
            file=sys.stderr,
        )
    if args.method == "all":
        run_all(args)
    else:
        run_method(args)


if __name__ == "__main__":
    main()
