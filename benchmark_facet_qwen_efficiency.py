#!/usr/bin/env python3
"""FACET gender-analysis setup: Qwen2.5-VL computational cost (oracle RAG variants)."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from benchmark_cost_lib import ResourceMonitor
import bair_efficient
import gender_analysis as ga
from llm_explainer import load_llm_model

METHODS = ["standard_rag", "bair", "mspoe", "longllmlingua", "madrag"]
METHOD_LABELS = {
    "standard_rag": "Standard RAG",
    "bair": "BAIR",
    "mspoe": "MS-PoE",
    "longllmlingua": "LongLLMLingua",
    "madrag": "MAD-RAG",
}


def load_entries(path: Path, num_samples: int, seed: int) -> List[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    valid = []
    for e in data:
        if not isinstance(e, dict):
            continue
        oid = e.get("uid") or e.get("filename")
        if not oid or not e.get("image_path") or not e.get("oracle_context"):
            continue
        if not Path(e["image_path"]).exists():
            continue
        valid.append(e)
    rng = random.Random(seed)
    rng.shuffle(valid)
    return valid[:num_samples]


def _entry_id(entry: dict) -> str:
    return str(entry.get("uid") or entry.get("filename") or "")


def _compressor_device(args: argparse.Namespace) -> str:
    if getattr(args, "longllmlingua_compressor_device", None):
        return str(args.longllmlingua_compressor_device)
    if args.compressor_gpu_id < 0:
        return "cpu"
    return f"cuda:{args.compressor_gpu_id}"


def _is_failed_generation(text: str) -> bool:
    value = (text or "").strip()
    return (
        not value
        or value == "[GENERATION_FAILED]"
        or value.startswith("[Error]")
        or value.startswith("[UNSUPPORTED")
        or value.lower().startswith("error:")
    )


def _count_generated_tokens(model_components: dict, text: str) -> int:
    if _is_failed_generation(text):
        return 0
    processor = model_components.get("processor")
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer is None:
        return 0
    try:
        encoded = tokenizer(text, add_special_tokens=False)
        return int(len(encoded.get("input_ids", [])))
    except Exception:
        return 0


def generate_one(
    method: str,
    model_components: dict,
    entry: dict,
    max_new_tokens: int,
    compressor_device: str,
    qwen_pixel_limit: int,
    alpha_v: float,
    alpha_t: float,
) -> str:
    question = entry.get("question") or ""
    instruction = entry.get("instruction") or ""
    image_path = str(entry["image_path"])
    ctx = (entry.get("oracle_context") or "").strip()

    if method == "standard_rag":
        return ga.generate_with_qwen_intervention(
            model_components=model_components,
            question=question,
            image_path=image_path,
            oracle_context=ctx,
            instruction=instruction,
            max_new_tokens=max_new_tokens,
            alpha_v=0.0,
            alpha_t=0.0,
            gamma_s=1.0,
            qwen_pixel_limit=qwen_pixel_limit if qwen_pixel_limit > 0 else None,
            mspoe_scaling=1.0,
            mspoe_text_only=False,
            use_madrag=False,
            include_experiment_notice=True,
        )
    if method == "bair":
        return ga.generate_with_qwen_intervention(
            model_components=model_components,
            question=question,
            image_path=image_path,
            oracle_context=ctx,
            instruction=instruction,
            max_new_tokens=max_new_tokens,
            alpha_v=alpha_v,
            alpha_t=alpha_t,
            gamma_s=1.0,
            qwen_pixel_limit=qwen_pixel_limit if qwen_pixel_limit > 0 else None,
            mspoe_scaling=1.0,
            mspoe_text_only=False,
            use_madrag=False,
            skip_failed=True,
            include_experiment_notice=True,
        )
    if method == "mspoe":
        return ga.generate_with_qwen_intervention(
            model_components=model_components,
            question=question,
            image_path=image_path,
            oracle_context=ctx,
            instruction=instruction,
            max_new_tokens=max_new_tokens,
            alpha_v=0.0,
            alpha_t=0.0,
            gamma_s=1.0,
            qwen_pixel_limit=qwen_pixel_limit if qwen_pixel_limit > 0 else None,
            mspoe_scaling=1.5,
            mspoe_text_only=False,
            use_madrag=False,
            include_experiment_notice=True,
        )
    if method == "madrag":
        return ga.generate_with_qwen_intervention(
            model_components=model_components,
            question=question,
            image_path=image_path,
            oracle_context=ctx,
            instruction=instruction,
            max_new_tokens=max_new_tokens,
            alpha_v=0.0,
            alpha_t=0.0,
            gamma_s=1.0,
            qwen_pixel_limit=qwen_pixel_limit if qwen_pixel_limit > 0 else None,
            mspoe_scaling=1.0,
            mspoe_text_only=False,
            use_madrag=True,
            include_experiment_notice=True,
        )
    if method == "longllmlingua":
        compressed = ga.compress_with_longllmlingua(ctx, question, instruction, compressor_device, 0.5)
        return ga.generate_with_qwen_intervention(
            model_components=model_components,
            question=question,
            image_path=image_path,
            oracle_context=compressed,
            instruction=instruction,
            max_new_tokens=max_new_tokens,
            alpha_v=0.0,
            alpha_t=0.0,
            gamma_s=1.0,
            qwen_pixel_limit=qwen_pixel_limit if qwen_pixel_limit > 0 else None,
            mspoe_scaling=1.0,
            mspoe_text_only=False,
            use_madrag=False,
            include_experiment_notice=True,
        )
    raise ValueError(method)


def run_method(args: argparse.Namespace) -> None:
    comp = _compressor_device(args)
    if args.method == "longllmlingua":
        print(f"LongLLMLingua compressor device: {comp!r}", flush=True)
        if torch.cuda.is_available() and comp.startswith("cuda:"):
            cidx = int(comp.split(":")[1])
            nvis = torch.cuda.device_count()
            if cidx >= nvis:
                raise SystemExit(
                    f"Compressor asks for {comp} but only {nvis} CUDA device(s) are visible. "
                    f"Typical fix: CUDA_VISIBLE_DEVICES=0,2 and --longllmlingua-compressor-device cuda:1."
                )

    entries = load_entries(args.input_json, args.num_samples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    method_dir = args.output_dir / args.method
    method_dir.mkdir(parents=True, exist_ok=True)

    model_components = load_llm_model(
        args.model_name,
        gpu_id=args.main_gpu_id,
        use_multi_gpu=args.use_multi_gpu,
    )
    rows, outputs = [], []
    t0 = time.perf_counter()
    with ResourceMonitor(args.monitor_gpu_id, interval=args.monitor_interval) as mon:
        for entry in entries:
            start = time.perf_counter()
            err, text = None, ""
            try:
                text = generate_one(
                    args.method,
                    model_components,
                    entry,
                    args.max_new_tokens,
                    comp,
                    args.qwen_pixel_limit,
                    args.alpha_v,
                    args.alpha_t,
                )
            except Exception as exc:
                err = repr(exc)
            seconds = time.perf_counter() - start
            generated_tokens = _count_generated_tokens(model_components, text)
            if err is None and _is_failed_generation(text):
                err = "GENERATION_FAILED"
            seconds_per_128_generated_tokens = (
                seconds * 128.0 / generated_tokens if err is None and generated_tokens > 0 else np.nan
            )
            rows.append(
                {
                    "method": args.method,
                    "uid": _entry_id(entry),
                    "seconds": seconds,
                    "generated_tokens": generated_tokens,
                    "seconds_per_128_generated_tokens": seconds_per_128_generated_tokens,
                    "error": err,
                }
            )
            outputs.append(
                {
                    "uid": _entry_id(entry),
                    "answer": text,
                    "generated_tokens": generated_tokens,
                    "error": err,
                }
            )
    total = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    ok = df[df["error"].isna()]
    summary = {
        "suite": "facet_qwen",
        "model": args.model_name,
        "method": args.method,
        "method_label": METHOD_LABELS[args.method],
        "alpha_v": float(args.alpha_v) if args.method == "bair" else np.nan,
        "alpha_t": float(args.alpha_t) if args.method == "bair" else np.nan,
        "bair_efficient_mode": bool(bair_efficient.enabled()),
        "bair_shared_prefix": False,
        "num_requested": args.num_samples,
        "num_success": int(len(ok)),
        "num_error": int(df["error"].notna().sum()),
        "total_seconds": float(total),
        "mean_seconds_per_sample": float(ok["seconds"].mean()) if not ok.empty else np.nan,
        "median_seconds_per_sample": float(ok["seconds"].median()) if not ok.empty else np.nan,
        "std_seconds_per_sample": float(ok["seconds"].std(ddof=0)) if len(ok) > 1 else 0.0,
        "mean_generated_tokens": float(ok["generated_tokens"].mean()) if not ok.empty else np.nan,
        "mean_seconds_per_128_generated_tokens": (
            float(ok["seconds_per_128_generated_tokens"].mean()) if not ok.empty else np.nan
        ),
        "median_seconds_per_128_generated_tokens": (
            float(ok["seconds_per_128_generated_tokens"].median()) if not ok.empty else np.nan
        ),
        **mon.summary(),
    }
    df.to_csv(method_dir / "per_sample.csv", index=False)
    (method_dir / "outputs.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    (method_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
            "--model-name",
            args.model_name,
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
            "--qwen-pixel-limit",
            str(args.qwen_pixel_limit),
            "--alpha-v",
            str(args.alpha_v),
            "--alpha-t",
            str(args.alpha_t),
            "--compressor-gpu-id",
            str(args.compressor_gpu_id),
            "--expect-conda-env",
            "",
        ]
        if args.use_multi_gpu:
            cmd.append("--use-multi-gpu")
        if args.longllmlingua_compressor_device:
            cmd.extend(["--longllmlingua-compressor-device", args.longllmlingua_compressor_device])
        print(f"\n=== FACET Qwen {METHOD_LABELS[m]} ===", flush=True)
        subprocess.run(cmd, check=False)
        sp = args.output_dir / m / "summary.json"
        if sp.is_file():
            summaries.append(json.loads(sp.read_text(encoding="utf-8")))
    if summaries:
        pd.DataFrame(summaries).to_csv(args.output_dir / "efficiency_summary.csv", index=False)
        (args.output_dir / "efficiency_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"Saved {args.output_dir / 'efficiency_summary.csv'}")


def main() -> None:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="FACET Qwen2.5-VL efficiency benchmark.")
    p.add_argument("--method", choices=METHODS + ["all"], default="all")
    p.add_argument(
        "--input-json",
        type=Path,
        default=base / "outputs" / "generation_results_facet_qwen" / "analysis_results_Qwen_Qwen2.5_VL_3B_Instruct_with_instruction_2_intervention_av0.5.json",
    )
    p.add_argument("--output-dir", type=Path, default=base / "outputs" / "computational_cost_runs" / "facet_qwen")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--main-gpu-id", type=int, default=0)
    p.add_argument("--use-multi-gpu", action="store_true", help="Load Qwen with device_map='balanced' to avoid single-device allocator warmup failures.")
    p.add_argument("--monitor-gpu-id", type=int, default=None)
    p.add_argument("--compressor-gpu-id", type=int, default=2)
    p.add_argument("--longllmlingua-compressor-device", default=None)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--monitor-interval", type=float, default=0.2)
    p.add_argument("--qwen-pixel-limit", type=int, default=28 * 28 * 50)
    p.add_argument("--alpha-v", type=float, default=0.5)
    p.add_argument("--alpha-t", type=float, default=1.0, help=argparse.SUPPRESS)
    p.add_argument("--expect-conda-env", default="")
    args = p.parse_args()
    args.alpha_t = 1.0
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
