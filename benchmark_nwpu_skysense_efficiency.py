#!/usr/bin/env python3
"""NWPU + SkySenseGPT (GeoChat family) computational cost — same five method names as other suites."""

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
from llm_explainer import load_llm_model
import nwpu_analysis as na

METHODS = ["standard_rag", "bair", "mspoe", "longllmlingua", "madrag"]
METHOD_LABELS = {
    "standard_rag": "Standard RAG",
    "bair": "BAIR",
    "mspoe": "MS-PoE",
    "longllmlingua": "LongLLMLingua",
    "madrag": "MAD-RAG",
}

DEFAULT_QUESTION = (
    "You are an expert in remote sensing and geospatial analysis. "
    "Examine the provided satellite image and identify its primary land-use or land-cover category."
)
DEFAULT_INSTRUCTION = "Use the image as primary evidence and use retrieved context as supporting information."


def load_entries(path: Path, num_samples: int, seed: int) -> List[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    valid = [
        e
        for e in data
        if isinstance(e, dict)
        and e.get("uid")
        and e.get("image_path")
        and e.get("nwpu_context")
        and Path(e["image_path"]).exists()
    ]
    rng = random.Random(seed)
    rng.shuffle(valid)
    return valid[:num_samples]


def _rag_context(raw_ctx: str, mode: str) -> str:
    if mode == "gt_only":
        return na.extract_gt_only_context(raw_ctx)
    return raw_ctx


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
    tokenizer = model_components.get("tokenizer") or model_components.get("processor")
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if tokenizer is None:
        return 0
    try:
        encoded = tokenizer(text, add_special_tokens=False)
        if isinstance(encoded, dict):
            return int(len(encoded.get("input_ids", [])))
        if hasattr(encoded, "input_ids"):
            return int(len(encoded.input_ids))
    except Exception:
        return 0
    return 0


def generate_one(
    method: str,
    model_name: str,
    model_components: dict,
    entry: dict,
    question: str,
    instruction: str,
    max_new_tokens: int,
    compressor_device: str,
    context_mode: str,
    skysense_max_retries: int,
    alpha_v: float,
    alpha_t: float,
) -> str:
    image_path = str(entry["image_path"])
    ctx = _rag_context((entry.get("nwpu_context") or "").strip(), context_mode)

    if method == "standard_rag":
        return na.generate_standard(
            model_components=model_components,
            image_path=image_path,
            question=question,
            instruction=instruction,
            context=ctx,
            max_new_tokens=max_new_tokens,
        )
    if method == "bair":
        return na.run_intervention(
            model_name=model_name,
            model_components=model_components,
            image_path=image_path,
            question=question,
            instruction=instruction,
            context=ctx,
            max_new_tokens=max_new_tokens,
            alpha_v=alpha_v,
            alpha_t=alpha_t,
            gamma_s=1.0,
            mspoe_scaling=1.0,
            mspoe_text_only=False,
            use_madrag=False,
            skysense_max_retries=skysense_max_retries,
            allow_quality_fallback=False,
        )
    if method == "mspoe":
        return na.run_intervention(
            model_name=model_name,
            model_components=model_components,
            image_path=image_path,
            question=question,
            instruction=instruction,
            context=ctx,
            max_new_tokens=max_new_tokens,
            alpha_v=0.0,
            alpha_t=0.0,
            gamma_s=1.0,
            mspoe_scaling=1.5,
            mspoe_text_only=False,
            use_madrag=False,
            skysense_max_retries=skysense_max_retries,
        )
    if method == "madrag":
        return na.run_intervention(
            model_name=model_name,
            model_components=model_components,
            image_path=image_path,
            question=question,
            instruction=instruction,
            context=ctx,
            max_new_tokens=max_new_tokens,
            alpha_v=0.0,
            alpha_t=0.0,
            gamma_s=1.0,
            mspoe_scaling=1.0,
            mspoe_text_only=False,
            use_madrag=True,
            skysense_max_retries=skysense_max_retries,
        )
    if method == "longllmlingua":
        compressed = na.compress_with_longllmlingua(ctx, question, instruction, compressor_device, ratio=0.5)
        return na.generate_standard(
            model_components=model_components,
            image_path=image_path,
            question=question,
            instruction=instruction,
            context=compressed,
            max_new_tokens=max_new_tokens,
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

    entries = load_entries(args.dataset_json, args.num_samples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    method_dir = args.output_dir / args.method
    method_dir.mkdir(parents=True, exist_ok=True)

    model_components = load_llm_model(args.model_name, gpu_id=args.main_gpu_id)
    rows, outputs = [], []
    t0 = time.perf_counter()
    with ResourceMonitor(args.monitor_gpu_id, interval=args.monitor_interval) as mon:
        for entry in entries:
            start = time.perf_counter()
            err, text = None, ""
            try:
                text = generate_one(
                    args.method,
                    args.model_name,
                    model_components,
                    entry,
                    args.question,
                    args.instruction,
                    args.max_new_tokens,
                    comp,
                    args.context_mode,
                    args.skysense_max_retries,
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
                    "uid": entry.get("uid"),
                    "seconds": seconds,
                    "generated_tokens": generated_tokens,
                    "seconds_per_128_generated_tokens": seconds_per_128_generated_tokens,
                    "error": err,
                }
            )
            outputs.append(
                {
                    "uid": entry.get("uid"),
                    "answer": text,
                    "generated_tokens": generated_tokens,
                    "error": err,
                }
            )
    total = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    ok = df[df["error"].isna()]
    summary = {
        "suite": "nwpu_skysense",
        "model": args.model_name,
        "context_mode": args.context_mode,
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
            "--dataset-json",
            str(args.dataset_json),
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
            "--context-mode",
            args.context_mode,
            "--skysense-max-retries",
            str(args.skysense_max_retries),
            "--alpha-v",
            str(args.alpha_v),
            "--alpha-t",
            str(args.alpha_t),
            "--question",
            args.question,
            "--instruction",
            args.instruction,
            "--compressor-gpu-id",
            str(args.compressor_gpu_id),
            "--expect-conda-env",
            "",
        ]
        if args.longllmlingua_compressor_device:
            cmd.extend(["--longllmlingua-compressor-device", args.longllmlingua_compressor_device])
        print(f"\n=== NWPU SkySense {METHOD_LABELS[m]} ===", flush=True)
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
    p = argparse.ArgumentParser(description="NWPU SkySenseGPT efficiency benchmark.")
    p.add_argument("--method", choices=METHODS + ["all"], default="all")
    p.add_argument("--dataset-json", type=Path, default=base / "data" / "generated" / "nwpu_retrieval_dataset.json")
    p.add_argument("--output-dir", type=Path, default=base / "outputs" / "computational_cost_runs" / "nwpu_skysense")
    p.add_argument("--model-name", default="ll-13/SkySenseGPT-7B-clip-lora")
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--main-gpu-id", type=int, default=0)
    p.add_argument("--monitor-gpu-id", type=int, default=None)
    p.add_argument("--compressor-gpu-id", type=int, default=2)
    p.add_argument("--longllmlingua-compressor-device", default=None)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--monitor-interval", type=float, default=0.2)
    p.add_argument("--context-mode", choices=["full", "gt_only"], default="full")
    p.add_argument("--skysense-max-retries", type=int, default=2)
    p.add_argument("--alpha-v", type=float, default=0.5)
    p.add_argument("--alpha-t", type=float, default=1.0, help=argparse.SUPPRESS)
    p.add_argument("--question", default=DEFAULT_QUESTION)
    p.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
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
