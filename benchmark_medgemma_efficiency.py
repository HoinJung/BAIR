#!/usr/bin/env python3
"""
Benchmark MedGemma IU-Chest efficiency for RAG/intervention methods.

Runs each method in a fresh Python process so peak RAM/VRAM measurements are
not contaminated by models left over from previous methods.

Recommended one-shot runner (sets CUDA_VISIBLE_DEVICES like other BAIR scripts): see
``run_medgemma_efficiency_benchmark.sh``.

BAIR and MAD-RAG both do one calibration forward and one generation forward.
The BAIR intervention is applied on the generation prefill attention row; the
patched attention hooks bypass single-token cached decode.
"""

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
from PIL import Image

from benchmark_cost_lib import ResourceMonitor
import medical_analysis as ma
from bottleneck_intervention import set_bottleneck_intervention
import bair_efficient


METHODS = ["standard_rag", "bair", "mspoe", "longllmlingua", "madrag"]
METHOD_LABELS = {
    "standard_rag": "Standard RAG",
    "bair": "BAIR",
    "mspoe": "MS-PoE",
    "longllmlingua": "LongLLMLingua",
    "madrag": "MAD-RAG",
}


def load_entries(path: Path, num_samples: int, seed: int) -> List[dict]:
    with path.open("r") as f:
        data = json.load(f)
    valid = [
        e
        for e in data
        if e.get("uid")
        and e.get("image_path")
        and (e.get("retrieved_context") or e.get("context"))
        and Path(e["image_path"]).exists()
    ]
    rng = random.Random(seed)
    rng.shuffle(valid)
    return valid[:num_samples]


def _question(entry: dict) -> str:
    return entry.get("question") or "Based on the visual evidence, what are the primary impressions for this chest radiograph?"


def _context(entry: dict) -> str:
    return entry.get("retrieved_context") or entry.get("context") or ""


def generate_madrag(entry: dict, max_new_tokens: int, instruction: Optional[str], device: str) -> str:
    model, processor, num_visual_tokens = ma.load_medgemma_intervention_model(device)
    image = Image.open(entry["image_path"]).convert("RGB")
    question = _question(entry)

    clean_prompt = ma.build_full_prompt(question, context=None, instruction=instruction)
    clean_conv = [{"role": "user", "content": [{"type": "text", "text": clean_prompt}, {"type": "image"}]}]
    clean_text = processor.apply_chat_template(clean_conv, tokenize=False, add_generation_prompt=True)
    clean_inputs = processor(images=image, text=clean_text, return_tensors="pt", padding=True)
    clean_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in clean_inputs.items()}

    set_bottleneck_intervention(
        True,
        num_visual_tokens=num_visual_tokens,
        calibration_run=True,
        reset_layer=True,
        alpha_v=0.0,
        alpha_t=0.0,
        use_madrag=True,
    )
    with torch.no_grad():
        model(**clean_inputs, use_cache=True)

    full_prompt = ma.build_full_prompt(question, context=_context(entry), instruction=instruction)
    gen_conv = [{"role": "user", "content": [{"type": "text", "text": full_prompt}, {"type": "image"}]}]
    gen_text = processor.apply_chat_template(gen_conv, tokenize=False, add_generation_prompt=True)
    question_suffix = f"\n\nQuestion: {question}"
    prefix = gen_text.split(question_suffix)[0] if question_suffix in gen_text else gen_text.rsplit(question, 1)[0]
    question_tokens = len(processor.tokenizer(gen_text)["input_ids"]) - len(processor.tokenizer(prefix)["input_ids"])

    inputs = processor(images=image, text=gen_text, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    bair_efficient.reuse_medgemma_pixel_values_if_efficient(clean_inputs, inputs)
    set_bottleneck_intervention(
        True,
        num_visual_tokens=num_visual_tokens,
        calibration_run=False,
        reset_layer=True,
        alpha_v=0.0,
        alpha_t=0.0,
        question_tokens=max(0, question_tokens),
        use_madrag=True,
    )
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


def generate_one(
    method: str,
    entry: dict,
    max_new_tokens: int,
    instruction: str,
    device: str,
    compressor_device: str,
    *,
    alpha_v: float,
    alpha_t: float,
) -> str:
    question = _question(entry)
    context = _context(entry)
    image_path = str(entry["image_path"])
    if method == "standard_rag":
        # Use the same direct/custom model path as intervention baselines.  The
        # HF pipeline has different preprocessing and runtime overhead, which
        # makes wall-clock comparisons with MS-PoE/BAIR misleading.
        return ma.generate_with_mspoe(
            image_path,
            question,
            context,
            max_new_tokens=max_new_tokens,
            instruction=instruction,
            device=device,
            scaling_factor=1.0,
            text_only=False,
        )
    if method == "bair":
        return ma.generate_with_medgemma_intervention(
            image_path,
            question,
            context,
            max_new_tokens=max_new_tokens,
            instruction=instruction,
            alpha_v=alpha_v,
            alpha_t=alpha_t,
            device=device,
        )
    if method == "mspoe":
        return ma.generate_with_mspoe(
            image_path, question, context, max_new_tokens=max_new_tokens, instruction=instruction, device=device, scaling_factor=1.5, text_only=False
        )
    if method == "longllmlingua":
        compressed = ma.compress_with_longllmlingua(context, question, instruction, device=compressor_device, ratio=0.5)
        return ma.generate_with_medgemma(image_path, question, compressed, max_new_tokens=max_new_tokens, instruction=instruction)
    if method == "madrag":
        return generate_madrag(entry, max_new_tokens=max_new_tokens, instruction=instruction, device=device)
    raise ValueError(f"Unknown method: {method}")


def _compressor_device(args) -> str:
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


def _count_generated_tokens(text: str) -> int:
    if _is_failed_generation(text):
        return 0
    processor = ma.INTERVENTION_PROCESSOR
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer is None:
        return 0
    try:
        encoded = tokenizer(text, add_special_tokens=False)
        return int(len(encoded.get("input_ids", [])))
    except Exception:
        return 0


def run_method(args) -> None:
    device = f"cuda:{args.medgemma_gpu_id}" if torch.cuda.is_available() else "cpu"
    ma.DEVICE_ID = args.medgemma_gpu_id
    ma.DEVICE = device
    compressor_dev = _compressor_device(args)
    if args.method == "longllmlingua":
        print(f"LongLLMLingua compressor device: {compressor_dev!r}", flush=True)
        if torch.cuda.is_available() and compressor_dev.startswith("cuda:"):
            cidx = int(compressor_dev.split(":")[1])
            nvis = torch.cuda.device_count()
            if cidx >= nvis:
                raise SystemExit(
                    f"Compressor asks for {compressor_dev} but only {nvis} CUDA device(s) are visible. "
                    f"You probably have CUDA_VISIBLE_DEVICES restricting the bus (e.g. only GPU 0). "
                    f"Fix: run without masking, or use CUDA_VISIBLE_DEVICES=0,2 and "
                    f"--longllmlingua-compressor-device cuda:1 (second visible = physical 2)."
                )
        if compressor_dev == "cpu" and not args.quiet:
            print(
                "Warning: LongLLMLingua on CPU — expect very slow compression (~minutes/sample for Llama-2-7b).",
                file=sys.stderr,
                flush=True,
            )
    if (
        not args.quiet
        and args.method == "longllmlingua"
        and torch.cuda.is_available()
        and compressor_dev.startswith("cuda:")
        and device.startswith("cuda:")
        and compressor_dev == device
    ):
        print(
            "Warning: MedGemma and LongLLMLingua share the same GPU; expect VRAM pressure or OOM.",
            file=sys.stderr,
            flush=True,
        )
    entries = load_entries(args.input_json, args.num_samples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    method_dir = args.output_dir / args.method
    method_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = []
    outputs = []
    load_start = time.perf_counter()
    with ResourceMonitor(args.monitor_gpu_id, interval=args.monitor_interval) as monitor:
        # Generation functions lazily load models on first call; track total and per-sample latency.
        for entry in entries:
            start = time.perf_counter()
            error = None
            text = ""
            try:
                text = generate_one(
                    args.method,
                    entry,
                    args.max_new_tokens,
                    args.instruction,
                    device,
                    compressor_dev,
                    alpha_v=args.alpha_v,
                    alpha_t=args.alpha_t,
                )
            except Exception as exc:
                error = repr(exc)
            seconds = time.perf_counter() - start
            generated_tokens = _count_generated_tokens(text)
            if error is None and _is_failed_generation(text):
                error = "GENERATION_FAILED"
            seconds_per_128_generated_tokens = (
                seconds * 128.0 / generated_tokens if error is None and generated_tokens > 0 else np.nan
            )
            sample_rows.append(
                {
                    "method": args.method,
                    "uid": entry.get("uid"),
                    "seconds": seconds,
                    "generated_tokens": generated_tokens,
                    "seconds_per_128_generated_tokens": seconds_per_128_generated_tokens,
                    "error": error,
                }
            )
            outputs.append(
                {
                    "uid": entry.get("uid"),
                    "method": args.method,
                    "answer": text,
                    "generated_tokens": generated_tokens,
                    "error": error,
                }
            )
    total_seconds = time.perf_counter() - load_start

    samples_df = pd.DataFrame(sample_rows)
    ok = samples_df[samples_df["error"].isna()]
    summary = {
        "suite": "medgemma_iuchest",
        "model": ma.MEDGEMMA_MODEL_ID,
        "method": args.method,
        "method_label": METHOD_LABELS.get(args.method, args.method),
        "alpha_v": float(args.alpha_v),
        "alpha_t": float(args.alpha_t),
        "bair_efficient_mode": bool(bair_efficient.enabled()),
        "bair_shared_prefix": False,
        "num_requested": int(args.num_samples),
        "num_success": int(len(ok)),
        "num_error": int(samples_df["error"].notna().sum()),
        "total_seconds": float(total_seconds),
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
        **monitor.summary(),
    }

    samples_df.to_csv(method_dir / "per_sample.csv", index=False)
    with (method_dir / "outputs.json").open("w") as f:
        json.dump(outputs, f, indent=2)
    with (method_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def run_all(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for method in METHODS:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--method",
            method,
            "--input-json",
            str(args.input_json),
            "--output-dir",
            str(args.output_dir),
            "--num-samples",
            str(args.num_samples),
            "--seed",
            str(args.seed),
            "--medgemma-gpu-id",
            str(args.medgemma_gpu_id),
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
        ]
        if args.longllmlingua_compressor_device:
            cmd.extend(["--longllmlingua-compressor-device", args.longllmlingua_compressor_device])
        cmd.extend(["--expect-conda-env", ""])
        cmd.extend(["--alpha-v", str(args.alpha_v), "--alpha-t", str(args.alpha_t)])
        if args.quiet:
            cmd.append("--quiet")
        print(f"\n=== Running {METHOD_LABELS[method]} ===", flush=True)
        subprocess.run(cmd, check=args.strict)
        summary_path = args.output_dir / method / "summary.json"
        if summary_path.exists():
            with summary_path.open("r") as f:
                summaries.append(json.load(f))
    if summaries:
        df = pd.DataFrame(summaries)
        df.to_csv(args.output_dir / "efficiency_summary.csv", index=False)
        with (args.output_dir / "efficiency_summary.json").open("w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nSaved summary: {args.output_dir / 'efficiency_summary.csv'}")


def main() -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Benchmark MedGemma IU-Chest computational cost and inference time.",
        epilog="Use: conda run -n lingua python %(prog)s --medgemma-gpu-id 0 --compressor-gpu-id 2",
    )
    parser.add_argument("--method", choices=METHODS + ["all"], default="all")
    parser.add_argument("--input-json", type=Path, default=base / "outputs" / "generation_results_medgemma" / "iuchest_medgemma_results_new_bair_av0.5_mid.json")
    parser.add_argument("--output-dir", type=Path, default=base / "outputs" / "medgemma_efficiency_benchmark")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--medgemma-gpu-id", type=int, default=0, help="CUDA device index for MedGemma (pipeline / intervention model).")
    parser.add_argument(
        "--monitor-gpu-id",
        type=int,
        default=None,
        help="GPU index for nvidia-smi utilization monitoring (default: same as --medgemma-gpu-id).",
    )
    parser.add_argument(
        "--compressor-gpu-id",
        type=int,
        default=2,
        help="CUDA device index for LongLLMLingua compressor. Ignored if --longllmlingua-compressor-device is set. Use -1 for CPU.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--monitor-interval", type=float, default=0.2)
    parser.add_argument(
        "--longllmlingua-compressor-device",
        default=None,
        help=(
            "Optional override for compressor device (e.g. cuda:2, cpu). If unset, uses cuda:{--compressor-gpu-id}. "
            "With CUDA_VISIBLE_DEVICES=0,2, physical GPU 2 is usually cuda:1 inside the process."
        ),
    )
    parser.add_argument("--instruction", default=ma.DEFAULT_GENERATION_RESULTS_JSON and "You are a radiologist. When context is provided, refer to it to accurately describe the image. If no context is provided, describe the image based on your knowledge.")
    parser.add_argument(
        "--alpha-v",
        type=float,
        default=0.5,
        help="BAIR bottleneck visual strength (only used when --method bair).",
    )
    parser.add_argument(
        "--alpha-t",
        type=float,
        default=1.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-fatal stderr warnings (compressor/GPU sharing, conda env).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="When --method all, exit non-zero if any subprocess fails.",
    )
    parser.add_argument(
        "--expect-conda-env",
        default="lingua",
        help="If set, print a warning when CONDA_DEFAULT_ENV does not match (non-fatal). Empty string disables.",
    )
    args = parser.parse_args()
    args.alpha_t = 1.0
    if args.monitor_gpu_id is None:
        args.monitor_gpu_id = args.medgemma_gpu_id
    if (
        not args.quiet
        and args.expect_conda_env
        and os.environ.get("CONDA_DEFAULT_ENV") != args.expect_conda_env
    ):
        print(
            f"Warning: CONDA_DEFAULT_ENV={os.environ.get('CONDA_DEFAULT_ENV')!r}; "
            f"expected {args.expect_conda_env!r}. Prefer: conda run -n {args.expect_conda_env} python ...",
            file=sys.stderr,
            flush=True,
        )

    if args.method == "all":
        run_all(args)
    else:
        run_method(args)


if __name__ == "__main__":
    main()
