#!/usr/bin/env python3
"""
Bias profile analysis for BAIR gtonly baselines.

Purpose:
- Measure where oracle/no-retrieval answers align inside the report context
  (primacy/recency behavior).
- Compare three similarity metrics:
  1) ROUGE-L recall
  2) Generic BERTScore recall (roberta-large)
  3) RadBERT token-level recall

Outputs:
- A 3x2 grid figure (metrics x models) with mean profile curves
  (No Retrieval vs Oracle).
- A JSON summary with peak positions and sample counts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
from scipy.interpolate import interp1d
from transformers import AutoModel, AutoTokenizer


@dataclass
class ModelConfig:
    name: str
    baseline_json: Path


def ensure_nltk_punkt():
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def interpolate_profile(values: List[float], bins: int) -> Optional[np.ndarray]:
    if values is None or len(values) < 2:
        return None
    x_current = np.linspace(0, 1, len(values))
    x_target = np.linspace(0, 1, bins)
    fn = interp1d(x_current, values, kind="linear", fill_value="extrapolate")
    return fn(x_target)


class RadBERTRecallScorer:
    def __init__(self, model_name: str = "StanfordAIMI/RadBERT", device: str = "cpu"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def score(self, refs: List[str], cands: List[str]) -> List[float]:
        out = []
        with torch.no_grad():
            for ref, cand in zip(refs, cands):
                if not ref.strip() or not cand.strip():
                    out.append(0.0)
                    continue
                ref_in = self.tokenizer(ref, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                cand_in = self.tokenizer(cand, return_tensors="pt", truncation=True, max_length=512).to(self.device)

                ref_emb = self.model(**ref_in).last_hidden_state
                cand_emb = self.model(**cand_in).last_hidden_state

                ref_emb = F.normalize(ref_emb, p=2, dim=2)
                cand_emb = F.normalize(cand_emb, p=2, dim=2)

                sim = torch.bmm(ref_emb, cand_emb.transpose(1, 2))
                max_per_ref_token, _ = torch.max(sim, dim=2)
                out.append(torch.mean(max_per_ref_token).item())
        return out


def build_rouge_profile(context: str, target_text: str, bins: int, scorer) -> Optional[np.ndarray]:
    if not context.strip() or not target_text.strip():
        return None
    sents = sent_tokenize(context)
    if len(sents) < 2:
        return None
    vals = [scorer.score(target=s, prediction=target_text)["rougeL"].recall for s in sents]
    return interpolate_profile(vals, bins)


def build_bert_profile(context: str, target_text: str, bins: int, scorer) -> Optional[np.ndarray]:
    if not context.strip() or not target_text.strip():
        return None
    sents = sent_tokenize(context)
    if len(sents) < 2:
        return None
    # Recall of target (ref) covered by each sentence (cand)
    refs = [target_text] * len(sents)
    cands = sents
    _, recalls, _ = scorer.score(cands, refs)
    vals = recalls.cpu().numpy().tolist()
    return interpolate_profile(vals, bins)


def build_radbert_profile(context: str, target_text: str, bins: int, scorer: RadBERTRecallScorer) -> Optional[np.ndarray]:
    if not context.strip() or not target_text.strip():
        return None
    sents = sent_tokenize(context)
    if len(sents) < 2:
        return None
    # Match old rogue_analysis orientation:
    # ref=sentence, cand=target_text -> recall of sentence covered by target.
    refs = sents
    cands = [target_text] * len(sents)
    vals = scorer.score(refs, cands)
    return interpolate_profile(vals, bins)


def context_text(entry: dict) -> str:
    # Prefer exact GT report for alignment with report-level analysis.
    return str(entry.get("gt_report") or entry.get("ground_truth_report") or "")


def mean_and_se(stacked: np.ndarray):
    mean = np.mean(stacked, axis=0)
    se = np.std(stacked, axis=0) / np.sqrt(max(1, stacked.shape[0]))
    return mean, se


def peak_position_percent(profile: np.ndarray) -> float:
    idx = int(np.argmax(profile))
    if len(profile) <= 1:
        return 0.0
    return 100.0 * idx / (len(profile) - 1)


def run_analysis(
    models: List[ModelConfig],
    bins: int,
    limit: int,
    output_dir: Path,
    metrics_requested: List[str],
):
    ensure_nltk_punkt()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    metrics: List[tuple[str, Callable[[str, str], Optional[np.ndarray]]]] = []

    if "rouge" in metrics_requested:
        try:
            from rouge_score import rouge_scorer
        except ImportError as e:
            raise ImportError("Missing dependency 'rouge-score'. Install with: pip install rouge-score") from e
        rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        metrics.append(("ROUGE-L", lambda c, t: build_rouge_profile(c, t, bins, rouge)))

    if "bert" in metrics_requested:
        try:
            from bert_score import BERTScorer
        except ImportError as e:
            raise ImportError("Missing dependency 'bert-score'. Install with: pip install bert-score") from e
        bert = BERTScorer(model_type="roberta-large", lang="en", rescale_with_baseline=True, device=device)
        metrics.append(("BERTScore", lambda c, t: build_bert_profile(c, t, bins, bert)))

    if "radbert" in metrics_requested:
        radbert = RadBERTRecallScorer(device=device)
        metrics.append(("RadBERT", lambda c, t: build_radbert_profile(c, t, bins, radbert)))

    if not metrics:
        raise ValueError("No metrics selected. Use --metrics with one or more of: rouge bert radbert")

    x = np.linspace(0, 100, bins)
    fig, axes = plt.subplots(len(metrics), len(models), figsize=(7 * len(models), 4 * len(metrics)), sharex=True)
    if len(models) == 1:
        axes = np.array(axes).reshape(len(metrics), 1)
    if len(metrics) == 1:
        axes = np.array(axes).reshape(1, len(models))

    summary = {"models": {}}

    for col, model in enumerate(models):
        entries = load_json(model.baseline_json)
        if limit > 0:
            entries = entries[:limit]

        summary["models"][model.name] = {}

        for row, (metric_name, profile_fn) in enumerate(metrics):
            prof_oracle = []
            prof_noret = []

            peak_oracle = []
            peak_noret = []

            for entry in entries:
                ctx = context_text(entry)
                if not ctx.strip():
                    continue

                o_ans = str(entry.get("oracle_answer", ""))
                n_ans = str(entry.get("no_retrieval_answer", ""))

                p = profile_fn(ctx, o_ans)
                if p is not None:
                    prof_oracle.append(p)
                    peak_oracle.append(peak_position_percent(p))

                p = profile_fn(ctx, n_ans)
                if p is not None:
                    prof_noret.append(p)
                    peak_noret.append(peak_position_percent(p))

            ax = axes[row, col]

            if prof_noret:
                arr = np.vstack(prof_noret)
                m, e = mean_and_se(arr)
                ax.plot(x, m, color="gray", linestyle="--", linewidth=2, label="No Retrieval")
                ax.fill_between(x, m - e, m + e, color="gray", alpha=0.12)

            if prof_oracle:
                arr = np.vstack(prof_oracle)
                m, e = mean_and_se(arr)
                ax.plot(x, m, color="red", linewidth=2, label="Oracle")
                ax.fill_between(x, m - e, m + e, color="red", alpha=0.14)

            if row == 0:
                ax.set_title(model.name, fontweight="bold")
            if col == 0:
                ax.set_ylabel(metric_name)
            if row == len(metrics) - 1:
                ax.set_xlabel("Position in report (%)")
            ax.set_ylim(0, 1.0)
            ax.grid(alpha=0.25)
            if row == 0 and col == len(models) - 1:
                ax.legend(loc="upper right")

            summary["models"][model.name][metric_name] = {
                "n_oracle": len(prof_oracle),
                "n_no_retrieval": len(prof_noret),
                "peak_oracle_mean_pct": float(np.mean(peak_oracle)) if peak_oracle else None,
                "peak_no_retrieval_mean_pct": float(np.mean(peak_noret)) if peak_noret else None,
            }

    fig.tight_layout()
    out_png = output_dir / "gtonly_bias_profiles_metrics_grid.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    out_json = output_dir / "gtonly_bias_profiles_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved figure: {out_png}")
    print(f"Saved summary: {out_json}")


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Bias profile analysis for gtonly baselines.")
    parser.add_argument(
        "--medgemma-json",
        type=str,
        default=str(base / "generation_results_medgemma_gtonly" / "iuchest_medgemma_results_baselines_ctx_gt_only.json"),
    )
    parser.add_argument(
        "--chexagent-json",
        type=str,
        default=str(base / "generation_results_chexagent_gtonly" / "iuchest_chexagent_results_baselines_ctx_gt_only.json"),
    )
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit per model (0 = full).")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["rouge", "bert", "radbert"],
        help="Subset of metrics to run: rouge bert radbert",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["medgemma", "chexagent"],
        default=["chexagent"],
        help="Which model baselines to analyze.",
    )
    parser.add_argument("--output-dir", type=str, default=str(base / "gtonly_bias_analysis"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = []
    if "medgemma" in args.models:
        models.append(ModelConfig("MedGemma gtonly", Path(args.medgemma_json)))
    if "chexagent" in args.models:
        models.append(ModelConfig("CheXagent gtonly", Path(args.chexagent_json)))
    if not models:
        raise ValueError("No models selected. Use --models with medgemma and/or chexagent.")
    run_analysis(
        models=models,
        bins=args.bins,
        limit=args.limit,
        output_dir=output_dir,
        metrics_requested=[m.lower() for m in args.metrics],
    )


if __name__ == "__main__":
    main()
