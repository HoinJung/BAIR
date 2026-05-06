#!/usr/bin/env python3
"""
Coincidence analysis for BAIR gtonly baselines (single-threshold version).

What this does:
- Computes report-position profiles using ROUGE-L / BERTScore / RadBERT.
- Runs grid search over ONE threshold (no high/low gap).
- Tests N-segment partition sizes N in {2, 4, 5}.
- Uses SRR-BERT-Upper full-mode original-label correctness, consistent with
  the Sankey setup.

Strict segment rule (single threshold):
- "Seg-k-Only": segment k max >= th AND all other segment max < th
- "Multiple": more than one segment max >= th
- "None": no segment max >= th
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
from scipy.interpolate import interp1d
from transformers import AutoModel, AutoTokenizer
from transformers import BertForSequenceClassification, BertTokenizer


@dataclass
class ModelInput:
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


def parse_labels(s, id_to_string) -> FrozenSet[str]:
    if not s or (isinstance(s, float) and pd.isna(s)):
        return frozenset()
    out = set()
    for lab in str(s).split(";"):
        lab = lab.strip().lower()
        if not lab:
            continue
        if lab.startswith("label_"):
            try:
                idx = int(lab.split("_")[1])
                if idx in id_to_string:
                    out.add(id_to_string[idx])
                    continue
            except ValueError:
                pass
        out.add(lab)
    return frozenset(out)


def compute_f1(pred: FrozenSet[str], gt: FrozenSet[str]) -> float:
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    tp = len(pred & gt)
    p = tp / len(pred)
    r = tp / len(gt)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def compute_accuracy(pred: FrozenSet[str], gt: FrozenSet[str]) -> float:
    return 1.0 if pred == gt else 0.0


def segment_maxes(profile: np.ndarray, n_segments: int) -> List[float]:
    seg_size = len(profile) // n_segments
    out = []
    for i in range(n_segments):
        s = i * seg_size
        e = (i + 1) * seg_size if i < n_segments - 1 else len(profile)
        vals = profile[s:e]
        out.append(float(np.max(vals)) if len(vals) else 0.0)
    return out


def classify_single_threshold(profile: np.ndarray, n_segments: int, threshold: float) -> str:
    maxes = segment_maxes(profile, n_segments)
    hot = [i for i, v in enumerate(maxes) if v >= threshold]
    if len(hot) == 0:
        return "None"
    if len(hot) > 1:
        return "Multiple"
    return f"Seg-{hot[0] + 1}-Only"


class RadBERTRecallScorer:
    def __init__(self, model_name: str = "StanfordAIMI/RadBERT", device: str = "cpu"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def score(self, refs: List[str], cands: List[str]) -> List[float]:
        vals = []
        with torch.no_grad():
            for ref, cand in zip(refs, cands):
                if not ref.strip() or not cand.strip():
                    vals.append(0.0)
                    continue
                ref_in = self.tokenizer(ref, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                cand_in = self.tokenizer(cand, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                ref_emb = F.normalize(self.model(**ref_in).last_hidden_state, p=2, dim=2)
                cand_emb = F.normalize(self.model(**cand_in).last_hidden_state, p=2, dim=2)
                sim = torch.bmm(ref_emb, cand_emb.transpose(1, 2))
                best, _ = torch.max(sim, dim=2)
                vals.append(torch.mean(best).item())
        return vals


def interpolate(values: List[float], bins: int) -> Optional[np.ndarray]:
    if len(values) < 2:
        return None
    x_cur = np.linspace(0, 1, len(values))
    x_tgt = np.linspace(0, 1, bins)
    fn = interp1d(x_cur, values, kind="linear", fill_value="extrapolate")
    return fn(x_tgt)


def profile_rouge(context: str, target_text: str, bins: int, scorer) -> Optional[np.ndarray]:
    sents = sent_tokenize(context)
    if len(sents) < 2:
        return None
    vals = [scorer.score(target=s, prediction=target_text)["rougeL"].recall for s in sents]
    return interpolate(vals, bins)


def profile_bert(context: str, target_text: str, bins: int, scorer) -> Optional[np.ndarray]:
    sents = sent_tokenize(context)
    if len(sents) < 2:
        return None
    refs = [target_text] * len(sents)
    cands = sents
    _, r, _ = scorer.score(cands, refs)
    return interpolate(r.cpu().numpy().tolist(), bins)


def profile_radbert(context: str, target_text: str, bins: int, scorer: RadBERTRecallScorer) -> Optional[np.ndarray]:
    sents = sent_tokenize(context)
    if len(sents) < 2:
        return None
    # Match old rogue_analysis orientation:
    # ref=sentence, cand=target_text -> recall of sentence covered by target.
    refs = sents
    cands = [target_text] * len(sents)
    return interpolate(scorer.score(refs, cands), bins)


class SRREvaluator:
    REPO_LABEL_TO_ID = {
        "Pleural Effusion": 0,
        "Upper abdominal finding": 1,
        "Widened cardiac silhouette": 2,
        "Lung Finding": 3,
        "No Finding": 4,
        "Widened aortic contour": 5,
        "Pleural Thickening": 6,
        "Vascular finding": 7,
        "Consolidation": 8,
        "Pneumothorax": 9,
        "Subdiaphragmatic gas": 10,
        "Masslike opacity": 11,
        "Chest wall finding": 12,
        "Focal air space opacity": 13,
        "Segmental collapse": 14,
        "Fracture": 15,
        "Mediastinal mass": 16,
        "Solitary masslike opacity": 17,
        "Support Devices": 18,
        "Mediastinal finding": 19,
        "Pleural finding": 20,
        "Air space opacity": 21,
        "Diffuse air space opacity": 22,
        "Multiple masslike opacities": 23,
        "Musculoskeletal finding": 24,
    }

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache = {}
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r") as f:
                    self.cache = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: cache file is malformed, starting fresh: {self.cache_path}")
                self.cache = {}

        self.model_name = "StanfordAIMI/SRR-BERT-Upper"
        self.tokenizer_name = "microsoft/BiomedVLP-CXR-BERT-general"
        self.max_len = 128
        self.no_findings = "no finding"
        self.id_to_string = {v: k.lower() for k, v in self.REPO_LABEL_TO_ID.items()}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.new_preds = 0

    def _lazy_load(self):
        if self.model is None:
            self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
            self.model = BertForSequenceClassification.from_pretrained(self.model_name).to(self.device).eval()

    def _hash(self, text: str):
        return hashlib.md5(f"full_{text}".encode("utf-8")).hexdigest()

    def _predict_once(self, text: str) -> List[str]:
        self._lazy_load()
        inp = self.tokenizer(
            str(text).strip(),
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inp).logits
            pred = (torch.sigmoid(logits)[0].cpu().numpy() > 0.5).astype(int)
        labels = [self.id_to_string[i] for i, f in enumerate(pred) if f and i in self.id_to_string]
        return labels if labels else [self.no_findings]

    def predict_set(self, text: str) -> FrozenSet[str]:
        if not text or not str(text).strip():
            return frozenset()
        text = str(text).strip()
        h = self._hash(text)
        if h in self.cache:
            return parse_labels(";".join(self.cache[h]), self.id_to_string)
        labels = self._predict_once(text)
        self.cache[h] = list(labels)
        self.new_preds += 1
        s = set(labels)
        if len(s) > 1 and self.no_findings in s:
            s.discard(self.no_findings)
        return frozenset(s)

    def save_cache(self):
        if self.new_preds > 0:
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f)


def load_gt_sets(gt_csv: Path, id_to_string) -> Dict[str, FrozenSet[str]]:
    df = pd.read_csv(gt_csv)
    df["uid"] = df["uid"].astype(str).str.strip().str.replace(".0", "", regex=False)
    if "pseudo_labels_full" in df.columns:
        return df.set_index("uid")["pseudo_labels_full"].map(lambda s: parse_labels(s, id_to_string)).to_dict()
    if "pseudo_labels" in df.columns:
        return df.set_index("uid")["pseudo_labels"].map(lambda s: parse_labels(s, id_to_string)).to_dict()
    raise ValueError(f"Unsupported GT csv schema: {gt_csv}")


def run(
    models: List[ModelInput],
    gt_csv: Path,
    thresholds: List[float],
    n_values: List[int],
    bins: int,
    score_metric: str,
    correct_threshold: float,
    output_dir: Path,
    cache_file: Path,
    limit: int,
    metrics_requested: List[str],
    min_last_count: int,
):
    ensure_nltk_punkt()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    metrics = {}
    if "rouge" in metrics_requested:
        try:
            from rouge_score import rouge_scorer
        except ImportError as e:
            raise ImportError("Missing dependency 'rouge-score'. Install with: pip install rouge-score") from e
        rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        metrics["ROUGE-L"] = lambda c, t: profile_rouge(c, t, bins, rouge)
    if "bert" in metrics_requested:
        try:
            from bert_score import BERTScorer
        except ImportError as e:
            raise ImportError("Missing dependency 'bert-score'. Install with: pip install bert-score") from e
        bert = BERTScorer(model_type="roberta-large", lang="en", rescale_with_baseline=True, device=device)
        metrics["BERTScore"] = lambda c, t: profile_bert(c, t, bins, bert)
    if "radbert" in metrics_requested:
        radbert = RadBERTRecallScorer(device=device)
        metrics["RadBERT"] = lambda c, t: profile_radbert(c, t, bins, radbert)
    if not metrics:
        raise ValueError("No metrics selected. Use --metrics with one or more of: rouge bert radbert")

    srr = SRREvaluator(cache_file)
    gt_sets = load_gt_sets(gt_csv, srr.id_to_string)
    score_fn = compute_f1 if score_metric == "f1" else compute_accuracy

    rows = []

    for model in models:
        entries = load_json(model.baseline_json)
        if limit > 0:
            entries = entries[:limit]

        prepared = []
        for e in entries:
            uid = str(e.get("uid", "")).strip().replace(".0", "")
            if not uid or uid not in gt_sets:
                continue
            gt = gt_sets[uid]
            if len(gt) == 0:
                continue
            ctx = str(e.get("gt_report") or e.get("ground_truth_report") or "")
            anchor = str(e.get("gt_problems") or e.get("ground_truth_problems") or "")
            ans = str(e.get("oracle_answer", ""))
            if not ctx.strip() or not anchor.strip() or anchor.strip().lower() == "normal" or not ans.strip():
                continue

            pred = srr.predict_set(ans)
            score = score_fn(pred, gt)
            is_correct = score >= correct_threshold

            metric_profiles = {}
            for m_name, fn in metrics.items():
                p = fn(ctx, anchor)
                if p is not None:
                    metric_profiles[m_name] = p
            if not metric_profiles:
                continue
            prepared.append({"uid": uid, "is_correct": is_correct, "profiles": metric_profiles})

        # Grid search with cached profiles (fast loop)
        for metric_name in metrics.keys():
            subset = [x for x in prepared if metric_name in x["profiles"]]
            if not subset:
                continue
            for n_seg in n_values:
                segment_labels = [f"Seg-{i + 1}-Only" for i in range(n_seg)] + ["Multiple", "None"]
                for th in thresholds:
                    bucket = {k: [] for k in segment_labels}
                    for item in subset:
                        cat = classify_single_threshold(item["profiles"][metric_name], n_seg, th)
                        bucket[cat].append(1 if item["is_correct"] else 0)

                    row = {
                        "model": model.name,
                        "metric": metric_name,
                        "n_segments": n_seg,
                        "threshold": th,
                        "n_total": len(subset),
                    }
                    for label in segment_labels:
                        vals = bucket[label]
                        row[f"n_{label}"] = len(vals)
                        row[f"acc_{label}"] = float(np.mean(vals)) if vals else np.nan
                    if n_seg >= 2:
                        row["delta_last_minus_first"] = (
                            row.get(f"acc_Seg-{n_seg}-Only", np.nan) - row.get("acc_Seg-1-Only", np.nan)
                        )
                    else:
                        row["delta_last_minus_first"] = np.nan
                    rows.append(row)

    srr.save_cache()

    df = pd.DataFrame(rows)
    out_csv = output_dir / "coincidence_single_threshold_gridsearch.csv"
    df.to_csv(out_csv, index=False)

    # Legacy best threshold summary per model/metric/N by max weighted segment accuracy.
    best_rows = []
    for (m, metric_name, n_seg), g in df.groupby(["model", "metric", "n_segments"], dropna=False):
        if g.empty:
            continue
        # Weighted mean over all single segments + multiple (exclude none).
        score_series = []
        for _, r in g.iterrows():
            num = 0.0
            den = 0.0
            for k in [f"Seg-{i + 1}-Only" for i in range(int(n_seg))] + ["Multiple"]:
                n_k = r.get(f"n_{k}", 0)
                a_k = r.get(f"acc_{k}", np.nan)
                if pd.notna(a_k) and n_k > 0:
                    num += n_k * a_k
                    den += n_k
            score_series.append((num / den) if den > 0 else np.nan)
        g = g.copy()
        g["weighted_acc"] = score_series
        g = g.sort_values("weighted_acc", ascending=False)
        best_rows.append(g.iloc[0].to_dict())

    best_df = pd.DataFrame(best_rows)
    out_best = output_dir / "coincidence_single_threshold_best_configs.csv"
    best_df.to_csv(out_best, index=False)

    # New best config focused on "last segment highest and plausible accuracy".
    # Plausible: enough support in last segment (n_last >= min_last_count).
    # Highest: last segment accuracy >= every other single-segment accuracy.
    focused_rows = []
    for (m, metric_name, n_seg), g in df.groupby(["model", "metric", "n_segments"], dropna=False):
        if g.empty:
            continue
        g = g.copy()
        last_label = f"Seg-{int(n_seg)}-Only"
        last_acc_col = f"acc_{last_label}"
        last_n_col = f"n_{last_label}"
        if last_acc_col not in g.columns or last_n_col not in g.columns:
            continue

        def _is_last_highest(row):
            last_acc = row.get(last_acc_col, np.nan)
            if pd.isna(last_acc):
                return False
            for i in range(1, int(n_seg)):
                other_acc = row.get(f"acc_Seg-{i}-Only", np.nan)
                if pd.notna(other_acc) and other_acc > last_acc:
                    return False
            return True

        g["last_is_highest"] = g.apply(_is_last_highest, axis=1)
        g["last_acc"] = g[last_acc_col]
        g["last_count"] = g[last_n_col].fillna(0).astype(int)
        g["first_acc"] = g.get("acc_Seg-1-Only", np.nan)
        g["delta_last_minus_first"] = g["last_acc"] - g["first_acc"]
        g["is_plausible"] = g["last_count"] >= int(min_last_count)

        # Primary target: plausible + last highest.
        cand = g[(g["is_plausible"]) & (g["last_is_highest"]) & (g["last_acc"].notna())]
        # Fallback 1: last highest (ignore count).
        if cand.empty:
            cand = g[(g["last_is_highest"]) & (g["last_acc"].notna())]
        # Fallback 2: any valid last acc.
        if cand.empty:
            cand = g[g["last_acc"].notna()]
        if cand.empty:
            continue

        cand = cand.sort_values(
            by=["last_acc", "last_count", "delta_last_minus_first"],
            ascending=[False, False, False],
        )
        focused_rows.append(cand.iloc[0].to_dict())

    focused_df = pd.DataFrame(focused_rows)
    out_focused = output_dir / "coincidence_last_segment_best_configs.csv"
    focused_df.to_csv(out_focused, index=False)

    # Visualization: bars over segments for chosen focused configs.
    if not focused_df.empty:
        n_rows = len(focused_df)
        fig_h = max(2, 1.4 * n_rows)
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, fig_h), squeeze=False)
        for r, (_, row) in enumerate(focused_df.iterrows()):
            ax = axes[r, 0]
            n_seg = int(row["n_segments"])
            seg_names = [f"Seg-{i}" for i in range(1, n_seg + 1)]
            acc_vals = [row.get(f"acc_Seg-{i}-Only", np.nan) for i in range(1, n_seg + 1)]
            counts = [int(row.get(f"n_Seg-{i}-Only", 0) or 0) for i in range(1, n_seg + 1)]
            x = np.arange(n_seg)
            colors = ["#4C78A8"] * n_seg
            colors[-1] = "#E45756"  # highlight last segment
            bars = ax.bar(x, acc_vals, color=colors, alpha=0.9)
            for i, b in enumerate(bars):
                y = b.get_height()
                if pd.notna(y):
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        y + 0.015,
                        f"{y:.3f}\n(n={counts[i]})",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
            ax.set_xticks(x, seg_names)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Accuracy")
            ax.grid(axis="y", alpha=0.25)
            ax.set_title(
                f"{row['model']} | {row['metric']} | N={n_seg}, th={row['threshold']:.2f} "
                f"| Last={row.get('acc_Seg-' + str(n_seg) + '-Only', np.nan):.3f}",
                fontsize=10,
            )
        fig.tight_layout()
        out_plot = output_dir / "coincidence_last_segment_best_configs_bar.png"
        fig.savefig(out_plot, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    else:
        out_plot = output_dir / "coincidence_last_segment_best_configs_bar.png"

    print(f"Saved grid search: {out_csv}")
    print(f"Saved best configs: {out_best}")
    print(f"Saved focused best configs: {out_focused}")
    if focused_df.empty:
        print("No focused rows for bar plot (file not generated).")
    else:
        print(f"Saved focused bar plot: {out_plot}")

    # Requested dedicated visualization:
    # RadBERT + threshold=0.4, separated plots for each model with larger fonts.
    rad = df[(df["metric"] == "RadBERT") & (np.isclose(df["threshold"], 0.4))]
    if not rad.empty:
        for model_name, g_model in rad.groupby("model"):
            g_model = g_model.sort_values("n_segments")
            fig, axes = plt.subplots(1, len(g_model), figsize=(7 * max(1, len(g_model)), 3), squeeze=False)
            for i, (_, row) in enumerate(g_model.iterrows()):
                ax = axes[0, i]
                n_seg = int(row["n_segments"])
                seg_names = [f"Seg-{k}" for k in range(1, n_seg + 1)]
                acc_vals = [row.get(f"acc_Seg-{k}-Only", np.nan) for k in range(1, n_seg + 1)]
                counts = [int(row.get(f"n_Seg-{k}-Only", 0) or 0) for k in range(1, n_seg + 1)]
                x = np.arange(n_seg)
                colors = ["#4C78A8"] * n_seg
                colors[-1] = "#E45756"
                bars = ax.bar(x, acc_vals, color=colors, alpha=0.92, edgecolor="black", linewidth=0.8)
                for k, b in enumerate(bars):
                    y = b.get_height()
                    if pd.notna(y):
                        ax.text(
                            b.get_x() + b.get_width() / 2,
                            y + 0.02,
                            f"{y:.3f}\n(n={counts[k]})",
                            ha="center",
                            va="bottom",
                            fontsize=14,
                            fontweight="bold",
                        )
                ax.set_xticks(x, seg_names, fontsize=14, fontweight="bold")
                ax.set_ylim(0, 1.0)
                ax.tick_params(axis="y", labelsize=13)
                ax.set_ylabel("Accuracy", fontsize=16, fontweight="bold")
                ax.set_title(f"N={n_seg}", fontsize=17, fontweight="bold")
                ax.grid(axis="y", alpha=0.25)
            fig.suptitle(f"{model_name} | RadBERT | threshold=0.4", fontsize=19, fontweight="bold")
            fig.tight_layout()
            safe_name = model_name.lower().replace(" ", "_")
            out_sep = output_dir / f"coincidence_radbert_th0.4_{safe_name}_separated.png"
            fig.savefig(out_sep, dpi=170, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"Saved RadBERT@0.4 separated plot: {out_sep}")


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Single-threshold coincidence analysis for gtonly.")
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
    parser.add_argument(
        "--gt-csv",
        type=str,
        default=str(base / "indiana_reports_with_pseudo_labels_dual.csv"),
    )
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--score-metric", type=str, choices=["f1", "accuracy"], default="f1")
    parser.add_argument("--correct-threshold", type=float, default=1.0)
    parser.add_argument("--n-values", type=int, nargs="+", default=[5])
    parser.add_argument("--th-start", type=float, default=0.10)
    parser.add_argument("--th-end", type=float, default=0.95)
    parser.add_argument("--th-step", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0, help="Optional per-model entry cap; 0 = full.")
    parser.add_argument("--output-dir", type=str, default=str(base / "gtonly_coincidence_analysis"))
    parser.add_argument("--cache-file", type=str, default=str(base / "srr_bert_f1_eval_cache.json"))
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["rouge", "bert", "radbert"],
        help="Subset of metrics to run: rouge bert radbert",
    )
    parser.add_argument(
        "--min-last-count",
        type=int,
        default=20,
        help="Minimum sample count for plausible last-segment best config.",
    )
    args = parser.parse_args()

    thresholds = list(np.round(np.arange(args.th_start, args.th_end + 1e-9, args.th_step), 4))
    models = [
        ModelInput("MedGemma gtonly", Path(args.medgemma_json)),
        ModelInput("CheXagent gtonly", Path(args.chexagent_json)),
    ]
    run(
        models=models,
        gt_csv=Path(args.gt_csv),
        thresholds=thresholds,
        n_values=args.n_values,
        bins=args.bins,
        score_metric=args.score_metric,
        correct_threshold=args.correct_threshold,
        output_dir=Path(args.output_dir),
        cache_file=Path(args.cache_file),
        limit=args.limit,
        metrics_requested=[m.lower() for m in args.metrics],
        min_last_count=args.min_last_count,
    )


if __name__ == "__main__":
    main()
