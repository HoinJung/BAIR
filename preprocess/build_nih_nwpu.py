#!/usr/bin/env python3
"""
Build NWPU retrieval-style dataset for BAIR experiments.

For each NWPU test image:
- Ground-truth class document is fixed at Document 3.
- The other four documents are chosen by RemoteCLIP similarity.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    import open_clip
except Exception:
    open_clip = None

from transformers import CLIPModel, CLIPProcessor

try:
    from llmlingua import PromptCompressor
except Exception:
    PromptCompressor = None

from bair.bottleneck_intervention import NWPURAGInterventionManager


def load_remoteclip(model_name: str, checkpoint: str | None, device: str):
    if open_clip is not None:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained="openai",
        )
        if checkpoint:
            state = torch.load(checkpoint, map_location=device)
            msg = model.load_state_dict(state, strict=False)
            print(f"Loaded RemoteCLIP checkpoint: {checkpoint} | {msg}")
        model = model.to(device).eval()
        tokenizer = open_clip.tokenize
        return {"backend": "open_clip", "model": model, "preprocess": preprocess, "tokenizer": tokenizer}

    # Fallback path when open_clip is not installed.
    print("[Info] open_clip not available. Falling back to transformers CLIPModel.")
    hf_model = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(hf_model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(hf_model)
    return {"backend": "hf_clip", "model": model, "processor": processor}


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def encode_texts(bundle: Dict[str, Any], texts: List[str], device: str, batch_size: int) -> torch.Tensor:
    model = bundle["model"]
    all_feats = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in tqdm(
            range(0, len(texts), batch_size),
            total=total_batches,
            desc="Encoding class documents",
            unit="batch",
        ):
            chunk = texts[i : i + batch_size]
            if bundle["backend"] == "open_clip":
                toks = bundle["tokenizer"](chunk).to(device)
                feats = model.encode_text(toks)
            else:
                proc = bundle["processor"](text=chunk, return_tensors="pt", padding=True, truncation=True).to(device)
                feats = model.get_text_features(**proc)
            all_feats.append(normalize(feats).cpu())
    return torch.cat(all_feats, dim=0)


def encode_image(bundle: Dict[str, Any], image_path: Path, device: str) -> torch.Tensor:
    model = bundle["model"]
    with torch.no_grad():
        if bundle["backend"] == "open_clip":
            img = bundle["preprocess"](Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            feat = model.encode_image(img)
        else:
            proc = bundle["processor"](images=Image.open(image_path).convert("RGB"), return_tensors="pt").to(device)
            feat = model.get_image_features(**proc)
        feat = normalize(feat)
    return feat.cpu()


def make_doc(label: str, desc: str) -> str:
    return f"Class: {label.replace('_', ' ')}\nDescription: {desc.strip()}"


def parse_docs(context: str) -> List[str]:
    parts = re.split(r"--- Document \d+ ---", context or "")
    return [p.strip() for p in parts if p.strip()]


def stable_hash(value: str) -> str:
    return hashlib.sha1((value or "").encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NWPU retrieval dataset with RemoteCLIP.")
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--nwpu-test-root", type=str, default=str(repo_root / "data" / "raw" / "NWPU" / "test" / "test"))
    parser.add_argument("--schema-json", type=str, default=str(repo_root / "data" / "metadata" / "nwpu_label_schema.json"))
    parser.add_argument("--database-fixed-json", type=str, default=str(repo_root / "data" / "metadata" / "nwpu_database_fixed.json"))
    parser.add_argument("--output-json", type=str, default=str(repo_root / "data" / "generated" / "nwpu_retrieval_dataset.json"))
    parser.add_argument("--model-name", type=str, default="ViT-B-32")
    parser.add_argument("--remoteclip-path", type=str, default=None)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--precompute-longllmlingua", action="store_true")
    parser.add_argument(
        "--llmlingua-question",
        type=str,
        default="You are an expert in remote sensing and geospatial analysis. Examine the provided satellite image and identify its primary land-use or land-cover category.",
    )
    parser.add_argument(
        "--llmlingua-instruction",
        type=str,
        default="Use the image as primary evidence and use retrieved context as supporting information.",
    )
    parser.add_argument("--llmlingua-compression-ratio", type=float, default=0.5)
    parser.add_argument("--llmlingua-device-id", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    print(f"[Stage] Loading schema/database and building corpus (seed={args.seed})")

    schema = json.loads(Path(args.schema_json).read_text(encoding="utf-8"))
    db = json.loads(Path(args.database_fixed_json).read_text(encoding="utf-8"))
    db_map = {d["class_key"]: d for d in db}
    raw_to_final: Dict[str, str] = schema["raw_to_final"]
    final_labels: List[str] = schema["final_labels"]

    # Build text corpus from final labels.
    corpus_labels: List[str] = []
    corpus_texts: List[str] = []
    for fl in final_labels:
        item = db_map.get(fl)
        if not item:
            continue
        corpus_labels.append(fl)
        corpus_texts.append(make_doc(fl, item["description"]))
    if len(corpus_texts) < 5:
        raise RuntimeError("Need at least 5 class documents to build 5-document contexts.")

    bundle = load_remoteclip(args.model_name, args.remoteclip_path, device)
    print(f"[Stage] Encoding {len(corpus_texts)} class documents for retrieval scoring")
    text_embs = encode_texts(bundle, corpus_texts, device=device, batch_size=args.batch_size)

    compressed_corpus_texts: List[str] = list(corpus_texts)
    if args.precompute_longllmlingua:
        if PromptCompressor is None:
            raise RuntimeError("llmlingua is required for --precompute-longllmlingua")
        llm_device = f"cuda:{args.llmlingua_device_id}" if torch.cuda.is_available() else "cpu"
        print(f"[Stage] Precomputing LongLLMLingua class documents on {llm_device} ({len(corpus_texts)} docs)")
        compressor = PromptCompressor(
            model_name="NousResearch/Llama-2-7b-hf",
            model_config={"torch_dtype": torch.bfloat16},
            device_map=llm_device,
        )
        manager = NWPURAGInterventionManager()
        compressed_corpus_texts = []
        for doc in tqdm(corpus_texts, desc="Compressing class documents"):
            compressed_doc = manager.compress_longllmlingua(
                compressor=compressor,
                context_docs=[doc],
                instruction=args.llmlingua_instruction,
                question=args.llmlingua_question,
                rate=args.llmlingua_compression_ratio,
            )
            compressed_corpus_texts.append(compressed_doc)

    test_root = Path(args.nwpu_test_root)
    image_paths = sorted([p for p in test_root.rglob("*.jpg") if p.is_file()])
    image_paths = [p for p in image_paths if (raw_to_final.get(p.parent.name.strip().lower()) or "").strip()]
    print(f"[Stage] Building retrieval rows from {len(image_paths)} NWPU images")

    records: List[Dict] = []
    for img_path in tqdm(image_paths, desc="Building NWPU retrieval rows"):
        raw_label = img_path.parent.name.strip().lower()
        final_label = (raw_to_final.get(raw_label) or "").strip()
        if not final_label:
            continue  # excluded class
        if final_label not in corpus_labels:
            continue

        gt_idx = corpus_labels.index(final_label)
        gt_doc = corpus_texts[gt_idx]

        img_emb = encode_image(bundle, img_path, device=device)
        sims = torch.matmul(img_emb, text_embs.T).squeeze(0)  # [num_docs]
        ranked = torch.argsort(sims, descending=True).tolist()

        distractor_indices = [idx for idx in ranked if idx != gt_idx][:4]
        if len(distractor_indices) < 4:
            fallback = [i for i in range(len(corpus_labels)) if i not in distractor_indices and i != gt_idx]
            rng.shuffle(fallback)
            distractor_indices.extend(fallback[: 4 - len(distractor_indices)])

        doc_indices = [
            distractor_indices[0],
            distractor_indices[1],
            gt_idx,
            distractor_indices[2],
            distractor_indices[3],
        ]
        docs_original = [corpus_texts[i] for i in doc_indices]
        docs_for_context = [compressed_corpus_texts[i] for i in doc_indices]
        context_original = "\n\n".join([f"--- Document {i + 1} ---\n{d}" for i, d in enumerate(docs_original)])
        context = "\n\n".join([f"--- Document {i + 1} ---\n{d}" for i, d in enumerate(docs_for_context)])
        rel_path = str(img_path.relative_to(test_root))
        uid = rel_path.replace("/", "__")

        row = {
            "uid": uid,
            "image_path": str(img_path),
            "image_relpath": rel_path,
            "raw_label": raw_label,
            "ground_truth_label": final_label,
            "ground_truth_document": gt_doc,
            "nwpu_context": context,
            # Compatibility key used by older BAIR scripts.
            "nih_context": context,
        }
        if args.precompute_longllmlingua:
            row["nwpu_context_original"] = context_original
            row["nwpu_context_longllmlingua"] = context
            row["nwpu_longllmlingua_meta"] = {
                "question": args.llmlingua_question,
                "instruction": args.llmlingua_instruction,
                "compression_ratio": args.llmlingua_compression_ratio,
                "context_mode": "full",
                "source": "class_document_precompression",
                "class_doc_count": len(corpus_texts),
                "class_doc_hash": stable_hash("\n".join(corpus_texts)),
            }
        records.append(row)
        if args.max_samples > 0 and len(records) >= args.max_samples:
            break

    Path(args.output_json).write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} rows -> {args.output_json}")


if __name__ == "__main__":
    main()
