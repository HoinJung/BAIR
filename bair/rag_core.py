from __future__ import annotations

import os
import random
import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import faiss
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch


@dataclass
class RetrievedPassage:
    page_id: int
    title: str
    extract: str
    professions: List[str]
    score: float


def load_siglip(model_name: str = "google/siglip-base-patch16-224"):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return processor, model


def _normalize(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype("float32")
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def embed_texts(processor, model, texts: Sequence[str]) -> np.ndarray:
    inputs = processor(text=list(texts), padding=True, truncation=True, return_tensors="pt")
    with np.errstate(all="ignore"):
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
    return _normalize(feats.detach().cpu().numpy())


def embed_image(processor, model, image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with np.errstate(all="ignore"):
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
    return _normalize(feats.detach().cpu().numpy())


def fetch_pages_for_index(db_path: str) -> Tuple[np.ndarray, List[str]]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, title, extract
            FROM pages
            WHERE length(trim(extract)) >= 40
            ORDER BY id
            """
        )
        rows = cur.fetchall()
        ids = np.array([int(r[0]) for r in rows], dtype=np.int64)
        texts = [f"{r[1]}. {r[2]}" for r in rows]
        return ids, texts
    finally:
        conn.close()


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def persist_index(index: faiss.Index, ids: np.ndarray, index_path: str, ids_path: str, model_name: str | None = None) -> None:
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    os.makedirs(os.path.dirname(ids_path), exist_ok=True)
    faiss.write_index(index, index_path)
    np.save(ids_path, ids)
    # Write simple meta for rebuild decisions
    if model_name:
        meta_path = index_path + ".meta.json"
        try:
            import json
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"model": model_name}, f)
        except Exception:
            pass


def load_index(index_path: str, ids_path: str) -> Tuple[faiss.Index, np.ndarray]:
    return faiss.read_index(index_path), np.load(ids_path)

def index_model_mismatch(index_path: str, expected_model: str) -> bool:
    meta_path = index_path + ".meta.json"
    if not os.path.exists(meta_path):
        return True
    try:
        import json
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("model") != expected_model
    except Exception:
        return True


def fetch_professions_for_pages(db_path: str, page_ids: Sequence[int]) -> Dict[int, List[str]]:
    if not page_ids:
        return {}
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        q = ",".join(["?"] * len(page_ids))
        cur.execute(
            f"""
            SELECT p.id, pf.name
            FROM pages p
            JOIN page_professions pp ON pp.page_id = p.id
            JOIN professions pf ON pf.id = pp.profession_id
            WHERE p.id IN ({q})
            """,
            list(page_ids),
        )
        out: Dict[int, List[str]] = {}
        for pid, name in cur.fetchall():
            out.setdefault(int(pid), []).append(str(name))
        return out
    finally:
        conn.close()


def fetch_page_meta(db_path: str, page_ids: Sequence[int]) -> Dict[int, Tuple[str, str]]:
    if not page_ids:
        return {}
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        q = ",".join(["?"] * len(page_ids))
        cur.execute(
            f"SELECT id, title, extract FROM pages WHERE id IN ({q})",
            list(page_ids),
        )
        return {int(r[0]): (str(r[1]), str(r[2])) for r in cur.fetchall()}
    finally:
        conn.close()


def retrieve(db_path: str, index: faiss.Index, mapped_ids: np.ndarray, image_emb: np.ndarray, top_k: int = 5) -> List[RetrievedPassage]:
    scores, idxs = index.search(image_emb, top_k)
    idxs = idxs[0].tolist()
    scrs = scores[0].tolist()
    hit_ids = [int(mapped_ids[i]) for i in idxs]
    title_extract = fetch_page_meta(db_path, hit_ids)
    page_profs = fetch_professions_for_pages(db_path, hit_ids)
    results: List[RetrievedPassage] = []
    for pid, sc in zip(hit_ids, scrs):
        title, extract = title_extract.get(pid, ("", ""))
        profs = page_profs.get(pid, [])
        results.append(RetrievedPassage(page_id=pid, title=title, extract=extract, professions=sorted(set(profs)), score=float(sc)))
    return results


def synthesize_answer(question: str, passages: Sequence[RetrievedPassage]) -> str:
    lines: List[str] = [question]
    used = set()
    for p in passages:
        for sent in p.extract.split(". "):
            sent = sent.strip()
            if not sent or sent in used:
                continue
            if len(lines) >= 6:
                break
            used.add(sent)
            lines.append(f"- {sent}.")
        if len(lines) >= 6:
            break
    return "\n".join(lines)


def compute_recall_at_k(db_path: str, index: faiss.Index, mapped_ids: np.ndarray, processor, model, image_path: str, gt_profession: str, k: int = 5) -> Tuple[float, List[RetrievedPassage]]:
    emb = embed_image(processor, model, image_path)
    results = retrieve(db_path, index, mapped_ids, emb, top_k=k)
    gt = gt_profession.strip().lower()
    hit = any(gt in [p.strip().lower() for p in r.professions] for r in results)
    return (1.0 if hit else 0.0), results


