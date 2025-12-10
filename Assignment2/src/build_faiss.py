# build_faiss.py

import os
import json
import faiss
import numpy as np
from typing import List, Dict
from embedder import Embedder


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize embeddings for cosine similarity (FAISS IP mode)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-10)


def build_faiss_index(embeddings: np.ndarray):
    """
    Build a FAISS index using cosine similarity via Inner Product (IP)
    after normalizing vectors.
    """

    embeddings = embeddings.astype("float32")
    embeddings = normalize(embeddings)

    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)

    print(f"[FAISS] Built FlatIP index with {index.ntotal} vectors.")
    return index


def save_faiss_index(index, metadata: List[Dict],
                     index_path="faiss_index/index.faiss",
                     meta_path="faiss_index/meta.json"):
    """Save FAISS index + metadata."""
    
    os.makedirs("faiss_index", exist_ok=True)

    faiss.write_index(index, index_path)
    print(f"[FAISS] Index saved to {index_path}")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[FAISS] Metadata saved to {meta_path}")


def load_faiss_index(index_path="faiss_index/index.faiss",
                     meta_path="faiss_index/meta.json"):
    """Load FAISS index + metadata."""
    
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("[FAISS] Index + metadata loaded.")
    return index, metadata


def build_from_chunks(chunks: List[Dict],
                      index_path="faiss_index/index.faiss",
                      meta_path="faiss_index/meta.json",
                      batch_size=256):
    """
    Build FAISS index directly from chunk list.
    """
    print("[FAISS] Encoding embeddings for all chunks...")
    embedder = Embedder()

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode_docs(texts, batch_size=batch_size)

    index = build_faiss_index(embeddings)
    save_faiss_index(index, chunks, index_path, meta_path)

    print("[FAISS] Build complete.\n")
    return index


if __name__ == "__main__":
    # Demo
    dummy_chunks = [{"chunk_id": i, "text": f"hello world {i}"} for i in range(5)]
    build_from_chunks(dummy_chunks)
