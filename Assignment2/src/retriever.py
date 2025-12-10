# retriever.py

import json
import numpy as np
import faiss
import nltk
from rank_bm25 import BM25Okapi
from embedder import Embedder
from sentence_transformers import CrossEncoder

nltk.download("punkt", quiet=True)

class HybridRetriever:
    def __init__(
        self,
        faiss_index_path="faiss_index/index.faiss",
        metadata_path="faiss_index/meta.json",
        bm25_path="bm25_index/bm25.json",
        alpha=0.6,
        beta=0.4,
        reranker_model="cross-encoder/ms-marco-MiniLM-L6-v2"
    ):
        self.index = faiss.read_index(faiss_index_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        with open(bm25_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.tokenized_corpus = data["tokenized_corpus"]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self.alpha = alpha
        self.beta = beta

        self.embedder = Embedder()
        self.reranker = CrossEncoder(reranker_model)

    def _normalize(self, x):
        x = np.array(x)
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    def dense_search(self, query_emb, top_k=50):
        D, I = self.index.search(query_emb, top_k)
        return I[0], D[0]

    def bm25_search(self, query, top_k=50):
        tokens = nltk.word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokens)
        idx = np.argsort(scores)[::-1][:top_k]
        return idx, [scores[i] for i in idx]
    
    def score_details(self, query, candidates=30):
        q_emb = self.embedder.encode_query(query)
        d_idx, d_scores = self.dense_search(q_emb, top_k=candidates)
        b_idx, b_scores = self.bm25_search(query, top_k=candidates)

        d_norm = self._normalize(d_scores)
        b_norm = self._normalize(b_scores)

        return {
            "dense_idx": d_idx.tolist(),
            "dense_scores": d_norm.tolist(),
            "bm25_idx": b_idx.tolist(),
            "bm25_scores": b_norm.tolist()
        }


    def rerank(self, query, candidates):
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked]

    def retrieve(self, query, top_k=5, candidates=50):
        q_emb = self.embedder.encode_query(query)

        d_idx, d_scores = self.dense_search(q_emb, top_k=candidates)
        b_idx, b_scores = self.bm25_search(query, top_k=candidates)

        d_scores = self._normalize(d_scores)
        b_scores = self._normalize(b_scores)

        hybrid = {}

        for idx, s in zip(d_idx, d_scores):
            hybrid[idx] = hybrid.get(idx, 0) + self.alpha * s

        for idx, s in zip(b_idx, b_scores):
            hybrid[idx] = hybrid.get(idx, 0) + self.beta * s

        ranked = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)[:candidates]

        cands = [{"chunk_id": idx, "score": float(score), "text": self.metadata[idx]["text"]} 
                 for idx, score in ranked]

        reranked = self.rerank(query, cands)
        return reranked[:top_k]
