# embedder.py

import torch
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        print(f"[EMBED] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def encode_docs(self, texts, batch_size=32):
        """
        Embeddings for documents/chunks.
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

    def encode_query(self, text):
        """
        Query encoding (same as docs but separate for clarity).
        """
        return self.model.encode([text], convert_to_numpy=True)


if __name__ == "__main__":
    model = Embedder()
    emb = model.encode_query("How do I fix a numpy broadcasting error?")
    print(emb.shape)
