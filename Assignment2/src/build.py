# build.py

from load_data import load_ai_dataset as load_stackoverflow
from chunker import make_chunks, save_chunks
from build_faiss import build_from_chunks
from bm25 import build_bm25_index, save_bm25_index

def main():
    data = load_stackoverflow(max_docs=5000)
    chunks = make_chunks(data)
    save_chunks(chunks, "faiss_index/meta.json")
    build_from_chunks(chunks, "faiss_index/index.faiss", "faiss_index/meta.json")
    bm25, tok = build_bm25_index(chunks)
    save_bm25_index(tok, "bm25_index/bm25.json")

if __name__ == "__main__":
    main()
