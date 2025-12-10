# bm25.py

import json
import os
import nltk
from rank_bm25 import BM25Okapi
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from typing import List, Dict

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def tokenize_for_bm25(text: str) -> List[str]:
    tokens = nltk.word_tokenize(text.lower())
    out = []
    for t in tokens:
        if not t.isalpha():
            continue
        if t in stop_words:
            continue
        out.append(lemmatizer.lemmatize(t))
    return out

def build_bm25_index(chunks: List[Dict]):
    corpus = [c["text"] for c in chunks]
    tokenized = [tokenize_for_bm25(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized

def save_bm25_index(tokenized_corpus, path="bm25_index/bm25.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"tokenized_corpus": tokenized_corpus}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def load_bm25_index(path="bm25_index/bm25.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tokenized = data["tokenized_corpus"]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized

if __name__ == "__main__":
    chunks = [
        {"text": "How to fix list index out of range error in python?"},
        {"text": "This error occurs when accessing invalid index."}
    ]
    bm25, tok = build_bm25_index(chunks)
    save_bm25_index(tok)
    bm25_loaded, _ = load_bm25_index()
