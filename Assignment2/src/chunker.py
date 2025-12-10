# chunker.py

import json
from typing import List, Dict
import os

def make_chunks(data: List[Dict]) -> List[Dict]:
    """
    Converts SO entries into chunk dicts:
    {
        "chunk_id",
        "type": "question" | "answer",
        "text",
        "tags"
    }
    """
    chunks = []
    print("[CHUNK] Building chunks...")

    for item in data:

        qtext = (item["question_title"] + "\n" + item["question_body"]).strip()

        # Question chunk
        chunks.append({
            "chunk_id": f"q-{item['question_id']}",
            "type": "question",
            "text": qtext,
            "tags": item.get("tags", [])
        })

        # Answer chunks
        for ans in item["answers"]:
            chunks.append({
                "chunk_id": f"a-{ans['answer_id']}",
                "type": "answer",
                "text": ans["text"],
                "score": ans["score"],
                "tags": item.get("tags", [])
            })

    print(f"[CHUNK] Created {len(chunks)} total chunks.\n")
    return chunks



def save_chunks(chunks, path="faiss_index/meta.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

if __name__ == "__main__":
    from load_data import load_stackoverflow

    data = load_stackoverflow(max_docs=5000)
    chunks = make_chunks(data)
    save_chunks(chunks)
