# load_data.py

import json

def load_ai_dataset(path="../data/ai_dataset.json", min_ans_score=0, max_docs=None):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out = []
    count = 0

    for q in raw:
        if max_docs and count >= max_docs:
            break

        good = [a for a in q["answers"] if a["score"] >= min_ans_score]
        if not good:
            continue

        out.append({
            "question_id": q["id"],
            "question_title": q["title"] or "",
            "question_body": q["body"] or "",
            "answers": [{"answer_id": a["id"], "text": a["body"], "score": a["score"]} for a in good],
            "tags": []
        })

        count += 1

    return out

if __name__ == "__main__":
    print(load_ai_dataset()[:1])
