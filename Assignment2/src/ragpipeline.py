# ragpipeline.py

from retriever import HybridRetriever
from heatmap import save_heatmap

class RAGPipeline:
    def __init__(self, top_k=5):
        self.retriever = HybridRetriever()
        self.top_k = top_k

    def retrieve(self, query):
        return self.retriever.retrieve(query, top_k=self.top_k)

    def build_prompt(self, question, chunks):
        context = "\n\n".join(c["text"] for c in chunks)
        prompt = f"""
Answer the question and use the given context too
CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()
        return prompt

    def answer(self, question, llm):
        chunks = self.retrieve(question)
        scores = self.retriever.score_details(question, candidates=20)

        dense_labels = [str(i) for i in scores["dense_idx"]]
        bm25_labels = [str(i) for i in scores["bm25_idx"]]

        save_heatmap(scores["dense_scores"], dense_labels, "static/dense.png")
        save_heatmap(scores["bm25_scores"], bm25_labels, "static/bm25.png")

        prompt = self.build_prompt(question, chunks)
        response = llm.generate(prompt)
        llm_answer = llm.generate(question)

        return llm_answer, response, chunks, ["static/dense.png", "static/bm25.png"]