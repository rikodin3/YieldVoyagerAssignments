# app.py

from flask import Flask, render_template, request
from ragpipeline import RAGPipeline
from llm_openai import OpenAILLM

app = Flask(__name__)
pipeline = RAGPipeline(top_k=5)
llm = OpenAILLM()

@app.route("/", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def index():
    llm_answer = ""
    rag_answer = ""
    chunks = []
    heatmaps = []

    if request.method == "POST":
        query = request.form["query"]
        llm_answer, rag_answer, chunks, heatmaps = pipeline.answer(query, llm)

    return render_template(
        "index.html",
        llm_answer=llm_answer,
        rag_answer=rag_answer,
        chunks=chunks,
        heatmaps=heatmaps
    )
if __name__ == "__main__":
    app.run(debug=True)
