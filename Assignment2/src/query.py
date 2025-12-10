# query.py

from ragpipeline import RAGPipeline
from llm_openai import OpenAILLM

def main():
    pipeline = RAGPipeline(top_k=5)
    llm = OpenAILLM()

    while True:
        query = input("Query (or 'exit'): ")
        if query.lower() == "exit":
            break
        answer, chunks = pipeline.answer(query, llm)

        print("\nRetrieved Chunks:\n")
        for c in chunks:
            print(f"{c['chunk_id']}  score={c['score']:.4f}")
            print(c["text"])
            print()

        print("Answer:\n")
        print(answer)
        print()

if __name__ == "__main__":
    main()
