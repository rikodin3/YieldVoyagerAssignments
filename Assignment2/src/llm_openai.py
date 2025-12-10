# llm_openai.py

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class OpenAILLM:
    def __init__(self, model="gpt-4o-mini"):
        key = os.getenv("OPENAI_API_KEY")
        if key is None:
            raise ValueError("OPENAI_API_KEY is missing")
        self.client = OpenAI(api_key=key)
        self.model = model

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content
