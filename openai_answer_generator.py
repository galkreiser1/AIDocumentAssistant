from __future__ import annotations

import os

from openai import OpenAI


DEFAULT_MODEL = "gpt-5.4-mini"


class OpenAIAnswerGenerator:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        self.model = model
        self.client = OpenAI(api_key=api_key)

    def generate_answer(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )
        return response.output_text
