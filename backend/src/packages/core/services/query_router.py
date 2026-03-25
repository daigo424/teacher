from __future__ import annotations

from packages.core.config import settings
from packages.core.services.openai_client import client


def classify_question(question: str) -> str:
    prompt = f"""
Classify the following question into one of the categories:

- META: asking about the system itself, capabilities, scope
- CONTENT: asking about knowledge contained in documents

Question: {question}

Answer only one word: META or CONTENT
"""

    response = client.responses.create(model=settings.chat_model, input=prompt, temperature=0)
    return response.output_text.strip()
