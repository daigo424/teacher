from __future__ import annotations

import json

from packages.core.config import settings
from packages.core.schemas import RetrievedChunk
from packages.core.services.openai_client import client


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(f"[Source: {chunk.document_title} #{chunk.chunk_index}]\n{chunk.content}" for chunk in chunks)

    prompt = f"""
You are answering questions using the provided context.

Rules:
- Use the context as the primary source of truth.
- You may paraphrase, summarize, and combine information from multiple context chunks.
- Do NOT copy sentences unnecessarily.
- If the question is broad or vague, describe the main topic of the context in a natural way.
- If the answer cannot be supported by the context, say "I don't know".
- Keep the answer concise and helpful.

Context:
{context}

Question:
{question}

Return JSON with this shape:
{{
  "answer": "..."
}}
""".strip()

    response = client.chat.completions.create(
        model=settings.chat_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a precise RAG assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or '{"answer": "I don\'t know"}'
    data = json.loads(content)
    answer = data.get("answer")

    if not isinstance(answer, str):
        return "I don't know"

    return answer
