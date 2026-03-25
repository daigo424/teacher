from __future__ import annotations

from packages.core.config import settings
from packages.core.services.openai_client import client


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in response.data]


def embed_query(text: str) -> list[float]:
    response = client.embeddings.create(model=settings.embedding_model, input=text)
    return response.data[0].embedding
