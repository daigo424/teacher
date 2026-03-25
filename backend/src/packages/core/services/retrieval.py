from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from packages.core.config import settings
from packages.core.db.models import Chunk, Document
from packages.core.schemas import RetrievedChunk
from packages.core.services.embeddings import embed_query


def search_chunks(db: Session, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
    query_embedding = embed_query(question)
    k = top_k or settings.top_k

    distance = Chunk.embedding.cosine_distance(query_embedding)
    score = (1 - distance).label("score")

    stmt = (
        select(
            Document.title,
            Document.source_path,
            Chunk.chunk_index,
            Chunk.content,
            score,
        )
        .join(Document, Document.id == Chunk.document_id)
        .order_by(distance)
        .limit(k)
    )

    rows = db.execute(stmt).all()

    return [
        RetrievedChunk(
            document_title=row.title,
            source_path=row.source_path,
            chunk_index=row.chunk_index,
            score=float(row.score),
            content=row.content,
        )
        for row in rows
    ]
