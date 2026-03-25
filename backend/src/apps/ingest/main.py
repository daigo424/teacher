from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy import select

from packages.core.config import settings
from packages.core.db.models import Chunk, Document
from packages.core.db.session import SessionLocal
from packages.core.services.embeddings import embed_texts
from packages.core.services.text import chunk_text, estimate_tokens, read_text_file, sha256_text


def ingest_file(file_path: str, title: str | None = None) -> None:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(file_path)

    text = read_text_file(file_path)
    checksum = sha256_text(text)
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    embeddings = embed_texts(chunks)

    with SessionLocal() as db:
        existing = db.execute(select(Document).where(Document.source_path == str(path.resolve()))).scalar_one_or_none()

        if existing and existing.checksum == checksum:
            print(f"[SKIP] Unchanged file: {file_path}")
            return

        if existing:
            existing.title = title or path.stem
            existing.checksum = checksum
            existing.chunks.clear()
            document = existing
            print(f"[UPDATE] Re-ingesting changed file: {file_path}")
        else:
            document = Document(
                title=title or path.stem,
                source_path=str(path.resolve()),
                checksum=checksum,
            )
            db.add(document)
            print(f"[INSERT] Ingesting new file: {file_path}")

        db.flush()

        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            db.add(
                Chunk(
                    document_id=document.id,
                    chunk_index=idx,
                    content=chunk,
                    token_count=estimate_tokens(chunk),
                    metadata_json={"source_file": path.name},
                    embedding=embedding,
                )
            )

        db.commit()
        print(f"[DONE] document_id={document.id}, chunks={len(chunks)}")


def ingest_targets(file_path: str):
    if file_path == "all":
        # Retrieve the files in /files/dataset/**/*.md
        base_dir = Path("/files/dataset")
        return list(base_dir.rglob("*.md"))
    else:
        return [Path(file_path)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", nargs="?", default="all", help="Path to a UTF-8 text file or 'all'")
    parser.add_argument("--title", help="Document title", default=None)
    args = parser.parse_args()

    targets = ingest_targets(args.file_path)

    for path in targets:
        ingest_file(str(path), args.title)
