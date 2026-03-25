from packages.core.schemas import RetrievedChunk


def test_retrieved_chunk_schema_accepts_expected_fields():
    chunk = RetrievedChunk(
        document_title="Doc",
        source_path="/tmp/doc.md",
        chunk_index=1,
        score=0.99,
        content="text",
    )

    assert chunk.document_title == "Doc"
    assert chunk.model_dump()["score"] == 0.99
