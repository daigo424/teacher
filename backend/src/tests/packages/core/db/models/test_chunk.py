from packages.core.db.models import Chunk


def test_chunk_model_metadata():
    constraint_names = {constraint.name for constraint in Chunk.__table__.constraints}

    assert Chunk.__tablename__ == "chunks"
    assert "uq_chunks_document_id_chunk_index" in constraint_names
    assert Chunk.__table__.c.document_id.index
    assert Chunk.__table__.c.metadata.name == "metadata"
