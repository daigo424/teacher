from packages.core.db.models import Document


def test_document_model_metadata():
    assert Document.__tablename__ == "documents"
    assert Document.__table__.c.source_path.unique
    assert "chunks" in Document.__mapper__.relationships
