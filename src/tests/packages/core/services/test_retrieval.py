from types import SimpleNamespace

from packages.core.services import retrieval


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class FakeDB:
    def __init__(self, rows):
        self.rows = rows
        self.statement = None

    def execute(self, statement):
        self.statement = statement
        return FakeResult(self.rows)


def test_search_chunks_uses_default_top_k_and_maps_rows(monkeypatch):
    rows = [
        SimpleNamespace(
            title="Doc 1",
            source_path="/tmp/doc1.md",
            chunk_index=3,
            content="body",
            score=0.88,
        )
    ]
    fake_db = FakeDB(rows)

    monkeypatch.setattr(retrieval, "embed_query", lambda question: [0.1, 0.2])
    monkeypatch.setattr(retrieval.settings, "top_k", 4)

    results = retrieval.search_chunks(fake_db, "question")

    assert len(results) == 1
    assert results[0].document_title == "Doc 1"
    assert fake_db.statement._limit_clause.value == 4


def test_search_chunks_respects_explicit_top_k(monkeypatch):
    fake_db = FakeDB([])

    monkeypatch.setattr(retrieval, "embed_query", lambda question: [0.1, 0.2])
    monkeypatch.setattr(retrieval.settings, "top_k", 4)

    retrieval.search_chunks(fake_db, "question", top_k=2)

    assert fake_db.statement._limit_clause.value == 2
