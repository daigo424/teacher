from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.ingest import main as ingest_main


class FakeResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class FakeSession:
    def __init__(self, existing=None):
        self.existing = existing
        self.added = []
        self.flush_called = False
        self.commit_called = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, stmt):
        return FakeResult(self.existing)

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        self.flush_called = True
        for obj in self.added:
            if getattr(obj, "id", None) is None and hasattr(obj, "source_path"):
                obj.id = 101

    def commit(self):
        self.commit_called = True


def test_ingest_targets_returns_all_markdown_files(monkeypatch):
    expected = [Path("/files/dataset/a.md"), Path("/files/dataset/nested/b.md")]

    class FakeBaseDir:
        def __init__(self, path):
            self.path = path

        def rglob(self, pattern):
            assert self.path == "/files/dataset"
            assert pattern == "*.md"
            return expected

    monkeypatch.setattr(ingest_main, "Path", FakeBaseDir)

    assert ingest_main.ingest_targets("all") == expected


def test_ingest_targets_wraps_single_path():
    target = ingest_main.ingest_targets("notes/sample.md")

    assert target == [Path("notes/sample.md")]


def test_ingest_file_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest_main.ingest_file(str(tmp_path / "missing.md"))


def test_ingest_file_skips_when_checksum_is_unchanged(tmp_path, monkeypatch, capsys):
    file_path = tmp_path / "doc.md"
    file_path.write_text("sample", encoding="utf-8")
    existing = SimpleNamespace(checksum="same-checksum")
    fake_session = FakeSession(existing=existing)

    monkeypatch.setattr(ingest_main, "SessionLocal", lambda: fake_session)
    monkeypatch.setattr(ingest_main, "read_text_file", lambda path: "sample")
    monkeypatch.setattr(ingest_main, "sha256_text", lambda text: "same-checksum")
    monkeypatch.setattr(ingest_main, "chunk_text", lambda text, size, overlap: ["chunk"])
    monkeypatch.setattr(ingest_main, "embed_texts", lambda chunks: [[0.1, 0.2]])
    monkeypatch.setattr(ingest_main.settings, "chunk_size", 100)
    monkeypatch.setattr(ingest_main.settings, "chunk_overlap", 10)

    ingest_main.ingest_file(str(file_path))

    assert fake_session.added == []
    assert not fake_session.commit_called
    assert "[SKIP] Unchanged file" in capsys.readouterr().out


def test_ingest_file_updates_existing_document(tmp_path, monkeypatch, capsys):
    file_path = tmp_path / "doc.md"
    file_path.write_text("sample", encoding="utf-8")
    existing = SimpleNamespace(
        id=7,
        checksum="old",
        title="Old",
        chunks=["old chunk"],
    )
    fake_session = FakeSession(existing=existing)

    monkeypatch.setattr(ingest_main, "SessionLocal", lambda: fake_session)
    monkeypatch.setattr(ingest_main, "read_text_file", lambda path: "sample")
    monkeypatch.setattr(ingest_main, "sha256_text", lambda text: "new")
    monkeypatch.setattr(ingest_main, "chunk_text", lambda text, size, overlap: ["a", "b"])
    monkeypatch.setattr(ingest_main, "embed_texts", lambda chunks: [[0.1], [0.2]])
    monkeypatch.setattr(ingest_main, "estimate_tokens", lambda text: len(text))
    monkeypatch.setattr(ingest_main.settings, "chunk_size", 100)
    monkeypatch.setattr(ingest_main.settings, "chunk_overlap", 10)

    ingest_main.ingest_file(str(file_path), "New Title")

    assert existing.title == "New Title"
    assert existing.checksum == "new"
    assert existing.chunks == []
    assert fake_session.flush_called
    assert fake_session.commit_called
    chunk_records = [obj for obj in fake_session.added if hasattr(obj, "chunk_index")]
    assert len(chunk_records) == 2
    assert "[UPDATE] Re-ingesting changed file" in capsys.readouterr().out


def test_ingest_file_inserts_new_document(tmp_path, monkeypatch, capsys):
    file_path = tmp_path / "doc.md"
    file_path.write_text("sample", encoding="utf-8")
    fake_session = FakeSession(existing=None)

    monkeypatch.setattr(ingest_main, "SessionLocal", lambda: fake_session)
    monkeypatch.setattr(ingest_main, "read_text_file", lambda path: "sample")
    monkeypatch.setattr(ingest_main, "sha256_text", lambda text: "new")
    monkeypatch.setattr(ingest_main, "chunk_text", lambda text, size, overlap: ["only chunk"])
    monkeypatch.setattr(ingest_main, "embed_texts", lambda chunks: [[0.1]])
    monkeypatch.setattr(ingest_main, "estimate_tokens", lambda text: 3)
    monkeypatch.setattr(ingest_main.settings, "chunk_size", 100)
    monkeypatch.setattr(ingest_main.settings, "chunk_overlap", 10)

    ingest_main.ingest_file(str(file_path))

    document_records = [obj for obj in fake_session.added if hasattr(obj, "source_path")]
    chunk_records = [obj for obj in fake_session.added if hasattr(obj, "chunk_index")]
    assert len(document_records) == 1
    assert len(chunk_records) == 1
    assert document_records[0].title == "doc"
    assert chunk_records[0].document_id == 101
    assert "[INSERT] Ingesting new file" in capsys.readouterr().out
