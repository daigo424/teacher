from types import SimpleNamespace

from packages.core.schemas import RetrievedChunk
from packages.core.services import generation


def test_generate_answer_builds_prompt_and_parses_json(monkeypatch):
    captured = {}
    chunks = [
        RetrievedChunk(
            document_title="Doc",
            source_path="/tmp/doc.md",
            chunk_index=2,
            score=0.7,
            content="Important context",
        )
    ]

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":"It works"}'))])

    monkeypatch.setattr(generation.settings, "chat_model", "chat-x")
    monkeypatch.setattr(generation.client.chat.completions, "create", fake_create)

    assert generation.generate_answer("How?", chunks) == "It works"
    assert captured["model"] == "chat-x"
    assert "Important context" in captured["messages"][1]["content"]
    assert "Question:\nHow?" in captured["messages"][1]["content"]


def test_generate_answer_falls_back_when_model_returns_empty_content(monkeypatch):
    monkeypatch.setattr(
        generation.client.chat.completions,
        "create",
        lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))]),
    )

    assert generation.generate_answer("How?", []) == "I don't know"
