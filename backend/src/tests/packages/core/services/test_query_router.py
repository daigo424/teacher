from types import SimpleNamespace

from packages.core.services import query_router


def test_classify_question_returns_trimmed_output(monkeypatch):
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_text=" META \n")

    monkeypatch.setattr(query_router.settings, "chat_model", "chat-router")
    monkeypatch.setattr(query_router.client.responses, "create", fake_create)

    assert query_router.classify_question("何ができますか") == "META"
    assert captured["model"] == "chat-router"
    assert "Question: 何ができますか" in captured["input"]
