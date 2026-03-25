from types import SimpleNamespace

from packages.core.services import embeddings


def test_embed_texts_uses_embedding_model(monkeypatch):
    captured = {}

    def fake_create(model, input):
        captured["model"] = model
        captured["input"] = input
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1]), SimpleNamespace(embedding=[0.2])])

    monkeypatch.setattr(embeddings.settings, "embedding_model", "embedding-x")
    monkeypatch.setattr(embeddings.client.embeddings, "create", fake_create)

    assert embeddings.embed_texts(["a", "b"]) == [[0.1], [0.2]]
    assert captured == {"model": "embedding-x", "input": ["a", "b"]}


def test_embed_query_returns_first_embedding(monkeypatch):
    monkeypatch.setattr(
        embeddings.client.embeddings,
        "create",
        lambda model, input: SimpleNamespace(data=[SimpleNamespace(embedding=[0.3, 0.4])]),
    )
    monkeypatch.setattr(embeddings.settings, "embedding_model", "embedding-y")

    assert embeddings.embed_query("query") == [0.3, 0.4]
