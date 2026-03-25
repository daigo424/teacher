from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import ask as ask_router_module

client = TestClient(app)


def test_ask_returns_meta_answer(monkeypatch):
    monkeypatch.setattr(ask_router_module, "classify_question", lambda question: "META")
    app.dependency_overrides[ask_router_module.get_db] = lambda: object()

    response = client.post("/ask", json={"question": "何ができますか"})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == {
        "answer": "This system can answer questions about AI and machine learning.",
        "context_count": 0,
        "sources": [],
    }


def test_ask_returns_generated_content_answer(monkeypatch):
    captured = {}
    chunks = [
        {
            "document_title": "Doc 1",
            "source_path": "/tmp/doc1.md",
            "chunk_index": 0,
            "score": 0.9,
            "content": "chunk body",
        }
    ]

    monkeypatch.setattr(ask_router_module, "classify_question", lambda question: "CONTENT")
    monkeypatch.setattr(ask_router_module, "search_chunks", lambda db, question: chunks)

    def fake_generate_answer(question, found_chunks):
        captured["question"] = question
        captured["chunks"] = found_chunks
        return "generated answer"

    monkeypatch.setattr(ask_router_module, "generate_answer", fake_generate_answer)
    app.dependency_overrides[ask_router_module.get_db] = lambda: object()

    response = client.post("/ask", json={"question": "Explain transformers"})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert captured == {"question": "Explain transformers", "chunks": chunks}
    assert response.json() == {
        "answer": "generated answer",
        "context_count": 1,
        "sources": chunks,
    }


def test_ask_rejects_empty_question():
    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422
