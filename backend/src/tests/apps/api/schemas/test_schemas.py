import pytest
from pydantic import ValidationError

from apps.api.schemas import AskRequest, AskResponse


def test_ask_request_requires_non_empty_question():
    with pytest.raises(ValidationError):
        AskRequest(question="")


def test_ask_response_defaults_are_applied():
    first = AskResponse(answer="ok")
    second = AskResponse(answer="still ok")

    first.sources.append(
        {
            "document_title": "Doc",
            "source_path": "/tmp/doc.md",
            "chunk_index": 0,
            "score": 0.5,
            "content": "body",
        }
    )

    assert first.context_count == 0
    assert second.sources == []
