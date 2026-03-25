import importlib

import openai

import packages.core.services.openai_client as openai_client_module


def test_openai_client_uses_settings_api_key(monkeypatch):
    created = {}

    class FakeOpenAI:
        def __init__(self, api_key):
            created["api_key"] = api_key
            self.api_key = api_key

    monkeypatch.setattr(openai_client_module.settings, "openai_api_key", "reloaded-key")
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    reloaded = importlib.reload(openai_client_module)

    assert reloaded.client.api_key == "reloaded-key"
    assert created["api_key"] == "reloaded-key"

    importlib.reload(openai_client_module)
