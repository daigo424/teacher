from packages.core.config.settings import Settings


def test_settings_uses_defaults_when_optional_values_are_missing():
    settings = Settings(
        app_db_url="sqlite+pysqlite:///:memory:",
        openai_api_key="test-key",
        _env_file=None,
    )

    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.chat_model == "gpt-4o-mini"
    assert settings.top_k == 3
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
