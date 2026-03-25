from packages.core.config.settings import Settings


def test_settings_uses_defaults_when_optional_values_are_missing():
    settings = Settings(
        env="test",
        app_db_host="db",
        app_db_name="app_test",
        app_db_username="app_user",
        app_db_password="app_pass",
        app_db_port="5432",
        openai_api_key="test-key",
        _env_file=None,
    )

    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.chat_model == "gpt-4o-mini"
    assert settings.top_k == 3
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
