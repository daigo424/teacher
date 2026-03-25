from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    env: str
    app_db_host: str
    app_db_name: str
    app_db_username: str
    app_db_password: str
    app_db_port: str
    openai_api_key: str
    debug: bool = False
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    top_k: int = 3
    chunk_size: int = 500
    chunk_overlap: int = 50

    @property
    def app_db_url(self) -> str:
        return f"postgresql://{self.app_db_username}:{self.app_db_password}@{self.app_db_host}:{self.app_db_port}/{self.app_db_name}"


settings = Settings()  # type: ignore[call-arg]
