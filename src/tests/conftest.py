import os

os.environ["ENV"] = "test"
os.environ["APP_DB_URL"] = "sqlite+pysqlite:///:memory:"
os.environ["OPENAI_API_KEY"] = "test-api-key"
