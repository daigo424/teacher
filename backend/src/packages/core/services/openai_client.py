from openai import OpenAI

from packages.core.config import settings

client = OpenAI(api_key=settings.openai_api_key)
