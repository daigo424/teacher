from pydantic import BaseModel, Field

from packages.core.schemas import RetrievedChunk


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    context_count: int = Field(default=0)
    sources: list[RetrievedChunk] = Field(default=[])
