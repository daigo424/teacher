from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    document_title: str
    source_path: str
    chunk_index: int
    score: float
    content: str
