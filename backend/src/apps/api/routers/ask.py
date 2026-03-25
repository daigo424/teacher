from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from apps.api.schemas import AskRequest, AskResponse
from packages.core.db.session import get_db
from packages.core.services.generation import generate_answer
from packages.core.services.query_router import classify_question
from packages.core.services.retrieval import search_chunks

router = APIRouter(tags=["contact"])


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, db: Session = Depends(get_db)) -> AskResponse:
    if classify_question(req.question) == "META":
        return AskResponse(answer="This system can answer questions about AI and machine learning.")
    else:
        chunks = search_chunks(db, req.question)
        answer = generate_answer(req.question, chunks)
        return AskResponse(answer=answer, context_count=len(chunks), sources=chunks)
