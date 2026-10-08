"""Routes HTTP du module LLM : elles reçoivent la requête, appellent le service, renvoient du JSON."""
from fastapi import APIRouter

from app.modules.llm.schemas import (GenerateRequest, GenerateResponse, NextWordRequest,
                                     NextWordResponse, TextRequest, TokensResponse)
from app.modules.llm.service import llm_service

router = APIRouter(prefix="/api/llm", tags=["llm"])


@router.post("/tokens", response_model=TokensResponse)
def tokens(request: TextRequest):
    """Découpe le texte en tokens."""
    return llm_service.tokenize(request.text)


@router.post("/next-word", response_model=NextWordResponse)
def next_word(request: NextWordRequest):
    """Probabilités des k tokens les plus probables après le texte."""
    return {"candidates": llm_service.next_tokens(request.text, request.k)}


@router.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    """Génère la suite du texte, token par token."""
    completion = llm_service.generate(request.text, request.temperature, request.max_new_tokens)
    return {"prompt": request.text, "completion": completion}
