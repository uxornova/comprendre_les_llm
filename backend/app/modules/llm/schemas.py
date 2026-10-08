"""Formats des requêtes et des réponses : FastAPI valide automatiquement ce qu'envoie le front."""
from pydantic import BaseModel, Field

from app.config import settings


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class NextWordRequest(TextRequest):
    k: int = Field(default=10, ge=1, le=30)


class GenerateRequest(TextRequest):
    temperature: float = Field(default=0.8, gt=0, le=2)
    max_new_tokens: int = Field(default=40, ge=1, le=settings.max_new_tokens)


class TokensResponse(BaseModel):
    tokens: list[str]
    ids: list[int]
    token_count: int
    char_count: int


class Candidate(BaseModel):
    token: str
    probability: float


class NextWordResponse(BaseModel):
    candidates: list[Candidate]


class GenerateResponse(BaseModel):
    prompt: str
    completion: str
