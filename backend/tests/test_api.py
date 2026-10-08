"""Tests de l'API : lancer avec  make test"""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # le "with" déclenche le chargement du modèle
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["parameters_millions"] == 124


def test_tokens(client):
    r = client.post("/api/llm/tokens", json={"text": "Le chat mange la souris"})
    assert r.status_code == 200
    body = r.json()
    assert "".join(body["tokens"]) == "Le chat mange la souris"
    assert body["token_count"] == len(body["ids"])


def test_next_word_probabilities_are_sorted(client):
    r = client.post("/api/llm/next-word", json={"text": "The cat sat on the", "k": 5})
    probs = [c["probability"] for c in r.json()["candidates"]]
    assert len(probs) == 5
    assert probs == sorted(probs, reverse=True)


def test_generate(client):
    r = client.post("/api/llm/generate", json={"text": "Hello", "max_new_tokens": 5})
    assert r.status_code == 200
    assert r.json()["prompt"] == "Hello"


def test_empty_text_is_rejected(client):
    assert client.post("/api/llm/tokens", json={"text": ""}).status_code == 422


def test_timing_header(client):
    assert "x-process-time-ms" in client.get("/api/health").headers
