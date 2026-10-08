# Environnement Python (sur un poste 42 : make dev VENV=/goinfre/$USER/llm-venv)
VENV    ?= .venv
PY      := $(abspath $(VENV))/bin/python

.PHONY: install dev test up down logs clean

install:            ## Crée l'environnement Python et installe les dépendances
	python3 -m venv $(VENV)
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements.txt

dev:                ## Lance le back (qui sert aussi le front) sur http://localhost:8000
	cd backend && FRONTEND_DIR=../frontend/src $(PY) -m uvicorn app.main:app --reload --port 8000

test:               ## Lance les tests de l'API
	cd backend && $(PY) -m pytest -q

up:                 ## Lance tout avec Docker sur http://localhost:8080
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f backend

clean:
	find . -name __pycache__ -prune -exec rm -rf {} +
	rm -rf backend/.pytest_cache notebooks/cache
