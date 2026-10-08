# Comprendre les LLM

Projet de découverte des grands modèles de langage, en deux volets :

- **des notebooks** : architecture Transformer codée en NumPy, Hugging Face, dimensionnement RAM
  (local vs API) et un cas data science avec mise en cache des calculs longs ;
- **une application web** : un front qui dialogue avec un back Python servant GPT-2.

## Architecture

```
                        ┌──────────────────────── docker compose ─────────────────────────┐
  navigateur ──:8080──► │  nginx ── /      ──► fichiers du front (frontend/src)          │
                        │        └─ /api/  ──► backend:8000  (FastAPI + GPT-2)           │
                        └─────────────────────────────────────────────────────────────────┘
```

```
llm-demo/
├── backend/                      API Python (FastAPI)
│   ├── app/
│   │   ├── main.py               point d'entrée : chargement du modèle, routes, middleware
│   │   ├── config.py             configuration lue dans les variables d'environnement
│   │   ├── middleware/timing.py  mesure le temps de traitement de chaque requête
│   │   └── modules/
│   │       ├── health/           GET /api/health
│   │       └── llm/              routes.py → service.py (logique) + schemas.py (validation)
│   ├── tests/test_api.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/src/                 interface web (HTML/CSS/JS, sans framework)
│   ├── index.html, styles.css, main.js
│   ├── core/                     api.service.js (seul point de contact avec le back), journal.js
│   └── features/                 tokens.js, next-word.js, generate.js
├── nginx/nginx.conf              sert le front, redirige /api vers le back
├── notebooks/                    comprendre_les_llm.py (cellules # %%) et .ipynb
├── docker-compose.yml
├── Makefile
└── requirements.txt              back + notebooks
```

## Installation

```bash
git clone https://github.com/uxornova/comprendre_les_llm.git
cd comprendre_les_llm
make install          # crée .venv et installe tout (PyTorch version CPU)
cp .env.example .env  # optionnel
```

## Lancer l'application

| Commande | Ce qu'elle fait | Adresse |
|---|---|---|
| `make dev` | Back en mode rechargement automatique ; il sert aussi le front | http://localhost:8000 |
| `make up` / `make down` | Tout avec Docker (nginx + back) | http://localhost:8080 |
| `make test` | Tests de l'API (pytest) | |

La documentation interactive de l'API (Swagger) est sur `/docs`.
Dans la page, le panneau **« Journal des échanges »** affiche chaque requête envoyée par le front et la
réponse JSON du back, avec le temps total et le temps passé côté back.

| Route | Rôle |
|---|---|
| `GET /api/health` | Vérifie que le back répond (modèle, RAM utilisée) |
| `POST /api/llm/tokens` | Découpe un texte en tokens |
| `POST /api/llm/next-word` | Probabilités des mots suivants les plus probables |
| `POST /api/llm/generate` | Génère la suite du texte |

## Les notebooks

Dans VS Code, ouvrir `notebooks/comprendre_les_llm.py` et cliquer sur **Run Cell** au-dessus d'un bloc
`# %%` (ou `Shift+Entrée`). Le `.ipynb` a le même contenu, avec les résultats déjà affichés.
Interpréteur : `Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv`.

Les fonctions décorées par `@cache_resultat` enregistrent leur résultat dans `notebooks/cache/` :
au lancement suivant, il est relu en moins d'une seconde. Pour tout recalculer :
`FORCE_RECALCUL = True` dans la cellule de paramètres.

Optionnel (partie API) : définir `OPENAI_API_KEY` avant de lancer VS Code.
