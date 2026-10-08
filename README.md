# Comprendre les LLM

Notebook de démonstration : architecture Transformer (codée en NumPy), Hugging Face,
dimensionnement RAM (local vs API) et un cas data science avec mise en cache des calculs longs.

## Installation

```bash
git clone https://github.com/uxornova/comprendre_les_llm.git
cd comprendre_les_llm
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Deux versions au contenu identique :

- `comprendre_les_llm.py` : script Python découpé en cellules `# %%`. Dans VS Code, cliquer sur
  **Run Cell** au-dessus d'un bloc (ou `Shift+Entrée`) : il s'exécute dans la fenêtre interactive.
- `comprendre_les_llm.ipynb` : le même contenu en notebook Jupyter, avec les résultats déjà affichés.

Choisir l'interpréteur Python : `Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv`.

Optionnel (partie API) : `export OPENAI_API_KEY="sk-..."` avant de lancer VS Code.

## L'application web (front ↔ back)

Une petite application montre comment une interface web discute avec un serveur qui fait tourner le modèle :

```
 navigateur (front)                       serveur Python (back)
 app/frontend/index.html  ── POST JSON ──►  app/backend.py (FastAPI)
  boutons, graphiques      ◄── JSON ─────   GPT-2 chargé une fois au démarrage
```

```bash
uvicorn app.backend:app --reload
```

Puis ouvrir http://localhost:8000. Chaque requête et sa réponse JSON s'affichent dans le panneau
« Journal des échanges ». La documentation interactive de l'API est sur http://localhost:8000/docs.

| Route | Rôle |
|---|---|
| `GET /api/sante` | Vérifie que le back répond (modèle, RAM utilisée) |
| `POST /api/tokens` | Découpe un texte en tokens |
| `POST /api/mot-suivant` | Probabilités des 10 mots suivants les plus probables |
| `POST /api/generer` | Génère la suite du texte |

## Le cache

Les fonctions décorées par `@cache_resultat` enregistrent leur résultat dans `cache/`.
Au lancement suivant, il est relu en moins d'une seconde. Pour tout recalculer :
`FORCE_RECALCUL = True` dans la cellule de paramètres, ou supprimer le dossier `cache/`.
