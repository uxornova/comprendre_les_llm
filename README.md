# Comprendre les LLM

Notebook de démonstration : architecture Transformer (codée en NumPy), Hugging Face,
dimensionnement RAM (local vs API) et un cas data science avec mise en cache des calculs longs.

## Installation

```bash
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

## Le cache

Les fonctions décorées par `@cache_resultat` enregistrent leur résultat dans `cache/`.
Au lancement suivant, il est relu en moins d'une seconde. Pour tout recalculer :
`FORCE_RECALCUL = True` dans la cellule de paramètres, ou supprimer le dossier `cache/`.
