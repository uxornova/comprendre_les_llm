"""Back-end : une API web qui expose GPT-2.

Lancement (depuis le dossier llm-demo) :
    uvicorn app.backend:app --reload
puis ouvrir http://localhost:8000 dans le navigateur.

Le modèle est chargé UNE fois au démarrage du serveur (comme le cache du notebook :
on ne paie le coût qu'une fois), puis chaque requête du front ne fait que l'utiliser.
"""
import getpass
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Sur les postes 42 : modèles Hugging Face dans /goinfre (le /home est trop petit)
if os.path.isdir("/goinfre"):
    os.environ.setdefault("HF_HOME", f"/goinfre/{getpass.getuser()}/hf")

import psutil
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

NOM_MODELE = "gpt2"
DOSSIER_FRONT = Path(__file__).parent / "frontend"
modele = {}   # rempli au démarrage


@asynccontextmanager
async def demarrage(app: FastAPI):
    t0 = time.perf_counter()
    modele["tokenizer"] = AutoTokenizer.from_pretrained(NOM_MODELE)
    modele["llm"] = AutoModelForCausalLM.from_pretrained(NOM_MODELE).eval()
    modele["duree_chargement"] = time.perf_counter() - t0
    print(f"✅ {NOM_MODELE} chargé en {modele['duree_chargement']:.1f} s")
    yield
    modele.clear()


app = FastAPI(title="Comprendre les LLM — API", lifespan=demarrage)


# ── Formats des requêtes envoyées par le front ──────────────────────────────
class Texte(BaseModel):
    texte: str = Field(min_length=1, max_length=2000)


class DemandeMotSuivant(Texte):
    k: int = Field(default=10, ge=1, le=30)


class DemandeGeneration(Texte):
    temperature: float = Field(default=0.8, gt=0, le=2)
    nb_tokens: int = Field(default=40, ge=1, le=150)


# ── Routes de l'API ─────────────────────────────────────────────────────────
@app.get("/api/sante")
def sante():
    """Le front appelle cette route au chargement pour vérifier que le back répond."""
    llm = modele["llm"]
    return {
        "statut": "ok",
        "modele": NOM_MODELE,
        "parametres_millions": round(sum(p.numel() for p in llm.parameters()) / 1e6),
        "chargement_s": round(modele["duree_chargement"], 1),
        "ram_utilisee_go": round(psutil.Process().memory_info().rss / 1024**3, 2),
    }


@app.post("/api/tokens")
def tokens(demande: Texte):
    """Découpe le texte en tokens, comme le tokenizer de la partie 2 du notebook."""
    tok = modele["tokenizer"]
    ids = tok.encode(demande.texte)
    return {"tokens": [tok.decode([i]) for i in ids], "ids": ids,
            "nb_tokens": len(ids), "nb_caracteres": len(demande.texte)}


@app.post("/api/mot-suivant")
def mot_suivant(demande: DemandeMotSuivant):
    """Probabilités des k tokens les plus probables après le texte."""
    tok, llm = modele["tokenizer"], modele["llm"]
    entrees = tok(demande.texte, return_tensors="pt")
    with torch.no_grad():
        probas = torch.softmax(llm(**entrees).logits[0, -1], dim=-1)
    top = torch.topk(probas, demande.k)
    return {"candidats": [{"token": tok.decode([int(i)]), "probabilite": round(float(p), 4)}
                          for p, i in zip(top.values, top.indices)]}


@app.post("/api/generer")
def generer(demande: DemandeGeneration):
    """Génère la suite du texte, token par token."""
    tok, llm = modele["tokenizer"], modele["llm"]
    entrees = tok(demande.texte, return_tensors="pt")
    with torch.no_grad():
        sortie = llm.generate(**entrees, max_new_tokens=demande.nb_tokens, do_sample=True,
                              temperature=demande.temperature, top_p=0.95,
                              pad_token_id=tok.eos_token_id)
    suite = tok.decode(sortie[0, entrees["input_ids"].shape[1]:], skip_special_tokens=True)
    return {"texte_initial": demande.texte, "suite": suite}


# ── Le back sert aussi la page du front ─────────────────────────────────────
@app.get("/")
def page_accueil():
    return FileResponse(DOSSIER_FRONT / "index.html")
