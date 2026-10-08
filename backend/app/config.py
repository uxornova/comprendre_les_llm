"""Configuration lue dans les variables d'environnement (fichier .env), comme config/env.ts dans Matcha."""
import getpass
import os
from dataclasses import dataclass
from pathlib import Path

# Sur les postes 42 : modèles Hugging Face dans /goinfre (le /home est trop petit)
if os.path.isdir("/goinfre"):
    os.environ.setdefault("HF_HOME", f"/goinfre/{getpass.getuser()}/hf")


@dataclass(frozen=True)
class Settings:
    model_name: str = os.getenv("MODEL_NAME", "gpt2")
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "150"))
    # En développement, le back sert aussi le front (en production, c'est nginx)
    frontend_dir: Path | None = Path(os.environ["FRONTEND_DIR"]) if os.getenv("FRONTEND_DIR") else None


settings = Settings()
