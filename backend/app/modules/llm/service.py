"""Logique métier : tout ce qui touche au modèle de langage."""
import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import settings


class LLMService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.load_seconds = 0.0

    def load(self):
        """Chargé une seule fois au démarrage du serveur : les requêtes ne paient pas ce coût."""
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).eval()
        self.load_seconds = time.perf_counter() - t0

    @property
    def ready(self) -> bool:
        return self.model is not None

    def info(self) -> dict:
        return {
            "model": self.model_name,
            "parameters_millions": round(sum(p.numel() for p in self.model.parameters()) / 1e6),
            "load_seconds": round(self.load_seconds, 1),
            "ram_used_gb": round(psutil.Process().memory_info().rss / 1024**3, 2),
        }

    def tokenize(self, text: str) -> dict:
        ids = self.tokenizer.encode(text)
        return {"tokens": [self.tokenizer.decode([i]) for i in ids], "ids": ids,
                "token_count": len(ids), "char_count": len(text)}

    def next_tokens(self, text: str, k: int) -> list[dict]:
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits[0, -1], dim=-1)
        top = torch.topk(probs, k)
        return [{"token": self.tokenizer.decode([int(i)]), "probability": round(float(p), 4)}
                for p, i in zip(top.values, top.indices)]

    def generate(self, text: str, temperature: float, max_new_tokens: int) -> str:
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                         temperature=temperature, top_p=0.95,
                                         pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


llm_service = LLMService(settings.model_name)
