"""Traducao EN->PT-BR para modelos que so legendam em ingles (MarianMT)."""
from __future__ import annotations

from .common import pick_device

# Alternativas: "Helsinki-NLP/opus-mt-en-ROMANCE" (multilingue, precisa do prefixo
# >>por<<) ou um modelo maior tipo "facebook/nllb-200-distilled-600M".
DEFAULT_MT = "Helsinki-NLP/opus-mt-tc-big-en-pt"


class Translator:
    """Wrapper simples. `prefix` fica vazio por padrao; alguns checkpoints do OPUS
    exigem um token de variante (ex: ">>por<<" ou ">>pob<<" para pt-BR) — confira o
    model card do checkpoint que voce escolher."""

    def __init__(self, model_id: str = DEFAULT_MT, device: str | None = None, prefix: str = "") -> None:
        from transformers import MarianMTModel, MarianTokenizer

        self.device = pick_device(device)
        self.prefix = prefix
        self.tokenizer = MarianTokenizer.from_pretrained(model_id)
        self.model = MarianMTModel.from_pretrained(model_id).to(self.device).eval()

    def __call__(self, texts: list[str]) -> list[str]:
        import torch

        entrada = [f"{self.prefix} {t}".strip() if self.prefix else t for t in texts]
        batch = self.tokenizer(entrada, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.inference_mode():
            saida = self.model.generate(**batch, num_beams=4, max_new_tokens=96)
        return [self.tokenizer.decode(s, skip_special_tokens=True) for s in saida]
