"""Ferramentas para testar o pipeline SEM dataset, SEM GPU e SEM chave de API.

    python -m src dummy-data     # imagens sinteticas + legendas-modelo em paths.images_dir / labels_jsonl
    python -m src tiny-models    # LLM e encoder minusculos, aleatorios (dev.tiny_dir), sem baixar nada

O perfil configs/perfis/smoke.yaml junta as duas coisas e roda o pipeline inteiro
(treino das 5 variantes, predicao, avaliacao) em poucos minutos na CPU:

    python -m src run --config configs/perfis/smoke.yaml

As metricas desse teste nao significam nada — ele so prova que as pecas encaixam.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

from .common import write_jsonl
from .config import ROOT, Config, resolve_path

CORES = {"vermelho": (200, 40, 40), "azul": (40, 70, 200), "verde": (40, 160, 70), "amarelo": (230, 200, 40)}
NUMEROS = {1: "um", 2: "dois", 3: "tres", 4: "quatro"}
CENAS = ["cais", "patio de conteineres", "terminal"]


def _cena(rng: random.Random, tamanho: int = 256):
    from PIL import Image, ImageDraw

    cor_nome = rng.choice(list(CORES))
    n = rng.randint(1, 4)
    cena = rng.choice(CENAS)
    guindaste = rng.random() < 0.5
    navio = rng.random() < 0.5

    img = Image.new("RGB", (tamanho, tamanho), (70, 130, 180) if cena == "cais" else (150, 150, 150))
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(tamanho * 0.65), tamanho, tamanho], fill=(110, 110, 110))
    if navio:
        d.polygon([(20, 150), (230, 150), (210, 180), (40, 180)], fill=(30, 30, 30))
    for i in range(n):
        x = 20 + i * 55
        d.rectangle([x, 185, x + 45, 215], fill=CORES[cor_nome], outline=(0, 0, 0))
    if guindaste:
        d.rectangle([200, 40, 210, 180], fill=(240, 120, 20))
        d.rectangle([140, 40, 240, 50], fill=(240, 120, 20))

    plural = "conteiner" if n == 1 else "conteineres"
    base = f"{NUMEROS[n]} {plural} {cor_nome}{'s' if n > 1 else ''} no {cena}"
    extras = []
    if navio:
        extras.append("um navio atracado")
    if guindaste:
        extras.append("um guindaste portuario laranja")
    legendas = [
        f"{base}" + (f" com {' e '.join(extras)}" if extras else "") + ".",
        f"no {cena} ha {base.split(' no ')[0]}" + (f" e {extras[0]}" if extras else "") + ".",
        f"cena portuaria com {base}" + (f", {extras[-1]}" if extras else "") + ".",
    ]
    objetos = ["conteiner"] + (["navio"] if navio else []) + (["guindaste"] if guindaste else [])
    return img, legendas, objetos


def make_dummy_data(cfg: Config) -> None:
    rng = random.Random(int(cfg.seed))
    n = int(cfg.get_path("dev.n_images", 30))
    images_dir = resolve_path(cfg, "images_dir")
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(1, n + 1):
        img, legendas, objetos = _cena(rng)
        image_id = f"dummy_{i:04d}"
        img.save(images_dir / f"{image_id}.jpg", "JPEG", quality=90)
        rows.append(
            {
                "image_id": image_id,
                "modelo": "dummy",
                "legendas": legendas,
                "objetos": objetos,
                "fora_de_dominio": False,
                "observacao": "dado sintetico para teste do pipeline",
            }
        )
    write_jsonl(resolve_path(cfg, "labels_jsonl"), rows)
    print(f"{n} imagens sinteticas em {images_dir} e rotulos em {resolve_path(cfg, 'labels_jsonl')}.")


def make_tiny_models(cfg: Config) -> None:
    """Tokenizador de palavras + Llama de 2 camadas + SigLIP de 2 camadas, todos aleatorios."""
    from tokenizers import Tokenizer, normalizers, pre_tokenizers
    from tokenizers.models import WordLevel
    from tokenizers.processors import TemplateProcessing
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedTokenizerFast,
        SiglipImageProcessor,
        SiglipVisionConfig,
        SiglipVisionModel,
    )

    out = Path(cfg.get_path("dev.tiny_dir", "outputs/tiny"))
    out = out if out.is_absolute() else ROOT / out

    # vocabulario: tudo que as legendas sinteticas e o prompt podem conter
    rng = random.Random(0)
    textos = [cfg.model.prompt]
    for _ in range(400):
        textos.extend(_cena(rng, 32)[1])
    palavras = sorted({w for t in textos for w in re.findall(r"\w+|[^\w\s]", t.lower())})
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    for w in palavras:
        vocab.setdefault(w, len(vocab))

    tk = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tk.normalizer = normalizers.Sequence([normalizers.NFC(), normalizers.Lowercase()])
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.post_processor = TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<pad>")

    llm_conf = LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        attention_bias=True,
        mlp_bias=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        tie_word_embeddings=False,
    )
    llm = LlamaForCausalLM(llm_conf)
    llm.save_pretrained(str(out / "llm"))
    tok.save_pretrained(str(out / "llm"))

    vis_conf = SiglipVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, image_size=32, patch_size=8
    )
    SiglipVisionModel(vis_conf).save_pretrained(str(out / "vision"))
    SiglipImageProcessor(size={"height": 32, "width": 32}).save_pretrained(str(out / "vision"))
    print(f"Modelos minusculos em {out} (vocabulario de {len(vocab)} tokens).")
