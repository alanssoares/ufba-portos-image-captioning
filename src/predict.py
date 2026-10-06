"""Gera as legendas de uma variante sobre um split (padrao: teste).

Uso:
    python -m src predict --variant lora
    python -m src predict --variant gold                     # Gemini via API
    python -m src predict --variant all                      # todas de `variants`
    python -m src predict --variant finetune --set generation.num_beams=1

Saida: results/preds_<variante>.jsonl, uma linha por imagem:
    {"image_id", "variante", "modelo", "legenda", "segundos"}

Retomavel: imagens que ja tem predicao sao puladas (use --redo para regerar).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from .common import append_jsonl, find_image, index_by, pick_device, read_json, read_jsonl
from .config import Config, resolve_path
from .splits import load_split

LOCAL_VARIANTS = ("base", "pretrain", "finetune", "lora", "qlora")


# ---------------------------------------------------------------------------
# modelo gold (API)
# ---------------------------------------------------------------------------
class GeminiCaptioner:
    """VLM grande via API do Google (google-genai). A chave vem de gold_model.api_key_env."""

    def __init__(self, cfg: Config) -> None:
        from google import genai
        from google.genai import types

        from .prompts import GOLD_FORMATO, GOLD_SYSTEM_PROMPT

        g = cfg.gold_model
        chave = os.environ.get(g.api_key_env) or os.environ.get("GOOGLE_API_KEY")
        if not chave:
            raise SystemExit(f"Defina {g.api_key_env} com uma chave do Google AI Studio (aistudio.google.com/apikey).")
        self.client = genai.Client(api_key=chave)
        self.types = types
        self.name = g.model
        # Mesma instrucao de tarefa das variantes Qwen (model.instruction): a unica diferenca
        # entre gold e SLM e o system prompt de dominio (glossario + regras), se ligado.
        self.instrucao = f"{' '.join(str(cfg.model.instruction).split())} {GOLD_FORMATO}"
        extra = {}
        if g.get("temperature") is not None:
            extra["temperature"] = float(g.temperature)
        if g.get("max_output_tokens"):
            extra["max_output_tokens"] = int(g.max_output_tokens)
        self.config = types.GenerateContentConfig(
            system_instruction=GOLD_SYSTEM_PROMPT if g.use_domain_prompt else None, **extra
        )

    def __call__(self, path: Path) -> str:
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        resposta = self.client.models.generate_content(
            model=self.name,
            contents=[self.types.Part.from_bytes(data=path.read_bytes(), mime_type=mime), self.instrucao],
            config=self.config,
        )
        texto = (resposta.text or "").strip().strip('"').strip()
        if not texto:
            raise RuntimeError(f"resposta vazia ({getattr(resposta, 'prompt_feedback', None)})")
        return " ".join(texto.split())


class DummyCaptioner:
    """So para teste do pipeline sem chave de API: devolve a 1a legenda de referencia embaralhada."""

    def __init__(self, cfg: Config) -> None:
        import random

        self.labels = index_by(read_jsonl(resolve_path(cfg, "labels_jsonl")))
        self.rng = random.Random(int(cfg.seed))
        self.name = "dummy"

    def __call__(self, path: Path) -> str:
        palavras = self.labels.get(path.stem, {}).get("legendas", ["sem legenda"])[0].split()
        self.rng.shuffle(palavras)
        return " ".join(palavras)


GOLD_PROVIDERS = {"gemini": GeminiCaptioner, "dummy": DummyCaptioner}


# ---------------------------------------------------------------------------
def _pendentes(cfg: Config, out: Path, redo: bool, versao: str | None = None) -> list[Path]:
    """Imagens do split ainda sem predicao. Com `versao`, linhas de outra versao contam como pendentes."""
    split = cfg.generation.split
    images_dir = resolve_path(cfg, "images_dir")
    linhas = index_by(read_jsonl(out))
    if versao is not None:
        linhas = {k: r for k, r in linhas.items() if r.get("prompt_versao") == versao}
    feitos = set() if redo else set(linhas)
    caminhos = []
    for image_id in load_split(cfg, split):
        if image_id in feitos:
            continue
        p = find_image(images_dir, image_id)
        if p is not None:
            caminhos.append(p)
    if cfg.generation.limit:
        caminhos = caminhos[: int(cfg.generation.limit)]
    return caminhos


def gold_versao(cfg: Config) -> str:
    """Versao da especificacao recebida pelo gold: modelo + system prompt + instrucao."""
    from .prompts import GOLD_FORMATO, GOLD_SYSTEM_PROMPT, versao

    g = cfg.gold_model
    sistema = GOLD_SYSTEM_PROMPT if g.use_domain_prompt else ""
    return versao(str(g.provider), str(g.model), sistema, " ".join(str(cfg.model.instruction).split()), GOLD_FORMATO)


def cota_diaria_esgotada(exc: Exception) -> bool:
    """429 da cota DIARIA: tentar de novo antes do reset so desperdica chamadas."""
    texto = str(exc)
    return "RESOURCE_EXHAUSTED" in texto and "PerDay" in texto


def predict_gold(cfg: Config, redo: bool = False) -> Path:
    out = resolve_path(cfg, "results_dir") / "preds_gold.jsonl"
    versao_atual = gold_versao(cfg)
    caminhos = _pendentes(cfg, out, redo, versao=versao_atual)
    if not caminhos:
        print(f"gold: nada a fazer (todas as imagens na versao {versao_atual}).")
        return out
    g = cfg.gold_model
    if g.provider not in GOLD_PROVIDERS:
        raise SystemExit(f"gold_model.provider '{g.provider}' desconhecido. Use: {', '.join(GOLD_PROVIDERS)}")
    captioner = GOLD_PROVIDERS[g.provider](cfg)
    intervalo = float(g.get("min_interval_s") or 0) if g.provider != "dummy" else 0.0
    print(f"gold ({captioner.name}, versao {versao_atual}) sobre {len(caminhos)} imagens -> {out}")
    erros, ultima, interrompido = 0, 0.0, False
    for i, path in enumerate(caminhos, 1):
        legenda, duracao, tentativa = None, None, 0
        t0 = time.perf_counter()
        for tentativa in range(1, int(g.max_retries) + 1):
            espera = intervalo - (time.monotonic() - ultima)
            if espera > 0:
                time.sleep(espera)   # respeita o limite de requisicoes por minuto
            ultima = time.monotonic()
            try:
                t_chamada = time.perf_counter()
                legenda = captioner(path)
                duracao = time.perf_counter() - t_chamada   # so a chamada bem-sucedida
                break
            except Exception as exc:
                print(f"  [{i}] {path.stem}: tentativa {tentativa} falhou — {exc}")
                if cota_diaria_esgotada(exc):
                    interrompido = True
                    break
                time.sleep(float(g.retry_wait_s) * tentativa)
        if interrompido:
            break
        if legenda is None:
            erros += 1
            continue
        append_jsonl(out, {"image_id": path.stem, "variante": "gold", "modelo": captioner.name,
                           "prompt_versao": versao_atual, "legenda": legenda,
                           # `segundos` = latencia da chamada que deu certo (vai para s/img);
                           # `segundos_total` inclui o espacamento por cota e as novas tentativas.
                           "segundos": duracao, "segundos_total": time.perf_counter() - t0,
                           "tentativas": tentativa})
        print(f"  [{i}/{len(caminhos)}] {path.stem}: {legenda}")
    faltam = len(_pendentes(cfg, out, False, versao=versao_atual))
    if interrompido:
        print(f"gold: INTERROMPIDO — cota diaria esgotada. Faltam {faltam} imagens; "
              "rode o mesmo comando depois do reset (continua de onde parou).")
    else:
        print(f"gold: pronto. Erros: {erros}. Faltam: {faltam}.")
    return out


def predict_local(cfg: Config, variant: str, redo: bool = False) -> Path:
    import torch
    from PIL import Image

    from .vlm import load_variant

    out = resolve_path(cfg, "results_dir") / f"preds_{variant}.jsonl"
    caminhos = _pendentes(cfg, out, redo)
    if not caminhos:
        print(f"{variant}: nada a fazer.")
        return out

    device = pick_device(cfg.get("device"))
    gen = cfg.generation
    model = load_variant(variant, cfg, resolve_path(cfg, "models_dir"), device, load_4bit=bool(gen.load_4bit))
    meta = read_json(resolve_path(cfg, "models_dir") / variant / "vlm_config.json", {})
    from .transfer import splits_sha1

    treinado_com = meta.get("splits_sha1")
    if treinado_com and treinado_com != splits_sha1(resolve_path(cfg, "splits_json")):
        print(
            f"  AVISO: '{variant}' foi treinado com outro splits.json — o teste local pode conter imagens "
            "de treino. Importe o pacote do Colab (python -m src import) para alinhar."
        )
    kwargs = dict(
        max_new_tokens=int(gen.max_new_tokens),
        num_beams=int(gen.num_beams),
        do_sample=bool(gen.do_sample),
        repetition_penalty=float(gen.repetition_penalty),
        no_repeat_ngram_size=int(gen.no_repeat_ngram_size),
    )
    print(f"{variant} sobre {len(caminhos)} imagens ({device}) -> {out}")
    bs = int(gen.batch_size)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for inicio in range(0, len(caminhos), bs):
        lote = caminhos[inicio : inicio + bs]
        imagens = []
        for p in lote:
            with Image.open(p) as img:
                imagens.append(img.convert("RGB"))
        t0 = time.perf_counter()
        legendas = model.generate(imagens, **kwargs)
        seg = (time.perf_counter() - t0) / len(lote)
        for p, legenda in zip(lote, legendas):
            append_jsonl(
                out,
                {
                    "image_id": p.stem,
                    "variante": variant,
                    "modelo": f"{cfg.model.model_id}+{variant}",
                    "legenda": " ".join(legenda.split()),
                    "segundos": seg,
                },
            )
            print(f"  {p.stem}: {legenda}")
    if device == "cuda":
        print(f"  VRAM de pico na inferencia: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def run(cfg: Config, variant: str, redo: bool = False) -> None:
    alvos = list(cfg.variants) if variant == "all" else [variant]
    for v in alvos:
        if v == "gold":
            predict_gold(cfg, redo)
        elif v in LOCAL_VARIANTS:
            predict_local(cfg, v, redo)
        else:
            raise SystemExit(f"variante desconhecida '{v}'. Use: {', '.join(LOCAL_VARIANTS + ('gold', 'all'))}")
