"""Modelo de legenda: um VLM do Hugging Face (padrao: Qwen3-VL-2B-Instruct) + processador.

O Qwen3-VL ja e multimodal: encoder de visao (ViT, ~0,4 B) -> "merger" (conector que leva
as features visuais para o espaco do LLM, ~0,1 B) -> LLM Qwen3 (~1,7 B). As variantes do
experimento diferem no que treina (ver docs/metodologia-variantes.md):

    conector (merger)       -> model.connector_modules (padrao: visual.merger, visual.deepstack_merger_list)
    LLM completo / parcial  -> training.*.llm_mode = full | partial
    adaptadores LoRA        -> training.*.llm_mode = lora (+ quantize_4bit no QLoRA)

Formato de treino (chat template do modelo):

    <|im_start|>user <imagem> {instrucao}<|im_end|>
    <|im_start|>assistant\\n{legenda}<|im_end|>      <- loss so aqui

Cada variante e salva em outputs/models/<variante>/:

    vlm_config.json      de onde vem cada peso (base_ref), instrucao, resolucao, split usado
    model/               modelo completo (finetune / partial)
    delta.safetensors    so os pesos treinados fora do LLM (conector; pretrain / lora / qlora)
    adapter/             adaptador PEFT (lora / qlora)
    processor/           processador (tokenizador + imagem)
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

import torch

from .config import ROOT, Config

VLM_CONFIG = "vlm_config.json"
DELTA_FILE = "delta.safetensors"

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# ---------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------
def bf16_nativo() -> bool:
    """bf16 de verdade (Ampere+). A T4 (sm_75) so emula bf16 — la usamos fp16."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def resolve_dtype(name: str | None, device: str) -> torch.dtype:
    if name in (None, "auto"):
        if device == "cuda":
            return torch.bfloat16 if bf16_nativo() else torch.float16
        return torch.float32
    if name not in DTYPES:
        raise ValueError(f"dtype desconhecido '{name}'. Use auto | bf16 | fp16 | fp32")
    if device == "cpu" and DTYPES[name] == torch.float16:
        return torch.float32  # fp16 na CPU e lento e instavel
    return DTYPES[name]


def resolve_model_ref(ref: str) -> str:
    """Caminho local (absoluto ou relativo ao repo) ou id do Hugging Face Hub."""
    p = Path(ref)
    if p.is_absolute() and p.exists():
        return str(p)
    if (ROOT / p).exists():
        return str(ROOT / p)
    return ref


def _from_pretrained(cls, ref: str, dtype: torch.dtype | None = None, **kwargs):
    """from_pretrained com o nome certo do argumento de dtype para a versao instalada."""
    import transformers
    from packaging.version import Version

    if dtype is not None:
        chave = "dtype" if Version(transformers.__version__) >= Version("4.56.0") else "torch_dtype"
        kwargs[chave] = dtype
    return cls.from_pretrained(ref, **kwargs)


def count_params(module: torch.nn.Module) -> tuple[int, int]:
    total = treinaveis = 0
    vistos = set()
    for p in module.parameters():
        if id(p) in vistos:  # embeddings amarrados ao lm_head contam uma vez
            continue
        vistos.add(id(p))
        n = p.numel() * (2 if p.__class__.__name__ == "Params4bit" else 1)  # 4 bits: 2 valores/byte
        total += n
        if p.requires_grad:
            treinaveis += n
    return total, treinaveis


def language_model(model):
    """O LLM (decoder) dentro do VLM — onde ficam .layers e .norm."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    lm = getattr(base, "language_model", None)
    if lm is None:
        lm = base.model.language_model
    return lm


def unwrap(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


# ---------------------------------------------------------------------------
# carga
# ---------------------------------------------------------------------------
def load_processor(ref: str):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(resolve_model_ref(ref))


SKIP_PADRAO = ["visual", "lm_head"]   # encoder de visao (com os mergers) e lm_head ficam em 16 bits


def bnb_skip_patterns(nomes: list[str]) -> list[str]:
    """Padroes de `llm_int8_skip_modules` que funcionam em qualquer versao do transformers.

    transformers 4.x pulava o modulo se o nome aparecesse em QUALQUER posicao do caminho
    ("visual" casava com "model.visual.merger.linear_fc1"). A 5.x compara do INICIO
    (`re.match`) ou pelo fim (`endswith`), e "visual" deixou de casar — o encoder de visao e
    os mergers eram quantizados em 4 bits. Para cada nome, mantemos o nome puro (4.x) e
    acrescentamos o regex `(.*\.)?nome` (5.x), que casa o componente em qualquer profundidade.
    """
    padroes: list[str] = []
    for nome in nomes:
        for p in (nome, rf"(.*\.)?{re.escape(nome)}"):
            if p not in padroes:
                padroes.append(p)
    return padroes


def modulos_quantizados_indevidos(model, nomes: list[str], classe_4bit=None) -> list[str]:
    """Modulos 4 bits cujo caminho contem um componente que deveria ficar em 16 bits."""
    if classe_4bit is None:
        try:
            import bitsandbytes as bnb
        except ImportError:
            return []
        classe_4bit = bnb.nn.Linear4bit
    componentes = [re.compile(rf"(^|\.){re.escape(n)}(\.|$)") for n in nomes]
    return [
        nome for nome, mod in model.named_modules()
        if isinstance(mod, classe_4bit) and any(c.search(nome) for c in componentes)
    ]


def load_model(ref: str, dtype: torch.dtype, device: str, attn: str | None, quant: dict | None = None):
    """Carrega a classe de arquitetura declarada no config (ex: Qwen3VLForConditionalGeneration)."""
    import transformers
    from transformers import AutoConfig

    ref = resolve_model_ref(ref)
    arquitetura = AutoConfig.from_pretrained(ref).architectures[0]
    cls = getattr(transformers, arquitetura, None) or transformers.AutoModelForImageTextToText

    kwargs: dict[str, Any] = {}
    if attn:
        kwargs["attn_implementation"] = attn
    if quant:
        from transformers import BitsAndBytesConfig

        skip = list(quant.get("skip_modules", SKIP_PADRAO))
        compute = resolve_dtype(quant.get("compute_dtype", "auto"), device)
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant.get("quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(quant.get("double_quant", True)),
            bnb_4bit_compute_dtype=compute,
            # encoder de visao e lm_head ficam em 16 bits (QLoRA classico so quantiza o LLM)
            llm_int8_skip_modules=bnb_skip_patterns(skip),
        )
        kwargs["device_map"] = {"": torch.cuda.current_device()}
        dtype = compute
    model = _from_pretrained(cls, ref, dtype, **kwargs)
    if quant:
        indevidos = modulos_quantizados_indevidos(model, skip)
        if indevidos:
            raise RuntimeError(
                f"{len(indevidos)} modulos de {skip} foram quantizados em 4 bits, mas deveriam ficar em "
                f"16 bits (ex: {indevidos[:3]}). Verifique quant.skip_modules e a versao do transformers."
            )
    if not quant:
        model = model.to(device)
    return model


def apply_delta(model, variant_dir: Path) -> list[str]:
    """Carrega pesos treinados fora do LLM (conector etc.). Devolve as chaves aplicadas."""
    from safetensors.torch import load_file

    caminho = variant_dir / DELTA_FILE
    if not caminho.exists():
        return []
    estado = load_file(str(caminho))
    alvo = unwrap(model)
    _, inesperadas = alvo.load_state_dict(estado, strict=False)
    if inesperadas:
        raise RuntimeError(f"{caminho}: chaves que nao existem no modelo: {inesperadas[:5]}...")
    return list(estado)


# ---------------------------------------------------------------------------
# o modelo de legenda
# ---------------------------------------------------------------------------
class Captioner:
    """Monta os batches no formato de chat do VLM, calcula a loss e gera legendas."""

    def __init__(self, model, processor, meta: dict) -> None:
        self.model = model
        self.processor = processor
        self.meta = meta
        tok = processor.tokenizer
        self.marker_ids = tok(meta["response_marker"], add_special_tokens=False).input_ids
        self.suffix = meta.get("response_suffix") or tok.eos_token or ""
        self.max_side = int(meta.get("image_max_side") or 0)
        self._prompt = self._build_prompt()

    # -- entrada ---------------------------------------------------------
    def _build_prompt(self) -> str:
        msgs = []
        if self.meta.get("system_prompt"):
            msgs.append({"role": "system", "content": [{"type": "text", "text": self.meta["system_prompt"]}]})
        msgs.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.meta["instruction"]}]})
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _resize(self, img):
        from PIL import Image

        img = img.convert("RGB")
        if self.max_side and max(img.size) > self.max_side:
            img = img.copy()
            img.thumbnail((self.max_side, self.max_side), Image.LANCZOS)
        return img

    def _marker_end(self, row: list[int]) -> int | None:
        m = self.marker_ids
        for i in range(len(row) - len(m), -1, -1):
            if row[i : i + len(m)] == m:
                return i + len(m)
        return None

    def train_batch(self, images, captions: list[str]) -> dict:
        """input_ids / attention_mask / pixel_values / image_grid_thw / labels (so a legenda conta)."""
        self.processor.tokenizer.padding_side = "right"
        textos = [self._prompt + c.strip() + self.suffix for c in captions]
        enc = self.processor(text=textos, images=[self._resize(i) for i in images], padding=True, return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i, row in enumerate(enc["input_ids"].tolist()):
            fim = self._marker_end(row)
            if fim is None:
                raise RuntimeError(
                    f"marcador de resposta {self.meta['response_marker']!r} nao encontrado — confira "
                    "model.response_marker para o chat template deste modelo"
                )
            labels[i, :fim] = -100
        enc["labels"] = labels
        return dict(enc)

    def loss(self, images, captions: list[str], device: str):
        batch = {k: v.to(device) for k, v in self.train_batch(images, captions).items()}
        return self.model(**batch).loss

    # -- saida -----------------------------------------------------------
    @torch.inference_mode()
    def generate(self, images, **gen_kwargs) -> list[str]:
        self.processor.tokenizer.padding_side = "left"
        enc = self.processor(
            text=[self._prompt] * len(images),
            images=[self._resize(i) for i in images],
            padding=True,
            return_tensors="pt",
        )
        dispositivo = next(p for p in self.model.parameters()).device
        enc = {k: v.to(dispositivo) for k, v in enc.items()}
        out = self.model.generate(**enc, pad_token_id=self.processor.tokenizer.pad_token_id, **gen_kwargs)
        novos = out[:, enc["input_ids"].shape[1] :]
        textos = self.processor.batch_decode(novos, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [" ".join(t.split()) for t in textos]


# ---------------------------------------------------------------------------
# metadados, salvamento e carga de variantes
# ---------------------------------------------------------------------------
def base_meta(cfg: Config) -> dict:
    m = cfg.model
    return {
        "base_ref": {"kind": "hub", "value": m.model_id},
        "adapter": False,
        "adapter_quantized_base": False,
        "instruction": m.instruction,
        "system_prompt": m.get("system_prompt"),
        "response_marker": m.response_marker,
        "response_suffix": m.get("response_suffix"),
        "image_max_side": int(m.image_max_side),
    }


def ref_path(meta_ref: dict, models_dir: Path) -> str:
    if meta_ref["kind"] == "local":
        return str(models_dir / meta_ref["value"])
    return meta_ref["value"]


def read_meta(variant_dir: Path) -> dict:
    path = variant_dir / VLM_CONFIG
    if not path.exists():
        raise SystemExit(f"{variant_dir} nao e uma variante treinada (falta {VLM_CONFIG}). Treine-a ou importe o pacote.")
    return json.loads(path.read_text(encoding="utf-8"))


def write_meta(variant_dir: Path, meta: dict) -> None:
    variant_dir.mkdir(parents=True, exist_ok=True)
    meta = {**meta, "salvo_em": dt.datetime.now().isoformat(timespec="seconds")}
    (variant_dir / VLM_CONFIG).write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def save_variant(cap: Captioner, out_dir: Path, meta: dict, delta_keys: list[str], save_dtype: torch.dtype) -> None:
    """Salva sem alterar o modelo em treino (pode ser um checkpoint intermediario)."""
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    base = unwrap(cap.model)

    if meta.get("weights") == "full":
        pesos = {k: v.detach().to("cpu", save_dtype) for k, v in base.state_dict().items()}
        base.save_pretrained(str(out_dir / "model"), state_dict=pesos, safe_serialization=True)
        cap.processor.save_pretrained(str(out_dir / "model"))
        del pesos
    else:
        if delta_keys:
            estado = base.state_dict()
            delta = {k: estado[k].detach().to("cpu", torch.float32).contiguous() for k in delta_keys}
            save_file(delta, str(out_dir / DELTA_FILE))
        if meta.get("adapter"):
            cap.model.save_pretrained(str(out_dir / "adapter"))

    cap.processor.save_pretrained(str(out_dir / "processor"))
    write_meta(out_dir, meta)


def load_variant(variant: str, cfg: Config, models_dir: Path, device: str, load_4bit: bool = False) -> Captioner:
    """Carrega uma variante para inferencia."""
    vdir = models_dir / variant
    meta = read_meta(vdir)
    dtype = resolve_dtype(cfg.model.dtype, device)

    quant = None
    if meta.get("adapter_quantized_base") or load_4bit:
        if device == "cuda":
            quant = dict(cfg.training.qlora.get("quant", {}))
        else:
            print(f"  ({variant}: sem CUDA — carregando sem quantizacao)")

    model = load_model(ref_path(meta["base_ref"], models_dir), dtype, device, cfg.model.attn_implementation, quant)
    apply_delta(model, vdir)
    if meta.get("adapter"):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(vdir / "adapter"), is_trainable=False)
    model.eval()
    proc_dir = vdir / "processor"
    processor = load_processor(str(proc_dir) if proc_dir.exists() else ref_path(meta["base_ref"], models_dir))
    return Captioner(model, processor, meta)
