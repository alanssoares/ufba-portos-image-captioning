"""VLM estilo LLaVA: encoder de visao + projetor + SLM PT-BR (so texto).

O Manaca-1B e um modelo de linguagem puro — nao enxerga imagens. Para legendar, a
imagem passa por um encoder de visao (SigLIP), vira uma sequencia de vetores, e um
projetor (MLP) leva esses vetores para o espaco de embeddings do LLM. A sequencia
que o LLM ve e:

    [img_1 ... img_M] [<s> descricao da imagem do porto:] [legenda ... </s>]

A loss so e calculada sobre os tokens da legenda.

Cada variante e salva numa pasta propria (outputs/models/<variante>/):

    vlm_config.json         metadados: de onde vem cada peso, prompt, pooling...
    projector.safetensors   pesos do projetor (sempre)
    llm/                    LLM completo (finetune / partial)
    adapter/                adaptador PEFT (lora / qlora)
    vision/                 encoder de visao (so se train_vision=true)
    tokenizer/, image_processor/
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .config import ROOT, Config

VLM_CONFIG = "vlm_config.json"
PROJECTOR_FILE = "projector.safetensors"

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
    """Caminho local relativo ao repo (ex: outputs/tiny/llm) ou id do Hugging Face Hub."""
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


def count_params(module: nn.Module) -> tuple[int, int]:
    total = treinaveis = 0
    for p in module.parameters():
        n = p.numel()
        # parametros de 4 bits do bitsandbytes guardam 2 valores por byte
        if p.__class__.__name__ == "Params4bit":
            n *= 2
        total += n
        if p.requires_grad:
            treinaveis += n
    return total, treinaveis


# ---------------------------------------------------------------------------
# componentes
# ---------------------------------------------------------------------------
def build_projector(kind: str, d_in: int, d_out: int) -> nn.Module:
    if kind == "linear":
        return nn.Linear(d_in, d_out)
    if kind == "mlp2x_gelu":
        return nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out))
    raise ValueError(f"projetor desconhecido '{kind}'. Use linear | mlp2x_gelu")


def load_vision(ref: str, dtype: torch.dtype):
    """Retorna (modelo, processador, tem_cls). Suporta SigLIP e CLIP."""
    from transformers import AutoConfig, AutoImageProcessor

    ref = resolve_model_ref(ref)
    tipo = AutoConfig.from_pretrained(ref).model_type
    if tipo.startswith("siglip"):
        from transformers import SiglipVisionModel as Cls

        tem_cls = False
    elif tipo.startswith("clip"):
        from transformers import CLIPVisionModel as Cls

        tem_cls = True
    else:
        raise ValueError(f"encoder de visao '{ref}' ({tipo}) nao suportado — use SigLIP ou CLIP")
    modelo = _from_pretrained(Cls, ref, dtype)
    processador = AutoImageProcessor.from_pretrained(ref)
    return modelo, processador, tem_cls


def load_tokenizer(ref: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(resolve_model_ref(ref))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_llm(ref: str, dtype: torch.dtype, device: str, attn: str, quant: dict | None = None):
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {}
    if attn:
        kwargs["attn_implementation"] = attn
    if quant:
        from transformers import BitsAndBytesConfig

        compute = resolve_dtype(quant.get("compute_dtype", "auto"), device)
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant.get("quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(quant.get("double_quant", True)),
            bnb_4bit_compute_dtype=compute,
        )
        kwargs["device_map"] = {"": torch.cuda.current_device()}
        dtype = compute
    modelo = _from_pretrained(AutoModelForCausalLM, resolve_model_ref(ref), dtype, **kwargs)
    if not quant:
        modelo = modelo.to(device)
    return modelo


# ---------------------------------------------------------------------------
# o modelo
# ---------------------------------------------------------------------------
class PortoVLM(nn.Module):
    def __init__(self, vision, image_processor, projector, llm, tokenizer, meta: dict) -> None:
        super().__init__()
        self.vision = vision
        self.image_processor = image_processor
        self.projector = projector
        self.llm = llm
        self.tokenizer = tokenizer
        self.meta = meta
        self.train_vision = False
        self.feature_layer = int(meta.get("vision_feature_layer", -1))
        self.pool = int(meta.get("image_pool", 1))
        self.tem_cls = bool(meta.get("vision_has_cls", False))
        self.max_caption_tokens = int(meta.get("max_caption_tokens", 64))

        bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        self.prompt_ids = bos + tokenizer(meta["prompt"], add_special_tokens=False).input_ids

    # -- imagem -------------------------------------------------------------
    def preprocess(self, images) -> torch.Tensor:
        return self.image_processor(images=images, return_tensors="pt")["pixel_values"]

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vparam = next(self.vision.parameters())
        with torch.set_grad_enabled(self.train_vision and torch.is_grad_enabled()):
            out = self.vision(pixel_values=pixel_values.to(vparam.device, vparam.dtype), output_hidden_states=True)
            feats = out.hidden_states[self.feature_layer]
        if self.tem_cls:
            feats = feats[:, 1:]
        if self.pool > 1:
            b, n, d = feats.shape
            lado = int(math.isqrt(n))
            if lado * lado == n:
                grade = feats.transpose(1, 2).reshape(b, d, lado, lado)
                alvo = max(1, lado // self.pool)
                feats = F.adaptive_avg_pool2d(grade.float(), alvo).to(feats.dtype).flatten(2).transpose(1, 2)
        pparam = next(self.projector.parameters())
        return self.projector(feats.to(pparam.device, pparam.dtype))

    # -- texto --------------------------------------------------------------
    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(ids)

    def build_inputs(self, pixel_values: torch.Tensor, captions: list[str] | None = None):
        """Monta inputs_embeds / attention_mask (e labels, se houver legendas)."""
        img = self.encode_images(pixel_values)
        b, m, _ = img.shape
        dev = img.device
        pad = self.tokenizer.pad_token_id
        eos = self.tokenizer.eos_token_id

        if captions is None:
            ids = torch.tensor(self.prompt_ids, device=dev).unsqueeze(0).expand(b, -1)
            txt = self._embed(ids)
            embeds = torch.cat([img.to(txt.dtype), txt], dim=1)
            mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=dev)
            return embeds, mask, None

        seqs, alvos = [], []
        for legenda in captions:
            cap = self.tokenizer(legenda, add_special_tokens=False).input_ids[: self.max_caption_tokens]
            cap = cap + [eos]
            seqs.append(self.prompt_ids + cap)
            alvos.append([-100] * len(self.prompt_ids) + cap)
        comprimento = max(len(s) for s in seqs)
        ids = torch.full((b, comprimento), pad, dtype=torch.long, device=dev)
        labels = torch.full((b, comprimento), -100, dtype=torch.long, device=dev)
        mask_txt = torch.zeros((b, comprimento), dtype=torch.long, device=dev)
        for i, (s, a) in enumerate(zip(seqs, alvos)):
            ids[i, : len(s)] = torch.tensor(s, device=dev)
            labels[i, : len(a)] = torch.tensor(a, device=dev)
            mask_txt[i, : len(s)] = 1

        txt = self._embed(ids)
        embeds = torch.cat([img.to(txt.dtype), txt], dim=1)
        mask = torch.cat([torch.ones((b, m), dtype=torch.long, device=dev), mask_txt], dim=1)
        labels = torch.cat([torch.full((b, m), -100, dtype=torch.long, device=dev), labels], dim=1)
        return embeds, mask, labels

    def forward(self, pixel_values: torch.Tensor, captions: list[str]):
        embeds, mask, labels = self.build_inputs(pixel_values, captions)
        return self.llm(inputs_embeds=embeds, attention_mask=mask, labels=labels, use_cache=False)

    @torch.inference_mode()
    def generate(self, pixel_values: torch.Tensor, **gen_kwargs) -> list[str]:
        embeds, mask, _ = self.build_inputs(pixel_values)
        out = self.llm.generate(
            inputs_embeds=embeds,
            attention_mask=mask,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **gen_kwargs,
        )
        # So com inputs_embeds, o generate devolve apenas os tokens novos.
        return [t.strip() for t in self.tokenizer.batch_decode(out, skip_special_tokens=True)]


# ---------------------------------------------------------------------------
# construcao, salvamento e carga
# ---------------------------------------------------------------------------
def base_meta(cfg: Config) -> dict:
    m = cfg.model
    return {
        "llm_base": {"kind": "hub", "value": m.llm_id},
        "vision_source": {"kind": "hub", "value": m.vision_id},
        "adapter": False,
        "adapter_quantized_base": False,
        "projector": m.projector,
        "image_pool": int(m.image_pool),
        "vision_feature_layer": int(m.vision_feature_layer),
        "prompt": m.prompt,
        "max_caption_tokens": int(m.max_caption_tokens),
    }


def _ref(meta_ref: dict, models_dir: Path) -> str:
    if meta_ref["kind"] == "local":
        return str(models_dir / meta_ref["value"])
    return meta_ref["value"]


def assemble(
    meta: dict,
    models_dir: Path,
    device: str,
    llm_dtype: torch.dtype,
    vision_dtype: torch.dtype,
    attn: str,
    quant: dict | None = None,
    projector_state: dict | None = None,
    projector_dtype: torch.dtype = torch.float32,
    adapter_dir: Path | None = None,
    adapter_trainable: bool = False,
    seed: int = 42,
) -> PortoVLM:
    vision, processor, tem_cls = load_vision(_ref(meta["vision_source"], models_dir), vision_dtype)
    vision = vision.to(device).eval()
    for p in vision.parameters():
        p.requires_grad_(False)

    llm_ref = _ref(meta["llm_base"], models_dir)
    tokenizer = load_tokenizer(llm_ref)
    llm = load_llm(llm_ref, llm_dtype, device, attn, quant)
    if adapter_dir is not None:
        from peft import PeftModel

        llm = PeftModel.from_pretrained(llm, str(adapter_dir), is_trainable=adapter_trainable)

    torch.manual_seed(seed)  # inicializacao do projetor reproduzivel (importa na variante "base")
    projector = build_projector(meta["projector"], vision.config.hidden_size, llm.config.hidden_size)
    if projector_state is not None:
        projector.load_state_dict(projector_state)
    projector = projector.to(device=device, dtype=projector_dtype)

    meta = {**meta, "vision_has_cls": tem_cls}
    return PortoVLM(vision, processor, projector, llm, tokenizer, meta)


def read_meta(variant_dir: Path) -> dict:
    path = variant_dir / VLM_CONFIG
    if not path.exists():
        raise SystemExit(f"{variant_dir} nao e uma variante treinada (falta {VLM_CONFIG}). Treine-a antes.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_projector_state(variant_dir: Path) -> dict:
    from safetensors.torch import load_file

    return load_file(str(variant_dir / PROJECTOR_FILE))


def save_variant(model: PortoVLM, out_dir: Path, meta: dict, save_dtype: torch.dtype = torch.bfloat16) -> None:
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    estado = {k: v.detach().float().cpu().contiguous() for k, v in model.projector.state_dict().items()}
    save_file(estado, str(out_dir / PROJECTOR_FILE))

    if meta.get("llm_weights") == "full":
        # Copia em save_dtype na CPU — nao altera o modelo em treino (pode ser um checkpoint intermediario).
        llm = model.llm
        pesos = {k: v.detach().to("cpu", save_dtype) for k, v in llm.state_dict().items()}
        llm.save_pretrained(str(out_dir / "llm"), state_dict=pesos, safe_serialization=True)
        del pesos
        model.tokenizer.save_pretrained(str(out_dir / "llm"))
    elif meta.get("adapter"):
        model.llm.save_pretrained(str(out_dir / "adapter"))
    if meta.get("vision_trained"):
        model.vision.save_pretrained(str(out_dir / "vision"), safe_serialization=True)
        model.image_processor.save_pretrained(str(out_dir / "vision"))

    model.tokenizer.save_pretrained(str(out_dir / "tokenizer"))
    model.image_processor.save_pretrained(str(out_dir / "image_processor"))
    meta = {**meta, "salvo_em": dt.datetime.now().isoformat(timespec="seconds")}
    (out_dir / VLM_CONFIG).write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_variant(variant: str, cfg: Config, models_dir: Path, device: str, load_4bit: bool = False) -> PortoVLM:
    """Carrega uma variante treinada para inferencia."""
    vdir = models_dir / variant
    meta = read_meta(vdir)
    dtype = resolve_dtype(cfg.model.dtype, device)

    quant = None
    if meta.get("adapter_quantized_base") or load_4bit:
        if device == "cuda":
            quant = dict(cfg.training.qlora.get("quant", {}))
        else:
            print(f"  ({variant}: sem CUDA — carregando o LLM sem quantizacao)")

    model = assemble(
        meta,
        models_dir,
        device,
        llm_dtype=dtype,
        vision_dtype=dtype,
        attn=cfg.model.attn_implementation,
        quant=quant,
        projector_state=load_projector_state(vdir),
        projector_dtype=dtype,
        adapter_dir=(vdir / "adapter") if meta.get("adapter") else None,
    )
    model.eval()
    return model
