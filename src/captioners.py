"""Modelos de captioning. Cada captioner expoe .name, .lang e .caption(PIL.Image).

Modelos que legendam em ingles (blip, florence2) sao traduzidos depois, no
run_captioning; os que legendam direto em PT-BR (qwen25vl, claude) tem lang="pt".

VRAM aproximada em fp16: blip-large ~0.9 GB, florence2-large ~1.6 GB,
qwen25vl-3b em 4 bits ~2.5 GB (aperta numa GPU de 4 GB — use imagens menores ou
rode no Colab).
"""
from __future__ import annotations

import base64

from .common import pick_device


def _load(cls, model_id: str, dtype=None, **kwargs):
    """from_pretrained tolerante a mudanca de nome do parametro de dtype."""
    if dtype is None:
        return cls.from_pretrained(model_id, **kwargs)
    try:
        return cls.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError:
        return cls.from_pretrained(model_id, torch_dtype=dtype, **kwargs)


class BlipCaptioner:
    """Baseline classico encoder-decoder (Salesforce BLIP)."""

    lang = "en"

    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-large", device: str | None = None) -> None:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.name = model_id.split("/")[-1]
        self.device = pick_device(device)
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = _load(BlipForConditionalGeneration, model_id, self.dtype).to(self.device).eval()

    def caption(self, image) -> str:
        import torch

        inputs = self.processor(images=image, return_tensors="pt").to(self.device, self.dtype)
        with torch.inference_mode():
            out = self.model.generate(**inputs, num_beams=5, max_new_tokens=40, min_length=8, length_penalty=1.0)
        return self.processor.decode(out[0], skip_special_tokens=True).strip()


class Florence2Captioner:
    """Florence-2 com a task <MORE_DETAILED_CAPTION> (legendas bem mais ricas que BLIP)."""

    lang = "en"

    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-large",
        device: str | None = None,
        task: str = "<MORE_DETAILED_CAPTION>",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._patch_flash_attn()
        self.name = model_id.split("/")[-1]
        self.device = pick_device(device)
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.task = task
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = _load(AutoModelForCausalLM, model_id, self.dtype, trust_remote_code=True).to(self.device).eval()

    @staticmethod
    def _patch_flash_attn() -> None:
        """O codigo remoto do Florence-2 tenta importar flash_attn mesmo quando ele
        nao e usado (nao ha wheel para Windows). Removemos o import da lista."""
        try:
            import transformers.dynamic_module_utils as dmu
        except ImportError:
            return
        if getattr(dmu, "_flash_attn_patched", False):
            return
        original = dmu.get_imports

        def sem_flash_attn(filename):
            return [imp for imp in original(filename) if imp != "flash_attn"]

        dmu.get_imports = sem_flash_attn
        dmu._flash_attn_patched = True

    def caption(self, image) -> str:
        import torch

        inputs = self.processor(text=self.task, images=image, return_tensors="pt").to(self.device, self.dtype)
        with torch.inference_mode():
            ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
                do_sample=False,
            )
        bruto = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(bruto, task=self.task, image_size=image.size)
        return str(parsed.get(self.task, "")).strip()


class QwenVLCaptioner:
    """VLM multilingue — legenda direto em PT-BR. Carrega em 4 bits por padrao."""

    lang = "pt"

    INSTRUCAO = (
        "Descreva esta imagem de ambiente portuario em uma unica frase em portugues do Brasil, "
        "com 12 a 30 palavras. Use termos tecnicos portuarios quando tiver certeza visual e nao "
        "invente nomes de navios, empresas ou cargas."
    )

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str | None = None,
        load_4bit: bool = True,
        instrucao: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.name = model_id.split("/")[-1]
        self.device = pick_device(device)
        self.instrucao = instrucao or self.INSTRUCAO
        self.processor = AutoProcessor.from_pretrained(model_id)
        kwargs = {}
        if load_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            kwargs["device_map"] = "auto"
        self.model = _load(Qwen2_5_VLForConditionalGeneration, model_id, torch.float16, **kwargs).eval()
        if not kwargs:
            self.model = self.model.to(self.device)

    def caption(self, image) -> str:
        import torch

        mensagens = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.instrucao}]}]
        texto = self.processor.apply_chat_template(mensagens, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[texto], images=[image], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
        novos = ids[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(novos, skip_special_tokens=True).strip()


class ClaudeCaptioner:
    """Teto de qualidade: VLM via API, legenda direto em PT-BR."""

    lang = "pt"

    def __init__(self, model_id: str = "claude-opus-5", fallbacks: bool = True) -> None:
        import anthropic

        from .prompts import SYSTEM_PROMPT

        self.name = model_id
        self.model_id = model_id
        self.fallbacks = fallbacks
        self.system = SYSTEM_PROMPT
        self.client = anthropic.Anthropic()

    def caption(self, image) -> str:
        import io

        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=92)
        dados = base64.standard_b64encode(buffer.getvalue()).decode()
        kwargs = dict(
            model=self.model_id,
            max_tokens=1000,
            system=self.system,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": dados}},
                        {"type": "text", "text": "Escreva UMA unica legenda para esta imagem. Responda so a legenda."},
                    ],
                }
            ],
        )
        if self.fallbacks:
            resposta = self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
            )
        else:
            resposta = self.client.messages.create(**kwargs)
        if resposta.stop_reason == "refusal":
            raise RuntimeError(f"requisicao recusada ({getattr(resposta, 'stop_details', None)})")
        return " ".join(b.text for b in resposta.content if b.type == "text").strip()


REGISTRY = {
    "blip": BlipCaptioner,
    "blip-base": lambda **kw: BlipCaptioner(model_id="Salesforce/blip-image-captioning-base", **kw),
    "florence2": Florence2Captioner,
    "florence2-base": lambda **kw: Florence2Captioner(model_id="microsoft/Florence-2-base", **kw),
    "qwen25vl": QwenVLCaptioner,
    "claude": ClaudeCaptioner,
}


def get_captioner(nome: str, **kwargs):
    if nome not in REGISTRY:
        raise SystemExit(f"Modelo '{nome}' desconhecido. Disponiveis: {', '.join(REGISTRY)}")
    return REGISTRY[nome](**kwargs)
