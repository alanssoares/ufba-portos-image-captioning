"""Treino das variantes: base, pretrain, finetune, lora, qlora.

Uso:
    python -m src train --stage base        # so materializa o modelo base (projetor aleatorio)
    python -m src train --stage pretrain    # pre-treino adaptativo de dominio (so o projetor)
    python -m src train --stage finetune    # fine-tuning completo (projetor + LLM inteiro)
    python -m src train --stage lora        # LoRA no LLM + projetor
    python -m src train --stage qlora       # LLM em 4 bits (NF4) + LoRA + projetor

    # ponto de partida: pre-treinado (padrao) ou base
    python -m src train --stage lora --set training.lora.init_from=base

Cada estagio usa training.common + training.<estagio> da config. A saida vai para
outputs/models/<estagio>/ (ver src/vlm.py) junto com train_summary.json (parametros
treinaveis, memoria de pico, tempo, losses) e train_log.jsonl.
"""
from __future__ import annotations

import datetime as dt
import math
from contextlib import nullcontext as _nada
import time
from pathlib import Path

from .common import append_jsonl, find_image, index_by, pick_device, read_jsonl, set_seed, write_json
from .config import Config, resolve_path, stage_config
from .splits import load_split

STAGES = ("base", "pretrain", "finetune", "lora", "qlora")


# ---------------------------------------------------------------------------
# dados
# ---------------------------------------------------------------------------
def build_samples(cfg: Config, split: str, per_image: str | int, max_images: int = 0) -> list[tuple[Path, str]]:
    labels = index_by(read_jsonl(resolve_path(cfg, "labels_jsonl")))
    images_dir = resolve_path(cfg, "images_dir")
    ids = load_split(cfg, split)
    if max_images:
        ids = ids[:max_images]
    amostras = []
    faltando = 0
    for image_id in ids:
        caminho = find_image(images_dir, image_id)
        legendas = labels.get(image_id, {}).get("legendas", [])
        if caminho is None or not legendas:
            faltando += 1
            continue
        escolhidas = legendas if per_image == "all" else legendas[: int(per_image)]
        amostras.extend((caminho, legenda) for legenda in escolhidas)
    if faltando:
        print(f"  ({split}: {faltando} imagens sem arquivo ou sem legenda foram ignoradas)")
    return amostras


class Collate:
    """Abre as imagens e aplica o processador do encoder. Classe (e nao closure) para
    funcionar com num_workers > 0 no Windows, onde os workers sao criados por spawn."""

    def __init__(self, processor) -> None:
        self.processor = processor

    def __call__(self, batch):
        from PIL import Image

        imagens = []
        for caminho, _ in batch:
            with Image.open(caminho) as img:
                imagens.append(img.convert("RGB"))
        pixel_values = self.processor(images=imagens, return_tensors="pt")["pixel_values"]
        return pixel_values, [legenda for _, legenda in batch]


def make_loader(samples, model, batch_size: int, shuffle: bool, num_workers: int, seed: int):
    import torch
    from torch.utils.data import DataLoader

    gen = torch.Generator().manual_seed(seed)
    return DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=Collate(model.image_processor),
        generator=gen,
    )


# ---------------------------------------------------------------------------
# otimizacao
# ---------------------------------------------------------------------------
def make_optimizer(name: str, params, lr: float, weight_decay: float, device: str):
    import torch

    if name in ("adamw_8bit", "paged_adamw_8bit"):
        if device != "cuda":
            print(f"  (otimizador {name} exige CUDA — usando adamw padrao)")
        else:
            import bitsandbytes as bnb

            cls = bnb.optim.PagedAdamW8bit if name == "paged_adamw_8bit" else bnb.optim.AdamW8bit
            return cls(params, lr=lr, weight_decay=weight_decay)
    elif name != "adamw":
        raise SystemExit(f"otimizador desconhecido '{name}'. Use adamw | adamw_8bit | paged_adamw_8bit")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def configure_trainable(model, sc: Config, quantized: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(False)

    modo = sc.llm_mode
    if modo == "full":
        for p in model.llm.parameters():
            p.requires_grad_(True)
    elif modo == "partial":
        n = int(sc.unfreeze_last_n_layers)
        if n <= 0:
            raise SystemExit("llm_mode=partial exige unfreeze_last_n_layers > 0")
        corpo = model.llm.model
        for camada in corpo.layers[-n:]:
            for p in camada.parameters():
                p.requires_grad_(True)
        for p in corpo.norm.parameters():
            p.requires_grad_(True)
        if sc.train_lm_head:
            for p in model.llm.lm_head.parameters():
                p.requires_grad_(True)
    elif modo == "lora":
        from peft import LoraConfig, get_peft_model

        if quantized:
            from peft import prepare_model_for_kbit_training

            model.llm = prepare_model_for_kbit_training(
                model.llm,
                use_gradient_checkpointing=bool(sc.gradient_checkpointing),
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        model.llm = get_peft_model(
            model.llm,
            LoraConfig(
                r=int(sc.r),
                lora_alpha=int(sc.alpha),
                lora_dropout=float(sc.dropout),
                target_modules=list(sc.target_modules),
                bias=sc.get("lora_bias", "none"),
                task_type="CAUSAL_LM",
            ),
        )
    elif modo != "frozen":
        raise SystemExit(f"llm_mode desconhecido '{modo}'. Use frozen | full | partial | lora")

    if sc.train_projector:
        for p in model.projector.parameters():
            p.requires_grad_(True)
    if sc.train_vision:
        model.train_vision = True
        for p in model.vision.parameters():
            p.requires_grad_(True)

    if sc.gradient_checkpointing:
        model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.llm.config.use_cache = False


# ---------------------------------------------------------------------------
# materializacao sem carregar pesos (variante base)
# ---------------------------------------------------------------------------
def materialize_base(cfg: Config, out_dir: Path) -> dict:
    """Projetor aleatorio (semente fixa) + referencias aos pesos originais — nao carrega o LLM."""
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoImageProcessor

    from .vlm import PROJECTOR_FILE, VLM_CONFIG, base_meta, build_projector, load_tokenizer, resolve_model_ref

    meta = base_meta(cfg)
    vconf = AutoConfig.from_pretrained(resolve_model_ref(cfg.model.vision_id))
    lconf = AutoConfig.from_pretrained(resolve_model_ref(cfg.model.llm_id))
    d_vis = getattr(vconf, "vision_config", vconf).hidden_size
    torch.manual_seed(int(cfg.seed))
    projector = build_projector(cfg.model.projector, d_vis, lconf.hidden_size)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.float().contiguous() for k, v in projector.state_dict().items()}, str(out_dir / PROJECTOR_FILE))
    load_tokenizer(cfg.model.llm_id).save_pretrained(str(out_dir / "tokenizer"))
    AutoImageProcessor.from_pretrained(resolve_model_ref(cfg.model.vision_id)).save_pretrained(str(out_dir / "image_processor"))
    meta.update(
        variant="base",
        llm_weights="base",
        init_from=None,
        salvo_em=dt.datetime.now().isoformat(timespec="seconds"),
    )
    write_json(out_dir / VLM_CONFIG, meta)
    return meta


# ---------------------------------------------------------------------------
# treino
# ---------------------------------------------------------------------------
def validation_loss(model, loader, device: str, amp_dtype, use_amp: bool) -> float | None:
    import torch

    if loader is None or len(loader) == 0:
        return None
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for pixel_values, legendas in loader:
            ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else _nada()
            with ctx:
                loss = model(pixel_values.to(device), legendas).loss
            total += float(loss) * len(legendas)
            n += len(legendas)
    model.train()
    if not model.train_vision:
        model.vision.eval()
    return total / max(n, 1)


def run(cfg: Config, stage: str) -> None:
    if stage not in STAGES:
        raise SystemExit(f"estagio desconhecido '{stage}'. Use um de: {', '.join(STAGES)}")

    import torch
    from transformers import get_cosine_schedule_with_warmup

    from .vlm import (
        assemble,
        base_meta,
        count_params,
        load_projector_state,
        read_meta,
        resolve_dtype,
        save_variant,
    )

    sc = stage_config(cfg, stage)
    set_seed(int(cfg.seed))
    models_dir = resolve_path(cfg, "models_dir")
    out_dir = models_dir / stage
    device = pick_device(cfg.get("device"))
    inicio = time.perf_counter()

    if stage == "base" or (int(sc.epochs) == 0 and not sc.init_from):
        meta = materialize_base(cfg, out_dir)
        write_json(out_dir / "train_summary.json", {"stage": stage, "epochs": 0, "nota": "sem treino", "meta": meta})
        print(f"Variante '{stage}' materializada em {out_dir} (projetor aleatorio, semente {cfg.seed}).")
        return

    # -- precisao --------------------------------------------------------------
    compute_dtype = resolve_dtype(cfg.model.dtype, device)
    use_amp = device == "cuda" and compute_dtype != torch.float32
    if device != "cuda":
        llm_dtype = torch.float32
    elif sc.get("weights_dtype"):
        llm_dtype = resolve_dtype(sc.weights_dtype, device)
    elif sc.llm_mode in ("full", "partial"):
        llm_dtype = torch.float32
    else:
        llm_dtype = compute_dtype
    vision_dtype = torch.float32 if (sc.train_vision or device != "cuda") else compute_dtype

    quant = None
    nota_quant = "nenhuma"
    if sc.quantize_4bit:
        if device == "cuda":
            quant = dict(sc.quant)
            nota_quant = f"{quant.get('quant_type', 'nf4')} 4 bits (bitsandbytes)"
        elif sc.get("fallback_no_cuda"):
            nota_quant = "nenhuma (fallback: sem CUDA)"
            print(f"  AVISO: sem CUDA — '{stage}' vai treinar SEM quantizacao (fallback_no_cuda=true).")
        else:
            raise SystemExit(
                f"'{stage}' usa 4 bits (bitsandbytes), que exige GPU CUDA. Para testar sem GPU use "
                f"--set training.{stage}.fallback_no_cuda=true (o resultado NAO e quantizado de verdade)."
            )

    # -- ponto de partida ------------------------------------------------------
    meta = base_meta(cfg)
    projector_state = None
    if sc.init_from:
        origem = models_dir / sc.init_from
        meta_origem = read_meta(origem)
        if meta_origem.get("adapter"):
            raise SystemExit(f"init_from={sc.init_from} e uma variante com adaptador; use pretrain, base ou finetune.")
        for chave in ("llm_base", "vision_source", "projector", "image_pool", "vision_feature_layer", "prompt", "max_caption_tokens"):
            if chave in meta_origem and meta_origem[chave] != meta.get(chave) and chave not in ("llm_base", "vision_source"):
                print(f"  (usando {chave}={meta_origem[chave]!r} de '{sc.init_from}' em vez do valor da config)")
            meta[chave] = meta_origem.get(chave, meta.get(chave))
        projector_state = load_projector_state(origem)

    print(f"Estagio '{stage}' | dispositivo {device} | LLM {llm_dtype} | amp {compute_dtype if use_amp else 'off'} | quantizacao {nota_quant}")
    print(f"  ponto de partida: {sc.init_from or 'pesos originais'} | llm_mode={sc.llm_mode}")

    model = assemble(
        meta,
        models_dir,
        device,
        llm_dtype=llm_dtype,
        vision_dtype=vision_dtype,
        attn=cfg.model.attn_implementation,
        quant=quant,
        projector_state=projector_state,
        projector_dtype=torch.float32,
        seed=int(cfg.seed),
    )
    configure_trainable(model, sc, quantized=quant is not None)
    total_params, treinaveis = count_params(model)
    print(f"  parametros: {total_params / 1e6:.1f} M total, {treinaveis / 1e6:.2f} M treinaveis ({100 * treinaveis / total_params:.2f}%)")

    # -- dados ---------------------------------------------------------------
    per_image = sc.captions_per_image
    treino = build_samples(cfg, "train", per_image, int(sc.max_train_images))
    valid = build_samples(cfg, "val", per_image)
    if not treino:
        raise SystemExit("Nenhum exemplo de treino. Rode `label` e `split` antes.")
    bs = int(sc.batch_size)
    loader = make_loader(treino, model, bs, True, int(sc.num_workers), int(cfg.seed))
    val_loader = make_loader(valid, model, bs, False, int(sc.num_workers), int(cfg.seed)) if valid else None
    print(f"  exemplos: {len(treino)} treino, {len(valid)} validacao | batch {bs} x acumulacao {sc.grad_accum}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(sc.optim, params, float(sc.lr), float(sc.weight_decay), device)
    accum = max(1, int(sc.grad_accum))
    passos_epoca = math.ceil(len(loader) / accum)
    total_passos = passos_epoca * int(sc.epochs)
    if int(sc.max_steps):
        total_passos = min(total_passos, int(sc.max_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(float(sc.warmup_ratio) * total_passos), max(total_passos, 1))
    todos_fp32 = all(p.dtype == torch.float32 for p in params)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and compute_dtype == torch.float16 and todos_fp32)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # -- metadados de saida ----------------------------------------------------
    meta.update(variant=stage, init_from=sc.init_from, llm_mode=sc.llm_mode, vision_trained=bool(sc.train_vision))
    if sc.llm_mode in ("full", "partial"):
        meta["llm_weights"] = "full"
        meta["llm_base"] = {"kind": "local", "value": f"{stage}/llm"}
    elif sc.llm_mode == "lora":
        meta["llm_weights"] = "adapter"
        meta["adapter"] = True
        meta["adapter_quantized_base"] = quant is not None
    if sc.train_vision:
        meta["vision_source"] = {"kind": "local", "value": f"{stage}/vision"}
    save_dtype = resolve_dtype(sc.get("save_dtype", "bf16"), "cuda") if device == "cuda" else torch.float32

    log_path = out_dir / "train_log.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    model.train()
    model.vision.eval()
    passo = 0
    melhor = math.inf
    historico = []
    ultima_loss = None
    parar = False
    for epoca in range(1, int(sc.epochs) + 1):
        acumulado, n_micro = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for i, (pixel_values, legendas) in enumerate(loader):
            ctx = torch.autocast(device_type="cuda", dtype=compute_dtype) if use_amp else _nada()
            with ctx:
                loss = model(pixel_values.to(device), legendas).loss / accum
            scaler.scale(loss).backward()
            acumulado += float(loss) * accum
            n_micro += 1

            if (i + 1) % accum == 0 or (i + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, float(sc.max_grad_norm))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                passo += 1
                ultima_loss = acumulado / n_micro
                if passo % int(sc.log_every) == 0 or passo == 1:
                    registro = {"epoca": epoca, "passo": passo, "loss": ultima_loss, "lr": scheduler.get_last_lr()[0]}
                    append_jsonl(log_path, registro)
                    print(f"  epoca {epoca} passo {passo}/{total_passos} loss {ultima_loss:.4f} lr {registro['lr']:.2e}")
                acumulado, n_micro = 0.0, 0
                if int(sc.max_steps) and passo >= int(sc.max_steps):
                    parar = True
                    break

        val = validation_loss(model, val_loader, device, compute_dtype, use_amp) if sc.eval_every_epoch else None
        historico.append({"epoca": epoca, "train_loss": ultima_loss, "val_loss": val})
        append_jsonl(log_path, {"epoca": epoca, "fim_epoca": True, "train_loss": ultima_loss, "val_loss": val})
        print(f"  == epoca {epoca}: train {ultima_loss:.4f} | val {val if val is None else round(val, 4)}")

        if sc.save == "best" and val is not None:
            if val < melhor:
                melhor = val
                save_variant(model, out_dir, meta, save_dtype)
                print(f"     melhor validacao ate agora — salvo em {out_dir}")
        if parar:
            break

    if sc.save != "best" or melhor == math.inf:
        save_variant(model, out_dir, meta, save_dtype)

    resumo = {
        "stage": stage,
        "init_from": sc.init_from,
        "llm_mode": sc.llm_mode,
        "quantizacao": nota_quant,
        "dispositivo": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "llm_dtype_treino": str(llm_dtype),
        "amp": str(compute_dtype) if use_amp else None,
        "params_total": total_params,
        "params_treinaveis": treinaveis,
        "exemplos_treino": len(treino),
        "exemplos_validacao": len(valid),
        "passos": passo,
        "melhor_val_loss": None if melhor == math.inf else melhor,
        "historico": historico,
        "vram_pico_gb": (torch.cuda.max_memory_allocated() / 1024**3) if device == "cuda" else None,
        "tempo_s": time.perf_counter() - inicio,
        "hiperparametros": sc.to_dict(),
        "concluido_em": dt.datetime.now().isoformat(timespec="seconds"),
    }
    write_json(out_dir / "train_summary.json", resumo)
    print(f"Pronto: '{stage}' em {resumo['tempo_s'] / 60:.1f} min -> {out_dir}")
