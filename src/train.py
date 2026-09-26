"""Treino das variantes: base, pretrain, finetune, lora, qlora (rode no Colab).

Uso:
    python -m src train --stage base        # so registra o modelo original (nada e treinado)
    python -m src train --stage pretrain    # pre-treino continuado de dominio: so o conector visao->LLM
    python -m src train --stage finetune    # fine-tuning completo: conector + LLM inteiro
    python -m src train --stage lora        # LoRA no LLM + conector
    python -m src train --stage qlora       # LLM em 4 bits (NF4) + LoRA + conector

    # ponto de partida: pre-treinado (padrao) ou base
    python -m src train --stage lora --set training.lora.init_from=base

Cada estagio usa training.common + training.<estagio> da config. A saida vai para
outputs/models/<estagio>/ (ver src/vlm.py) junto com train_summary.json (parametros
treinaveis, memoria de pico, tempo, losses) e train_log.jsonl.
"""
from __future__ import annotations

import datetime as dt
import math
import time
from contextlib import nullcontext as _nada
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
    """Abre as imagens. Classe (e nao closure) para funcionar com num_workers > 0 no Windows."""

    def __call__(self, batch):
        from PIL import Image

        imagens = []
        for caminho, _ in batch:
            with Image.open(caminho) as img:
                imagens.append(img.convert("RGB"))
        return imagens, [legenda for _, legenda in batch]


def make_loader(samples, batch_size: int, shuffle: bool, num_workers: int, seed: int):
    import torch
    from torch.utils.data import DataLoader

    gen = torch.Generator().manual_seed(seed)
    return DataLoader(samples, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=Collate(), generator=gen)


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


def _casa(nome: str, padroes) -> bool:
    return any(p in nome for p in padroes)


def configure_trainable(model, sc: Config, quantized: bool):
    """Congela tudo e libera o que o estagio treina. Devolve o modelo (pode virar PeftModel)."""
    from .vlm import language_model

    for p in model.parameters():
        p.requires_grad_(False)

    modo = sc.llm_mode
    if modo == "full":
        for p in language_model(model).parameters():
            p.requires_grad_(True)
    elif modo == "partial":
        n = int(sc.unfreeze_last_n_layers)
        if n <= 0:
            raise SystemExit("llm_mode=partial exige unfreeze_last_n_layers > 0")
        lm = language_model(model)
        for camada in lm.layers[-n:]:
            for p in camada.parameters():
                p.requires_grad_(True)
        for p in lm.norm.parameters():
            p.requires_grad_(True)
    elif modo == "lora":
        from peft import LoraConfig, get_peft_model

        if quantized:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=bool(sc.gradient_checkpointing),
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        alvos = sc.target_modules
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(sc.r),
                lora_alpha=int(sc.alpha),
                lora_dropout=float(sc.dropout),
                target_modules=alvos if isinstance(alvos, str) else list(alvos),
                bias=sc.get("lora_bias", "none"),
                task_type="CAUSAL_LM",
            ),
        )
    elif modo != "frozen":
        raise SystemExit(f"llm_mode desconhecido '{modo}'. Use frozen | full | partial | lora")

    # conector visao->LLM e, opcionalmente, o encoder de visao inteiro
    for nome, p in model.named_parameters():
        if "lora_" in nome:
            continue
        if sc.train_connector and _casa(nome, sc.connector_modules):
            p.requires_grad_(True)
        if sc.train_vision and ".visual." in f".{nome}":
            p.requires_grad_(True)

    if sc.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.config.use_cache = False
    return model


def trained_delta_keys(model) -> list[str]:
    """Nomes (no modelo original, sem prefixo PEFT) dos pesos treinaveis que NAO sao LoRA."""
    from .vlm import unwrap

    base = unwrap(model)
    return [n for n, p in base.named_parameters() if p.requires_grad and "lora_" not in n]


# ---------------------------------------------------------------------------
# treino
# ---------------------------------------------------------------------------
def validation_loss(cap, loader, device: str, amp_dtype, use_amp: bool) -> float | None:
    import torch

    if loader is None or len(loader) == 0:
        return None
    cap.model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for imagens, legendas in loader:
            ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else _nada()
            with ctx:
                loss = cap.loss(imagens, legendas, device)
            total += float(loss) * len(legendas)
            n += len(legendas)
    cap.model.train()
    return total / max(n, 1)


def run(cfg: Config, stage: str) -> None:
    if stage not in STAGES:
        raise SystemExit(f"estagio desconhecido '{stage}'. Use um de: {', '.join(STAGES)}")

    from .transfer import splits_sha1
    from .vlm import base_meta, write_meta

    sc = stage_config(cfg, stage)
    set_seed(int(cfg.seed))
    models_dir = resolve_path(cfg, "models_dir")
    out_dir = models_dir / stage
    fingerprint = splits_sha1(resolve_path(cfg, "splits_json"))

    if stage == "base" or (int(sc.epochs) == 0 and not sc.init_from):
        meta = {**base_meta(cfg), "variant": stage, "weights": "base", "init_from": None, "splits_sha1": fingerprint}
        write_meta(out_dir, meta)
        write_json(out_dir / "train_summary.json", {"stage": stage, "epochs": 0, "nota": "modelo original, sem treino", "meta": meta})
        print(f"Variante '{stage}' registrada em {out_dir}: {cfg.model.model_id} sem treino.")
        return

    import torch
    from transformers import get_cosine_schedule_with_warmup

    from .vlm import Captioner, apply_delta, count_params, load_model, load_processor, read_meta, ref_path, resolve_dtype, save_variant

    device = pick_device(cfg.get("device"))
    inicio = time.perf_counter()

    # -- precisao --------------------------------------------------------------
    compute_dtype = resolve_dtype(cfg.model.dtype, device)
    use_amp = device == "cuda" and compute_dtype != torch.float32
    if device != "cuda":
        model_dtype = torch.float32
    elif sc.get("weights_dtype"):
        model_dtype = resolve_dtype(sc.weights_dtype, device)
    elif sc.llm_mode in ("full", "partial"):
        model_dtype = torch.float32
    else:
        model_dtype = compute_dtype

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
                f"'{stage}' usa 4 bits (bitsandbytes), que exige GPU CUDA. Rode no Colab "
                f"(ou --set training.{stage}.fallback_no_cuda=true so para teste)."
            )

    # -- ponto de partida ------------------------------------------------------
    meta = base_meta(cfg)
    herdadas: list[str] = []
    origem = None
    if sc.init_from:
        origem = models_dir / sc.init_from
        meta_origem = read_meta(origem)
        if meta_origem.get("adapter"):
            raise SystemExit(f"init_from={sc.init_from} tem adaptador LoRA; use base, pretrain ou finetune.")
        for chave in ("base_ref", "instruction", "system_prompt", "response_marker", "response_suffix", "image_max_side"):
            if chave in meta_origem:
                if chave != "base_ref" and meta_origem[chave] != meta.get(chave):
                    print(f"  (usando {chave}={meta_origem[chave]!r} de '{sc.init_from}' em vez do valor da config)")
                meta[chave] = meta_origem[chave]

    base_ref = ref_path(meta["base_ref"], models_dir)
    print(f"Estagio '{stage}' | {device} | pesos {model_dtype} | amp {compute_dtype if use_amp else 'off'} | quantizacao {nota_quant}")
    print(f"  ponto de partida: {sc.init_from or 'modelo original'} ({base_ref}) | llm_mode={sc.llm_mode}")

    model = load_model(base_ref, model_dtype, device, cfg.model.attn_implementation, quant)
    if origem is not None:
        herdadas = apply_delta(model, origem)
    processor = load_processor(str(origem / "processor") if origem is not None and (origem / "processor").exists() else base_ref)

    model = configure_trainable(model, sc, quantized=quant is not None)
    # pesos treinaveis em fp32 (estabilidade + GradScaler), exceto se o usuario pediu bf16 explicito
    if device == "cuda" and sc.get("weights_dtype") != "bf16":
        for p in model.parameters():
            if p.requires_grad and p.dtype in (torch.float16, torch.bfloat16):
                p.data = p.data.float()
    cap = Captioner(model, processor, meta)
    total_params, treinaveis = count_params(model)
    print(f"  parametros: {total_params / 1e6:.1f} M total, {treinaveis / 1e6:.2f} M treinaveis ({100 * treinaveis / total_params:.2f}%)")
    if treinaveis == 0:
        raise SystemExit("Nada para treinar — confira llm_mode, train_connector e connector_modules.")

    # -- dados ---------------------------------------------------------------
    treino = build_samples(cfg, "train", sc.captions_per_image, int(sc.max_train_images))
    valid = build_samples(cfg, "val", sc.captions_per_image)
    if not treino:
        raise SystemExit("Nenhum exemplo de treino. Rode `label` e `split` antes (e confira data/images).")
    bs = int(sc.batch_size)
    loader = make_loader(treino, bs, True, int(sc.num_workers), int(cfg.seed))
    val_loader = make_loader(valid, bs, False, int(sc.num_workers), int(cfg.seed)) if valid else None
    print(f"  exemplos: {len(treino)} treino, {len(valid)} validacao | batch {bs} x acumulacao {sc.grad_accum}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(sc.optim, params, float(sc.lr), float(sc.weight_decay), device)
    accum = max(1, int(sc.grad_accum))
    total_passos = math.ceil(len(loader) / accum) * int(sc.epochs)
    if int(sc.max_steps):
        total_passos = min(total_passos, int(sc.max_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(float(sc.warmup_ratio) * total_passos), max(total_passos, 1))
    todos_fp32 = all(p.dtype == torch.float32 for p in params)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and compute_dtype == torch.float16 and todos_fp32)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # -- metadados de saida ----------------------------------------------------
    meta.update(variant=stage, init_from=sc.init_from, llm_mode=sc.llm_mode, splits_sha1=fingerprint)
    if sc.llm_mode in ("full", "partial"):
        meta["weights"] = "full"
        meta["base_ref"] = {"kind": "local", "value": f"{stage}/model"}
        delta_keys: list[str] = []
    else:
        meta["weights"] = "delta"
        meta["adapter"] = sc.llm_mode == "lora"
        meta["adapter_quantized_base"] = quant is not None
        delta_keys = sorted(set(herdadas) | set(trained_delta_keys(model)))
    save_dtype = resolve_dtype(sc.get("save_dtype", "bf16"), "cuda") if device == "cuda" else torch.float32

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    log_path.write_text("", encoding="utf-8")

    model.train()
    passo = 0
    melhor = math.inf
    historico = []
    ultima_loss = None
    parar = False
    for epoca in range(1, int(sc.epochs) + 1):
        acumulado, n_micro = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for i, (imagens, legendas) in enumerate(loader):
            ctx = torch.autocast(device_type="cuda", dtype=compute_dtype) if use_amp else _nada()
            with ctx:
                loss = cap.loss(imagens, legendas, device) / accum
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

        val = validation_loss(cap, val_loader, device, compute_dtype, use_amp) if sc.eval_every_epoch else None
        historico.append({"epoca": epoca, "train_loss": ultima_loss, "val_loss": val})
        append_jsonl(log_path, {"epoca": epoca, "fim_epoca": True, "train_loss": ultima_loss, "val_loss": val})
        print(f"  == epoca {epoca}: train {ultima_loss:.4f} | val {val if val is None else round(val, 4)}")

        if sc.save == "best" and val is not None and val < melhor:
            melhor = val
            save_variant(cap, out_dir, meta, delta_keys, save_dtype)
            print(f"     melhor validacao ate agora — salvo em {out_dir}")
        if parar:
            break

    if sc.save != "best" or melhor == math.inf:
        save_variant(cap, out_dir, meta, delta_keys, save_dtype)

    resumo = {
        "stage": stage,
        "modelo": cfg.model.model_id,
        "init_from": sc.init_from,
        "llm_mode": sc.llm_mode,
        "quantizacao": nota_quant,
        "dispositivo": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "pesos_dtype_treino": str(model_dtype),
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
