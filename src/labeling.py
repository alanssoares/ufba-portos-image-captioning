"""Rotulagem: legendas de referencia PT-BR geradas pelo Claude (padrao: claude-opus-5-5).

As legendas daqui sao a REFERENCIA de todo o experimento: alimentam o pre-treino e
os fine-tunings (splits de treino/validacao) e sao o gabarito da avaliacao (split
de teste). Por isso o rotulador nunca aparece como linha da tabela comparativa —
seria medir o modelo contra ele mesmo.

Uso:
    python -m src label                          # todas as imagens ainda sem rotulo
    python -m src label --set labeling.limit=5   # teste barato
    python -m src label --set labeling.model=claude-sonnet-5

Requer a variavel de ambiente indicada em labeling.api_key_env (ANTHROPIC_API_KEY).
Retomavel: imagens ja rotuladas sao puladas (use --redo para regerar).

Cada linha grava `prompt_versao` (hash de src/prompts.py): todas as referencias do
experimento devem ter a mesma versao. Se o prompt mudar, o comando avisa quais linhas
estao em versao antiga — regere-as com --redo.
"""
from __future__ import annotations

import base64
import datetime as dt
import json
import os
import time
from pathlib import Path

from .common import append_jsonl, index_by, list_images, read_jsonl
from .config import Config, resolve_path
from .prompts import PROMPT_VERSAO, SYSTEM_PROMPT, USER_PROMPT, VERSOES_COMPATIVEIS

# Fallback do lado do servidor: se um classificador recusar a requisicao, a API
# reroteia para outro modelo em vez de devolver stop_reason="refusal".
FALLBACK_BETA = "server-side-fallback-2026-07-01"

MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def schema(n_captions: int) -> dict:
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "legendas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": n_captions,
                    "maxItems": n_captions,
                },
                "objetos": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "elementos portuarios visiveis, em minusculas",
                },
                "fora_de_dominio": {"type": "boolean"},
                "observacao": {"type": "string"},
            },
            "required": ["legendas", "objetos", "fora_de_dominio", "observacao"],
            "additionalProperties": False,
        },
    }


def encode_image(path: Path) -> tuple[str, str]:
    return MEDIA_TYPES.get(path.suffix.lower(), "image/jpeg"), base64.standard_b64encode(path.read_bytes()).decode()


class ClaudeLabeler:
    def __init__(self, cfg: Config) -> None:
        import anthropic

        lab = cfg.labeling
        chave = os.environ.get(lab.api_key_env)
        if not chave:
            raise SystemExit(f"Defina a variavel de ambiente {lab.api_key_env} com a chave da Anthropic.")
        self.client = anthropic.Anthropic(api_key=chave)
        self.model = lab.model
        self.n_captions = int(lab.n_captions)
        self.max_tokens = int(lab.max_tokens)
        self.fallbacks = bool(lab.fallbacks)

    def __call__(self, path: Path) -> dict:
        media_type, data = encode_image(path)
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                        {"type": "text", "text": USER_PROMPT},
                    ],
                }
            ],
            output_config={"format": schema(self.n_captions)},
        )
        if self.fallbacks:
            response = self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
        else:
            response = self.client.messages.create(**kwargs)

        if response.stop_reason == "refusal":
            raise RuntimeError(f"requisicao recusada pelo modelo ({getattr(response, 'stop_details', None)})")
        textos = [b.text for b in response.content if b.type == "text"]
        if not textos:
            raise RuntimeError("resposta sem bloco de texto")
        payload = json.loads(textos[-1])
        payload["modelo"] = getattr(response, "model", self.model)
        return payload


def versoes_desatualizadas(rows: list[dict]) -> list[str]:
    """image_ids rotulados com versao incompativel do prompt (ou sem versao registrada).

    Versoes so ortograficamente diferentes (prompts.VERSOES_EQUIVALENTES) sao compativeis.
    """
    return [r["image_id"] for r in rows if r.get("prompt_versao") not in VERSOES_COMPATIVEIS]


def avisar_versoes(rows: list[dict]) -> None:
    antigas = versoes_desatualizadas(rows)
    if antigas:
        amostra = ", ".join(antigas[:10]) + (" ..." if len(antigas) > 10 else "")
        print(f"ATENCAO: {len(antigas)} de {len(rows)} rotulos foram gerados com outra versao do prompt "
              f"(atual: {PROMPT_VERSAO}): {amostra}. Regere com `python -m src label --redo` "
              "para manter as referencias homogeneas.")


def run(cfg: Config, redo: bool = False) -> None:
    images_dir = resolve_path(cfg, "images_dir")
    out = resolve_path(cfg, "labels_jsonl")
    lab = cfg.labeling

    if not redo:
        avisar_versoes(read_jsonl(out))
    ja_feitos = set() if redo else set(index_by(read_jsonl(out)))
    pendentes = [p for p in list_images(images_dir) if p.stem not in ja_feitos]
    if lab.limit:
        pendentes = pendentes[: int(lab.limit)]
    if not pendentes:
        print(f"Nada a rotular em {images_dir} — todas as imagens ja tem legenda (ou a pasta esta vazia).")
        return

    labeler = ClaudeLabeler(cfg)
    print(f"{len(pendentes)} imagens para rotular com {lab.model} -> {out}")
    erros = 0
    for i, path in enumerate(pendentes, 1):
        payload = None
        for tentativa in range(1, int(lab.max_retries) + 1):
            try:
                payload = labeler(path)
                break
            except Exception as exc:  # rede, rate limit, recusa...
                print(f"[{i}/{len(pendentes)}] {path.stem}: tentativa {tentativa} falhou — {exc}")
                time.sleep(float(lab.retry_wait_s) * tentativa)
        if payload is None:
            erros += 1
            continue
        append_jsonl(
            out,
            {
                "image_id": path.stem,
                "modelo": payload["modelo"],
                "legendas": [t.strip() for t in payload["legendas"] if t.strip()],
                "objetos": payload["objetos"],
                "fora_de_dominio": payload["fora_de_dominio"],
                "observacao": payload["observacao"],
                "prompt_versao": PROMPT_VERSAO,
                "rotulado_em": dt.datetime.now().isoformat(timespec="seconds"),
            },
        )
        marca = " [FORA DE DOMINIO]" if payload["fora_de_dominio"] else ""
        print(f"[{i}/{len(pendentes)}] {path.stem}{marca}: {payload['legendas'][0]}")

    print(f"\nPronto. Rotulos em {out}. Erros: {erros}.")
