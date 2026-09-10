"""Gera rascunhos de legendas PT-BR com a API da Anthropic (Claude).

Os rascunhos NAO sao o ground truth: eles entram no app de revisao
(`python -m streamlit run src/review_app.py`) e viram referencia so depois da
revisao humana.

Uso:
    python -m src.draft_captions                    # todas as imagens ainda sem rascunho
    python -m src.draft_captions --limit 5          # teste barato
    python -m src.draft_captions --model claude-sonnet-5 --n-captions 3

Requer ANTHROPIC_API_KEY no ambiente (ou perfil ativo via `ant auth login`).
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

from .common import DRAFTS_JSONL, IMAGES_DIR, append_jsonl, index_by, list_images, read_jsonl
from .prompts import SYSTEM_PROMPT, USER_PROMPT

# Fallback do lado do servidor: se um classificador recusar a requisicao, a API
# reroteia para outro modelo em vez de devolver stop_reason="refusal".
FALLBACK_BETA = "server-side-fallback-2026-07-01"


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
    media = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    return media.get(path.suffix.lower(), "image/jpeg"), base64.standard_b64encode(path.read_bytes()).decode()


def call_model(client, args, path: Path) -> dict:
    media_type, data = encode_image(path)
    kwargs = dict(
        model=args.model,
        max_tokens=2000,
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
        output_config={"format": schema(args.n_captions)},
    )
    if args.fallbacks:
        response = client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
    else:
        response = client.messages.create(**kwargs)

    if response.stop_reason == "refusal":
        detalhe = getattr(response, "stop_details", None)
        raise RuntimeError(f"requisicao recusada pelo modelo ({detalhe})")

    textos = [b.text for b in response.content if b.type == "text"]
    if not textos:
        raise RuntimeError("resposta sem bloco de texto")
    return json.loads(textos[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=IMAGES_DIR)
    parser.add_argument("--out", type=Path, default=DRAFTS_JSONL)
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--n-captions", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="processa no maximo N imagens (0 = todas)")
    parser.add_argument("--redo", action="store_true", help="regera mesmo o que ja tem rascunho")
    parser.add_argument("--no-fallbacks", dest="fallbacks", action="store_false")
    parser.set_defaults(fallbacks=True)
    args = parser.parse_args()

    import anthropic

    client = anthropic.Anthropic()

    ja_feitos = set() if args.redo else set(index_by(read_jsonl(args.out)))
    pendentes = [p for p in list_images(args.images) if p.stem not in ja_feitos]
    if args.limit:
        pendentes = pendentes[: args.limit]
    if not pendentes:
        print("Nada a fazer — todas as imagens ja tem rascunho.")
        return

    print(f"{len(pendentes)} imagens para rascunhar com {args.model}.")
    erros = 0
    for i, path in enumerate(pendentes, 1):
        try:
            payload = call_model(client, args, path)
        except Exception as exc:
            print(f"[{i}/{len(pendentes)}] {path.stem}: ERRO — {exc}")
            erros += 1
            time.sleep(2)
            continue
        append_jsonl(
            args.out,
            {
                "image_id": path.stem,
                "modelo": args.model,
                "legendas": payload["legendas"],
                "objetos": payload["objetos"],
                "fora_de_dominio": payload["fora_de_dominio"],
                "observacao": payload["observacao"],
                "revisado": False,
            },
        )
        marca = " [FORA DE DOMINIO]" if payload["fora_de_dominio"] else ""
        print(f"[{i}/{len(pendentes)}] {path.stem}{marca}: {payload['legendas'][0]}")

    print(f"\nPronto. Rascunhos em {args.out}. Erros: {erros}.")
    print("Proximo passo: python -m streamlit run src/review_app.py")


if __name__ == "__main__":
    main()
