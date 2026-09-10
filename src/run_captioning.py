"""Roda um modelo de captioning sobre a pasta de imagens e grava as predicoes.

Uso:
    python -m src.run_captioning --model blip --translate
    python -m src.run_captioning --model florence2 --translate --limit 10
    python -m src.run_captioning --model qwen25vl          # ja gera PT-BR
    python -m src.run_captioning --model claude            # teto de qualidade (API)

Saida (results/preds_<modelo>.jsonl), uma linha por imagem:
    {"image_id", "modelo", "legenda", "legenda_en", "traduzido_por", "lang"}

Modelos em ingles sem --translate geram legenda_en e deixam `legenda` vazia — a
avaliacao PT-BR precisa da traducao, entao praticamente sempre use --translate.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image

from .captioners import REGISTRY, get_captioner
from .common import IMAGES_DIR, RESULTS_DIR, append_jsonl, index_by, list_images, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(REGISTRY))
    parser.add_argument("--images", type=Path, default=IMAGES_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda | cpu (padrao: detecta)")
    parser.add_argument("--translate", action="store_true", help="traduz EN->PT nos modelos em ingles")
    parser.add_argument("--mt-model", default=None, help="checkpoint de traducao (padrao: opus-mt-tc-big-en-pt)")
    parser.add_argument("--mt-prefix", default="", help='prefixo de variante do OPUS, ex: ">>por<<"')
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()

    out = args.out or RESULTS_DIR / f"preds_{args.model}.jsonl"

    kwargs = {} if args.model == "claude" else {"device": args.device}
    captioner = get_captioner(args.model, **kwargs)

    tradutor = None
    if captioner.lang == "en" and args.translate:
        from .translate import DEFAULT_MT, Translator

        tradutor = Translator(args.mt_model or DEFAULT_MT, device=args.device, prefix=args.mt_prefix)

    ja_feitos = set() if args.redo else set(index_by(read_jsonl(out)))
    imagens = [p for p in list_images(args.images) if p.stem not in ja_feitos]
    if args.limit:
        imagens = imagens[: args.limit]
    if not imagens:
        print("Nada a fazer — todas as imagens ja tem predicao para este modelo.")
        return

    print(f"{captioner.name} ({captioner.lang}) sobre {len(imagens)} imagens -> {out}")
    inicio = time.perf_counter()
    erros = 0
    for i, path in enumerate(imagens, 1):
        try:
            with Image.open(path) as img:
                bruta = captioner.caption(img.convert("RGB"))
        except Exception as exc:
            print(f"[{i}/{len(imagens)}] {path.stem}: ERRO — {exc}")
            erros += 1
            continue

        if captioner.lang == "pt":
            legenda, legenda_en, mt = bruta, "", ""
        elif tradutor is not None:
            legenda, legenda_en, mt = tradutor([bruta])[0], bruta, tradutor.model.name_or_path
        else:
            legenda, legenda_en, mt = "", bruta, ""

        append_jsonl(
            out,
            {
                "image_id": path.stem,
                "modelo": captioner.name,
                "lang": captioner.lang,
                "legenda": legenda,
                "legenda_en": legenda_en,
                "traduzido_por": mt,
            },
        )
        print(f"[{i}/{len(imagens)}] {path.stem}: {legenda or legenda_en}")

    elapsed = time.perf_counter() - inicio
    feitas = len(imagens) - erros
    media = f"{elapsed / feitas:.2f}s/img" if feitas else "n/a"
    print(f"\n{feitas} legendas em {elapsed:.1f}s ({media}). Erros: {erros}. Arquivo: {out}")


if __name__ == "__main__":
    main()
