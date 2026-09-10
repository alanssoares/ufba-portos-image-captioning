"""Avalia legendas geradas contra as referencias humanas (PT-BR).

Uso:
    python -m src.evaluate --preds results/preds_blip.jsonl
    python -m src.evaluate --preds results/preds_*.jsonl --out results/comparativo.md
    python -m src.evaluate --preds results/preds_blip.jsonl --no-clipscore --no-bertscore

Imagens marcadas como `descartada` no ground truth sao ignoradas, assim como
predicoes sem legenda em portugues (modelo em ingles rodado sem --translate).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from .common import CAPTIONS_JSONL, IMAGES_DIR, RESULTS_DIR, index_by, read_jsonl
from .metrics_ptbr import bertscore_pt, clipscore_pt, diversidade, metricas_ngrama

ORDEM = [
    "N",
    "BLEU-1",
    "BLEU-4",
    "ROUGE-L",
    "CIDEr",
    "BERTScore-F1",
    "CLIPScore",
    "RefCLIPScore",
    "Distinct-1",
    "Distinct-2",
    "Tam.medio",
]


def carregar_refs(path: Path) -> dict[str, list[str]]:
    refs = {}
    for row in read_jsonl(path):
        if row.get("descartada") or not row.get("legendas"):
            continue
        refs[row["image_id"]] = row["legendas"]
    return refs


def avaliar(preds_path: Path, refs: dict[str, list[str]], args) -> dict:
    linhas = index_by(read_jsonl(preds_path))
    hyps = {k: v["legenda"].strip() for k, v in linhas.items() if k in refs and v.get("legenda", "").strip()}

    sem_ref = [k for k in linhas if k not in refs]
    sem_pt = [k for k, v in linhas.items() if k in refs and not v.get("legenda", "").strip()]
    if not hyps:
        raise SystemExit(
            f"{preds_path.name}: nenhuma predicao avaliavel. "
            f"{len(sem_pt)} sem legenda PT (faltou --translate?), {len(sem_ref)} sem referencia."
        )

    modelo = next(iter(linhas.values())).get("modelo", preds_path.stem)
    resultado = {"modelo": modelo, "arquivo": preds_path.name, "N": len(hyps)}
    resultado.update(metricas_ngrama(refs, hyps))
    resultado.update(diversidade(hyps))

    if args.bertscore:
        resultado.update(bertscore_pt(refs, hyps, device=args.device))
    if args.clipscore:
        imagens = {k: args.images / f"{k}.jpg" for k in hyps}
        imagens = {k: p for k, p in imagens.items() if p.exists()}
        if imagens:
            resultado.update(clipscore_pt(imagens, hyps, refs, device=args.device))
        else:
            print(f"  (CLIPScore pulado: nenhuma imagem encontrada em {args.images})")

    if sem_pt or sem_ref:
        print(f"  ignoradas: {len(sem_pt)} sem legenda PT, {len(sem_ref)} sem referencia humana")
    return resultado


def tabela_markdown(resultados: list[dict]) -> str:
    colunas = ["modelo"] + [c for c in ORDEM if any(c in r for r in resultados)]
    linhas = ["| " + " | ".join(colunas) + " |", "|" + "|".join(["---"] * len(colunas)) + "|"]
    for r in resultados:
        celulas = []
        for c in colunas:
            v = r.get(c, "")
            celulas.append(v if isinstance(v, str) else (f"{v:.0f}" if c == "N" else f"{v:.4f}"))
        linhas.append("| " + " | ".join(celulas) + " |")
    return "\n".join(linhas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preds", nargs="+", required=True, help="um ou mais results/preds_*.jsonl")
    parser.add_argument("--refs", type=Path, default=CAPTIONS_JSONL)
    parser.add_argument("--images", type=Path, default=IMAGES_DIR)
    parser.add_argument("--out", type=Path, default=None, help="tabela markdown de saida")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-bertscore", dest="bertscore", action="store_false")
    parser.add_argument("--no-clipscore", dest="clipscore", action="store_false")
    parser.set_defaults(bertscore=True, clipscore=True)
    args = parser.parse_args()

    refs = carregar_refs(args.refs)
    if not refs:
        raise SystemExit(f"Nenhuma referencia em {args.refs}. Revise as legendas antes de avaliar.")
    print(f"{len(refs)} imagens com referencia humana ({sum(len(v) for v in refs.values())} legendas).\n")

    caminhos = sorted({Path(p) for padrao in args.preds for p in glob.glob(padrao)})
    if not caminhos:
        raise SystemExit(f"Nenhum arquivo de predicao casou com {args.preds}")

    resultados = []
    for caminho in caminhos:
        print(f"Avaliando {caminho.name}...")
        resultado = avaliar(caminho, refs, args)
        resultados.append(resultado)
        destino = RESULTS_DIR / f"metrics_{caminho.stem.replace('preds_', '')}.json"
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")

    tabela = tabela_markdown(resultados)
    print("\n" + tabela)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(tabela + "\n", encoding="utf-8")
        print(f"\nTabela salva em {args.out}")


if __name__ == "__main__":
    main()
