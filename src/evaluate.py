"""Compara as variantes contra as legendas de referencia (Claude) no split de teste.

Uso:
    python -m src evaluate
    python -m src evaluate --set evaluation.bertscore=false --set evaluation.clipscore=false

    # avaliar arquivos avulsos (ex: data/sample) sem config de split:
    python -m src evaluate --preds "data/sample/preds_*.jsonl" --refs data/sample/refs.jsonl

Saidas em results/:
    comparativo.md    tabela de metricas + custo de treino + exemplos qualitativos
    comparativo.csv   mesma tabela, para planilha/graficos
    metrics_<variante>.json
"""
from __future__ import annotations

import csv
import glob
from pathlib import Path

from .common import find_image, index_by, read_json, read_jsonl, write_json
from .config import Config, resolve_path
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

DESCRICAO = {
    "base": "Qwen3-VL-2B original, zero-shot (sem treino)",
    "pretrain": "pre-treino continuado de dominio (so o conector visao->LLM)",
    "finetune": "fine-tuning completo (conector + LLM)",
    "lora": "LoRA no LLM + conector",
    "qlora": "QLoRA (LLM 4 bits) + conector",
    "gold": "VLM grande via API (referencia externa)",
}


def refs_from_labels_file(path: Path) -> dict[str, list[str]]:
    refs = {}
    for row in read_jsonl(path):
        if row.get("descartada") or row.get("fora_de_dominio") or not row.get("legendas"):
            continue
        refs[row["image_id"]] = row["legendas"]
    return refs


def avaliar(preds_path: Path, refs: dict[str, list[str]], images_dir: Path | None, ev: Config | dict) -> dict:
    linhas = index_by(read_jsonl(preds_path))
    hyps = {k: v["legenda"].strip() for k, v in linhas.items() if k in refs and v.get("legenda", "").strip()}
    nome = preds_path.stem.replace("preds_", "")
    faltando = len([k for k in refs if k not in hyps])
    if not hyps:
        print(f"  {preds_path.name}: nenhuma predicao avaliavel — pulando.")
        return {}

    primeira = next(iter(linhas.values()))
    resultado = {"variante": primeira.get("variante", nome), "modelo": primeira.get("modelo", nome), "N": len(hyps)}
    resultado.update(metricas_ngrama(refs, hyps))
    resultado.update(diversidade(hyps))
    tempos = [float(v["segundos"]) for v in linhas.values() if "segundos" in v]
    if tempos:
        resultado["s/img"] = sum(tempos) / len(tempos)

    if ev.get("bertscore"):
        resultado.update(bertscore_pt(refs, hyps, device=ev.get("device")))
    if ev.get("clipscore") and images_dir is not None:
        imagens = {k: find_image(images_dir, k) for k in hyps}
        imagens = {k: p for k, p in imagens.items() if p is not None}
        if imagens:
            resultado.update(clipscore_pt(imagens, hyps, refs, device=ev.get("device")))
        else:
            print(f"  (CLIPScore pulado: nenhuma imagem encontrada em {images_dir})")
    if faltando:
        print(f"  {nome}: {faltando} imagens do teste sem predicao")
    return resultado


def _fmt(c: str, v) -> str:
    if isinstance(v, str):
        return v
    if v is None:
        return "—"
    return f"{v:.0f}" if c == "N" else f"{v:.4f}"


def tabela_markdown(resultados: list[dict]) -> str:
    colunas = ["variante"] + [c for c in ORDEM if any(c in r for r in resultados)]
    linhas = ["| " + " | ".join(colunas) + " |", "|" + "|".join(["---"] * len(colunas)) + "|"]
    for r in resultados:
        linhas.append("| " + " | ".join(_fmt(c, r.get(c, "")) for c in colunas) + " |")
    return "\n".join(linhas)


def _tamanho_mb(pasta: Path) -> float:
    return sum(p.stat().st_size for p in pasta.rglob("*") if p.is_file()) / 1024**2 if pasta.exists() else 0.0


def tabela_custos(models_dir: Path, variantes: list[str]) -> str:
    linhas = [
        "| variante | ponto de partida | quantizacao | params treinaveis | % | VRAM pico (GB) | tempo (min) | melhor val loss | disco (MB) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for v in variantes:
        if v == "gold":
            continue
        s = read_json(models_dir / v / "train_summary.json", {}) or {}
        if not s:
            continue
        total = s.get("params_total") or 0
        tr = s.get("params_treinaveis") or 0
        linhas.append(
            "| "
            + " | ".join(
                [
                    v,
                    str(s.get("init_from") or "—"),
                    str(s.get("quantizacao") or "—"),
                    f"{tr / 1e6:.2f} M" if tr else "0",
                    f"{100 * tr / total:.2f}" if total else "—",
                    _fmt("x", s.get("vram_pico_gb")),
                    _fmt("x", (s.get("tempo_s") or 0) / 60) if s.get("tempo_s") else "—",
                    _fmt("x", s.get("melhor_val_loss")),
                    f"{_tamanho_mb(models_dir / v):.0f}",
                ]
            )
            + " |"
        )
    return "\n".join(linhas)


def exemplos(refs: dict[str, list[str]], preds: dict[str, dict], n: int) -> str:
    blocos = []
    for image_id in sorted(refs)[:n]:
        linhas = [f"### {image_id}", "", f"- **referencia (Claude):** {refs[image_id][0]}"]
        for nome, linhas_pred in preds.items():
            if image_id in linhas_pred:
                linhas.append(f"- **{nome}:** {linhas_pred[image_id].get('legenda', '')}")
        blocos.append("\n".join(linhas))
    return "\n\n".join(blocos)


def run(cfg: Config, preds_glob: list[str] | None = None, refs_path: Path | None = None) -> list[dict]:
    from .splits import references

    results_dir = resolve_path(cfg, "results_dir")
    ev = cfg.evaluation
    images_dir = resolve_path(cfg, "images_dir")

    if refs_path is not None:
        refs = refs_from_labels_file(refs_path)
        origem = str(refs_path)
    else:
        refs = references(cfg, ev.split)
        origem = f"{resolve_path(cfg, 'labels_jsonl').name} (split {ev.split})"
    if not refs:
        raise SystemExit("Nenhuma referencia para avaliar. Rode label + split antes.")

    if preds_glob:
        caminhos = sorted({Path(p) for padrao in preds_glob for p in glob.glob(padrao)})
    else:
        caminhos = [results_dir / f"preds_{v}.jsonl" for v in cfg.variants if (results_dir / f"preds_{v}.jsonl").exists()]
    if not caminhos:
        raise SystemExit(f"Nenhum arquivo de predicao encontrado em {results_dir}. Rode `predict` antes.")

    print(f"{len(refs)} imagens com referencia ({sum(len(v) for v in refs.values())} legendas) de {origem}.\n")
    resultados, preds = [], {}
    for caminho in caminhos:
        print(f"Avaliando {caminho.name}...")
        r = avaliar(caminho, refs, images_dir, ev)
        if not r:
            continue
        resultados.append(r)
        preds[r["variante"]] = index_by(read_jsonl(caminho))
        write_json(results_dir / f"metrics_{r['variante']}.json", r)

    if not resultados:
        raise SystemExit("Nenhuma predicao avaliavel.")

    tabela = tabela_markdown(resultados)
    print("\n" + tabela)

    colunas = ["variante", "modelo"] + [c for c in ORDEM + ["BLEU-2", "BLEU-3", "BERTScore-P", "BERTScore-R", "Vocabulario", "s/img"] if any(c in r for r in resultados)]
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "comparativo.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=colunas, extrasaction="ignore")
        w.writeheader()
        w.writerows(resultados)

    variantes = [r["variante"] for r in resultados]
    partes = [
        "# Comparativo das variantes",
        "",
        f"Referencias: {origem} — rotulador `{cfg.labeling.model}`. "
        f"Modelo gold: `{cfg.gold_model.model}` ({cfg.gold_model.provider}). SLM: `{cfg.model.model_id}`.",
        "",
        "| variante | o que e |",
        "|---|---|",
        *[f"| {v} | {DESCRICAO.get(v, '')} |" for v in variantes],
        "",
        "## Metricas",
        "",
        tabela,
        "",
        "> CIDEr calcula o IDF no proprio conjunto avaliado: com N < ~100 o valor e instavel.",
        "",
        "## Custo de treino",
        "",
        tabela_custos(resolve_path(cfg, "models_dir"), variantes),
        "",
        "## Exemplos",
        "",
        exemplos(refs, preds, int(ev.n_examples)),
        "",
    ]
    (results_dir / "comparativo.md").write_text("\n".join(partes), encoding="utf-8")
    print(f"\nRelatorio: {results_dir / 'comparativo.md'} | CSV: {results_dir / 'comparativo.csv'}")
    return resultados
