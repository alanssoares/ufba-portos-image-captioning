"""Divisao treino/validacao/teste sobre as imagens rotuladas.

Uso:
    python -m src split
    python -m src split --set split.test=0.2 --redo

Estavel: se data/splits.json ja existe, as atribuicoes anteriores sao mantidas e so
as imagens novas sao sorteadas (com a mesma proporcao). Assim o conjunto de teste nao
muda por baixo dos modelos ja treinados. --redo refaz tudo do zero.
"""
from __future__ import annotations

import datetime as dt
import random
from pathlib import Path

from .common import index_by, read_json, read_jsonl, write_json
from .config import Config, resolve_path

SPLITS = ("train", "val", "test")


def eligible_ids(labels: dict[str, dict], exclude_ood: bool) -> list[str]:
    ids = []
    for image_id, row in labels.items():
        if not row.get("legendas"):
            continue
        if exclude_ood and row.get("fora_de_dominio"):
            continue
        ids.append(image_id)
    return sorted(ids)


def assign(ids: list[str], ratios: dict[str, float], seed: int) -> dict[str, list[str]]:
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("as proporcoes do split somam zero")
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_test = round(n * ratios["test"] / total)
    n_val = round(n * ratios["val"] / total)
    # Com poucas imagens, garanta ao menos 1 no teste e 1 na validacao quando possivel.
    if n >= 3:
        n_test = max(n_test, 1)
        n_val = max(n_val, 1)
    out = {"test": ids[:n_test], "val": ids[n_test : n_test + n_val], "train": ids[n_test + n_val :]}
    return {k: sorted(v) for k, v in out.items()}


def make_splits(cfg: Config, redo: bool = False) -> dict:
    labels = index_by(read_jsonl(resolve_path(cfg, "labels_jsonl")))
    out_path = resolve_path(cfg, "splits_json")
    sp = cfg.split

    if sp.get("fixed"):
        fixed = read_json(Path(sp.fixed))
        if not fixed:
            raise SystemExit(f"split.fixed aponta para {sp.fixed}, que nao existe ou esta vazio")
        write_json(out_path, fixed)
        print(f"Split fixo copiado de {sp.fixed}.")
        return fixed

    ids = eligible_ids(labels, bool(sp.exclude_out_of_domain))
    if not ids:
        raise SystemExit("Nenhuma imagem rotulada. Rode `python -m src label` (ou `dummy-data`) antes.")

    ratios = {k: float(sp[k]) for k in SPLITS}
    atual = None if redo else read_json(out_path)
    if atual:
        ja = {i for k in SPLITS for i in atual.get(k, [])}
        novos = [i for i in ids if i not in ja]
        extra = assign(novos, ratios, int(sp.seed) + len(ja)) if novos else {k: [] for k in SPLITS}
        result = {k: sorted(set(atual.get(k, [])) & set(ids) | set(extra[k])) for k in SPLITS}
        removidos = len(ja - set(ids))
        print(f"Split existente mantido; {len(novos)} imagens novas distribuidas, {removidos} removidas.")
    else:
        result = assign(ids, ratios, int(sp.seed))

    data = {
        **result,
        "seed": int(sp.seed),
        "proporcoes": ratios,
        "atualizado_em": dt.datetime.now().isoformat(timespec="seconds"),
    }
    write_json(out_path, data)
    print(" | ".join(f"{k}: {len(result[k])}" for k in SPLITS) + f"  -> {out_path}")
    return data


def load_split(cfg: Config, split: str) -> list[str]:
    data = read_json(resolve_path(cfg, "splits_json"))
    if not data:
        raise SystemExit("data/splits.json nao existe. Rode `python -m src split` antes.")
    if split not in data:
        raise SystemExit(f"split '{split}' nao existe em splits.json")
    return list(data[split])


def references(cfg: Config, split: str) -> dict[str, list[str]]:
    """image_id -> legendas de referencia (do rotulador) para as imagens do split."""
    labels = index_by(read_jsonl(resolve_path(cfg, "labels_jsonl")))
    return {i: labels[i]["legendas"] for i in load_split(cfg, split) if i in labels and labels[i].get("legendas")}
