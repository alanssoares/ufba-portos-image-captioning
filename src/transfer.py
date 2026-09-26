"""Leva os modelos treinados do Colab para a maquina local (onde roda a inferencia).

No Colab, depois do treino:
    python -m src export                       # -> <paths.root>/exports/modelos.zip

Na maquina local, depois de baixar o zip (pelo Drive no navegador ou Drive para desktop):
    python -m src import --from C:/Users/voce/Downloads/modelos.zip
    python -m src import --from "G:/Meu Drive/ufba-portos-captioning"   # pasta sincronizada tambem serve

O pacote leva, alem de outputs/models/<variante>/, os arquivos que definem o
experimento — data/labels.jsonl e data/splits.json — para que a inferencia local use
EXATAMENTE o mesmo conjunto de teste do treino. Na importacao, arquivos locais que
seriam sobrescritos e forem diferentes ganham uma copia .bak-<data> (nada e apagado).
Imagens so vao junto com transfer.include_images=true (ou --with-images).
"""
from __future__ import annotations

import datetime as dt
import filecmp
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from .config import Config, resolve_path

MANIFEST = "export_manifest.json"
# Arquivos grandes e ja comprimidos: guardar sem recompressao (rapido).
SEM_COMPRESSAO = {".safetensors", ".bin", ".jpg", ".jpeg", ".png", ".webp"}


def splits_sha1(splits_path: Path) -> str | None:
    """Impressao digital do split (so as listas de ids) — detecta teste diferente do treino."""
    if not splits_path.exists():
        return None
    data = json.loads(splits_path.read_text(encoding="utf-8"))
    chave = json.dumps({k: sorted(data.get(k, [])) for k in ("train", "val", "test")}, sort_keys=True)
    return hashlib.sha1(chave.encode("utf-8")).hexdigest()


def _entradas(cfg: Config, variants: list[str], with_images: bool) -> list[tuple[Path, str]]:
    """(arquivo local, caminho dentro do pacote). Caminhos do pacote sao sempre os padroes."""
    entradas = []
    models_dir = resolve_path(cfg, "models_dir")
    for v in variants:
        vdir = models_dir / v
        if not (vdir / "vlm_config.json").exists():
            print(f"  (variante '{v}' nao treinada — fica fora do pacote)")
            continue
        for p in sorted(vdir.rglob("*")):
            if p.is_file():
                entradas.append((p, f"outputs/models/{v}/{p.relative_to(vdir).as_posix()}"))
    for chave, destino in (("labels_jsonl", "data/labels.jsonl"), ("splits_json", "data/splits.json"), ("metadata_csv", "data/metadata.csv")):
        p = resolve_path(cfg, chave)
        if p.exists():
            entradas.append((p, destino))
    if with_images:
        images_dir = resolve_path(cfg, "images_dir")
        for p in sorted(images_dir.iterdir()) if images_dir.exists() else []:
            if p.is_file():
                entradas.append((p, f"data/images/{p.name}"))
    return entradas


def export(cfg: Config, out: Path | None = None, with_images: bool | None = None) -> Path:
    tr = cfg.transfer
    with_images = bool(tr.include_images) if with_images is None else with_images
    variants = [v for v in tr.variants]
    out = Path(out) if out else resolve_path(cfg, "exports_dir") / tr.export_name
    out.parent.mkdir(parents=True, exist_ok=True)

    entradas = _entradas(cfg, variants, with_images)
    incluidas = sorted({dest.split("/")[2] for _, dest in entradas if dest.startswith("outputs/models/")})
    if not incluidas:
        raise SystemExit("Nenhuma variante treinada para exportar. Rode `train` antes.")

    manifesto = {
        "criado_em": dt.datetime.now().isoformat(timespec="seconds"),
        "variantes": incluidas,
        "splits_sha1": splits_sha1(resolve_path(cfg, "splits_json")),
        "com_imagens": with_images,
        "model_id": cfg.model.model_id,
    }
    total = sum(p.stat().st_size for p, _ in entradas)
    print(f"Exportando {', '.join(incluidas)} ({len(entradas)} arquivos, {total / 1024**3:.2f} GB) -> {out}")
    with zipfile.ZipFile(out, "w", allowZip64=True) as zf:
        for p, dest in entradas:
            modo = zipfile.ZIP_STORED if p.suffix.lower() in SEM_COMPRESSAO else zipfile.ZIP_DEFLATED
            zf.write(p, dest, compress_type=modo)
        zf.writestr(MANIFEST, json.dumps(manifesto, ensure_ascii=False, indent=2))
    print(f"Pronto: {out} ({out.stat().st_size / 1024**3:.2f} GB). Baixe e rode `python -m src import --from <zip>` localmente.")
    return out


def _destino(cfg: Config, rel: str) -> Path:
    """Mapeia o caminho padrao do pacote para os caminhos (possivelmente customizados) da config local."""
    partes = rel.split("/")
    if rel.startswith("outputs/models/"):
        return resolve_path(cfg, "models_dir").joinpath(*partes[2:])
    if rel == "data/labels.jsonl":
        return resolve_path(cfg, "labels_jsonl")
    if rel == "data/splits.json":
        return resolve_path(cfg, "splits_json")
    if rel == "data/metadata.csv":
        return resolve_path(cfg, "metadata_csv")
    if rel.startswith("data/images/"):
        return resolve_path(cfg, "images_dir") / partes[-1]
    raise ValueError(rel)


def _backup_se_diferente(src: Path, dst: Path, carimbo: str) -> None:
    if dst.exists() and dst.is_file() and not filecmp.cmp(src, dst, shallow=False):
        bak = dst.with_name(f"{dst.name}.bak-{carimbo}")
        shutil.copy2(dst, bak)
        print(f"  {dst.name} local era diferente — copia de seguranca em {bak.name}")


def import_(cfg: Config, source: Path) -> None:
    source = Path(source)
    if not source.exists():
        raise SystemExit(f"{source} nao existe")
    carimbo = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    with tempfile.TemporaryDirectory() as tmp:
        if source.is_file():
            with zipfile.ZipFile(source) as zf:
                zf.extractall(tmp)
            base = Path(tmp)
        else:
            base = source  # pasta com a mesma estrutura (ex: Drive para desktop)

        manifesto = {}
        if (base / MANIFEST).exists():
            manifesto = json.loads((base / MANIFEST).read_text(encoding="utf-8"))

        arquivos = [p for p in base.rglob("*") if p.is_file() and p.name != MANIFEST]
        rels = [p.relative_to(base).as_posix() for p in arquivos]
        validos = [(p, r) for p, r in zip(arquivos, rels) if r.startswith(("outputs/models/", "data/labels.jsonl", "data/splits.json", "data/metadata.csv", "data/images/"))]
        if not any(r.startswith("outputs/models/") for _, r in validos):
            raise SystemExit(f"{source} nao tem outputs/models/<variante>/ — e o pacote gerado por `python -m src export`?")

        copiados = 0
        for p, rel in validos:
            if rel.startswith("data/images/") and _destino(cfg, rel).exists():
                continue
            dst = _destino(cfg, rel)
            if dst.resolve() == p.resolve():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not rel.startswith("outputs/models/"):
                _backup_se_diferente(p, dst, carimbo)
            shutil.copy2(p, dst)
            copiados += 1

    models_dir = resolve_path(cfg, "models_dir")
    variantes = sorted(d.name for d in models_dir.iterdir() if (d / "vlm_config.json").exists()) if models_dir.exists() else []
    print(f"{copiados} arquivos importados. Variantes disponiveis em {models_dir}: {', '.join(variantes) or 'nenhuma'}")
    atual = splits_sha1(resolve_path(cfg, "splits_json"))
    if manifesto.get("splits_sha1") and atual != manifesto["splits_sha1"]:
        print("  AVISO: o splits.json local nao e o mesmo usado no treino.")
    if not any(resolve_path(cfg, "images_dir").glob("*")):
        print(f"  AVISO: {resolve_path(cfg, 'images_dir')} esta vazia — a inferencia precisa das imagens do teste.")
    print("Proximo passo: python -m src run --config configs/perfis/local_4gb.yaml")
