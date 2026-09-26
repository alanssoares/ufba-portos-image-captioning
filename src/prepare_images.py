"""Normaliza as imagens brutas e mantem o manifesto de proveniencia.

Uso:
    python -m src prepare
    python -m src prepare --set prepare.max_side=768

O que faz:
  - percorre data/raw (recursivamente);
  - corrige orientacao EXIF, converte para RGB, redimensiona o lado maior;
  - deduplica por SHA-1 do arquivo original;
  - salva como data/images/psa_XXXX.jpg;
  - cria/atualiza data/metadata.csv, onde as colunas de licenca devem ser
    preenchidas a mao (fonte, url, licenca) — obrigatorio para uso academico.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
from pathlib import Path

from PIL import Image, ImageOps

from .common import IMAGE_EXTS
from .config import Config, resolve_path

FIELDS = [
    "image_id",
    "sha1",
    "arquivo_origem",
    "largura",
    "altura",
    "fonte",
    "url",
    "licenca",
    "data_coleta",
    "observacoes",
]


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def save_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def next_index(rows: list[dict[str, str]]) -> int:
    best = 0
    for row in rows:
        try:
            best = max(best, int(row["image_id"].split("_")[-1]))
        except (ValueError, KeyError):
            continue
    return best + 1


def run(cfg: Config) -> None:
    src_dir = resolve_path(cfg, "raw_dir")
    dst_dir = resolve_path(cfg, "images_dir")
    manifest = resolve_path(cfg, "metadata_csv")
    max_side = int(cfg.prepare.max_side)
    quality = int(cfg.prepare.quality)
    prefix = cfg.prepare.prefix

    dst_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(manifest)
    seen = {row["sha1"]: row for row in rows}
    idx = next_index(rows)
    hoje = dt.date.today().isoformat()

    sources = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS) if src_dir.exists() else []
    if not sources:
        print(f"Nenhuma imagem encontrada em {src_dir}. Coloque os arquivos originais la primeiro.")
        return

    novas = duplicadas = falhas = 0
    for src in sources:
        digest = sha1_of(src)
        if digest in seen:
            duplicadas += 1
            continue
        try:
            with Image.open(src) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                if max_side and max(img.size) > max_side:
                    img.thumbnail((max_side, max_side), Image.LANCZOS)
                image_id = f"{prefix}_{idx:04d}"
                img.save(dst_dir / f"{image_id}.jpg", "JPEG", quality=quality)
                largura, altura = img.size
        except Exception as exc:  # imagem corrompida, formato exotico, etc.
            print(f"  ! falhou em {src.name}: {exc}")
            falhas += 1
            continue

        row = {
            "image_id": image_id,
            "sha1": digest,
            "arquivo_origem": str(src.relative_to(src_dir)),
            "largura": str(largura),
            "altura": str(altura),
            "fonte": "",
            "url": "",
            "licenca": "",
            "data_coleta": hoje,
            "observacoes": "",
        }
        rows.append(row)
        seen[digest] = row
        idx += 1
        novas += 1

    save_manifest(manifest, rows)
    print(f"{novas} novas, {duplicadas} duplicadas ignoradas, {falhas} falhas. Total no manifesto: {len(rows)}.")
    faltando = [r["image_id"] for r in rows if not r["licenca"]]
    if faltando:
        print(f"ATENCAO: {len(faltando)} imagens sem licenca preenchida em {manifest.name}.")
