"""Caminhos, IO e utilidades compartilhadas pelo pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"
IMAGES_DIR = DATA / "images"
METADATA_CSV = DATA / "metadata.csv"
DRAFTS_JSONL = DATA / "drafts.jsonl"
CAPTIONS_JSONL = DATA / "captions.jsonl"
RESULTS_DIR = ROOT / "results"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def index_by(rows: Iterable[dict[str, Any]], key: str = "image_id") -> dict[str, dict[str, Any]]:
    """Ultima ocorrencia vence — permite reprocessar sem limpar o arquivo."""
    return {row[key]: row for row in rows}


def list_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def pick_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
