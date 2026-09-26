"""Configuracao do experimento: YAML em camadas + overrides pela linha de comando.

Ordem de precedencia (a ultima vence):

    configs/default.yaml  ->  --config perfil1.yaml  ->  --config perfil2.yaml  ->  --set a.b=c

Exemplos:

    python -m src run --config configs/perfis/colab_l4.yaml
    python -m src train --stage lora --set training.lora.r=32 --set training.lora.lr=1e-4

Os valores de `--set` sao interpretados como YAML (`true`, `3`, `1e-4`, `[a, b]`
viram bool, int, float e lista). Caminhos em `paths.*` sao relativos a
`paths.root`, que por sua vez e relativo a raiz do repositorio.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


class Config(dict):
    """dict com acesso por atributo (cfg.training.lora.r) e por caminho pontilhado."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"config nao tem a chave '{name}'") from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self, default=str))


def _wrap(value: Any) -> Any:
    if isinstance(value, dict) and not isinstance(value, Config):
        return Config({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_override(expr: str) -> tuple[list[str], Any]:
    if "=" not in expr:
        raise ValueError(f"override invalido '{expr}': use chave.pontilhada=valor")
    key, raw = expr.split("=", 1)
    value = yaml.safe_load(raw) if raw.strip() != "" else ""
    if isinstance(value, str):
        # O YAML 1.1 le "1e-4" como texto; aqui tratamos como numero.
        try:
            value = float(value)
        except ValueError:
            pass
    return key.strip().split("."), value


def apply_override(cfg: dict, keys: list[str], value: Any) -> None:
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    # Um perfil pode herdar de outro: `herda: perfis/colab_t4.yaml`
    parent = data.pop("herda", None)
    if parent:
        parent_path = (path.parent / parent) if not Path(parent).is_absolute() else Path(parent)
        data = deep_merge(load_yaml(parent_path), data)
    return data


def load_config(
    configs: Iterable[str | Path] | None = None,
    overrides: Iterable[str] | None = None,
    base: str | Path | None = DEFAULT_CONFIG,
) -> Config:
    cfg: dict = load_yaml(base) if base else {}
    for extra in configs or []:
        cfg = deep_merge(cfg, load_yaml(extra))
    for expr in overrides or []:
        keys, value = parse_override(expr)
        apply_override(cfg, keys, value)
    return _wrap(cfg)


def resolve_path(cfg: Config, key: str) -> Path:
    """Caminho absoluto de cfg.paths.<key>, relativo a paths.root (que e relativo ao repo)."""
    root = Path(cfg.paths.get("root", "."))
    if not root.is_absolute():
        root = ROOT / root
    value = Path(cfg.paths[key])
    return value if value.is_absolute() else root / value


def stage_config(cfg: Config, stage: str) -> Config:
    """Hiperparametros efetivos de um estagio: training.common + training.<stage>."""
    training = cfg.get("training", {})
    if stage not in training:
        raise KeyError(f"estagio '{stage}' nao existe em training.* da config")
    return _wrap(deep_merge(training.get("common", {}), training[stage]))


def dump(cfg: Config) -> str:
    return yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False)
