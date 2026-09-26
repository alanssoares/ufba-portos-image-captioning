"""Ponto de entrada unico: python -m src <comando> [--config perfil.yaml ...] [--set chave=valor ...]

Comandos:
    prepare       normaliza data/raw -> data/images + manifesto
    label         legendas de referencia com o Claude (labeling.*)
    split         treino/validacao/teste (split.*)
    train         --stage base|pretrain|finetune|lora|qlora
    predict       --variant base|pretrain|finetune|lora|qlora|gold|all
    evaluate      metricas + relatorio results/comparativo.md
    run           executa pipeline.steps em ordem (--from, --only, --skip)
    show-config   imprime a config efetiva (depois de perfis e --set)
    export        empacota modelos treinados + labels + splits (Colab -> local)
    import        instala um pacote exportado (--from zip ou pasta)
    dummy-data    imagens e rotulos sinteticos (teste sem dataset)

Exemplos:
    python -m src run --config configs/perfis/smoke.yaml
    python -m src run --config configs/perfis/colab_l4.yaml --from train:pretrain
    python -m src train --stage qlora --set training.qlora.lora.r=8
    python -m src show-config --config configs/perfis/local_4gb.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import Config, dump, load_config, resolve_path


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", action="append", default=[], help="perfil YAML adicional (pode repetir)")
    p.add_argument("--set", "-s", dest="overrides", action="append", default=[], help="override chave.pontilhada=valor")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for nome in ("prepare", "show-config", "dummy-data"):
        _add_common(sub.add_parser(nome))

    p = sub.add_parser("label")
    _add_common(p)
    p.add_argument("--redo", action="store_true")

    p = sub.add_parser("split")
    _add_common(p)
    p.add_argument("--redo", action="store_true")

    p = sub.add_parser("train")
    _add_common(p)
    p.add_argument("--stage", required=True, choices=["base", "pretrain", "finetune", "lora", "qlora"])

    p = sub.add_parser("predict")
    _add_common(p)
    p.add_argument("--variant", required=True)
    p.add_argument("--redo", action="store_true")

    p = sub.add_parser("evaluate")
    _add_common(p)
    p.add_argument("--preds", nargs="+", default=None, help="glob(s) de preds_*.jsonl (padrao: variantes da config)")
    p.add_argument("--refs", type=Path, default=None, help="arquivo de referencias (padrao: labels + split de teste)")

    p = sub.add_parser("export")
    _add_common(p)
    p.add_argument("--out", type=Path, default=None, help="caminho do zip (padrao: <root>/exports/modelos.zip)")
    p.add_argument("--with-images", action="store_true", default=None)

    p = sub.add_parser("import")
    _add_common(p)
    p.add_argument("--from", dest="source", type=Path, required=True, help="zip exportado ou pasta com a mesma estrutura")

    p = sub.add_parser("run")
    _add_common(p)
    p.add_argument("--from", dest="start", default=None, help="comeca neste passo (ex: train:lora)")
    p.add_argument("--only", nargs="+", default=None, help="roda so estes passos")
    p.add_argument("--skip", nargs="+", default=[], help="pula estes passos")
    p.add_argument("--redo", action="store_true", help="refaz predicoes/rotulos existentes")
    return parser


def execute_step(cfg: Config, step: str, redo: bool = False) -> None:
    nome, _, alvo = step.partition(":")
    if nome == "prepare":
        from .prepare_images import run

        run(cfg)
    elif nome == "label":
        from .labeling import run

        run(cfg, redo=redo)
    elif nome == "split":
        from .splits import make_splits

        make_splits(cfg, redo=False)
    elif nome == "train":
        from .train import run

        run(cfg, alvo)
    elif nome == "predict":
        from .predict import run

        run(cfg, alvo or "all", redo=redo)
    elif nome == "evaluate":
        from .evaluate import run

        run(cfg)
    elif nome == "export":
        from .transfer import export

        export(cfg)
    elif nome == "dummy-data":
        from .devtools import make_dummy_data

        make_dummy_data(cfg)
    else:
        raise SystemExit(f"passo desconhecido '{step}'")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)

    if args.cmd == "show-config":
        print(dump(cfg))
        return
    if args.cmd == "label":
        execute_step(cfg, "label", redo=args.redo)
    elif args.cmd == "split":
        from .splits import make_splits

        make_splits(cfg, redo=args.redo)
    elif args.cmd == "train":
        execute_step(cfg, f"train:{args.stage}")
    elif args.cmd == "predict":
        execute_step(cfg, f"predict:{args.variant}", redo=args.redo)
    elif args.cmd == "evaluate":
        from .evaluate import run

        run(cfg, preds_glob=args.preds, refs_path=args.refs)
    elif args.cmd == "export":
        from .transfer import export

        export(cfg, out=args.out, with_images=args.with_images)
    elif args.cmd == "import":
        from .transfer import import_

        import_(cfg, args.source)
    elif args.cmd == "run":
        passos = list(cfg.pipeline.steps)
        if args.only:
            passos = [p for p in passos if p in args.only]
        if args.start:
            if args.start not in passos:
                raise SystemExit(f"--from {args.start}: passo nao esta em pipeline.steps ({', '.join(passos)})")
            passos = passos[passos.index(args.start) :]
        passos = [p for p in passos if p not in args.skip]

        results_dir = resolve_path(cfg, "results_dir")
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "config_efetiva.yaml").write_text(dump(cfg), encoding="utf-8")
        print(f"Pipeline: {' -> '.join(passos)}\n")
        for passo in passos:
            t0 = time.perf_counter()
            print(f"\n===== {passo} =====")
            execute_step(cfg, passo, redo=args.redo)
            print(f"===== {passo}: {time.perf_counter() - t0:.1f}s =====")
    else:
        execute_step(cfg, args.cmd)


if __name__ == "__main__":
    main(sys.argv[1:])
