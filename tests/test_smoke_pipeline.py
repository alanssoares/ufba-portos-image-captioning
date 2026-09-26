"""Pipeline inteiro com modelos minusculos e dados sinteticos (CPU, ~1-3 min).

Pulado se torch/transformers/peft nao estiverem instalados.
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("pycocoevalcap")

from src.cli import main  # noqa: E402
from src.common import read_json, read_jsonl  # noqa: E402


def test_pipeline_smoke(tmp_path):
    root = tmp_path.as_posix()
    sets = [
        f"paths.root={root}",
        f"dev.tiny_dir={root}/tiny",
        f"model.llm_id={root}/tiny/llm",
        f"model.vision_id={root}/tiny/vision",
        "dev.n_images=16",
        "training.common.epochs=1",
    ]
    argv = ["run", "--config", "configs/perfis/smoke.yaml"]
    for s in sets:
        argv += ["--set", s]
    main(argv)

    for variante in ("base", "pretrain", "finetune", "lora", "qlora"):
        assert (tmp_path / "outputs" / "models" / variante / "vlm_config.json").exists()
        preds = read_jsonl(tmp_path / "results" / f"preds_{variante}.jsonl")
        assert preds and all("legenda" in p for p in preds)
    assert (tmp_path / "outputs" / "models" / "lora" / "adapter").exists()
    assert (tmp_path / "outputs" / "models" / "finetune" / "llm").exists()
    assert read_json(tmp_path / "outputs" / "models" / "finetune" / "train_summary.json")["init_from"] == "pretrain"
    relatorio = (tmp_path / "results" / "comparativo.md").read_text(encoding="utf-8")
    for variante in ("base", "pretrain", "finetune", "lora", "qlora", "gold"):
        assert f"| {variante} |" in relatorio
