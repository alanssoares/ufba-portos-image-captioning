"""Pipeline inteiro com o Qwen3-VL-2B de verdade, dados sinteticos e 2 passos por variante.

Precisa de GPU CUDA e baixa ~4,5 GB na primeira vez, entao so roda com RUN_SMOKE=1:

    RUN_SMOKE=1 pytest tests/test_smoke_pipeline.py        (PowerShell: $env:RUN_SMOKE=1; pytest ...)
"""
import os

import pytest

if os.environ.get("RUN_SMOKE") != "1":
    pytest.skip("smoke test desligado (defina RUN_SMOKE=1)", allow_module_level=True)

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("pycocoevalcap")
if not torch.cuda.is_available():
    pytest.skip("smoke test precisa de GPU CUDA", allow_module_level=True)

from src.cli import main  # noqa: E402
from src.common import read_json, read_jsonl  # noqa: E402


def test_pipeline_smoke(tmp_path):
    main(["run", "--config", "configs/perfis/smoke.yaml", "--set", f"paths.root={tmp_path.as_posix()}"])

    modelos = tmp_path / "outputs" / "models"
    for variante in ("base", "pretrain", "finetune", "lora", "qlora"):
        assert (modelos / variante / "vlm_config.json").exists()
        preds = read_jsonl(tmp_path / "results" / f"preds_{variante}.jsonl")
        assert preds and all(p["legenda"] for p in preds)
    assert (modelos / "pretrain" / "delta.safetensors").exists()
    assert (modelos / "lora" / "adapter").exists()
    assert (modelos / "finetune" / "model").exists()
    assert read_json(modelos / "qlora" / "vlm_config.json")["adapter_quantized_base"] is True
    assert (tmp_path / "exports" / "modelos.zip").exists()
    relatorio = (tmp_path / "results" / "comparativo.md").read_text(encoding="utf-8")
    for variante in ("base", "pretrain", "finetune", "lora", "qlora", "gold"):
        assert f"| {variante} |" in relatorio
