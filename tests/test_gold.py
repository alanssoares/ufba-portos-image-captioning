"""Gold: retomada por versao da especificacao e parada na cota diaria."""
from PIL import Image

from src.common import append_jsonl, write_json
from src.config import load_config
from src.predict import _pendentes, cota_diaria_esgotada, gold_versao


def _cfg(tmp_path, *extra):
    return load_config(overrides=[f"paths.root={tmp_path.as_posix()}", *extra])


def _setup(tmp_path, ids):
    (tmp_path / "data" / "images").mkdir(parents=True)
    for i in ids:
        Image.new("RGB", (8, 8)).save(tmp_path / "data" / "images" / f"{i}.jpg")
    write_json(tmp_path / "data" / "splits.json", {"train": [], "val": [], "test": ids})


def test_versao_muda_com_modelo_e_instrucao(tmp_path):
    base = gold_versao(_cfg(tmp_path))
    assert gold_versao(_cfg(tmp_path, "gold_model.model=outro")) != base
    assert gold_versao(_cfg(tmp_path, "model.instruction=outra coisa")) != base
    assert gold_versao(_cfg(tmp_path)) == base


def test_linhas_de_outra_versao_contam_como_pendentes(tmp_path):
    cfg = _cfg(tmp_path)
    _setup(tmp_path, ["a", "b", "c"])
    out = tmp_path / "results" / "preds_gold.jsonl"
    v = gold_versao(cfg)
    append_jsonl(out, {"image_id": "a", "legenda": "x"})                        # sem versao (antiga)
    append_jsonl(out, {"image_id": "b", "legenda": "x", "prompt_versao": v})    # atual
    pend = [p.stem for p in _pendentes(cfg, out, False, versao=v)]
    assert pend == ["a", "c"]
    assert [p.stem for p in _pendentes(cfg, out, False)] == ["c"]               # sem versao: comportamento antigo


def test_cota_diaria():
    diaria = Exception("429 RESOURCE_EXHAUSTED ... 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'")
    minuto = Exception("429 RESOURCE_EXHAUSTED ... 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'")
    assert cota_diaria_esgotada(diaria)
    assert not cota_diaria_esgotada(minuto)
    assert not cota_diaria_esgotada(Exception("503 UNAVAILABLE"))
