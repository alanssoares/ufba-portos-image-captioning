import json

from src.common import write_json, write_jsonl
from src.config import load_config
from src.transfer import export, import_, splits_sha1


def _fake_colab(root):
    write_jsonl(root / "data" / "labels.jsonl", [{"image_id": "a", "legendas": ["x"]}])
    write_json(root / "data" / "splits.json", {"train": ["a"], "val": [], "test": ["b"]})
    for v in ("base", "lora"):
        write_json(root / "outputs" / "models" / v / "vlm_config.json", {"variant": v})
        (root / "outputs" / "models" / v / "projector.safetensors").write_bytes(b"\0" * 16)
    (root / "outputs" / "models" / "lora" / "adapter").mkdir()
    (root / "outputs" / "models" / "lora" / "adapter" / "adapter_model.safetensors").write_bytes(b"1")


def test_export_import_ida_e_volta(tmp_path):
    colab, local = tmp_path / "colab", tmp_path / "local"
    _fake_colab(colab)
    zip_path = export(load_config(overrides=[f"paths.root={colab.as_posix()}"]))
    manifesto = json.loads(__import__("zipfile").ZipFile(zip_path).read("export_manifest.json"))
    assert manifesto["variantes"] == ["base", "lora"]

    # split local diferente: deve ganhar backup e ser substituido pelo do treino
    write_json(local / "data" / "splits.json", {"train": ["z"], "val": [], "test": ["y"]})
    cfg_local = load_config(overrides=[f"paths.root={local.as_posix()}"])
    import_(cfg_local, zip_path)

    assert (local / "outputs" / "models" / "lora" / "adapter" / "adapter_model.safetensors").exists()
    assert splits_sha1(local / "data" / "splits.json") == manifesto["splits_sha1"]
    assert list((local / "data").glob("splits.json.bak-*"))


def test_import_de_pasta(tmp_path):
    colab, local = tmp_path / "colab", tmp_path / "local"
    _fake_colab(colab)
    import_(load_config(overrides=[f"paths.root={local.as_posix()}"]), colab)
    assert (local / "outputs" / "models" / "base" / "vlm_config.json").exists()
