from src.common import write_jsonl
from src.config import load_config
from src.splits import assign, make_splits


def test_assign_proporcoes_e_disjuncao():
    ids = [f"img_{i:03d}" for i in range(100)]
    s = assign(ids, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (70, 15, 15)
    assert set(s["train"]) | set(s["val"]) | set(s["test"]) == set(ids)
    assert not set(s["train"]) & set(s["test"])


def test_assign_deterministico():
    ids = [str(i) for i in range(30)]
    r = {"train": 0.7, "val": 0.15, "test": 0.15}
    assert assign(ids, r, 7) == assign(ids, r, 7)


def _labels(path, n, ood=()):
    write_jsonl(path, [{"image_id": f"x_{i:03d}", "legendas": ["a"], "fora_de_dominio": i in ood} for i in range(n)])


def test_split_estavel_com_imagens_novas(tmp_path):
    cfg = load_config(overrides=[f"paths.root={tmp_path.as_posix()}"])
    labels = tmp_path / "data" / "labels.jsonl"
    _labels(labels, 40, ood={0})
    primeiro = make_splits(cfg)
    assert "x_000" not in primeiro["train"] + primeiro["val"] + primeiro["test"]
    _labels(labels, 60, ood={0})
    segundo = make_splits(cfg)
    assert set(primeiro["test"]) <= set(segundo["test"])   # teste nao muda por baixo dos modelos
    assert sum(len(segundo[k]) for k in ("train", "val", "test")) == 59
