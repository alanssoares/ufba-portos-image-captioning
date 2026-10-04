import csv

import pytest
from PIL import Image

from src import fetch_commons_dataset as fetch
from src.config import load_config
from src.prepare_images import run as prepare


def _cfg(root):
    return load_config(overrides=[f"paths.root={root.as_posix()}"])


def _img(path, cor=(200, 0, 0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1600, 900), cor).save(path)


def _commons(path, linhas):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fetch.CAMPOS, extrasaction="ignore")
        w.writeheader()
        w.writerows(linhas)


def _manifesto(root):
    with (root / "data" / "metadata.csv").open(encoding="utf-8", newline="") as fh:
        return {r["arquivo_origem"]: r for r in csv.DictReader(fh)}


def test_prepare_preenche_licenca_do_commons(tmp_path):
    _img(tmp_path / "data/raw/images/Porto_A.jpg", (10, 20, 30))
    _img(tmp_path / "data/raw/images/foto_propria.jpg", (40, 50, 60))
    _commons(tmp_path / "data/sources/commons_metadata.csv", [{
        "arquivo": "Porto_A.jpg", "fonte": "Wikimedia Commons", "autor": "Fulano",
        "url_pagina": "https://commons.wikimedia.org/wiki/File:Porto_A.jpg", "licenca": "CC BY-SA 4.0",
    }])

    prepare(_cfg(tmp_path))

    rows = _manifesto(tmp_path)
    a = rows["images/Porto_A.jpg"]   # caminho sempre com "/", tambem no Windows
    assert a["fonte"] == "Fulano, Wikimedia Commons"
    assert a["url"].endswith("File:Porto_A.jpg")
    assert a["licenca"] == "CC BY-SA 4.0"
    propria = [r for k, r in rows.items() if k.endswith("foto_propria.jpg")][0]
    assert propria["licenca"] == ""          # fora do Commons: continua para preencher a mao
    assert max(Image.open(tmp_path / "data/images" / f"{a['image_id']}.jpg").size) == 1024


def test_prepare_completa_linhas_antigas_sem_sobrescrever(tmp_path):
    _img(tmp_path / "data/raw/images/Porto_A.jpg")
    cfg = _cfg(tmp_path)
    prepare(cfg)                              # sem CSV de fontes: licenca vazia
    rows = _manifesto(tmp_path)
    assert all(r["licenca"] == "" for r in rows.values())

    # edicao manual na fonte + CSV do Commons chegando depois
    path = tmp_path / "data" / "metadata.csv"
    linhas = list(rows.values())
    linhas[0]["fonte"] = "anotado a mao"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    _commons(tmp_path / "data/sources/commons_metadata.csv", [{
        "arquivo": "Porto_A.jpg", "fonte": "Wikimedia Commons", "autor": "Fulano",
        "url_pagina": "u", "licenca": "CC BY 4.0",
    }])

    prepare(cfg)
    r = list(_manifesto(tmp_path).values())[0]
    assert r["licenca"] == "CC BY 4.0"
    assert r["fonte"] == "anotado a mao"       # valor existente nao e sobrescrito
    assert len(_manifesto(tmp_path)) == 1      # nada duplicado


def test_prepare_recusa_csv_do_commons_no_lugar_do_manifesto(tmp_path):
    _img(tmp_path / "data/raw/images/Porto_A.jpg")
    _commons(tmp_path / "data/metadata.csv", [{"arquivo": "Porto_A.jpg", "url_pagina": "u"}])
    original = (tmp_path / "data/metadata.csv").read_bytes()
    with pytest.raises(SystemExit, match="commons_metadata.csv"):
        prepare(_cfg(tmp_path))
    assert (tmp_path / "data/metadata.csv").read_bytes() == original   # nada perdido


def test_fetch_csv_mescla_e_ordena(tmp_path):
    caminho = str(tmp_path / "fontes.csv")
    fetch.gravar_csv(caminho, [{"arquivo": "b.jpg", "url_pagina": "pb"}, {"arquivo": "A.jpg", "url_pagina": "pa"}])
    lidas = fetch.ler_csv(caminho)
    assert [l["arquivo"] for l in lidas] == ["A.jpg", "b.jpg"]
    assert set(lidas[0]) == set(fetch.CAMPOS)


def test_fetch_recusa_manifesto_do_pipeline(tmp_path):
    caminho = tmp_path / "metadata.csv"
    caminho.write_text("image_id,sha1,arquivo_origem\npsa_0001,x,a.jpg\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        fetch.ler_csv(str(caminho))


def test_para_titulo():
    assert fetch.para_titulo("https://commons.wikimedia.org/wiki/File:Porto_de_Salvador,_BA.jpg") == "File:Porto de Salvador, BA.jpg"
    assert fetch.para_titulo("Arquivo:X_y.jpg") == "File:X y.jpg"
