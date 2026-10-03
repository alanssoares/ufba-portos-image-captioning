#!/usr/bin/env python3
"""
Baixa imagens do Wikimedia Commons a partir de uma lista de links
e gera o metadata.csv (fonte, URL, licença, autor...).

Rode a partir da raiz do repositório:
    python src/data/baixar_e_gerar_metadata.py data/sources/commons_links.txt
    python src/data/baixar_e_gerar_metadata.py data/sources/commons_links.txt --largura 1024

O arquivo de links tem um link (ou título) por linha. Aceita:
    https://commons.wikimedia.org/wiki/File:Exemplo.jpg
    File:Exemplo.jpg
    Exemplo.jpg
Linhas vazias e linhas começando com # são ignoradas.

Resultado (caminhos padrão, relativos ao diretório atual):
    data/raw/images/<arquivos>    <- imagens (não versionar)
    data/metadata.csv             <- metadados (versionar)

Para usar outros caminhos: --imagens PASTA  e  --metadata ARQUIVO.csv
"""
import argparse
import csv
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date

API = "https://commons.wikimedia.org/w/api.php"
# A Wikimedia exige um User-Agent identificável. TROQUE pelo seu contato.
USER_AGENT = "MeuDataset-Metadata/1.0 (seu-email@exemplo.com)"
BATCH = 50  # máximo de títulos por requisição à API

CAMPOS = [
    "arquivo", "fonte", "url_pagina", "url_arquivo", "autor", "creditos",
    "licenca", "url_licenca", "termos_de_uso", "atribuicao_obrigatoria",
    "mime", "largura", "altura", "sha256", "data_consulta",
]


def abrir(url, tentativas=4):
    """GET com User-Agent e retry simples para 429/5xx."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for n in range(tentativas):
        try:
            return urllib.request.urlopen(req, timeout=60)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and n < tentativas - 1:
                espera = int(e.headers.get("Retry-After", 2 ** (n + 1)))
                print(f"  HTTP {e.code}, aguardando {espera}s...")
                time.sleep(espera)
                continue
            raise
        except urllib.error.URLError:
            if n < tentativas - 1:
                time.sleep(2 ** (n + 1))
                continue
            raise


def para_titulo(linha):
    """Converte URL ou nome de arquivo em 'File:Nome.ext'."""
    linha = linha.strip()
    if linha.startswith("http"):
        p = urllib.parse.urlparse(linha)
        if "/wiki/" in p.path:
            linha = urllib.parse.unquote(p.path.split("/wiki/", 1)[1])
        else:
            linha = urllib.parse.parse_qs(p.query).get("title", [linha])[0]
    linha = linha.replace("_", " ")
    linha = re.sub(r"^(Arquivo|Ficheiro|Image):", "File:", linha, flags=re.I)
    if not linha.lower().startswith("file:"):
        linha = "File:" + linha
    return "File:" + linha[5:]


def limpar_html(texto):
    texto = re.sub(r"<[^>]+>", "", texto or "")
    return re.sub(r"\s+", " ", html.unescape(texto)).strip()


def consultar(titulos, largura):
    params = {
        "action": "query", "format": "json", "formatversion": "2",
        "redirects": "1", "prop": "imageinfo",
        "iiprop": "url|extmetadata|mime|size",
        "titles": "|".join(titulos),
    }
    if largura:
        params["iiurlwidth"] = str(largura)  # gera 'thumburl' redimensionada
    with abrir(API + "?" + urllib.parse.urlencode(params)) as r:
        return json.load(r)["query"]


def nome_seguro(titulo, usados):
    """Nome de arquivo local seguro e único."""
    base = titulo.removeprefix("File:").replace(" ", "_")
    base = re.sub(r"[^\w.\-]", "_", base)
    nome, ext = os.path.splitext(base)
    candidato, i = base, 2
    while candidato.lower() in usados:
        candidato = f"{nome}_{i}{ext}"
        i += 1
    usados.add(candidato.lower())
    return candidato


def baixar(url, destino):
    """Baixa para 'destino' e devolve o SHA-256."""
    h = hashlib.sha256()
    tmp = destino + ".part"
    with abrir(url) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 16):
            h.update(chunk)
            f.write(chunk)
    os.replace(tmp, destino)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("links", help="arquivo .txt com os links")
    ap.add_argument("--imagens", default=os.path.join("data", "raw", "images"),
                    help="pasta onde salvar as imagens (padrão: data/raw/images)")
    ap.add_argument("--metadata", default=os.path.join("data", "metadata.csv"),
                    help="caminho do CSV de metadados (padrão: data/metadata.csv)")
    ap.add_argument("--largura", type=int, default=0,
                    help="baixar versão redimensionada com essa largura em px "
                         "(padrão: 0 = original)")
    ap.add_argument("--pausa", type=float, default=1.0,
                    help="segundos entre downloads (padrão: 1.0)")
    args = ap.parse_args()

    pasta_img = args.imagens
    caminho_csv = args.metadata
    os.makedirs(pasta_img, exist_ok=True)
    os.makedirs(os.path.dirname(caminho_csv) or ".", exist_ok=True)

    with open(args.links, encoding="utf-8") as f:
        titulos = [para_titulo(l) for l in f if l.strip() and not l.startswith("#")]
    titulos = list(dict.fromkeys(titulos))
    print(f"{len(titulos)} imagens na lista.")

    linhas, falhas, usados = [], [], set()
    hoje = date.today().isoformat()

    for i in range(0, len(titulos), BATCH):
        lote = titulos[i:i + BATCH]
        dados = consultar(lote, args.largura)

        # título pedido -> título final (após normalização/redirecionamento)
        mapa = {t: t for t in lote}
        for chave in ("normalized", "redirects"):
            for m in dados.get(chave, []):
                for k, v in mapa.items():
                    if v == m["from"]:
                        mapa[k] = m["to"]
        paginas = {p["title"]: p for p in dados["pages"]}

        for pedido in lote:
            pag = paginas.get(mapa[pedido])
            if not pag or pag.get("missing") or "imageinfo" not in pag:
                falhas.append((pedido, "não encontrado no Commons"))
                continue

            info = pag["imageinfo"][0]
            meta = info.get("extmetadata", {})
            val = lambda k: limpar_html(meta.get(k, {}).get("value", ""))

            url_dl = info.get("thumburl") if args.largura else info["url"]
            url_dl = url_dl or info["url"]
            nome = nome_seguro(pag["title"], usados)
            destino = os.path.join(pasta_img, nome)

            print(f"Baixando {nome} ...")
            try:
                sha = baixar(url_dl, destino)
            except Exception as e:
                falhas.append((pedido, f"erro no download: {e}"))
                usados.discard(nome.lower())
                continue

            linhas.append({
                "arquivo": nome,
                "fonte": "Wikimedia Commons",
                "url_pagina": info["descriptionurl"],  # citar esta
                "url_arquivo": info["url"],
                "autor": val("Artist"),
                "creditos": val("Credit"),
                "licenca": val("LicenseShortName"),
                "url_licenca": val("LicenseUrl"),
                "termos_de_uso": val("UsageTerms"),
                "atribuicao_obrigatoria": val("AttributionRequired"),
                "mime": info.get("mime", ""),
                "largura": info.get("thumbwidth", info.get("width", "")),
                "altura": info.get("thumbheight", info.get("height", "")),
                "sha256": sha,
                "data_consulta": hoje,
            })
            time.sleep(args.pausa)

    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(linhas)

    print(f"\n{len(linhas)} imagens baixadas -> {pasta_img}")
    print(f"Metadados -> {caminho_csv}")

    if falhas:
        print("\nFalhas:")
        for t, motivo in falhas:
            print(f"  - {t}: {motivo}")

    sem_lic = [l["arquivo"] for l in linhas if not l["licenca"]]
    if sem_lic:
        print("\nATENÇÃO: sem licença identificada (confira a página manualmente):")
        for a in sem_lic:
            print("  -", a)

    if falhas or sem_lic:
        sys.exit(1)


if __name__ == "__main__":
    main()