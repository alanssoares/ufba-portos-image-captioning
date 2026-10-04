#!/usr/bin/env python3
"""
Baixa imagens do Wikimedia Commons a partir de uma lista de links e registra a
proveniencia de cada arquivo (fonte, URL, autor, licenca...) em um CSV de fontes.

Rode a partir da raiz do repositorio:
    python -m src.fetch_commons_dataset data/sources/commons_links.txt
    python -m src.fetch_commons_dataset data/sources/commons_links.txt --largura 2048
    python -m src.fetch_commons_dataset data/sources/commons_links.txt --contato voce@exemplo.com

O arquivo de links tem um link (ou titulo) por linha. Aceita:
    https://commons.wikimedia.org/wiki/File:Exemplo.jpg
    File:Exemplo.jpg
    Exemplo.jpg
Linhas vazias e linhas comecando com # sao ignoradas.

Resultado (caminhos padrao, relativos ao diretorio atual):
    data/raw/images/<arquivos>             <- imagens originais (nao versionar)
    data/sources/commons_metadata.csv      <- proveniencia do Commons (versionar)

Este CSV NAO e o manifesto do pipeline: `python -m src prepare` le o CSV de fontes
(paths.commons_csv) e preenche sozinho as colunas fonte/url/licenca do
data/metadata.csv, casando pelo nome do arquivo.

Rodar de novo e incremental: arquivos ja registrados e presentes no disco nao sao
baixados de novo, e o CSV e mesclado (nunca sobrescrito do zero).

A Wikimedia exige um User-Agent com contato: use --contato ou a variavel de
ambiente WIKIMEDIA_CONTACT.

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
USER_AGENT = "ufba-portos-image-captioning/1.0 ({contato})"
BATCH = 50  # maximo de titulos por requisicao a API

PADRAO_IMAGENS = os.path.join("data", "raw", "images")
PADRAO_METADATA = os.path.join("data", "sources", "commons_metadata.csv")

CAMPOS = [
    "arquivo", "fonte", "url_pagina", "url_arquivo", "autor", "creditos",
    "licenca", "url_licenca", "termos_de_uso", "atribuicao_obrigatoria",
    "mime", "largura", "altura", "sha256", "data_consulta",
]

_user_agent = USER_AGENT.format(contato="contato-nao-informado")


def abrir(url, tentativas=4):
    """GET com User-Agent e retry simples para 429/5xx."""
    req = urllib.request.Request(url, headers={"User-Agent": _user_agent})
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
    """Nome de arquivo local seguro e unico."""
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


def ler_csv(caminho):
    """Linhas ja registradas no CSV de fontes (vazio se nao existir)."""
    if not os.path.exists(caminho):
        return []
    with open(caminho, encoding="utf-8", newline="") as f:
        leitor = csv.DictReader(f)
        faltando = {"arquivo", "url_pagina"} - set(leitor.fieldnames or [])
        if faltando:
            sys.exit(f"{caminho} nao parece um CSV de fontes do Commons (faltam {sorted(faltando)}). "
                     "Use --metadata para apontar outro arquivo.")
        return list(leitor)


def gravar_csv(caminho, linhas):
    tmp = caminho + ".part"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(linhas, key=lambda l: l["arquivo"].lower()))
    os.replace(tmp, caminho)


def main():
    global _user_agent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("links", help="arquivo .txt com os links")
    ap.add_argument("--imagens", default=PADRAO_IMAGENS,
                    help=f"pasta onde salvar as imagens (padrao: {PADRAO_IMAGENS})")
    ap.add_argument("--metadata", default=PADRAO_METADATA,
                    help=f"CSV de fontes do Commons (padrao: {PADRAO_METADATA})")
    ap.add_argument("--largura", type=int, default=0,
                    help="baixar versao redimensionada com essa largura em px "
                         "(padrao: 0 = original)")
    ap.add_argument("--pausa", type=float, default=1.0,
                    help="segundos entre downloads (padrao: 1.0)")
    ap.add_argument("--contato", default=os.environ.get("WIKIMEDIA_CONTACT", ""),
                    help="e-mail ou URL para o User-Agent (padrao: $WIKIMEDIA_CONTACT)")
    args = ap.parse_args()

    if args.contato:
        _user_agent = USER_AGENT.format(contato=args.contato)
    else:
        print("AVISO: sem --contato / WIKIMEDIA_CONTACT; a Wikimedia pode limitar as requisicoes.")

    pasta_img = args.imagens
    caminho_csv = args.metadata
    os.makedirs(pasta_img, exist_ok=True)
    os.makedirs(os.path.dirname(caminho_csv) or ".", exist_ok=True)

    with open(args.links, encoding="utf-8") as f:
        titulos = [para_titulo(l) for l in f if l.strip() and not l.lstrip().startswith("#")]
    titulos = list(dict.fromkeys(titulos))
    print(f"{len(titulos)} imagens na lista.")

    # Registros anteriores: chave = pagina do Commons (estavel entre execucoes).
    registros = {l["url_pagina"]: l for l in ler_csv(caminho_csv)}
    usados = {l["arquivo"].lower() for l in registros.values()}
    novas, reaproveitadas, falhas = 0, 0, []
    hoje = date.today().isoformat()

    for i in range(0, len(titulos), BATCH):
        lote = titulos[i:i + BATCH]
        dados = consultar(lote, args.largura)

        # titulo pedido -> titulo final (apos normalizacao/redirecionamento)
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
                falhas.append((pedido, "nao encontrado no Commons"))
                continue

            info = pag["imageinfo"][0]
            anterior = registros.get(info["descriptionurl"])
            if anterior and os.path.exists(os.path.join(pasta_img, anterior["arquivo"])):
                reaproveitadas += 1
                continue

            meta = info.get("extmetadata", {})
            val = lambda k: limpar_html(meta.get(k, {}).get("value", ""))

            url_dl = info.get("thumburl") if args.largura else info["url"]
            url_dl = url_dl or info["url"]
            nome = anterior["arquivo"] if anterior else nome_seguro(pag["title"], usados)
            destino = os.path.join(pasta_img, nome)

            print(f"Baixando {nome} ...")
            try:
                sha = baixar(url_dl, destino)
            except Exception as e:
                falhas.append((pedido, f"erro no download: {e}"))
                if not anterior:
                    usados.discard(nome.lower())
                continue

            registros[info["descriptionurl"]] = {
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
            }
            novas += 1
            # grava a cada imagem: uma interrupcao nao perde a proveniencia ja baixada
            gravar_csv(caminho_csv, registros.values())
            time.sleep(args.pausa)

    gravar_csv(caminho_csv, registros.values())

    print(f"\n{novas} baixadas, {reaproveitadas} ja existentes -> {pasta_img}")
    print(f"Fontes ({len(registros)} registros) -> {caminho_csv}")
    print("Proximo passo: python -m src prepare")

    if falhas:
        print("\nFalhas:")
        for t, motivo in falhas:
            print(f"  - {t}: {motivo}")

    sem_lic = sorted(l["arquivo"] for l in registros.values() if not l["licenca"])
    if sem_lic:
        print("\nATENCAO: sem licenca identificada (confira a pagina manualmente):")
        for a in sem_lic:
            print("  -", a)

    if falhas or sem_lic:
        sys.exit(1)


if __name__ == "__main__":
    main()
