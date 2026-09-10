"""Gera notebooks/colab_pipeline.ipynb."""
import json
from pathlib import Path

REPO = "https://github.com/alanssoares/ufba-portos-image-captioning.git"
BRANCH = "main"

cells = []


def md(texto):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": texto.strip().splitlines(keepends=True)})


def code(texto):
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": texto.strip().splitlines(keepends=True),
        }
    )


md(f"""
# Image Captioning PT-BR — Porto de Salvador

Pipeline completo no Colab: preparar o dataset, rascunhar legendas, revisar, gerar predições
com vários modelos e comparar por métricas.

Este notebook **não duplica código**: ele clona o repositório
[`ufba-portos-image-captioning`]({REPO.replace('.git', '')}) e chama os mesmos módulos `src/*`
que rodam na sua máquina. Corrigiu um bug aqui? Vale lá também, e vice-versa.

**Antes de começar:** `Ambiente de execução → Alterar o tipo de ambiente de execução → GPU (T4)`.

A T4 tem 16 GB de VRAM — quatro vezes o que a sua GPU local tem. Aqui dá para rodar
Florence-2-large e Qwen2.5-VL sem apertar.

---
""")

md("## 1. Conferir a GPU")

code("""
!nvidia-smi
""")

md(f"""
## 2. Clonar o repositório (sempre a `main`)

O notebook roda contra a **`main`** — nunca contra branch de feature. Se você está
desenvolvendo numa branch, faça o merge antes de rodar aqui: o que o Colab executa é o que
está publicado.

Se a pasta já existir de uma execução anterior, esta célula atualiza com `git pull` em vez de
clonar de novo.

Se o repo for **privado**, o clone vai falhar. Nesse caso, compacte a pasta do projeto, faça
upload pelo painel de arquivos do Colab e descompacte em `/content` — o resto do notebook
funciona igual. Não coloque token do GitHub no notebook.
""")

code(f"""
import os

REPO = "{REPO}"
BRANCH = "{BRANCH}"
PROJETO = "/content/ufba-portos-image-captioning"

if os.path.exists(PROJETO):
    !git -C {{PROJETO}} checkout {{BRANCH}} && git -C {{PROJETO}} pull --ff-only
else:
    !git clone --branch {{BRANCH}} --single-branch {{REPO}} {{PROJETO}}

%cd {{PROJETO}}
!git log --oneline -1
""")

md("""
## 3. Instalar dependências

O Colab já traz PyTorch, então instalamos só o resto. Se aparecer o aviso pedindo para
reiniciar a sessão, reinicie e continue da célula 4 — o clone e o Drive continuam no lugar.
""")

code("""
!pip install -q -r requirements.txt

import torch
print("torch", torch.__version__, "| CUDA disponível:", torch.cuda.is_available())
""")

md("""
## 4. Onde os dados vivem (monte o Drive!)

Sessão do Colab morre e leva tudo junto. Como a revisão humana das legendas é a etapa mais
cara do projeto, os dados ficam no **Google Drive** — se a sessão cair, você recomeça sem
perder anotação.

Todos os scripts aceitam caminhos por flag, então basta apontá-los para o Drive.
""")

code("""
from pathlib import Path

USAR_DRIVE = True  # False = tudo em /content (some quando a sessão cair)

if USAR_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = Path("/content/drive/MyDrive/ufba-portos-captioning")
else:
    BASE = Path("/content/work")

RAW = BASE / "raw"                  # imagens originais que você subir
IMAGES = BASE / "images"            # normalizadas pelo prepare_images
MANIFEST = BASE / "metadata.csv"    # proveniência e licença
DRAFTS = BASE / "drafts.jsonl"      # rascunhos do Claude
CAPTIONS = BASE / "captions.jsonl"  # ground truth revisado
RESULTS = BASE / "results"          # predições e métricas

for pasta in (RAW, IMAGES, RESULTS):
    pasta.mkdir(parents=True, exist_ok=True)

print("Base:", BASE)
""")

md("""
## 5. Subir as imagens

Duas opções:

- **Drive (recomendado):** jogue os arquivos direto na pasta `ufba-portos-captioning/raw`
  do seu Drive, pelo navegador ou pelo app. Não precisa rodar a célula de upload.
- **Upload manual:** rode a célula abaixo e escolha os arquivos (some quando a sessão cair,
  a não ser que `USAR_DRIVE = True`).
""")

code("""
from google.colab import files
import shutil

enviados = files.upload()
for nome in enviados:
    shutil.move(nome, RAW / nome)
print(f"{len(enviados)} arquivos em {RAW}")
""")

md("""
## 6. Normalizar e catalogar

EXIF corrigido, RGB, lado maior em 1024 px, deduplicação por SHA-1, e um manifesto com uma
linha por imagem.

**Preencha `fonte`, `url` e `licenca` no `metadata.csv`** (dá para abrir direto no Drive) —
sem isso o dataset não é publicável nem citável na monografia.
""")

code("""
!python -m src.prepare_images --src "{RAW}" --dst "{IMAGES}" --manifest "{MANIFEST}" --max-side 1024

import pandas as pd
pd.read_csv(MANIFEST).head(10)
""")

md("""
## 7. Rascunhar legendas com o Claude

A chave vem dos **Secrets do Colab** (ícone da chave 🔑 na barra lateral): crie um secret
chamado `ANTHROPIC_API_KEY` e ative o acesso para este notebook. Nunca escreva a chave numa
célula — o notebook vai para o Git com o que estiver escrito nele.

Comece com `--limit 5` para ver o custo antes de rodar no dataset inteiro.
""")

code("""
import os
from google.colab import userdata

os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
print("Chave carregada dos Secrets do Colab.")
""")

code("""
!python -m src.draft_captions --images "{IMAGES}" --out "{DRAFTS}" --limit 5
""")

md("""
## 8. Revisar (a etapa que gera o dado de verdade)

O app Streamlit do repo não roda bem no Colab (precisaria de túnel), então aqui vai um
revisor mínimo em ipywidgets. Ele escreve **no mesmo formato** do `captions.jsonl`, então
você pode revisar parte aqui, parte no app local, sem conflito.

Legenda revisada por humano é o que separa um dataset de um monte de saída de LLM.
""")

code("""
import datetime as dt

import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image

from src.common import index_by, list_images, read_jsonl, write_jsonl

N_LEGENDAS = 3
imagens = list_images(IMAGES)
rascunhos = index_by(read_jsonl(DRAFTS))
estado = {"i": 0}


def salvar(image_id, legendas, objetos, descartada):
    final = index_by(read_jsonl(CAPTIONS))
    final[image_id] = {
        "image_id": image_id,
        "legendas": [t.strip() for t in legendas if t.strip()],
        "objetos": [o.strip().lower() for o in objetos.split(",") if o.strip()],
        "observacao": "",
        "descartada": bool(descartada),
        "revisor": "colab",
        "revisado_em": dt.datetime.now().isoformat(timespec="seconds"),
        "origem_rascunho": rascunhos.get(image_id, {}).get("modelo", ""),
    }
    write_jsonl(CAPTIONS, [final[k] for k in sorted(final)])


def mostrar():
    clear_output(wait=True)
    if not imagens:
        print(f"Nenhuma imagem em {IMAGES}.")
        return
    estado["i"] %= len(imagens)
    path = imagens[estado["i"]]
    image_id = path.stem
    final = index_by(read_jsonl(CAPTIONS))
    base = (final.get(image_id) or rascunhos.get(image_id) or {}).get("legendas", [])
    revisadas = sum(1 for p in imagens if p.stem in final)

    display(Image.open(path).copy())
    print(f"{image_id}  ({estado['i'] + 1}/{len(imagens)})  —  {revisadas} já revisadas")

    campos = [
        widgets.Textarea(value=base[i] if i < len(base) else "", layout=widgets.Layout(width="90%", height="60px"))
        for i in range(N_LEGENDAS)
    ]
    objetos = widgets.Text(
        value=", ".join((final.get(image_id) or rascunhos.get(image_id) or {}).get("objetos", [])),
        description="objetos:",
        layout=widgets.Layout(width="90%"),
    )
    descartar = widgets.Checkbox(value=False, description="descartar (fora do domínio)")
    b_salvar = widgets.Button(description="Salvar e próxima", button_style="primary")
    b_pular = widgets.Button(description="Pular")
    b_voltar = widgets.Button(description="Voltar")

    def ao_salvar(_):
        salvar(image_id, [c.value for c in campos], objetos.value, descartar.value)
        estado["i"] += 1
        mostrar()

    b_salvar.on_click(ao_salvar)
    b_pular.on_click(lambda _: (estado.update(i=estado["i"] + 1), mostrar()))
    b_voltar.on_click(lambda _: (estado.update(i=estado["i"] - 1), mostrar()))

    display(widgets.VBox(campos + [objetos, descartar, widgets.HBox([b_voltar, b_pular, b_salvar])]))


mostrar()
""")

md("""
## 9. Gerar as predições

Na T4 dá para rodar todos. `--translate` é obrigatório nos modelos que só legendam em inglês
(BLIP e Florence-2) — sem ele a legenda em português fica vazia e a avaliação ignora a linha.

| modelo | o que é | idioma |
|---|---|---|
| `blip` | baseline encoder-decoder clássico | EN → traduzido |
| `florence2` | Florence-2, task `<MORE_DETAILED_CAPTION>` | EN → traduzido |
| `qwen25vl` | Qwen2.5-VL-3B (4 bits) | PT-BR direto |
| `claude` | VLM via API — teto de qualidade | PT-BR direto |
""")

code("""
!python -m src.run_captioning --model blip --images "{IMAGES}" --out "{RESULTS}/preds_blip.jsonl" --translate
""")

code("""
!python -m src.run_captioning --model florence2 --images "{IMAGES}" --out "{RESULTS}/preds_florence2.jsonl" --translate
""")

code("""
!python -m src.run_captioning --model qwen25vl --images "{IMAGES}" --out "{RESULTS}/preds_qwen25vl.jsonl"
""")

code("""
!python -m src.run_captioning --model claude --images "{IMAGES}" --out "{RESULTS}/preds_claude.jsonl"
""")

md("""
## 10. Avaliar e comparar

BLEU / ROUGE-L / CIDEr (sem Java), BERTScore com BERTimbau, CLIPScore e RefCLIPScore
multilíngues, mais diversidade lexical.

Lembre do `N`: com menos de ~100 imagens o CIDEr é instável, porque calcula o IDF a partir
do próprio conjunto avaliado.
""")

code("""
!python -m src.evaluate --preds "{RESULTS}/preds_*.jsonl" --refs "{CAPTIONS}" --images "{IMAGES}" --out "{RESULTS}/comparativo.md"
""")

code("""
import json

import pandas as pd

metricas = [json.load(open(p, encoding="utf-8")) for p in sorted(RESULTS.glob("metrics_*.json"))]
pd.DataFrame(metricas).set_index("modelo").drop(columns=["arquivo"], errors="ignore").round(4)
""")

md("""
## 11. Olhar nos resultados (análise qualitativa)

A tabela diz *quanto* cada modelo erra; esta célula mostra *como* — é daqui que sai a
discussão sobre o gap de domínio, com exemplos de legenda genérica ("um barco grande perto
de um prédio") contra o vocabulário técnico das referências.
""")

code("""
from IPython.display import display
from PIL import Image

from src.common import index_by, read_jsonl

QUANTAS = 5

refs = {r["image_id"]: r for r in read_jsonl(CAPTIONS) if not r.get("descartada")}
preds = {p.stem.replace("preds_", ""): index_by(read_jsonl(p)) for p in sorted(RESULTS.glob("preds_*.jsonl"))}

for image_id in list(refs)[:QUANTAS]:
    caminho = IMAGES / f"{image_id}.jpg"
    if caminho.exists():
        foto = Image.open(caminho)
        foto.thumbnail((420, 420))
        display(foto)
    print(f"=== {image_id} ===")
    print(f"  [referência] {refs[image_id]['legendas'][0]}")
    for modelo, linhas in preds.items():
        legenda = linhas.get(image_id, {}).get("legenda", "—")
        print(f"  [{modelo}] {legenda}")
    print()
""")

md("""
## 12. Levar os resultados embora

Com `USAR_DRIVE = True` tudo já está salvo no Drive — não precisa fazer nada. A célula
abaixo é só para baixar o comparativo direto para a máquina.
""")

code("""
from google.colab import files

files.download(str(RESULTS / "comparativo.md"))
""")

md("""
---

## Notas

- **Custo de sessão:** a T4 gratuita desconecta por inatividade e tem cota diária. Rode as
  etapas caras (predições) em blocos, com os dados no Drive.
- **Fine-tuning:** este notebook cobre só inferência e avaliação, igual ao repo. LoRA no BLIP
  caberia na T4 e seria o passo natural depois de ter o dataset revisado — mas é implementação
  nova, não está aqui.
- **Divergência com o local:** se você mudar `src/*` aqui dentro do Colab, a mudança morre com
  a sessão — e a célula 2 vai reclamar do `git pull` num diretório sujo. Edite no repo, faça
  commit, merge na `main`, e reexecute a célula 2.
""")

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True, "name": "colab_pipeline.ipynb"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

destino = Path("notebooks/colab_pipeline.ipynb")
destino.parent.mkdir(parents=True, exist_ok=True)
destino.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{destino}: {len(cells)} células")
