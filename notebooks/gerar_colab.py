"""Gera notebooks/colab_pipeline.ipynb.  Uso: python notebooks/gerar_colab.py"""
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
# Image Captioning PT-BR — Porto de Salvador: SLM base × pré-treino × fine-tuning × LoRA × QLoRA

Este notebook roda o experimento completo no Colab:

1. **rotular** as imagens com o Claude (`claude-opus-5-5`) — legendas de referência de todos os splits;
2. **dividir** em treino / validação / teste;
3. **treinar** as variantes do SLM (Manacá-1B + encoder SigLIP + projetor):
   `base` → `pretrain` → `finetune` / `lora` / `qlora`;
4. **gerar** as legendas do teste com cada variante e com o **modelo gold** (Gemini Pro, via API);
5. **comparar** tudo por métricas (BLEU, ROUGE-L, CIDEr, BERTScore, CLIPScore) e custo de treino.

O notebook **não duplica código**: clona o repositório [`ufba-portos-image-captioning`]({REPO.replace('.git', '')})
e chama `python -m src ...`, exatamente como na sua máquina. Toda escolha é parâmetro de
`configs/default.yaml`, ajustável por perfil (`configs/perfis/*.yaml`) ou por `--set chave=valor`.

**GPU:** `Ambiente de execução → Alterar o tipo de ambiente de execução`.
O plano Google AI Pro inclui unidades de computação do Colab — dá para escolher **L4** ou **A100**.

| GPU | perfil | fine-tuning |
|---|---|---|
| T4 (15 GB) | `colab_t4` | **parcial** (últimas 8 camadas) — o completo não cabe |
| L4 (22,5 GB) | `colab_l4` | completo (fp32 + Adam 8 bits) |
| A100 (40 GB) | `colab_a100` | completo, com folga |
""")

md("## 1. GPU e perfil")

code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv

import subprocess

gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout
PERFIL = "colab_a100" if "A100" in gpu else "colab_l4" if "L4" in gpu else "colab_t4"
# PERFIL = "colab_t4"   # descomente para forçar
print("GPU:", gpu.strip() or "nenhuma", "| perfil:", PERFIL)
""")

md(f"""
## 2. Clonar o repositório (sempre a `{BRANCH}`)

Se a pasta já existir de uma execução anterior, a célula só faz `git pull`.
Se o repo for **privado**, o clone falha: compacte a pasta do projeto, suba pelo painel de
arquivos e descompacte em `/content`. Não coloque token do GitHub no notebook.
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
## 3. Dependências

O Colab já traz PyTorch. Se aparecer o aviso pedindo para reiniciar a sessão, reinicie e
continue da célula 4.
""")

code("""
!pip install -q -r requirements.txt

import torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
""")

md("""
## 4. Onde os dados e os modelos vivem

Sessão do Colab morre e leva tudo junto — por isso tudo (imagens, rótulos, modelos, resultados)
vai para o **Google Drive** por padrão. O fine-tuning completo ocupa ~3,5 GB; o resto é pequeno.

`ARGS` é repassado a todo comando: perfil de GPU + raiz dos caminhos. Para mudar qualquer
parâmetro, acrescente `--set chave=valor` (ex: `--set training.lora.r=32`).
""")

code("""
from pathlib import Path

USAR_DRIVE = True  # False = tudo em /content/work (some quando a sessão cair)

if USAR_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = Path("/content/drive/MyDrive/ufba-portos-captioning")
else:
    BASE = Path("/content/work")
BASE.mkdir(parents=True, exist_ok=True)

ARGS = f"--config configs/perfis/{PERFIL}.yaml --set paths.root={BASE}"
print("Raiz:", BASE)
!python -m src show-config {ARGS} | head -40
""")

md("""
## 5. Chaves de API (Secrets do Colab)

Crie, no ícone da chave 🔑 da barra lateral, os secrets **`ANTHROPIC_API_KEY`** (rotulagem) e
**`GEMINI_API_KEY`** (modelo gold — gere em aistudio.google.com/apikey; a assinatura Google AI Pro
não inclui a API, mas o AI Studio tem cota gratuita). Nunca escreva chave numa célula.
""")

code("""
import os
from google.colab import userdata

for nome in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
    try:
        os.environ[nome] = userdata.get(nome)
        print(f"{nome}: carregada")
    except Exception as exc:
        print(f"{nome}: ausente ({type(exc).__name__}) — os passos que dependem dela vão falhar")
""")

md("""
## 6. (Opcional) Teste de fumaça

Roda o pipeline inteiro com imagens sintéticas e modelos minúsculos, sem API e sem baixar nada,
em `/content/smoke`. Leva ~1–2 min e prova que o ambiente está certo antes de gastar GPU.
""")

code("""
!python -m src run --config configs/perfis/smoke.yaml --set paths.root=/content/smoke --set dev.tiny_dir=/content/smoke/tiny --set model.llm_id=/content/smoke/tiny/llm --set model.vision_id=/content/smoke/tiny/vision
""")

md("""
## 7. Subir as imagens

- **Drive (recomendado):** coloque os arquivos originais em `MyDrive/ufba-portos-captioning/data/raw/`.
- **Upload manual:** rode a célula abaixo.
""")

code("""
from google.colab import files
import shutil

RAW = BASE / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
enviados = files.upload()
for nome in enviados:
    shutil.move(nome, RAW / nome)
print(f"{len(enviados)} arquivos em {RAW}")
""")

md("""
## 8. Normalizar, rotular e dividir

`prepare` corrige EXIF, reduz para 1024 px, deduplica e cria o `metadata.csv` (preencha
`fonte`, `url`, `licenca`). `label` chama o Claude — comece com `--set labeling.limit=5`
para ver o custo. É retomável: imagens já rotuladas são puladas.
""")

code("""
!python -m src prepare {ARGS}
!python -m src label {ARGS} --set labeling.limit=5
""")

code("""
!python -m src label {ARGS}
!python -m src split {ARGS}
""")

md("""
## 9. Treinar as variantes

Cada célula grava `outputs/models/<variante>/` com `train_summary.json` (parâmetros treináveis,
VRAM de pico, tempo, losses). `finetune`, `lora` e `qlora` partem do **pré-treinado** por padrão;
para partir do base: `--set training.<estagio>.init_from=base`.
""")

code("""
!python -m src train --stage base {ARGS}
!python -m src train --stage pretrain {ARGS}
""")

code("""
!python -m src train --stage finetune {ARGS}
""")

code("""
!python -m src train --stage lora {ARGS}
""")

code("""
!python -m src train --stage qlora {ARGS}
""")

md("""
## 10. Gerar as legendas do teste

Todas as variantes locais + o modelo gold (Gemini). Retomável.
""")

code("""
!python -m src predict --variant all {ARGS}
""")

md("""
## 11. Comparar

Gera `results/comparativo.md` (métricas + custo de treino + exemplos) e `comparativo.csv`.
Com menos de ~100 imagens de teste o CIDEr é instável — o `N` vai junto na tabela.
""")

code("""
!python -m src evaluate {ARGS}

import pandas as pd
pd.read_csv(BASE / "results" / "comparativo.csv").set_index("variante").round(4)
""")

code("""
from IPython.display import Markdown, display
display(Markdown((BASE / "results" / "comparativo.md").read_text(encoding="utf-8")))
""")

md("""
## 12. Olhar nas imagens (análise qualitativa)
""")

code("""
from IPython.display import display
from PIL import Image

from src.common import index_by, read_json, read_jsonl

QUANTAS = 5
teste = read_json(BASE / "data" / "splits.json")["test"][:QUANTAS]
refs = index_by(read_jsonl(BASE / "data" / "labels.jsonl"))
preds = {p.stem.replace("preds_", ""): index_by(read_jsonl(p)) for p in sorted((BASE / "results").glob("preds_*.jsonl"))}

for image_id in teste:
    caminho = BASE / "data" / "images" / f"{image_id}.jpg"
    if caminho.exists():
        img = Image.open(caminho)
        img.thumbnail((512, 512))
        display(img)
    print(f"{image_id}\\n  referência (Claude): {refs[image_id]['legendas'][0]}")
    for nome, linhas in preds.items():
        if image_id in linhas:
            print(f"  {nome:>9}: {linhas[image_id]['legenda']}")
    print()
""")

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "L4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

destino = Path(__file__).with_name("colab_pipeline.ipynb")
destino.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"{destino} gerado com {len(cells)} celulas.")
