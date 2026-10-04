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
# Treino no Colab — Qwen3-VL-2B para legendas do Porto de Salvador

Divisão de trabalho do projeto:

- **máquina local:** preparar imagens, rotular (Claude), dividir, **inferência e avaliação**;
- **Colab (este notebook):** **treino** das variantes `base` → `pretrain` → `finetune` / `lora` / `qlora`
  e exportação de um pacote `modelos.zip` para a inferência local.

O notebook **não duplica código**: clona o repositório [`ufba-portos-image-captioning`]({REPO.replace('.git', '')})
e chama `python -m src ...`, como na sua máquina. Parâmetros em `configs/default.yaml`,
perfis em `configs/perfis/`, ajustes com `--set chave=valor`.

Funciona no navegador **e no VS Code** (extensão oficial *Google Colab*: abra este `.ipynb`,
*Select Kernel → Colab → New Colab Server* e escolha a GPU). No VS Code, os arquivos da sua
máquina **não** aparecem no servidor — os dados (imagens normalizadas, `labels.jsonl`,
`splits.json`) vêm do próprio repositório clonado, e os modelos treinados vão para o Google Drive.

| GPU | perfil | fine-tuning |
|---|---|---|
| T4 (15 GB) | `colab_t4` | **parcial** (últimas 8 camadas) — o completo não cabe |
| L4 (22,5 GB) | `colab_l4` | completo (pesos bf16 + Adam 8 bits) |
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

Se a pasta já existir, só faz `git pull`. Repo **privado**: suba a pasta do projeto compactada e
descompacte em `/content`. Não coloque token do GitHub no notebook.
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

O Colab já traz PyTorch. Se pedir para reiniciar a sessão, reinicie e continue da célula 4.

A célula também **remove o `torchao`** pré-instalado no Colab: o projeto não o usa (QLoRA usa
`bitsandbytes`), mas o `peft` aborta o LoRA quando encontra uma versão antiga dele
(`ImportError: Found an incompatible version of torchao`).
""")

code("""
!pip install -q -r requirements.txt
!pip uninstall -y -q torchao   # versao do Colab e incompativel com o peft; nao e usado no projeto

import torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
""")

md("""
## 4. Dados (do repositório) e Google Drive (saída)

Os **dados de entrada vêm do clone**: `data/images/` (normalizadas), `data/labels.jsonl` e
`data/splits.json` são versionados. Rode `prepare`, rotulagem e `split` na sua máquina,
faça commit **e push**, e o `git pull` da célula 2 traz tudo.

O **Drive guarda só a saída**, que precisa sobreviver ao fim da sessão:

```
MyDrive/ufba-portos-captioning/outputs/models/<variante>/   modelos treinados
MyDrive/ufba-portos-captioning/exports/modelos.zip          pacote para a inferência local
```

**VS Code:** se `drive.mount` falhar, rode o comando *Colab: Mount Google Drive to Server...*
(paleta de comandos) e execute a célula de novo.
""")

code("""
from pathlib import Path

BASE = Path("/content/drive/MyDrive/ufba-portos-captioning")
if not Path("/content/drive/MyDrive").exists():
    from google.colab import drive
    drive.mount("/content/drive")
BASE.mkdir(parents=True, exist_ok=True)

ARGS = (f"--config configs/perfis/{PERFIL}.yaml"
        f" --set paths.models_dir={BASE / 'outputs' / 'models'}"
        f" --set paths.exports_dir={BASE / 'exports'}")
for nome in ("data/labels.jsonl", "data/splits.json"):
    print(f"{nome}: {'ok' if Path(nome).exists() else 'FALTANDO (faça commit e push na sua máquina)'}")
print("imagens:", len(list(Path("data/images").glob("*.jpg"))))
print("saída no Drive:", BASE)
""")

md("""
## 5. (Opcional) Teste de fumaça

Pipeline inteiro com imagens sintéticas, 2 passos de treino por variante e sem chamar API,
em `/content/smoke`. Baixa o Qwen3-VL-2B (~4,5 GB, fica em cache para o treino de verdade) e
leva poucos minutos. Prova que o ambiente está certo antes do treino completo.
""")

code("""
!python -m src run --config configs/perfis/smoke.yaml --set paths.root=/content/smoke
""")

md("""
## 6. (Opcional) Rotular e dividir aqui

Só se você **não** fez isso localmente (o resultado fica no clone, que some com a sessão:
copie `data/labels.jsonl` e `data/splits.json` de volta para o repositório). Precisa do secret `ANTHROPIC_API_KEY` (ícone 🔑 no
navegador). No VS Code os Secrets do Colab podem não estar disponíveis — a célula pede a chave
digitada (não fica salva no notebook).
""")

code("""
import os
from getpass import getpass

try:
    from google.colab import userdata
    os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
except Exception:
    os.environ["ANTHROPIC_API_KEY"] = getpass("ANTHROPIC_API_KEY: ")

!python -m src label {ARGS}
!python -m src split {ARGS}
""")

md("""
## 7. Treinar as variantes

Cada célula grava `outputs/models/<variante>/` no Drive com `train_summary.json` (parâmetros
treináveis, VRAM de pico, tempo, losses). `finetune`, `lora` e `qlora` partem do **pré-treinado**;
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

code("""
import json
import pandas as pd

linhas = []
for v in ("pretrain", "finetune", "lora", "qlora"):
    p = BASE / "outputs" / "models" / v / "train_summary.json"
    if p.exists():
        s = json.loads(p.read_text(encoding="utf-8"))
        linhas.append({"variante": v, "treinaveis (M)": s["params_treinaveis"] / 1e6, "VRAM pico (GB)": s["vram_pico_gb"],
                       "tempo (min)": s["tempo_s"] / 60, "melhor val loss": s["melhor_val_loss"], "quantizacao": s["quantizacao"]})
pd.DataFrame(linhas).set_index("variante").round(3)
""")

md("""
## 8. Exportar para a inferência local

Gera `MyDrive/ufba-portos-captioning/exports/modelos.zip` (~4,5 GB com o fine-tuning completo)
com as variantes, `labels.jsonl` e `splits.json`. Na sua máquina:

```bash
python -m src import --from <caminho do modelos.zip baixado>
python -m src run --config configs/perfis/local_4gb.yaml
```

Com o *Google Drive para desktop* dá para apontar `--from` direto para a pasta
`ufba-portos-captioning` sincronizada, sem baixar o zip.
""")

code("""
!python -m src export {ARGS}
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
