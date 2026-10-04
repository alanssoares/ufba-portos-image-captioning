# Image Captioning no domínio portuário — Porto de Salvador (BA)

Projeto acadêmico (UFBA) que **monta um dataset próprio** de imagens do Porto de Salvador com
legendas em português do Brasil e **avalia um SLM multimodal (Qwen3-VL-2B)** adaptado ao
domínio portuário — comparando seis variantes do mesmo modelo, do original zero-shot até
fine-tuning completo, LoRA e QLoRA, contra um VLM de fronteira.

| | |
|---|---|
| **Dataset** | imagens do Porto de Salvador, normalizadas e com proveniência/licença por imagem ([`docs/dataset.md`](docs/dataset.md)) |
| **Referências** | 3 legendas por imagem geradas pelo **Claude** (`claude-opus-5-5`) com glossário portuário |
| **SLM** | **Qwen3-VL-2B-Instruct** (`Qwen/Qwen3-VL-2B-Instruct`, ~2,1 B: encoder de visão + conector + LLM Qwen3 de 1,7 B), multilíngue |
| **Variantes** | `base` · `pretrain` · `finetune` · `lora` · `qlora` · `gold` (Gemini Flash via API) |
| **Métricas** | BLEU, ROUGE-L, CIDEr, BERTScore (BERTimbau), CLIPScore/RefCLIPScore, diversidade + custo de treino |
| **Onde roda** | dados, inferência e avaliação **localmente**; treino no **Google Colab** (navegador ou VS Code) |
| **Configuração** | tudo em [`configs/default.yaml`](configs/default.yaml), com perfis por hardware e `--set chave=valor` |

## Sumário

1. [Contexto e hipótese](#1-contexto-e-hipótese)
2. [Arquitetura e variantes](#2-arquitetura-e-variantes)
3. [Instalação](#3-instalação)
4. [Teste rápido, sem dataset](#4-teste-rápido-sem-dataset)
5. [Montando o dataset próprio](#5-montando-o-dataset-próprio)
6. [Treino no Colab](#6-treino-no-colab)
7. [Inferência e avaliação local](#7-inferência-e-avaliação-local)
8. [Configuração e parâmetros](#8-configuração-e-parâmetros)
9. [Referência de comandos](#9-referência-de-comandos)
10. [Estrutura do repositório e saídas](#10-estrutura-do-repositório-e-saídas)
11. [Decisões de avaliação](#11-decisões-de-avaliação)
12. [Reprodutibilidade](#12-reprodutibilidade)
13. [Limites conhecidos e solução de problemas](#13-limites-conhecidos-e-solução-de-problemas)
14. [Documentação, licenças e créditos](#14-documentação-licenças-e-créditos)

---

## 1. Contexto e hipótese

Modelos genéricos de captioning descrevem uma foto do Tecon Salvador como *"um navio grande em
um porto"*. O vocabulário técnico — portêiner, transtêiner, reach stacker, berço de atracação —
simplesmente não aparece, e em português a situação é pior: a maioria dos modelos legenda em
inglês.

**Hipótese:** um VLM pequeno (~2 B de parâmetros), ajustado com poucas centenas de imagens do
domínio, aproxima-se de um VLM de fronteira na tarefa de legendar imagens portuárias em
português — a uma fração do custo.

**Perguntas de pesquisa**

1. Qual o gap de domínio do modelo original, e quanto cada etapa recupera? (`base` → `pretrain` → `finetune`)
2. Quanto do ganho do fine-tuning completo se recupera com LoRA, treinando ~1% dos parâmetros?
3. A quantização em 4 bits do QLoRA custa qualidade? (`qlora` × `lora`)
4. Qual a distância entre o SLM ajustado e um VLM de fronteira (`gold`)?
5. Onde os modelos erram — vocabulário técnico, confusões entre equipamentos parecidos, alucinações?

## 2. Arquitetura e variantes

O Qwen3-VL-2B já é multimodal: um encoder de visão, um conector ("merger") que transforma as
features visuais em tokens para o LLM, e o LLM Qwen3. O treino usa o chat template do modelo:

```
imagem (≤ 448 px) ─► encoder de visão ViT (~0,4 B) ─► conector merger (~0,1 B) ─► ~196 tokens de imagem
                                                                                     │
LLM Qwen3 (~1,7 B):  <|im_start|>user [imagem] {instrução}<|im_end|>                 ◄┘
                     <|im_start|>assistant\n{legenda}<|im_end|>     ← loss só aqui
```

| variante | ponto de partida | o que treina | quantização | pergunta que responde |
|---|---|---|---|---|
| `base` | pesos originais | nada — zero-shot com a instrução | — | gap de domínio do modelo original |
| `pretrain` | `base` | só o conector visão→LLM (~0,1 B) | — | quanto vale adaptar "como o LLM lê a imagem" |
| `finetune` | `pretrain`* | conector + **todos** os pesos do LLM (~1,7 B) | — | teto do SLM |
| `lora` | `pretrain`* | conector + adaptadores LoRA (~1% do LLM) | — | eficiência de parâmetros |
| `qlora` | `pretrain`* | conector + adaptadores LoRA | LLM em 4 bits (NF4) | custo da quantização |
| `gold` | — | nada — **Gemini Flash** via API (`gemini-3.8-flash`) | — | distância para um VLM grande de outro fornecedor |

\* `--set training.<estagio>.init_from=base` para partir do base.

Por que o Qwen3-VL e não o Manacá-1B (primeira escolha): o Manacá é só de texto e exigiria
treinar do zero a ponte visão–linguagem, o que pede centenas de milhares de pares — ver
[`docs/metodologia-variantes.md`](docs/metodologia-variantes.md#2-o-modelo-qwen3-vl-2b-instruct).

O **rotulador (Claude) não é uma linha da tabela**: as legendas dele são a referência, e medir um
modelo contra ele mesmo seria circular. O teto externo é o Gemini, de outro fornecedor.
Justificativas completas em [`docs/metodologia-variantes.md`](docs/metodologia-variantes.md).

## 3. Instalação

No Colab o notebook instala tudo sozinho. Na máquina local (Python 3.11):

```bash
uv venv --python 3.11
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
```

O PyTorch vai primeiro, no índice da CUDA certa (`cu124` para o driver 566.x). Sem GPU, troque o
índice por `cpu`.

Chaves de API, como variáveis de ambiente (nunca no código nem no notebook):

| variável | para quê | onde obter |
|---|---|---|
| `ANTHROPIC_API_KEY` | rotulagem (`label`) | console.anthropic.com |
| `GEMINI_API_KEY` | modelo gold (`predict --variant gold`) | aistudio.google.com/apikey — a assinatura Google AI Pro não inclui a API |

```bash
export ANTHROPIC_API_KEY=sk-ant-...          # PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY=...
```

## 4. Teste rápido, sem dataset

```bash
pytest                                                 # config, split, export/import (sem GPU)
python -m src run --config configs/perfis/smoke.yaml   # pipeline inteiro (precisa de GPU)
```

O smoke test gera imagens sintéticas e roda **o pipeline inteiro** — treino das 5 variantes com
2 passos cada, export, predição, avaliação — sem chamar nenhuma API (o gold vira um modelo
"dummy"). Usa o Qwen3-VL-2B de verdade (baixa ~4,5 GB) e precisa de GPU CUDA: rode-o pela célula
opcional do notebook no Colab (a T4 serve), antes do treino completo. As métricas não significam
nada; o teste prova que as peças encaixam. Via pytest: `RUN_SMOKE=1 pytest tests/test_smoke_pipeline.py`.

## 5. Montando o dataset próprio

O dataset é construído por um pipeline de três comandos. Guia completo — o que fotografar,
quantas imagens, fontes permitidas, licenças, formatos e datasheet — em
[`docs/dataset.md`](docs/dataset.md).

```
data/raw/  ──prepare──►  data/images/ + metadata.csv  ──label──►  labels.jsonl  ──split──►  splits.json
```

### 5.1 Coletar

Coloque os arquivos originais em `data/raw/` (qualquer formato de imagem, subpastas livres).
Cubra a variedade do porto: terminal de contêineres, granéis, cruzeiros, rebocadores,
infraestrutura, operação, vistas gerais — com pontos de vista, horários e climas variados.
Recomendado: **≥ 700 imagens** (≥ 100 no teste, para o CIDEr ficar estável). Use só imagens com
licença que permita uso acadêmico (fotos próprias, Wikimedia Commons, acervos com autorização).

Para imagens do **Wikimedia Commons**, liste os links em `data/sources/commons_links.txt` e rode:

```bash
python -m src.fetch_commons_dataset data/sources/commons_links.txt --contato voce@exemplo.com
```

Baixa os originais em `data/raw/images/` e registra autor, URL e licença em
`data/sources/commons_metadata.csv` (`paths.commons_csv`). É incremental: rodar de novo só baixa
o que é novo e mescla o CSV.

### 5.2 Normalizar — `prepare`

```bash
python -m src prepare
```

Corrige a orientação EXIF, converte para RGB, reduz o lado maior para 1024 px (`prepare.max_side`),
deduplica por SHA-1 e salva como `data/images/psa_XXXX.jpg`. Cria/atualiza `data/metadata.csv`
com uma linha por imagem. As colunas `fonte`, `url` e `licenca` vêm automaticamente do
`data/sources/commons_metadata.csv` (casando pelo nome do arquivo); **para imagens de outras
fontes, preencha-as à mão** — sem isso o dataset não é publicável nem citável. Valores já
preenchidos nunca são sobrescritos. Rodar de novo processa só as imagens novas.

### 5.3 Rotular — `label`

```bash
python -m src label --set labeling.limit=5    # confira qualidade e custo primeiro
python -m src label                           # todas as pendentes
```

O Claude recebe cada imagem com o glossário portuário ([`src/prompts.py`](src/prompts.py)) e
devolve, em JSON estruturado, **3 legendas** com focos fixos (operação · objetos e sua relação ·
visão geral), os **objetos** visíveis e uma flag **`fora_de_dominio`**. O foco é técnico: ação em
curso e identificação dos equipamentos — sem nomes próprios, estado de conservação, clima ou luz.
Regras completas em [`docs/dataset.md`](docs/dataset.md#3-legendas-rótulos). Retomável: imagens já
rotuladas são puladas.

### 5.4 Dividir — `split`

```bash
python -m src split
```

Treino/validação/teste 70/15/15 com semente fixa, excluindo imagens `fora_de_dominio`. O split é
**estável**: ao adicionar imagens, só as novas são sorteadas — o teste existente não muda por
baixo dos modelos já treinados.

### 5.5 O que vai para o git

`data/metadata.csv`, `data/labels.jsonl` e `data/splits.json` são versionados (definem o
experimento). `data/raw/` e `data/images/` ficam fora (peso e direitos de uso).

## 6. Treino no Colab

A GPU local (4 GB) não comporta o treino; ele roda no Colab — as unidades de computação do
Google AI Pro dão acesso a L4 e A100. Treinar no Hugging Face de graça não é viável (o ZeroGPU
gratuito dá 5 min de GPU por dia, com chamadas de 60 s).

**1. Suba os dados para o Drive:** copie `data/images/`, `data/labels.jsonl` e `data/splits.json`
para `MyDrive/ufba-portos-captioning/data/`.

**2. Abra [`notebooks/colab_pipeline.ipynb`](notebooks/colab_pipeline.ipynb)** no Colab — ou no
**VS Code** com a extensão oficial *Google Colab* (*Select Kernel → Colab → New Colab Server*,
escolhendo a GPU). O notebook detecta a GPU e escolhe o perfil, clona a `main`, instala as
dependências, monta o Drive, roda um smoke test opcional, treina as 5 variantes e exporta o pacote.
Em linha de comando, equivale a:

```bash
python -m src run --config configs/perfis/colab_l4.yaml --set paths.root=/content/drive/MyDrive/ufba-portos-captioning
```

| GPU | perfil | fine-tuning completo |
|---|---|---|
| T4 (15 GB) | `colab_t4` | não cabe — vira **parcial** (últimas 8 camadas), registrado no relatório |
| L4 (22,5 GB) | `colab_l4` | sim (pesos bf16 + Adam 8 bits paginado) |
| A100 (40 GB) | `colab_a100` | sim, com folga |

Cada variante é salva em `outputs/models/<variante>/` no Drive, com `train_summary.json`
(parâmetros treináveis, VRAM de pico, tempo, losses por época).

**3. Exporte:** a última etapa (`python -m src export`) gera
`MyDrive/ufba-portos-captioning/exports/modelos.zip` com as variantes **e** o `labels.jsonl` /
`splits.json` usados no treino (~4,5 GB com o fine-tuning completo).

> O notebook sempre clona a branch `main` do GitHub: faça push das mudanças antes de rodar.

## 7. Inferência e avaliação local

```bash
python -m src import --from C:/Users/<voce>/Downloads/modelos.zip
python -m src run --config configs/perfis/local_4gb.yaml
```

- `import` aceita o zip ou a pasta `ufba-portos-captioning` sincronizada pelo *Google Drive para
  desktop*. Traz as variantes e os `labels.jsonl` / `splits.json` do treino — arquivos locais
  diferentes ganham uma cópia `.bak-<data>`, nada é apagado.
- O perfil `local_4gb` roda `predict` das 5 variantes + `gold` e depois `evaluate`. Com 4 GB de
  VRAM, as variantes são carregadas em **4 bits**; para comparar sem quantização use
  `--set device=cpu` (lento) e declare a escolha na monografia.
- Cada variante guarda a impressão digital do split do treino; se o `splits.json` local for
  diferente, o `predict` avisa.

**Saída:** `results/comparativo.md`, com (1) a tabela de métricas, (2) o custo de treino de cada
variante — ponto de partida, quantização, parâmetros treináveis, VRAM de pico, tempo, melhor loss
de validação, tamanho em disco — e (3) exemplos qualitativos lado a lado. A mesma tabela sai em
`results/comparativo.csv` para gráficos.

## 8. Configuração e parâmetros

Toda escolha do experimento está em [`configs/default.yaml`](configs/default.yaml), comentada.
Precedência (a última vence):

```
configs/default.yaml  →  --config perfil.yaml (repetível)  →  --set chave.pontilhada=valor (repetível)
```

| seção | controla |
|---|---|
| `paths.*` | onde ficam dados, modelos, resultados e pacotes (relativos a `paths.root`) |
| `prepare.*` | tamanho máximo, qualidade JPEG, prefixo dos ids |
| `labeling.*` | modelo do rotulador, nº de legendas, tentativas, limite |
| `split.*` | proporções, semente, exclusão de fora-de-domínio, split fixo externo |
| `model.*` | VLM (`model_id`), instrução, prompt de sistema, resolução da imagem, marcador da resposta, dtype |
| `training.common` / `training.<estagio>` | épocas, lr, batch, acumulação, otimizador, gradient checkpointing, modo do LLM (`frozen`/`full`/`partial`/`lora`), conector (`connector_modules`), `init_from`, LoRA (`r`, `alpha`, `dropout`, `target_modules`), quantização |
| `generation.*` | beam search, tamanho, penalidades, inferência em 4 bits |
| `gold_model.*` | provedor e modelo do gold, uso do glossário de domínio |
| `evaluation.*` | split avaliado, BERTScore/CLIPScore, nº de exemplos |
| `transfer.*` | variantes e imagens no pacote de export |
| `pipeline.steps` | ordem dos passos do `run` |

Perfis prontos em [`configs/perfis/`](configs/perfis/): `local_4gb`, `colab_t4`, `colab_l4`,
`colab_a100`, `smoke`. Um perfil pode herdar de outro com `herda: outro.yaml`.

```bash
python -m src show-config --config configs/perfis/colab_t4.yaml    # config efetiva
python -m src label --set labeling.model=claude-sonnet-5            # rotulador mais barato
python -m src train --stage lora --set training.lora.r=32 --set training.lora.alpha=64
python -m src train --stage qlora --set training.qlora.init_from=base
python -m src train --stage finetune --set training.finetune.weights_dtype=bf16   # menos memória
python -m src predict --variant gold --set gold_model.model=gemini-3.1-pro-preview   # Pro: exige faturamento
python -m src evaluate --set evaluation.clipscore=false
```

Trocar o modelo é trocar `model.model_id` — qualquer VLM do Hugging Face com chat template
(ex.: `Qwen/Qwen2.5-VL-3B-Instruct`); se o template for de outra família, ajuste também
`model.response_marker` e `training.*.connector_modules`.

## 9. Referência de comandos

Um único ponto de entrada: `python -m src <comando> [--config perfil.yaml] [--set chave=valor]`.

| comando | o que faz | onde roda |
|---|---|---|
| `prepare` | `data/raw` → `data/images` + `metadata.csv` | local |
| `label [--redo]` | legendas de referência com o Claude | local (ou Colab) |
| `split [--redo]` | treino / validação / teste | local (ou Colab) |
| `train --stage base\|pretrain\|finetune\|lora\|qlora` | treina uma variante | Colab |
| `export [--out zip] [--with-images]` | empacota variantes + labels + splits | Colab |
| `import --from <zip ou pasta>` | instala o pacote | local |
| `predict --variant <v>\|all [--redo]` | legendas do teste (`gold` = Gemini) | local |
| `evaluate [--preds glob] [--refs arquivo]` | métricas e relatório | local |
| `run [--from X] [--only ...] [--skip ...]` | executa `pipeline.steps` do perfil | ambos |
| `show-config` | imprime a config efetiva | ambos |
| `dummy-data` | imagens e rótulos sintéticos (smoke test) | ambos |

`label` e `predict` são retomáveis (pulam o que já existe; `--redo` refaz).

## 10. Estrutura do repositório e saídas

```
configs/
  default.yaml              todos os parâmetros, comentados
  perfis/*.yaml             local_4gb, colab_t4, colab_l4, colab_a100, smoke
data/
  raw/                      imagens originais (fora do git)
  images/                   normalizadas, psa_XXXX.jpg (versionadas, ~18 MB)
  metadata.csv              manifesto do pipeline: proveniência e licença (versionado)
  sources/                  commons_links.txt + commons_metadata.csv (fontes do Commons, versionado)
  labels.jsonl              legendas de referência do Claude (versionado)
  splits.json               treino / validação / teste (versionado)
  sample/                   legendas fictícias (só texto) para testar a avaliação
docs/
  dataset.md                como montar, documentar e publicar o dataset + datasheet
  metodologia-variantes.md  o que é cada variante e por quê
  image-captioning-metricas.md   conceitos e métricas de captioning
  dominio-porto-salvador.md      domínio portuário e estudo de caso
notebooks/
  colab_pipeline.ipynb      treino no Colab (gerado por gerar_colab.py)
src/
  cli.py                    python -m src <comando>
  config.py                 YAML em camadas + --set
  prepare_images.py         normalização + manifesto
  labeling.py · prompts.py  rotulagem com o Claude e glossário portuário
  splits.py                 divisão estável
  vlm.py                    Qwen3-VL + processador; formato de chat; salvar/carregar variantes
  train.py                  estágios base / pretrain / finetune / lora / qlora
  transfer.py               export / import do pacote de modelos
  predict.py                geração das variantes + modelo gold (Gemini)
  metrics_ptbr.py           métricas adaptadas ao PT-BR
  evaluate.py               tabela comparativa e relatório
  devtools.py               dados sintéticos (smoke test)
tests/                      pytest
outputs/models/<variante>/  (fora do git)
  vlm_config.json           de onde vem cada peso, instrução, resolução, split usado
  model/                    modelo completo (finetune)
  delta.safetensors         conector treinado (pretrain, lora, qlora)
  adapter/                  adaptador PEFT (lora, qlora)
  processor/
  train_summary.json · train_log.jsonl
exports/modelos.zip         pacote Colab → local (fora do git)
results/
  preds_<variante>.jsonl    legendas geradas
  metrics_<variante>.json
  comparativo.md · comparativo.csv
  config_efetiva.yaml       config exata da última execução de `run`
```

## 11. Decisões de avaliação

- **Tokenizador próprio** em vez do `PTBTokenizer` do `pycocoevalcap` (que depende de Java e foi
  feito para o inglês): NFC + minúsculas + hífen vira espaço + pontuação removida, acentos
  preservados.
- **METEOR e SPICE ficaram de fora**: exigem Java, e o METEOR depende de WordNet em inglês.
- **CIDEr** calcula o IDF no próprio conjunto avaliado: com menos de ~100 imagens é instável; a
  tabela sempre traz o `N`.
- **BERTScore** usa BERTimbau (`neuralmind/bert-base-portuguese-cased`); **CLIPScore** usa o
  encoder de texto multilíngue alinhado ao CLIP ViT-B/32 — compara a legenda com a **imagem**,
  sem depender da referência. **RefCLIPScore** combina os dois.
- **Diversidade** (Distinct-1/2, tamanho médio) detecta modelo que repete sempre a mesma frase.
- **Mesma decodificação** para todas as variantes locais (`generation.*`: beam search 3,
  `no_repeat_ngram_size` 3, penalidade de repetição 1,2).
- **Checkpoint** escolhido pela menor loss de validação (`training.*.save: best`).

Para ver as métricas sem modelo nenhum, `data/sample/` traz legendas de mentira (só texto):

```bash
python -m src evaluate --preds "data/sample/preds_*.jsonl" --refs data/sample/refs.jsonl --set evaluation.bertscore=false --set evaluation.clipscore=false
```

## 12. Reprodutibilidade

- Sementes fixas: `seed` (treino) e `split.seed` (divisão).
- `data/labels.jsonl` e `data/splits.json` versionados; cada variante registra a impressão digital
  do split com que foi treinada.
- `results/config_efetiva.yaml` guarda a configuração exata de cada `run`; `train_summary.json`
  guarda hiperparâmetros, GPU, dtype, quantização e histórico de losses de cada variante.
- O notebook do Colab executa a mesma `main` do repositório — o código não é duplicado.

## 13. Limites conhecidos e solução de problemas

**Limites**

- **Referências de LLM:** as métricas medem proximidade ao rotulador (Claude), não a anotadores
  humanos. Revise uma amostra do teste à mão e reporte a concordância.
- **Inferência local em 4 bits:** muda a comparação em relação a 16 bits — declare, ou rode com
  `--set device=cpu`.
- **Fine-tuning na T4 é parcial** (últimas 8 camadas); o completo exige L4 ou A100.
- **Contaminação:** o Qwen3-VL pode ter visto fotos públicas do Porto na web (afeta sobretudo o
  `base`); fotos próprias no teste reduzem o risco.
- **CIDEr instável** com poucas imagens de teste.

**Problemas comuns**

| sintoma | causa provável | o que fazer |
|---|---|---|
| loss `nan` na T4 | fp16 com um modelo treinado em bf16 | `--set model.dtype=fp32` |
| `CUDA out of memory` no treino | batch ou resolução grandes | `--set training.<estagio>.batch_size=1` (e `grad_accum` maior) ou `--set model.image_max_side=336` |
| `marcador de resposta ... nao encontrado` | outro VLM com outro chat template | ajuste `model.response_marker` |
| erro ao carregar `qwen3_vl` | `transformers` antigo | `pip install -U "transformers>=4.57"` |
| `'qlora' usa 4 bits ... exige GPU CUDA` | bitsandbytes sem GPU | treine no Colab (ou `fallback_no_cuda=true` só para teste) |
| `predict` avisa que o split é outro | `splits.json` local ≠ do treino | `python -m src import --from <modelos.zip>` |
| `Nada a rotular` | imagens já rotuladas ou pasta vazia | confira `data/images/`; `--redo` refaz |
| `drive.mount` falha no VS Code | a extensão monta o Drive por comando | paleta: *Colab: Mount Google Drive to Server...* |
| push recusado por privacidade de e-mail | commits com e-mail privado | use o e-mail `...@users.noreply.github.com` do GitHub |

## 14. Documentação, licenças e créditos

- [`docs/dataset.md`](docs/dataset.md) — montagem, formatos, licenças e datasheet do dataset
- [`docs/metodologia-variantes.md`](docs/metodologia-variantes.md) — desenho do experimento
- [`docs/image-captioning-metricas.md`](docs/image-captioning-metricas.md) — conceitos e métricas
- [`docs/dominio-porto-salvador.md`](docs/dominio-porto-salvador.md) — domínio portuário

**Licenças**

- Código deste repositório: Apache 2.0 ([`LICENSE`](LICENSE)).
- Qwen3-VL-2B-Instruct: Apache 2.0 — cite o relatório técnico do Qwen3-VL.
- Imagens: licença individual de cada uma, registrada em `data/metadata.csv`.
- Legendas geradas pelo Claude e pelo Gemini: sujeitas aos termos de uso da Anthropic e do Google;
  confira-os antes de publicar o dataset ou os modelos treinados.

**Créditos:** Qwen3-VL (Qwen / Alibaba Cloud), BERTimbau (NeuralMind),
`pycocoevalcap`, Hugging Face `transformers` / `peft`, `bitsandbytes`.
