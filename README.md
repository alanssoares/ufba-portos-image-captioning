# Image Captioning no domínio portuário — Porto de Salvador (BA)

Avaliação de um **SLM em português do Brasil** para legendar imagens do porto, comparando
seis variantes do mesmo modelo — do base sem treino até fine-tuning completo, LoRA e QLoRA —
contra um VLM de fronteira.

- **Referências:** legendas geradas pelo **Claude** (`claude-opus-5-5`) para todas as imagens.
- **SLM:** **Manacá-1B** (`menezesbruno/manaca-1b-base`, Llama 1,7 B, PT-BR) + encoder **SigLIP**
  + projetor MLP, no estilo LLaVA.
- **Modelo gold:** **Gemini Pro** via API, avaliado contra as mesmas referências.
- Tudo parametrizado em [`configs/default.yaml`](configs/default.yaml).
- **Treino no Colab** ([`notebooks/colab_pipeline.ipynb`](notebooks/colab_pipeline.ipynb), no navegador
  ou no VS Code); **inferência e avaliação na máquina local**.

Documentação: [metodologia das variantes](docs/metodologia-variantes.md) ·
[conceitos e métricas](docs/image-captioning-metricas.md) ·
[domínio portuário](docs/dominio-porto-salvador.md).

## As variantes

| variante | o que é |
|---|---|
| `base` | Manacá + SigLIP com projetor aleatório — limite inferior, sem treino |
| `pretrain` | pré-treino adaptativo de domínio: só o projetor aprende (etapa 1 do LLaVA) |
| `finetune` | fine-tuning completo: projetor + todos os pesos do LLM |
| `lora` | LoRA no LLM (16 bits) + projetor |
| `qlora` | LoRA sobre o LLM quantizado em 4 bits (NF4) + projetor |
| `gold` | Gemini Pro via API — teto externo |

`finetune`, `lora` e `qlora` partem do `pretrain` por padrão
(`--set training.<estagio>.init_from=base` para partir do base). Detalhes e justificativas em
[`docs/metodologia-variantes.md`](docs/metodologia-variantes.md).

## Instalação (local)

No Colab o notebook instala tudo sozinho. Na máquina local:

```bash
uv venv --python 3.11
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
```

O PyTorch vai primeiro, no índice da CUDA certa (`cu124` para o driver 566.x). Sem GPU, troque
o índice por `cpu`. Chaves de API como variáveis de ambiente:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # rotulagem (Claude)
export GEMINI_API_KEY=...             # modelo gold (aistudio.google.com/apikey)
```

## Teste rápido, sem dataset

```bash
python -m src run --config configs/perfis/smoke.yaml
```

Gera imagens sintéticas e modelos minúsculos aleatórios (nada é baixado, nenhuma API é
chamada) e roda **o pipeline inteiro** — treino das 5 variantes, predição, avaliação — em
poucos minutos na CPU, em `outputs/smoke/`. As métricas não significam nada; o teste prova que
as peças encaixam. O mesmo teste roda com `pytest` (`tests/test_smoke_pipeline.py`).

## Fluxo de trabalho: dados e inferência locais, treino no Colab

```
 LOCAL                                   COLAB (GPU)                          LOCAL
 prepare -> label -> split  ── Drive ──►  train base/pretrain/finetune/  ── modelos.zip ──►  import -> predict -> evaluate
                                          lora/qlora -> export
```

**1. Local — dados**

```bash
python -m src prepare     # data/raw -> data/images (EXIF, 1024 px, dedup) + metadata.csv
python -m src label       # Claude rotula todas as imagens -> data/labels.jsonl
python -m src split       # treino / validação / teste -> data/splits.json
```

Copie `data/images/`, `data/labels.jsonl` e `data/splits.json` para
`MyDrive/ufba-portos-captioning/data/` no Google Drive.

**2. Colab — treino.** Abra [`notebooks/colab_pipeline.ipynb`](notebooks/colab_pipeline.ipynb) no Colab
(ou no VS Code com a extensão *Google Colab*, escolhendo um servidor com GPU) e rode as células.
Equivale a:

```bash
python -m src run --config configs/perfis/colab_l4.yaml --set paths.root=/content/drive/MyDrive/ufba-portos-captioning
```

que treina as 5 variantes e grava `exports/modelos.zip` no Drive.

**3. Local — inferência e avaliação**

```bash
python -m src import --from C:/Users/<voce>/Downloads/modelos.zip
python -m src run --config configs/perfis/local_4gb.yaml    # predict das 5 variantes + gold, evaluate
```

O pacote leva `labels.jsonl` e `splits.json` junto com os modelos, para que o teste local seja
exatamente o do treino (arquivos locais diferentes ganham uma cópia `.bak-<data>`; nada é apagado).
Cada variante guarda a impressão digital do split: se o `splits.json` local mudar, o `predict` avisa.

> Treinar no Hugging Face de graça não é viável: o ZeroGPU gratuito dá 5 min de GPU por dia com
> chamadas de 60 s (feito para demos), e o HF Jobs é pago.

### Comandos

Um único ponto de entrada: `python -m src <comando>`. Todo comando aceita
`--config <perfil.yaml>` (repetível) e `--set chave.pontilhada=valor` (repetível).

| comando | o que faz |
|---|---|
| `prepare` / `label` / `split` | dados (acima) |
| `train --stage base\|pretrain\|finetune\|lora\|qlora` | treina uma variante |
| `export` / `import --from` | pacote de modelos Colab → local |
| `predict --variant <v>\|all` | legendas do teste (`gold` = Gemini via API) |
| `evaluate` | `results/comparativo.md` e `.csv` |
| `run [--from X] [--only ...] [--skip ...]` | executa `pipeline.steps` do perfil em ordem |
| `show-config` | imprime a config efetiva |

`label` e `predict` são retomáveis (pulam o que já foi feito; `--redo` refaz). `split` é
estável: imagens novas são sorteadas sem mexer no teste existente.

### Exemplos de parâmetros

```bash
python -m src label --set labeling.limit=5                      # teste de custo
python -m src label --set labeling.model=claude-sonnet-5         # rotulador mais barato
python -m src train --stage lora --set training.lora.r=32 --set training.lora.alpha=64
python -m src train --stage qlora --set training.qlora.init_from=base
python -m src predict --variant gold --set gold_model.model=gemini-3.1-pro-preview
python -m src predict --variant all --config configs/perfis/local_4gb.yaml --set device=cpu
python -m src evaluate --set evaluation.clipscore=false
```

## Perfis (`configs/perfis/`)

| perfil | onde | observações |
|---|---|---|
| `local_4gb` | máquina local, GPU 4 GB | **só inferência + avaliação**; variantes carregadas em 4 bits (`--set device=cpu` para rodar sem quantizar, lento) |
| `colab_t4` | Colab T4 15 GB | treino + export; fp16; `finetune` **parcial** (últimas 8 camadas) — o completo não cabe |
| `colab_l4` | Colab L4 22,5 GB | treino + export; bf16; fine-tuning completo (fp32 + Adam 8 bits paginado) |
| `colab_a100` | Colab A100 40 GB | treino + export; bf16; tudo com folga |
| `smoke` | CPU | teste de fumaça do pipeline inteiro |

## Saídas

```
data/
  raw/                 imagens originais (fora do git)
  images/              normalizadas, psa_XXXX.jpg (fora do git)
  metadata.csv         proveniência e licença — preencha fonte/url/licenca
  labels.jsonl         legendas de referência (Claude)
  splits.json          treino / validação / teste
outputs/models/<variante>/   (fora do git)
  vlm_config.json      de onde vem cada peso, prompt, pooling
  projector.safetensors
  llm/ | adapter/      LLM completo (finetune) ou adaptador PEFT (lora, qlora)
  train_summary.json   parâmetros treináveis, VRAM de pico, tempo, losses
exports/modelos.zip     pacote Colab -> local (fora do git)
results/
  preds_<variante>.jsonl
  comparativo.md       métricas + custo de treino + exemplos
  comparativo.csv
  config_efetiva.yaml  config exata da última execução de `run`
```

## Estrutura do código

```
configs/default.yaml      todos os parâmetros, comentados
configs/perfis/*.yaml     sobreposições por hardware
src/cli.py                python -m src <comando>
src/config.py             YAML em camadas + --set
src/prepare_images.py     normalização + manifesto
src/labeling.py           rotulagem com o Claude
src/prompts.py            glossário portuário e instruções
src/splits.py             divisão estável
src/vlm.py                SigLIP + projetor + Manacá; salvar/carregar variantes
src/train.py              estágios base / pretrain / finetune / lora / qlora
src/predict.py            geração das variantes + modelo gold (Gemini)
src/metrics_ptbr.py       métricas adaptadas ao PT-BR
src/evaluate.py           tabela comparativa e relatório
src/transfer.py           export / import do pacote de modelos
src/devtools.py           dados sintéticos e modelos minúsculos (smoke test)
notebooks/gerar_colab.py  gera o notebook do Colab
tests/                    pytest
```

## Decisões de avaliação (adaptadas ao PT-BR)

- **Tokenizador próprio** em vez do `PTBTokenizer` do `pycocoevalcap` (que depende de Java e foi
  feito para o inglês): NFC + minúsculas + hífen vira espaço + pontuação removida, acentos
  preservados. Como o Manacá gera tudo em minúsculas, normalizar a caixa também evita penalizá-lo.
- **METEOR e SPICE ficaram de fora**: exigem Java, e o METEOR depende de WordNet em inglês.
- **CIDEr** calcula o IDF no próprio conjunto avaliado: com menos de ~100 imagens é instável.
  A tabela sempre traz o `N`.
- **BERTScore** usa BERTimbau; **CLIPScore** usa o encoder de texto multilíngue alinhado ao CLIP
  ViT-B/32 — compara a legenda com a **imagem**, sem depender da referência.

Para ver as métricas funcionando sem modelo nenhum, `data/sample/` traz legendas de mentira
(só texto, sem imagem):

```bash
python -m src evaluate --preds "data/sample/preds_*.jsonl" --refs data/sample/refs.jsonl --set evaluation.bertscore=false --set evaluation.clipscore=false
```

## Limites conhecidos

- **4 GB de VRAM**: localmente só inferência, com as variantes em 4 bits — declare na monografia
  (ou rode a inferência com `--set device=cpu`, sem quantizar). Todo treino vai para o Colab
  (L4 ou A100 com as unidades do Google AI Pro).
- **Referências de LLM**: as métricas medem proximidade ao rotulador (Claude), não a anotadores
  humanos — declare na monografia.
- **Gemini API**: a assinatura Google AI Pro não inclui a API; a chave vem do Google AI Studio.
