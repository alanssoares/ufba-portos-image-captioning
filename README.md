# Image Captioning no domínio portuário — Porto de Salvador (BA)

Avaliação de um **SLM em português do Brasil** para legendar imagens do porto, comparando
seis variantes do mesmo modelo — do base sem treino até fine-tuning completo, LoRA e QLoRA —
contra um VLM de fronteira.

- **Referências:** legendas geradas pelo **Claude** (`claude-opus-5-5`) para todas as imagens.
- **SLM:** **Manacá-1B** (`menezesbruno/manaca-1b-base`, Llama 1,7 B, PT-BR) + encoder **SigLIP**
  + projetor MLP, no estilo LLaVA.
- **Modelo gold:** **Gemini Pro** via API, avaliado contra as mesmas referências.
- Tudo parametrizado em [`configs/default.yaml`](configs/default.yaml); roda **localmente** e no
  **Colab** ([`notebooks/colab_pipeline.ipynb`](notebooks/colab_pipeline.ipynb)).

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

## Pipeline

Um único ponto de entrada: `python -m src <comando>`. Todo comando aceita
`--config <perfil.yaml>` (repetível) e `--set chave.pontilhada=valor` (repetível).

```bash
python -m src prepare                 # data/raw -> data/images (EXIF, 1024 px, dedup) + metadata.csv
python -m src label                   # Claude rotula todas as imagens -> data/labels.jsonl
python -m src split                   # treino / validação / teste -> data/splits.json
python -m src train --stage base      # materializa o modelo base
python -m src train --stage pretrain
python -m src train --stage finetune
python -m src train --stage lora
python -m src train --stage qlora
python -m src predict --variant all   # legendas do teste: 5 variantes + gold
python -m src evaluate                # results/comparativo.md e .csv
```

Ou tudo de uma vez, na ordem de `pipeline.steps`:

```bash
python -m src run --config configs/perfis/colab_l4.yaml
python -m src run --from train:lora            # retoma de um passo
python -m src run --only predict:gold evaluate # só alguns passos
python -m src show-config --config configs/perfis/local_4gb.yaml   # config efetiva
```

`label` e `predict` são retomáveis (pulam o que já foi feito; `--redo` refaz). `split` é
estável: imagens novas são sorteadas sem mexer no teste existente.

### Exemplos de parâmetros

```bash
python -m src label --set labeling.limit=5                      # teste de custo
python -m src label --set labeling.model=claude-sonnet-5         # rotulador mais barato
python -m src train --stage lora --set training.lora.r=32 --set training.lora.alpha=64
python -m src train --stage qlora --set training.qlora.init_from=base
python -m src predict --variant gold --set gold_model.model=gemini-3.1-pro-preview
python -m src evaluate --set evaluation.clipscore=false
```

## Perfis de hardware (`configs/perfis/`)

| perfil | GPU | observações |
|---|---|---|
| `local_4gb` | 4 GB local | pré-treino com LLM em 4 bits, **QLoRA**, inferência em 4 bits. `finetune` e `lora` ficam fora — rode no Colab |
| `colab_t4` | T4 15 GB | fp16; `finetune` **parcial** (últimas 8 camadas) — o completo não cabe |
| `colab_l4` | L4 22,5 GB | bf16; fine-tuning completo (fp32 + Adam 8 bits paginado) |
| `colab_a100` | A100 40 GB | bf16; tudo com folga |
| `smoke` | nenhuma | teste de fumaça (acima) |

Modelos treinados no Colab podem ser copiados para `outputs/models/` e avaliados localmente.

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

- **4 GB de VRAM**: só QLoRA e inferência em 4 bits. Fine-tuning completo e LoRA em 16 bits
  vão para o Colab (L4 ou A100 com as unidades do Google AI Pro).
- **Referências de LLM**: as métricas medem proximidade ao rotulador (Claude), não a anotadores
  humanos — declare na monografia.
- **Gemini API**: a assinatura Google AI Pro não inclui a API; a chave vem do Google AI Studio.
