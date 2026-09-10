# Image Captioning no domínio portuário — Porto de Salvador (BA)

Pipeline para montar um dataset próprio de imagens do porto, gerar legendas em
**português do Brasil** com diferentes modelos e compará-los com métricas automáticas.

Referências de domínio e de avaliação estão em [`docs/`](docs/):
[domínio portuário](docs/dominio-porto-salvador.md) e
[conceitos e métricas](docs/image-captioning-metricas.md).

## A hipótese do trabalho

Modelos genéricos de captioning descrevem uma foto do Tecon como *"a large ship at a dock"*.
O vocabulário técnico (portêiner, transtêiner, reach stacker, berço) simplesmente não aparece.
Medir esse **gap de domínio** — em quanto cada modelo fica abaixo da referência humana, e onde
exatamente ele erra — é o resultado central do projeto.

## Instalação

```bash
uv venv --python 3.11
```

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

```bash
uv pip install -r requirements.txt
```

O PyTorch vai primeiro, no índice da CUDA certa (`cu124` para o driver 566.x desta máquina);
o `requirements.txt` traz o resto. Sem GPU, troque o índice por `cpu` — tudo funciona, só
mais devagar.

Para os rascunhos via API, exporte a chave:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Pipeline

### 1. Coletar e normalizar imagens

Jogue os arquivos originais em `data/raw/` (subpastas são percorridas) e rode:

```bash
python -m src.prepare_images
```

Corrige orientação EXIF, converte para RGB, reduz o lado maior para 1024 px, deduplica por
SHA-1 e salva como `data/images/psa_XXXX.jpg`. Cria `data/metadata.csv` com uma linha por
imagem — **preencha as colunas `fonte`, `url` e `licenca` à mão**; sem isso o dataset não é
publicável nem citável.

### 2. Rascunhar as legendas

```bash
python -m src.draft_captions --limit 5     # comece pequeno para ver o custo
python -m src.draft_captions               # todas as pendentes
```

Usa Claude com o vocabulário portuário no *system prompt* e saída estruturada (3 legendas +
objetos visíveis + flag de fora-de-domínio) → `data/drafts.jsonl`. Está ligado o
`fallbacks` do lado do servidor: se um classificador recusar uma imagem, a API reroteia em
vez de devolver `stop_reason: "refusal"` (desligue com `--no-fallbacks`).

**Isto ainda não é ground truth.**

### 3. Revisar (a etapa que gera o dado real)

```bash
python -m streamlit run src/review_app.py
```

Imagem à esquerda, as 3 legendas editáveis à direita. Corrija, descarte o que não for cena
portuária, e salve — o resultado vai para `data/captions.jsonl`, que é a referência usada na
avaliação. Legenda revisada por humano é o que separa um dataset de um monte de saída de LLM.

### 4. Gerar predições

```bash
python -m src.run_captioning --model blip --translate
python -m src.run_captioning --model florence2 --translate
python -m src.run_captioning --model qwen25vl
python -m src.run_captioning --model claude
```

Saída em `results/preds_<modelo>.jsonl`. Modelos disponíveis:

| `--model` | O que é | VRAM (fp16) | Idioma |
|---|---|---|---|
| `blip` / `blip-base` | BLIP, baseline encoder-decoder clássico | ~0,9 / ~0,5 GB | EN → traduzido |
| `florence2` / `florence2-base` | Florence-2, task `<MORE_DETAILED_CAPTION>` | ~1,6 / ~0,5 GB | EN → traduzido |
| `qwen25vl` | Qwen2.5-VL-3B em 4 bits | ~2,5 GB | PT-BR direto |
| `claude` | VLM via API — teto de qualidade | — | PT-BR direto |

### 5. Avaliar

```bash
python -m src.evaluate --preds "results/preds_*.jsonl" --out results/comparativo.md
```

Imprime e salva uma tabela markdown com BLEU-1/4, ROUGE-L, CIDEr, BERTScore-F1, CLIPScore,
RefCLIPScore e diversidade (Distinct-1/2, tamanho médio).

Para ver as métricas funcionando antes de ter dataset, `data/sample/` traz um exemplo de
mentira: 4 imagens que existem só como `image_id`, com legendas de referência escritas à mão e
dois conjuntos de predições simuladas — uma com vocabulário técnico, outra genérica. **Não há
arquivo de imagem ali**, então rode sem CLIPScore (que precisaria da imagem de verdade):

```bash
python -m src.evaluate --preds "data/sample/*.jsonl" --refs data/sample/refs.jsonl --no-bertscore --no-clipscore
```

## Decisões de avaliação (adaptadas ao PT-BR)

- **Tokenizador próprio** em vez do `PTBTokenizer` do `pycocoevalcap`: o original depende de
  JAR do Stanford CoreNLP (Java) e foi feito para o inglês. Aqui é NFC + minúsculas + hífen
  vira espaço + pontuação removida, acentos preservados. Nenhuma etapa do projeto precisa de Java.
- **METEOR e SPICE ficaram de fora**: ambos exigem Java, e o METEOR ainda depende de WordNet
  em inglês para sinônimos. Em PT-BR o ganho não paga o atrito — está documentado em
  [`src/metrics_ptbr.py`](src/metrics_ptbr.py) para citar na monografia.
- **CIDEr** calcula o IDF a partir do próprio conjunto avaliado: com menos de ~100 imagens o
  valor é instável. A tabela sempre reporta o `N` junto.
- **BERTScore** usa BERTimbau (`neuralmind/bert-base-portuguese-cased`); **CLIPScore** usa o
  encoder de texto multilíngue alinhado ao CLIP ViT-B/32 — CLIPScore compara a legenda com a
  **imagem**, então funciona mesmo onde ainda não há referência humana.
- **A tradução EN→PT é uma variável do experimento**, não um detalhe: BLIP e Florence-2 são
  avaliados através do MarianMT. Por isso `legenda_en` fica gravada junto da legenda traduzida —
  dá para separar erro de percepção visual de erro de tradução. Alguns checkpoints do OPUS
  pedem prefixo de variante (`--mt-prefix ">>por<<"`); confira o model card do que você usar.

## Estrutura

```
data/
  raw/            imagens originais (fora do git)
  images/         normalizadas, psa_XXXX.jpg (fora do git)
  metadata.csv    proveniência e licença — versionado
  drafts.jsonl    rascunhos do Claude
  captions.jsonl  ground truth revisado — versionado
  sample/         legendas fictícias (só texto, sem imagens) para testar a avaliação
src/
  prepare_images.py   normalização + manifesto
  draft_captions.py   rascunhos via API
  review_app.py       revisão humana (Streamlit)
  captioners.py       registry dos modelos
  translate.py        MarianMT EN→PT
  run_captioning.py   roda um modelo sobre a pasta de imagens
  metrics_ptbr.py     métricas adaptadas ao PT-BR
  evaluate.py         comparação entre modelos
results/          predições e métricas
```

## Limites conhecidos

- **4 GB de VRAM** dão conta de inferência até ~1B em fp16. Fine-tuning (LoRA no BLIP, por
  exemplo) não cabe local — use Colab com T4.
- `qwen25vl` em 4 bits fica no limite dos 4 GB; se estourar, reduza `--max-side` na etapa 1
  ou rode com `--device cpu` (lento).
- O Florence-2 usa `trust_remote_code=True`; há um patch em `captioners.py` que remove o
  import de `flash_attn` (não existe wheel para Windows).
