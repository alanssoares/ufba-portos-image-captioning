# Metodologia: as variantes comparadas

Este documento explica **o que é cada variante**, por que o experimento foi desenhado assim e
quais cuidados valem registrar na monografia. A implementação está em `src/vlm.py`
(modelo), `src/train.py` (estágios) e `configs/default.yaml` (todos os hiperparâmetros).

---

## 1. De onde vêm as legendas: o rotulador

Todas as legendas de referência são geradas pelo **Claude** (`labeling.model`, padrão
`claude-opus-5-5`) com o glossário portuário no *system prompt* (`src/prompts.py`) e saída
estruturada: 3 legendas por imagem, objetos visíveis e uma flag de fora-do-domínio.

Essas legendas têm dois papéis:

- **treino e validação**: são os pares imagem–legenda usados no pré-treino e nos fine-tunings;
- **teste**: são o gabarito contra o qual todas as variantes são medidas.

Por isso o rotulador **não aparece** na tabela comparativa — seria medir o modelo contra ele
mesmo. O teto externo é o **modelo gold** (seção 4), de outro fornecedor.

> Limitação a declarar: as referências são de um LLM, não de anotadores humanos. As métricas
> medem aproximação ao estilo e ao vocabulário do rotulador. Uma amostra revisada à mão
> (ex.: 50 imagens do teste) dá uma estimativa de quanto isso pesa.

## 2. O modelo: SLM PT-BR + encoder de visão (estilo LLaVA)

O SLM escolhido é o **Manacá-1B** (`menezesbruno/manaca-1b-base`): arquitetura Llama,
1,7 B de parâmetros, 24 camadas, dimensão 2048, pré-treinado do zero em português do Brasil,
licença CC BY 4.0. É um modelo **base** (sem instrução) e **só de texto**. Particularidade: o
tokenizador converte tudo para minúsculas, então as legendas geradas saem em minúsculas — as
métricas já normalizam caixa, então isso não penaliza a comparação.

Como o Manacá não enxerga imagens, o modelo de legenda é montado assim:

```
imagem ─► SigLIP (congelado) ─► 196 patches ─► pooling 2×2 ─► 49 vetores
       ─► projetor MLP (768 → 2048 → 2048) ─► 49 "tokens de imagem"

LLM vê:  [49 tokens de imagem] [<s> descrição da imagem do porto:] [legenda ... </s>]
                                                                    └── loss só aqui
```

Encoder (`model.vision_id`), projetor (`model.projector`), pooling (`model.image_pool`),
camada de features (`model.vision_feature_layer`) e prompt (`model.prompt`) são parâmetros.
Trocar o SLM é trocar `model.llm_id` (qualquer `AutoModelForCausalLM`).

## 3. As variantes

| variante | ponto de partida | o que treina | quantização |
|---|---|---|---|
| **base** | pesos originais | nada — projetor aleatório (semente fixa) | — |
| **pretrain** | base | só o projetor | — |
| **finetune** | pretrain* | projetor + **todos** os pesos do LLM | — |
| **lora** | pretrain* | projetor + adaptadores LoRA no LLM | — |
| **qlora** | pretrain* | projetor + adaptadores LoRA | LLM em 4 bits (NF4) |
| **gold** | — | nada — VLM grande via API | — |

\* `training.<estágio>.init_from` escolhe o ponto de partida: `pretrain` (padrão) ou `base`.

### base — o limite inferior

O LLM nunca viu vetores de imagem e o projetor é aleatório: as legendas saem sem relação com a
imagem. É o ponto zero que mostra quanto cada etapa de treino acrescenta.

### pretrain — pré-treino adaptativo de domínio

Pré-treinar um SLM do zero exige bilhões de tokens; com algumas centenas de imagens isso não
faz sentido. O que se faz aqui é a **etapa 1 do LLaVA** (*feature alignment*): encoder e LLM
congelados, só o projetor aprende a traduzir o que o SigLIP vê para o "idioma" de embeddings do
Manacá, usando os pares imagem–legenda do Porto. É barato (~6 M de parâmetros treináveis) e é o
ponto de partida das três variantes seguintes.

### finetune — fine-tuning completo

Etapa 2 do LLaVA: projetor + todos os 1,7 B de parâmetros do LLM. É a referência de "quanto dá
para extrair do modelo" — e a mais cara em memória: pesos fp32 + gradientes + estados do Adam
(8 bits) passam de 17 GB. Não cabe na T4 do Colab gratuito; lá o perfil `colab_t4` faz um
fine-tuning **parcial** (últimas 8 camadas, `llm_mode: partial`), o que fica registrado no relatório.

### lora — Low-Rank Adaptation

O LLM fica congelado (em 16 bits) e matrizes de posto baixo (`r`, `alpha`, `target_modules`) são
treinadas em todas as projeções de atenção e MLP. Treina ~1% dos parâmetros. Pergunta que
responde: *quanto do ganho do fine-tuning completo se recupera com uma fração do custo?*

### qlora — LoRA sobre o LLM quantizado

Igual ao LoRA, mas o LLM congelado é carregado em **4 bits NF4** com dupla quantização
(bitsandbytes). Reduz a memória do LLM de ~3,4 GB para ~1 GB — é o que permite treinar numa GPU
local de 4 GB. Pergunta que responde: *a quantização custa qualidade?* (compare `qlora` × `lora`).

### gold — o teto externo

Um VLM grande via API (padrão: Gemini Pro, `gold_model.model`) legenda o teste com o mesmo
glossário do rotulador (`gold_model.use_domain_prompt`). Como é de outro fornecedor, é avaliado
contra as referências do Claude sem circularidade. Mostra a distância entre o SLM treinado e um
modelo de fronteira.

## 4. Comparabilidade — o que é mantido fixo

- mesmos splits (`data/splits.json`, estável entre execuções);
- mesmas legendas de treino, mesmo prompt, mesmo número de épocas por padrão;
- `finetune`, `lora` e `qlora` partem do mesmo `pretrain`;
- mesma decodificação (`generation.*`: beam search 3, `no_repeat_ngram_size` 3);
- seleção do checkpoint pela menor loss de validação (`training.*.save: best`).

O relatório (`results/comparativo.md`) traz, além das métricas, o **custo** de cada variante:
parâmetros treináveis, VRAM de pico, tempo de treino e tamanho em disco.

## 5. Cuidados para a monografia

- **Quantização na inferência**: o perfil `local_4gb` carrega *todas* as variantes em 4 bits
  para caberem na GPU. Isso muda a comparação — rode a inferência final no Colab ou declare.
- **fp16 na T4**: o Manacá foi treinado em bf16; em fp16 pode haver *overflow* (loss NaN). Se
  acontecer, `--set model.dtype=fp32`.
- **CIDEr** usa o IDF do próprio conjunto de teste: com N < ~100 é instável.
- **Sementes**: `seed` e `split.seed` fixos; registre a `results/config_efetiva.yaml` de cada rodada.
