# Metodologia: as variantes comparadas

Este documento explica **o que é cada variante**, por que o experimento foi desenhado assim e
quais cuidados valem registrar na monografia. A implementação está em `src/vlm.py`
(modelo), `src/train.py` (estágios) e `configs/default.yaml` (todos os hiperparâmetros).

---

## 1. De onde vêm as legendas: o rotulador

Todas as legendas de referência são geradas pelo **Claude** (`labeling.model`, padrão
`claude-opus-5-5`) com o glossário portuário no *system prompt* (`src/prompts.py`) e saída
estruturada: 3 legendas por imagem, objetos visíveis e uma flag de fora-do-domínio. Por que o
prompt tem o foco técnico atual (operação + equipamentos, sem cor, luz ou nomes) e o que isso
muda na avaliação: [`prompt-rotulagem.md`](prompt-rotulagem.md).

Essas legendas têm dois papéis:

- **treino e validação**: são os pares imagem–legenda usados no pré-treino e nos fine-tunings;
- **teste**: são o gabarito contra o qual todas as variantes são medidas.

Por isso o rotulador **não aparece** na tabela comparativa — seria medir o modelo contra ele
mesmo. O teto externo é o **modelo gold** (seção 3), de outro fornecedor.

> Limitação a declarar: as referências são de um LLM, não de anotadores humanos. As métricas
> medem aproximação ao estilo e ao vocabulário do rotulador. Uma amostra revisada à mão
> (ex.: 50 imagens do teste) dá uma estimativa de quanto isso pesa.

## 2. O modelo: Qwen3-VL-2B-Instruct

O SLM é o **Qwen3-VL-2B-Instruct** (`Qwen/Qwen3-VL-2B-Instruct`, Apache 2.0): um modelo
visão-linguagem pequeno, multilíngue (escreve em português), já treinado para descrever imagens.

```
imagem (lado maior ≤ 448 px) ─► encoder de visão ViT (~0,4 B, 24 blocos, patch 16)
       ─► conector "merger" + 3 "deepstack mergers" (~0,1 B): 2×2 patches → 1 token de imagem
       ─► LLM Qwen3 (~1,7 B, 28 camadas)

LLM vê:  <|im_start|>user [tokens de imagem] {instrução}<|im_end|>
         <|im_start|>assistant\n{legenda}<|im_end|>        ← loss só aqui
```

Tudo é parâmetro: modelo (`model.model_id`), instrução (`model.instruction`), resolução
(`model.image_max_side`, 448 px ≈ 196 tokens de imagem), módulos do conector
(`training.*.connector_modules`). Outro VLM com chat template — por exemplo
`Qwen/Qwen2.5-VL-3B-Instruct` — entra trocando `model.model_id` (e, se o template for diferente,
`model.response_marker`).

**Por que não o Manacá-1B** (primeira escolha, descartada): ele é só de texto. Seria preciso
treinar do zero um projetor entre um encoder de visão e o LLM, o que o LLaVA faz com ~558 mil
pares imagem–legenda; com algumas centenas de imagens do Porto o alinhamento não se forma, e todas
as variantes tenderiam a legendas genéricas. Um VLM que já legenda permite que as variantes
meçam o que interessa: **o ganho de adaptar ao domínio portuário**.

## 3. As variantes

| variante | ponto de partida | o que treina | quantização |
|---|---|---|---|
| **base** | pesos originais | nada — zero-shot com a instrução | — |
| **pretrain** | base | só o conector visão→LLM (~0,1 B) | — |
| **finetune** | pretrain* | conector + **todos** os pesos do LLM (~1,7 B) | — |
| **lora** | pretrain* | conector + adaptadores LoRA no LLM | — |
| **qlora** | pretrain* | conector + adaptadores LoRA | LLM em 4 bits (NF4) |
| **gold** | — | nada — VLM grande via API (Gemini Pro) | — |

\* `training.<estágio>.init_from` escolhe o ponto de partida: `pretrain` (padrão) ou `base`.

### base — o ponto de partida real

O Qwen3-VL original, zero-shot, só com a instrução em português. Diferente de um modelo sem
treino nenhum, ele já legenda — mas com vocabulário genérico ("um navio grande em um porto").
É a linha que mostra o **gap de domínio**.

### pretrain — pré-treino continuado de domínio

Pré-treinar um VLM do zero exige centenas de milhões de pares; com algumas centenas de imagens
isso não faz sentido. O que se faz aqui é um **pré-treino continuado** em que só o conector
aprende: encoder de visão e LLM congelados, pares imagem–legenda do Porto. O conector ajusta
"como o LLM lê a imagem" para as cenas portuárias (portêineres, pátios de contêineres, cais),
sem mexer no conhecimento de linguagem. É o ponto de partida das três variantes seguintes.

### finetune — fine-tuning completo

Conector + todos os ~1,7 B de parâmetros do LLM (o encoder segue congelado; `train_vision: true`
libera). É a referência de "quanto dá para extrair do modelo" — e a mais cara em memória: pesos
fp32 + gradientes + Adam 8 bits passam de 18 GB. Na T4 do Colab o perfil `colab_t4` faz um
fine-tuning **parcial** (últimas 8 camadas, `llm_mode: partial`); na L4 os pesos ficam em bf16;
na A100, fp32. O que foi usado fica registrado no relatório.

### lora — Low-Rank Adaptation

O LLM fica congelado (em 16 bits) e matrizes de posto baixo (`r`, `alpha`) são treinadas em
todas as projeções de atenção e MLP **do LLM** (a regex de `target_modules` exclui o encoder de
visão). Treina ~1% dos parâmetros do LLM, mais o conector. Pergunta que responde: *quanto do
ganho do fine-tuning completo se recupera com uma fração do custo?*

### qlora — LoRA sobre o LLM quantizado

Igual ao LoRA, mas o LLM congelado é carregado em **4 bits NF4** com dupla quantização
(bitsandbytes); encoder de visão e `lm_head` ficam em 16 bits (`quant.skip_modules`). Reduz a
memória do LLM de ~3,4 GB para ~1 GB. Pergunta que responde: *a quantização custa qualidade?*
(compare `qlora` × `lora`).

### gold — o teto externo

Um VLM grande via API (padrão: Gemini Pro, `gold_model.model`) legenda o teste com a **mesma
instrução de tarefa das variantes Qwen** (`model.instruction`) e, a mais, o glossário e as regras
do rotulador como *system prompt* (`gold_model.use_domain_prompt`) — ver
[`prompt-rotulagem.md`](prompt-rotulagem.md#42-mesma-especificação-para-todos-os-modelos-avaliados). Como é de outro fornecedor, é avaliado
contra as referências do Claude sem circularidade. Mostra a distância entre o SLM ajustado e um
modelo de fronteira.

## 4. Comparabilidade — o que é mantido fixo

- mesmos splits (`data/splits.json`, estável entre execuções; cada variante registra a
  impressão digital do split com que foi treinada);
- mesmas legendas de treino, mesma instrução, mesma resolução de imagem, mesmo número de épocas;
- `finetune`, `lora` e `qlora` partem do mesmo `pretrain`;
- mesma decodificação (`generation.*`: beam search 3, `no_repeat_ngram_size` 3);
- seleção do checkpoint pela menor loss de validação (`training.*.save: best`).

O relatório (`results/comparativo.md`) traz, além das métricas, o **custo** de cada variante:
parâmetros treináveis, VRAM de pico, tempo de treino e tamanho em disco.

## 5. Cuidados para a monografia

- **Onde roda cada etapa**: treino no Colab, inferência na máquina local. O pacote
  `modelos.zip` (`python -m src export` / `import`) leva junto `labels.jsonl` e `splits.json`, e o
  `predict` avisa se o teste local divergir do usado no treino.
- **Quantização na inferência**: o perfil `local_4gb` carrega *todas* as variantes em 4 bits
  para caberem na GPU. Isso muda a comparação — rode com `--set device=cpu` (sem quantizar) ou declare.
- **fp16 na T4**: o Qwen3-VL foi treinado em bf16; em fp16 pode haver *overflow* (loss NaN). Se
  acontecer, `--set model.dtype=fp32`.
- **Contaminação**: o Qwen3-VL foi pré-treinado em dados da web e pode já ter visto fotos
  públicas do Porto de Salvador (ex.: Wikimedia Commons). Isso afeta sobretudo a variante `base`;
  vale mencionar, e fotos próprias no teste reduzem o risco.
- **CIDEr** usa o IDF do próprio conjunto de teste: com N < ~100 é instável.
- **Sementes**: `seed` e `split.seed` fixos; registre a `results/config_efetiva.yaml` de cada rodada.
