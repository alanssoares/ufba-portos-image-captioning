# Custos de treino por variante

Resultado da célula de resumo do notebook do Colab (`notebooks/colab_pipeline.ipynb`, seção 7),
lido de `outputs/models/<variante>/train_summary.json`. Treino com o dataset real:
68 imagens de treino × 3 legendas, 15 de validação (`data/splits.json`, seed 42), referências
na versão de prompt `09101c626775`.

- **Data do treino:** 2026-10-04
- **GPU / perfil:** NVIDIA L4 (22,5 GB) no Colab → perfil `colab_l4` (fine-tuning completo, pesos bf16 + Adam 8 bits)
- **Modelo:** `Qwen/Qwen3-VL-2B-Instruct`; `finetune`, `lora` e `qlora` partem do `pretrain`

| variante | treináveis (M) | VRAM pico (GB) | tempo (min) | melhor val loss | quantização |
|---|---:|---:|---:|---:|---|
| pretrain | 100,714 | 8,007 | 1,728 | 2,153 | nenhuma |
| finetune | 1821,289 | 9,180 | 3,438 | 1,638 | nenhuma |
| lora | 118,147 | 8,267 | 3,644 | 1,359 | nenhuma |
| qlora | 118,147 | 8,199 | 2,731 | 1,394 | nf4 4 bits (bitsandbytes) |

`base` não aparece porque não é treinado (Qwen original, zero-shot).

## Leitura

- **Parâmetros treináveis.** O `pretrain` treina só os conectores visuais (~101 M). O `finetune`
  treina conector + LLM completo (~1,82 B), cerca de 15× mais que `lora`/`qlora` (~118 M:
  conector + adaptadores LoRA).
- **Val loss.** `lora` (1,359) e `qlora` (1,394) ficam abaixo do `finetune` (1,638), com uma
  fração dos parâmetros. Com só 68 imagens de treino, o ajuste completo tende a sobreajustar,
  e os adaptadores funcionam como regularização. O `pretrain` (2,153) mostra que ajustar só o
  conector não basta para aprender o estilo das referências.
- **Val loss não é a métrica final.** Ela mede a verossimilhança das referências de
  validação; a comparação entre variantes vem das métricas de legenda no teste (BLEU,
  ROUGE-L, CIDEr, BERTScore, CLIPScore) em `results/comparativo.md`.
- **VRAM.** As diferenças são pequenas (8,0 a 9,2 GB), e todas usam menos da metade dos 22,5 GB da L4. O `qlora` quase não economiza memória
  em relação ao `lora` (8,20 × 8,27 GB): num modelo de 2 B, o peso do LLM em 16 bits é uma
  parte pequena do pico, que é dominado por ativações, imagens e o encoder de visão (mantido
  em 16 bits). A economia do QLoRA aparece em modelos maiores ou na inferência em GPUs pequenas.
- **Tempo.** Todos os treinos ficaram abaixo de 4 minutos. Com um dataset tão pequeno, os
  tempos têm variação alta entre execuções (cache, carregamento do modelo) e servem só como
  ordem de grandeza.
- **Ressalva geral.** Uma única execução por variante, sem repetição de sementes: diferenças
  pequenas de val loss (ex: `lora` × `qlora`) não devem ser tratadas como significativas.
