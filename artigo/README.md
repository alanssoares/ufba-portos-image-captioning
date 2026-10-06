# Artigo (resumo estendido, formato SBC)

Resumo estendido de até 4 páginas (sem contar as referências) com a metodologia e os
resultados do experimento, para a Unidade de PLN da disciplina.

| arquivo | conteúdo |
|---|---|
| `main.tex` | texto do artigo |
| `referencias.tex` | referências no formato autor-ano da SBC |
| `sbc-template.sty` | estilo SBC (reimplementação: A4, Times 12 pt, margens 3,5/2,5/3/3 cm). Pode ser trocado pelo `sbc-template.sty` oficial sem mudar o `main.tex` |

## Compilar

```bash
cd artigo
pdflatex main.tex && pdflatex main.tex
```

No Overleaf, envie os três arquivos e compile com pdfLaTeX. Com o pacote `babel-portuguese`
instalado, o documento usa hifenização em português; sem ele, cai para inglês e só os
rótulos (Tabela, Figura, Referências) ficam traduzidos.

Os números das tabelas vêm de `results/comparativo.csv` e `results/treino_custos.md`.
