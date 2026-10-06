# Image Captioning: Conceitos e Métricas de Avaliação

> Documento de referência geral sobre a tarefa de **Image Captioning** (geração automática de
> legendas para imagens) e as métricas usadas para avaliar a qualidade das legendas geradas.
> O conteúdo é descrito de forma genérica, aplicável a qualquer domínio de imagens; ao final há
> uma seção com considerações específicas para o domínio portuário deste projeto.

---

## 1. O que é Image Captioning

Image Captioning é a tarefa de gerar automaticamente uma descrição textual (legenda) que resuma
o conteúdo visual de uma imagem — objetos presentes, atributos, ações e relações entre eles.
É uma tarefa multimodal: combina **visão computacional** (entender a imagem) com **processamento
de linguagem natural** (produzir texto fluente e coerente).

Formalmente, dado uma imagem `I`, o modelo aprende a gerar uma sequência de palavras
`y = (y_1, y_2, ..., y_T)` que descreva `I`, geralmente maximizando a probabilidade condicional
`P(y | I)` durante o treinamento.

### 1.1 Arquiteturas comuns

- **Encoder-Decoder clássico (CNN + RNN/LSTM)**: uma CNN (ex: ResNet, VGG) extrai um vetor de
  características da imagem; um RNN/LSTM gera a legenda palavra a palavra a partir desse vetor.
- **Encoder-Decoder com Atenção** (ex: *Show, Attend and Tell*): o decoder "olha" para diferentes
  regiões da imagem em cada passo de geração, em vez de usar um único vetor global.
- **Transformers** (ex: modelos baseados em ViT + Transformer decoder): substituem RNNs por
  self-attention, permitindo paralelização e melhor captura de dependências de longo alcance.
- **Modelos multimodais pré-treinados** (ex: BLIP, BLIP-2, GIT, Flamingo, LLaVA): pré-treinados em
  grandes pares imagem-texto e depois ajustados (fine-tuning) para captioning; hoje representam o
  estado da arte.
- **Abordagens baseadas em CLIP**: usam embeddings de imagem/texto alinhados (ex: CLIP) como base
  para geração ou recuperação de legendas.

### 1.2 Componentes de um dataset de captioning

- **Imagens**: o conjunto de imagens a serem legendadas.
- **Legendas de referência (ground truth)**: geralmente múltiplas legendas humanas por imagem
  (datasets consolidados como MS COCO Captions e Flickr30k usam 5 legendas/imagem), pois a mesma
  imagem admite várias descrições corretas.
- **Splits**: divisão em treino/validação/teste.

---

## 2. Por que a avaliação é difícil

Diferente de tarefas de classificação, não existe uma única legenda "correta" para uma imagem —
várias descrições diferentes podem estar igualmente certas. Isso torna a avaliação automática
desafiadora, pois é preciso comparar o texto gerado com uma ou mais referências humanas,
tolerando variações de vocabulário, ordem das palavras e nível de detalhe.

Por isso, a prática usual é combinar **múltiplas métricas automáticas** (cada uma captura um
aspecto diferente da qualidade) com, quando possível, **avaliação humana**.

---

## 3. Métricas automáticas baseadas em sobreposição de texto

Essas métricas comparam a legenda gerada com uma ou mais legendas de referência, medindo
sobreposição de palavras/n-gramas. São as mais tradicionais e ainda amplamente reportadas.

| Métrica | O que mede | Características |
|---|---|---|
| **BLEU** (1 a 4) | Precisão de n-gramas (1 a 4 palavras) entre legenda gerada e referências, com penalidade de brevidade | Rápida e simples, mas favorece frases curtas e não capta sinônimos/paráfrases; originalmente criada para tradução automática |
| **ROUGE-L** | Maior subsequência comum (LCS) entre legenda gerada e referência | Foca em recall/estrutura; comum também em sumarização |
| **METEOR** | Alinhamento palavra a palavra considerando sinônimos, stemming e paráfrases, além de precisão/recall combinados | Correlaciona melhor com julgamento humano que o BLEU, mas é mais custosa computacionalmente |
| **CIDEr** | Similaridade de n-gramas ponderada por TF-IDF, comparando a legenda com **múltiplas** referências | Desenvolvida especificamente para captioning; dá menos peso a palavras muito comuns (ex: "a", "uma", "imagem") |
| **SPICE** | Compara grafos de cena (objetos, atributos, relações) extraídos do texto gerado e das referências, em vez de n-gramas | Correlaciona melhor com julgamento humano sobre conteúdo semântico, mas ignora fluência/gramática |

**Observação prática**: BLEU, ROUGE-L e METEOR vêm originalmente de tradução automática/sumarização;
CIDEr e SPICE foram desenhadas especificamente para captioning e tendem a ser mais informativas
nesse contexto. É comum reportar um conjunto (BLEU-4, METEOR, ROUGE-L, CIDEr, SPICE) para permitir
comparação com a literatura.

---

## 4. Métricas baseadas em embeddings / modelos neurais

Métricas mais recentes tentam capturar similaridade **semântica** em vez de apenas sobreposição
lexical, o que ajuda a valorizar paráfrases corretas que as métricas de n-gramas penalizariam.

- **BERTScore**: usa embeddings contextuais de um modelo tipo BERT para calcular similaridade
  semântica entre a legenda gerada e a referência, token a token.
- **CLIPScore**: mede a similaridade (cosseno) entre o embedding da **imagem** e o embedding do
  **texto gerado** no espaço do CLIP — vantagem de **não depender de legendas de referência**,
  útil quando o dataset tem poucas ou nenhuma legenda humana por imagem.
- **RefCLIPScore**: combina CLIPScore (imagem-texto) com similaridade em relação às referências,
  buscando equilibrar fidelidade visual e proximidade das legendas humanas.

Essas métricas costumam correlacionar melhor com avaliação humana do que as métricas puramente
lexicais, especialmente para captar legendas corretas porém formuladas de forma diferente da
referência.

---

## 5. Avaliação humana

Métricas automáticas são proxies imperfeitos. Quando possível, complementa-se com avaliação
humana em critérios como:

- **Adequação/Correção (fidelidade)**: a legenda descreve corretamente o que está na imagem?
- **Fluência/Gramaticalidade**: o texto é natural e bem formado?
- **Completude/Detalhamento**: a legenda cobre os elementos relevantes da cena ou é genérica
  demais (ex: "uma imagem de um lugar")?
- **Ausência de alucinação**: a legenda não menciona objetos/atributos que não estão presentes
  na imagem.

Em projetos acadêmicos com recursos limitados, uma avaliação humana simplificada (ex: escala
Likert de 1–5 aplicada a uma amostra das legendas geradas) já agrega valor significativo à
avaliação puramente automática.

---

## 6. Outras dimensões de avaliação úteis

Além da qualidade textual, vale considerar:

- **Diversidade lexical**: o modelo gera legendas variadas ou repete sempre os mesmos padrões
  genéricos? (métricas como *Distinct-n*, tamanho médio de vocabulário usado).
- **Cobertura de objetos/conceitos**: quão bem a legenda menciona os objetos relevantes da cena
  (pode ser medido com detectores de objetos + comparação com termos da legenda).
- **Viés e generalização**: o modelo performa de forma consistente em diferentes subgrupos de
  imagens (ex: iluminação, ângulo, época do dia), ou só funciona bem nos casos mais comuns do
  conjunto de treino?
- **Custo computacional / latência**: relevante se houver interesse em uso prático (ex:
  aplicações em tempo real).

---

## 7. Considerações para avaliação de imagens de um porto (opcional/específico)

Caso a avaliação precise ser adaptada ao domínio portuário deste projeto, alguns pontos adicionais
podem ser úteis (sem substituir as métricas gerais acima, que continuam sendo a base):

- **Vocabulário de domínio**: verificar se as legendas usam corretamente termos técnicos do
  domínio (ver vocabulário em `docs/dominio-porto-salvador.md`, seção 5), como "navio
  porta-contêineres", "guindaste de cais", "rebocador" etc., em vez de termos genéricos demais
  (ex: "barco", "máquina").
- **Distinção de categorias visuais próximas**: portos têm elementos visualmente parecidos que
  o modelo pode confundir (ex: transtêiner vs. portêiner; navio graneleiro vs. porta-contêineres;
  silo vs. tanque). Uma análise de erro qualitativa sobre esses pares é mais informativa do que
  olhar só a métrica agregada.
- **CLIPScore como complemento útil**: como o dataset deste projeto é pequeno (imagens do Porto de
  Salvador com licença aberta — ver [`dataset.md`](dataset.md)) e pode ter poucas
  legendas de referência por imagem, métricas que não dependem de referências (CLIPScore) ajudam a
  avaliar legendas mesmo com poucos exemplos anotados manualmente.
- **Contexto de cena vs. detalhe técnico**: decidir, na avaliação, se o objetivo é legendas mais
  gerais ("navio atracado em um porto") ou mais técnicas ("navio porta-contêineres atracado ao
  cais, sob um portêiner") — o nível de granularidade esperado deve ser refletido
  também na forma como as legendas de referência foram escritas, para que a comparação automática
  faça sentido. Neste projeto a decisão foi pelo nível **técnico, sem nomes próprios**
  (ver [`prompt-rotulagem.md`](prompt-rotulagem.md)).

---

## 8. Recomendação prática

Para um projeto acadêmico como este, uma combinação razoável costuma ser:

1. **BLEU-4, METEOR, ROUGE-L e CIDEr** — para comparabilidade com a literatura de captioning
   (neste projeto, METEOR ficou de fora; ver seção 9).
2. **CLIPScore** — como métrica adicional independente de referências, útil dado o volume
   pequeno/moderado de dados.
3. **Análise de erro qualitativa em amostra** — inspeção manual de um subconjunto de legendas
   geradas, observando confusões de vocabulário de domínio e possíveis alucinações.

---

## 9. Métricas usadas neste projeto (como ler o `comparativo.md`)

As seções anteriores são conceituais. Esta descreve **exatamente o que o código calcula**
(`src/metrics_ptbr.py`, chamado por `python -m src evaluate`) e como interpretar cada coluna
de `results/comparativo.md` neste experimento.

### 9.1 Antes das métricas: como o texto é preparado

As métricas de n-gramas não comparam o texto bruto. Gerada e referências passam pela mesma
tokenização (`tokenize_pt`):

1. normalização Unicode (NFC) e **minúsculas**;
2. **hífen vira espaço** ("porta-contêineres" → "porta contêineres");
3. pontuação removida;
4. **acentos mantidos**.

Consequência prática: "contêineres" e "conteineres" (ou "contêiners") são **palavras diferentes**
para BLEU, ROUGE-L e CIDEr. Um erro de grafia custa como uma palavra errada — foi por isso que o
prompt do gold precisou ser acentuado (ver [`prompt-rotulagem.md`](prompt-rotulagem.md), seção 4.5).

Não usamos o `PTBTokenizer` do `pycocoevalcap` porque ele chama o Stanford CoreNLP (Java) e foi
feito para o inglês.

### 9.2 Coluna por coluna

Cada imagem de teste tem **3 referências** (os 3 focos: operação, objetos, visão geral); cada
modelo gera **1 legenda** por imagem. `N` é o número de imagens avaliadas.

| coluna | o que mede | como é calculada aqui | faixa | maior é melhor? |
|---|---|---|---|---|
| **N** | imagens avaliadas | imagens do split de teste com predição e referência | — | — |
| **BLEU-1** | precisão de palavras isoladas em relação às referências | `pycocoevalcap`, nível de corpus (contagens somadas em todas as imagens), contagem limitada pelo máximo entre as 3 referências, com penalidade de brevidade | 0 a 1 | sim |
| **BLEU-4** | precisão de sequências de até 4 palavras (média geométrica de 1- a 4-gramas) | idem | 0 a 1 | sim |
| **ROUGE-L** | maior subsequência comum (ordem das palavras, sem exigir contiguidade) | `pycocoevalcap`, F-measure com β = 1,2 (pesa mais o recall), melhor referência por imagem, média nas imagens | 0 a 1 | sim |
| **CIDEr** | consenso com as referências, por n-gramas de 1 a 4 ponderados por TF-IDF | `pycocoevalcap` (`Cider`, não o CIDEr-D), **IDF calculado nas próprias imagens de teste**, escala ×10 | 0 a ~10 (na prática 0 a ~1,5) | sim |
| **BERTScore-F1** | similaridade semântica entre a legenda e a referência mais próxima | BERTimbau (`neuralmind/bert-base-portuguese-cased`), camada 9, **sem reescala por baseline**, melhor referência por imagem | ~0,5 a 1 (valores comprimidos) | sim |
| **CLIPScore** | alinhamento entre a **imagem** e a legenda, **sem usar referência** | `2,5 × max(cos(imagem, legenda), 0)`; imagem no CLIP ViT-B/32, texto no encoder multilíngue destilado (`clip-ViT-B-32-multilingual-v1`) | 0 a 2,5 (na prática ~0,6 a 0,9) | sim, com ressalva (9.4) |
| **RefCLIPScore** | combina o CLIPScore com a similaridade da legenda às referências | média harmônica entre o CLIPScore e a maior similaridade de cosseno legenda × referência (encoder de texto multilíngue) | 0 a ~1,4 | sim |
| **Distinct-1** | variedade de vocabulário entre todas as legendas do modelo | palavras únicas ÷ total de palavras, somando as 15 legendas | 0 a 1 | depende (9.3) |
| **Distinct-2** | variedade de pares de palavras | bigramas únicos ÷ total de bigramas | 0 a 1 | depende (9.3) |
| **Tam.medio** | tamanho médio das legendas | média de palavras (após a tokenização) por legenda | — | o ideal é ficar perto das referências (~20,5) |

Não calculamos **METEOR** nem **SPICE**: ambos dependem de Java, e o METEOR usa o WordNet em
inglês — em PT-BR o ganho não compensa.

### 9.3 Como ler as famílias de métricas

**Sobreposição de palavras (BLEU, ROUGE-L, CIDEr).** Medem o quanto o modelo escreve *como as
referências*. São sensíveis a vocabulário e grafia: acertar "portêiner" vale, escrever "guindaste
grande" ou "portêinê" não vale, mesmo que a cena esteja certa. O BLEU-4 é baixo em captioning
em geral (bons modelos ficam em torno de 0,2 a 0,4 nos benchmarks em inglês, como o COCO) e ainda mais aqui, com referências técnicas
e 3 focos diferentes.

**Semântica (BERTScore).** Dá crédito a sinônimos e paráfrases. Sem a reescala por baseline,
todos os valores ficam numa faixa estreita (0,65 a 0,79 neste experimento): diferenças de
0,02 a 0,05 já são relevantes.

**Imagem × texto (CLIPScore, RefCLIPScore).** Não dependem só das referências — avaliam se o
texto "combina" com a imagem. Ver a ressalva importante abaixo.

**Diversidade (Distinct-1/2) e tamanho.** Servem para detectar um modelo que repete sempre a
mesma frase genérica (Distinct baixo). Não são métricas de qualidade: neste experimento,
todas as variantes têm Distinct parecido (~0,33 / ~0,67), ou seja, nenhuma colapsou. O tamanho
médio mostra se o modelo respeita a regra de 12 a 30 palavras e o estilo das referências.

### 9.4 Cuidados específicos deste experimento

1. **CIDEr instável com N pequeno.** O IDF é calculado sobre as 15 imagens de teste: uma palavra
   rara em 15 legendas ganha peso alto por acaso. Compare variantes entre si, nunca com valores
   da literatura, e reporte sempre o N.

2. **O CLIPScore premia o que a especificação proíbe.** No comparativo de out/2026, o `base`
   (zero-shot) teve o **maior** CLIPScore (0,81), acima das variantes treinadas (0,66 a 0,72) e do
   gold (0,71). Os exemplos mostram por quê: o `base` descreve **cor** ("gancho vermelho"),
   **marcas e textos legíveis** ("Taylor 9975C", "Delmas", "Maersk") e **céu/clima** — tudo
   visualmente bem ancorado e premiado pelo CLIP, mas proibido pela definição técnica da tarefa
   (ver [`prompt-rotulagem.md`](prompt-rotulagem.md)). Além disso, jargão como "portêiner" e
   "spreader" em português é pouco representado no encoder multilíngue. Neste projeto, portanto,
   o CLIPScore mede em parte **aderência visual genérica**, não aderência à tarefa: use-o como
   sinal complementar, e leia a divergência com as métricas de referência como evidência da
   troca descrita em `prompt-rotulagem.md`, seção 3.2.

3. **As métricas de referência medem proximidade ao rotulador.** As referências são do Claude;
   um modelo que escreve parecido com ele pontua mais. O gold tem vantagem adicional por receber
   o mesmo glossário. Ver as limitações em [`prompt-rotulagem.md`](prompt-rotulagem.md), seção 7.

4. **Uma execução, 15 imagens.** Diferenças pequenas (ex: `lora` × `qlora`) estão dentro do ruído.
   Os números absolutos variam com o sorteio das imagens de teste e com a semente do treino; só
   diferenças grandes e consistentes entre métricas (ex: `base` → `lora`/`qlora` → `gold`) devem
   ser tratadas como resultado.

5. **Tempo (`s/img`).** Para as variantes locais, mede a inferência na mesma máquina. Para o gold
   (API remota), a latência não é comparável e fica fora da tabela — ver
   [`metodologia-variantes.md`](metodologia-variantes.md).

### 9.5 Leitura resumida do comparativo de out/2026

- Nas métricas de referência (BLEU-4, ROUGE-L, CIDEr, BERTScore) a ordem é a mesma:
  `base` < `pretrain` < `finetune` < `lora` ≈ `qlora` < `gold`.
- `lora` e `qlora` superam o `finetune` completo, coerente com a val loss do treino
  ([`../results/treino_custos.md`](../results/treino_custos.md)): com 68 imagens de treino, ajustar
  todos os pesos sobreajusta, e os adaptadores atuam como regularização.
- `lora` e `qlora` estão empatados dentro do ruído.
- O CLIPScore inverte a ordem pelos motivos do item 9.4.2.
- Qualitativamente, o SLM adaptado acerta o vocabulário de domínio (reach stacker, spreader,
  contêiner-tanque), mas ainda erra grafia ("contêiners", "portêinê") e às vezes inventa detalhes
  — erros que o gold não comete.

---

## 10. Fontes e leitura complementar

- Vinyals et al., *Show and Tell: A Neural Image Caption Generator* (2015).
- Xu et al., *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention* (2015).
- Papineni et al., *BLEU: a Method for Automatic Evaluation of Machine Translation* (2002).
- Banerjee & Lavie, *METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with
  Human Judgments* (2005).
- Vedantam et al., *CIDEr: Consensus-based Image Description Evaluation* (2015).
- Anderson et al., *SPICE: Semantic Propositional Image Caption Evaluation* (2016).
- Zhang et al., *BERTScore: Evaluating Text Generation with BERT* (2020).
- Hessel et al., *CLIPScore: A Reference-free Evaluation Metric for Image Captioning* (2021).
