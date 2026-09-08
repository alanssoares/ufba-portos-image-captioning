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
- **CLIPScore como complemento útil**: como o dataset deste projeto é pequeno (imagens do Wikimedia
  Commons sobre o Porto de Salvador, conforme `docs/dataset-licensing.md`) e pode ter poucas
  legendas de referência por imagem, métricas que não dependem de referências (CLIPScore) ajudam a
  avaliar legendas mesmo com poucos exemplos anotados manualmente.
- **Contexto de cena vs. detalhe técnico**: decidir, na avaliação, se o objetivo é legendas mais
  gerais ("navio atracado em um porto") ou mais técnicas ("navio porta-contêineres atracado no
  Tecon Salvador, com portêiner ao lado") — o nível de granularidade esperado deve ser refletido
  também na forma como as legendas de referência foram escritas, para que a comparação automática
  faça sentido.

---

## 8. Recomendação prática

Para um projeto acadêmico como este, uma combinação razoável costuma ser:

1. **BLEU-4, METEOR, ROUGE-L e CIDEr** — para comparabilidade com a literatura de captioning.
2. **CLIPScore** — como métrica adicional independente de referências, útil dado o volume
   pequeno/moderado de dados.
3. **Análise de erro qualitativa em amostra** — inspeção manual de um subconjunto de legendas
   geradas, observando confusões de vocabulário de domínio e possíveis alucinações.

---

## 9. Fontes e leitura complementar

- Vinyals et al., *Show and Tell: A Neural Image Caption Generator* (2015).
- Xu et al., *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention* (2015).
- Papineni et al., *BLEU: a Method for Automatic Evaluation of Machine Translation* (2002).
- Banerjee & Lavie, *METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with
  Human Judgments* (2005).
- Vedantam et al., *CIDEr: Consensus-based Image Description Evaluation* (2015).
- Anderson et al., *SPICE: Semantic Propositional Image Caption Evaluation* (2016).
- Zhang et al., *BERTScore: Evaluating Text Generation with BERT* (2020).
- Hessel et al., *CLIPScore: A Reference-free Evaluation Metric for Image Captioning* (2021).
