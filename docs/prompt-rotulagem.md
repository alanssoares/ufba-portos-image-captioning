# Prompt de rotulagem: problema, decisões e impactos

> Registro de decisão (out/2026). Explica por que o prompt que gera as legendas de referência
> mudou, o que foi considerado, o que foi implementado e o que isso muda na interpretação dos
> resultados. Leitura recomendada para quem for rodar o pipeline, revisar os rótulos ou
> escrever a seção de metodologia.

---

## 1. Contexto

As legendas de referência (`data/labels.jsonl`) são o **gabarito de todo o experimento**:
alimentam o pré-treino e os fine-tunings (splits de treino e validação) e são a referência das
métricas no split de teste (BLEU, ROUGE-L, CIDEr, BERTScore, CLIPScore). Elas são geradas por
um LLM (Claude) a partir do *system prompt* de [`src/prompts.py`](../src/prompts.py), com
3 legendas por imagem — veja [`dataset.md`](dataset.md#3-legendas-rótulos).

Como as métricas medem **proximidade às referências**, o prompt de rotulagem define, na prática,
**qual tarefa está sendo avaliada**. Mudar o prompt muda o que conta como "acertar".

## 2. O problema

Ao rodar a rotulagem nas primeiras 5 imagens (versão inicial do prompt), apareceram cinco
problemas:

| # | problema | exemplo / causa | efeito |
|---|---|---|---|
| 1 | **Atributos subjetivos e não técnicos** nas legendas | "contêiner marrom **enferrujado**", "com **luz de fim de tarde**", "em dia **nublado**" | ruído nas referências: o rotulador menciona isso de forma inconsistente, e as métricas de n-gramas punem ou premiam esses termos de forma arbitrária |
| 2 | **Foco difuso** | a regra "3 legendas com focos diferentes (elementos, ação, enquadramento)" deixava o enquadramento competir com a operação | legendas descreviam a cena, e não a operação portuária e os equipamentos — que é o que o projeto quer avaliar |
| 3 | **Contexto fixo do Porto de Salvador** | o prompt descrevia CODEBA, Tecon e Elevador Lacerda, mas só ~7 das 99 imagens do dataset são de Salvador (o resto vem de Hamburgo, Roterdã, Gdynia, Japão...) | risco de o rotulador "puxar" a legenda para o contexto errado |
| 4 | **Nomes próprios e textos legíveis** | nome do navio, armador ou terminal pintados no casco ou nos guindastes | premiaria OCR e memorização (um modelo que reconhece navios de fotos públicas), não compreensão da cena; agrava o risco de contaminação já declarado |
| 5 | **Quantidades sem certeza** | "**quatro** portêineres" quando não dava para contar | afirmação não verificável na referência |

Na discussão das correções surgiram mais dois problemas, de **desenho experimental**:

| # | problema | efeito |
|---|---|---|
| 6 | **Gold e SLM recebiam especificações diferentes.** O Gemini (gold) recebia o *system prompt* completo (glossário + regras), mais uma instrução "escreva no foco da 1ª legenda". As variantes Qwen recebiam só a instrução curta de `model.instruction`, que não dizia para evitar cor, luz, conservação ou nomes | o Qwen **base (zero-shot)** perderia pontos por *estilo*, não por *conhecimento*; o gold sairia artificialmente favorecido. E o gold escrevia num foco único, enquanto os SLMs treinam nos três focos misturados |
| 7 | **O prompt não era versionado** | se o prompt mudar no meio da rotulagem, o `labels.jsonl` passa a ter referências de "tarefas" diferentes sem que ninguém perceba |

## 3. O que foi discutido

### 3.1 A mudança faz sentido metodologicamente?

**Sim, desde que aplicada igualmente a todos os modelos avaliados.** Argumentos:

- **Validade da tarefa.** Legendas "genéricas", no estilo COCO, medem uma descrição visual
  ampla. O objetivo do projeto é medir se o modelo reconhece **a operação portuária e os
  equipamentos**. Referências que especificam isso fazem as métricas medirem o que interessa.
  É a mesma lógica de datasets de domínio, como os de sensoriamento remoto (por exemplo, o
  RSICD), que usam vocabulário controlado e foco em objetos e relações.
- **Menos ruído.** Ferrugem, luz e clima são atributos que o rotulador ora menciona, ora não.
  Removê-los reduz a variância entre referências sem perder informação técnica.
- **Omitir nomes evita atalhos.** O modelo é avaliado por entender a cena, não por ler o casco.
- **Focos fixos aumentam a cobertura.** Três referências complementares (operação · objetos e
  relações · visão geral) dão mais sinal ao treino com `captions_per_image: all`.

### 3.2 Riscos e trocas reconhecidos

- **Viés de vocabulário.** Referências com jargão (portêiner, transtêiner, spreader) penalizam no
  BLEU e no CIDEr um modelo que escreve "guindaste", mesmo quando está correto. O BERTScore
  atenua isso em parte. É uma intensificação da limitação já declarada: **as métricas medem
  proximidade ao rotulador**.
- **Cor deixa de ser descrita.** Cor é informação visual ancorada, e o CLIPScore costuma
  premiá-la. Restringi-la é uma escolha de definição de tarefa (técnica, não descritiva), e
  precisa estar escrita assim na metodologia.
- **CIDEr com focos diferentes.** O CIDEr usa a média de similaridade com as referências; uma
  legenda que acerta um único foco tem teto mais baixo. O efeito vale para todos os modelos
  (o ranking se mantém), mas os valores absolutos ficam menores do que em datasets com
  referências parafraseadas.

## 4. A solução

### 4.1 Prompt de rotulagem (`src/prompts.py`)

O prompt foi reorganizado em blocos reutilizáveis:

| bloco | conteúdo | usado por |
|---|---|---|
| `GLOSSARIO` | termos por categoria — embarcações, **partes do navio**, infraestrutura, equipamentos (com spreader), carga (com peças de peação, contêiner refrigerado), áreas, pessoas e **ações** (içamento, movimentação, peação, atracação...). Diz explicitamente que as imagens vêm de vários portos e que o porto não deve ser presumido | rotulador e gold |
| `PRIORIDADES` | (a) operação em curso e quem a executa → (b) identificação técnica dos objetos → (c) relação espacial; cenário só como localização operacional | rotulador e gold |
| `REGRAS_COMUNS` (1–7) | só o visível; **sem nomes próprios nem textos legíveis**; termo técnico só com certeza; verbo técnico, e "carregamento"/"descarga" só se o sentido for visível; **sem conservação, estética, clima, céu ou luz** (cor só para distinguir objetos iguais); quantidades e tamanhos só com certeza; 12 a 30 palavras | rotulador e gold |
| `REGRAS_ROTULAGEM` (8–10) | 3 focos fixos: (1) operação + equipamento, (2) objetos e relações espaciais, (3) visão geral; `objetos` com termos do glossário, singular, sem cor; `observacao` registra textos omitidos e dúvidas de identificação | só rotulador |

### 4.2 Mesma especificação para todos os modelos avaliados

- **`model.instruction`** (`configs/default.yaml`) passou a ser a **instrução de tarefa única**:
  vale para todas as variantes Qwen **e** para o gold. Ela espelha as regras comuns: operação +
  equipamentos, termos técnicos só com certeza, sem nomes, conservação, clima ou iluminação.
- **Gold** (`src/predict.py`): recebe `model.instruction` + "responda apenas com a legenda", e,
  se `gold_model.use_domain_prompt: true`, o `GOLD_SYSTEM_PROMPT` (glossário + prioridades +
  regras comuns, **sem** as regras de 3 legendas e JSON). O foco fixo na 1ª legenda foi removido.
- **A única diferença intencional entre gold e Qwen é o glossário no *system prompt*.** Com
  `use_domain_prompt: false`, os dois recebem exatamente a mesma instrução.

### 4.3 Versionamento do prompt

- `PROMPT_VERSAO` = hash SHA-256 (12 caracteres) de `SYSTEM_PROMPT` + `USER_PROMPT`,
  insensível a CRLF/LF.
- Cada linha do `labels.jsonl` grava `prompt_versao`.
- `python -m src label` avisa quantas linhas estão em versão diferente da atual e sugere
  `--redo`.
- Versão em uso nesta rodada: **`09101c626775`**.

### 4.4 Antes e depois

| imagem | prompt antigo | prompt novo |
|---|---|---|
| `psa_0002` | Spreader vermelho de um guindaste portuário encaixa-se sobre o teto de um contêiner marrom **enferrujado**, ao lado de um contêiner refrigerado branco. | Spreader de um guindaste portuário encaixa-se sobre o teto de um contêiner empilhado durante a movimentação de contêineres. |
| `psa_0005` | Navio porta-contêineres carregado está atracado junto a **quatro** portêineres de lanças erguidas, enquanto um rebocador encosta no seu costado **com luz de fim de tarde**. | Navio porta-contêineres carregado permanece atracado ao cais junto a portêineres com lanças erguidas, enquanto um rebocador encosta no seu costado. |

## 5. Como as referências desta rodada foram geradas

Por decisão de custo, as referências **não** foram geradas pela API (`python -m src label`). Elas
foram escritas pelo Claude (`claude-opus-5-5`) **numa sessão interativa**: cada imagem
normalizada (`data/images/psa_XXXX.jpg`) foi inspecionada, e as legendas foram escritas
seguindo o mesmo `SYSTEM_PROMPT`, no mesmo formato do `labels.jsonl`. As linhas trazem
`"origem": "sessao interativa (sem API)"`.

Antes de entrar no arquivo, cada lote passou por uma validação automática:

- exatamente 3 legendas;
- 12 a 30 palavras por legenda;
- nenhum termo proibido (ferrugem, sujeira, nublado, céu, entardecer...);
- nenhum início do tipo "Uma imagem de".

Diferenças em relação ao pipeline por API, que precisam ser declaradas:

- não houve saída estruturada por JSON Schema nem registro da chamada à API; a
  reprodutibilidade é a do arquivo versionado, não a de uma re-execução;
- o identificador `claude-opus-5-5` é o modelo configurado na sessão; não é possível garantir
  qual modelo efetivamente serviu cada resposta.

Resultado da rodada (99 imagens, `psa_0001` a `psa_0099`):

| item | valor |
|---|---|
| linhas no `labels.jsonl` | 99 (uma por imagem, nenhuma faltando) |
| versão do prompt | `09101c626775` em todas as linhas |
| legendas | 297 (3 por imagem), nenhuma duplicada |
| tamanho das legendas | 13 a 26 palavras (média 20,5) |
| `fora_de_dominio` | 1 (`psa_0014`) → 98 imagens entram no split |
| imagens do Porto de Salvador (segundo a fonte) | 7, anotadas em `observacao` |
| termos de cor, clima ou luz nas legendas | 0 (verificado por busca) |
| objetos mais frequentes | contêiner (82), pátio de contêineres (52), spreader (43), portêiner (38), navio porta-contêineres (30), cais (28) |

Critérios adotados durante a rotulagem:

- imagens **sem ambiente portuário** são marcadas `fora_de_dominio` (ex: `psa_0014`, um içamento
  em área industrial) e saem do split;
- **unidades de carga do domínio fora do porto** (contêiner-tanque numa fábrica, pátio
  retroportuário, terminal ferroviário) ficam no domínio, com a ressalva em `observacao`;
- cenas noturnas não mencionam a hora do dia (regra 5); o fato fica em `observacao`;
- quando o tipo exato do equipamento não é visualmente certo, usa-se o termo genérico, com a
  dúvida registrada em `observacao` (ex: `psa_0024`, "guindaste de pórtico", que pode ser RTG ou
  straddle carrier).

## 6. Impactos

| onde | impacto |
|---|---|
| **Referências** | mais homogêneas e técnicas; vocabulário controlado pelo glossário; sem nomes próprios |
| **Treino** (pretrain / finetune / lora / qlora) | os modelos aprendem o estilo técnico das referências; como a instrução mudou, modelos treinados com a instrução antiga **não são comparáveis** e devem ser re-treinados (nenhum havia sido treinado até esta mudança) |
| **Qwen base (zero-shot)** | passa a receber a especificação da tarefa na instrução; ainda não vê o glossário, então tende a usar termos genéricos e a ser penalizado nas métricas de n-gramas |
| **Gold (Gemini)** | mesma instrução das variantes + glossário; deixa de ter foco fixo, o que corrige a vantagem anterior |
| **Métricas** | valores absolutos de CIDEr e BLEU tendem a ser menores (focos diferentes, jargão); a comparação **entre modelos** fica mais justa. BERTScore e CLIPScore ajudam a separar "errou o termo" de "errou a cena" |
| **Dataset publicado** | o `labels.jsonl` deve ter uma única `prompt_versao`; o datasheet deve citar a versão |

## 7. Limitações a declarar na apresentação e no relatório

1. Referências geradas por LLM, sem anotação humana; as métricas medem proximidade ao estilo e
   ao vocabulário do rotulador.
2. A tarefa foi **definida como técnica**: sem cor (salvo para distinguir objetos), luz, clima
   ou estado de conservação. Modelos que descrevem esses atributos são penalizados por definição.
3. Viés de vocabulário a favor do glossário.
4. Referências desta rodada geradas em sessão interativa, sem a trilha da API (seção 5).
5. Dataset majoritariamente de portos estrangeiros (~7 de 99 imagens de Salvador).

## 8. Próximos passos recomendados

- [ ] **Revisão humana de uma amostra do teste** (ex: 30 a 50 imagens) por alguém com
      conhecimento portuário, reportando a taxa de acerto dos termos técnicos e da ação descrita.
- [ ] Opcional: incluir na tabela a linha **"base + system prompt"**
      (`--set model.system_prompt=...` com o `GOLD_SYSTEM_PROMPT`), para separar o ganho de
      conhecer o estilo do ganho de conhecimento de domínio.
- [ ] Opcional: reportar as métricas também contra a **referência 1** isolada (foco na operação).
- [ ] Ao ampliar o dataset, manter a mesma `prompt_versao`, ou regerar tudo com `--redo` se o
      prompt mudar.

## 9. Arquivos alterados

| arquivo | mudança |
|---|---|
| `src/prompts.py` | glossário ampliado; prioridades; regras comuns × regras de rotulagem; `GOLD_SYSTEM_PROMPT`; `PROMPT_VERSAO` |
| `src/labeling.py` | grava `prompt_versao`; avisa sobre versões misturadas |
| `src/predict.py` | gold usa `model.instruction` + `GOLD_SYSTEM_PROMPT` |
| `configs/default.yaml` | nova `model.instruction` (compartilhada por Qwen e gold) |
| `tests/test_prompts.py` | testes de versão, separação das regras e instrução |
| `docs/dataset.md`, `docs/metodologia-variantes.md`, `README.md` | regras de rotulagem e do gold atualizadas, com link para este documento |
| `data/labels.jsonl` | referências na versão `09101c626775` |
