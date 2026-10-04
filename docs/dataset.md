# Dataset próprio — Porto de Salvador (BA)

Guia para **montar, documentar e publicar** o dataset de imagens do Porto de Salvador com
legendas em PT-BR. Este documento é também o *datasheet* do dataset: preencha as seções marcadas
com ☐ quando a coleta estiver pronta.

---

## 1. Visão geral do pipeline de dados

```
 fontes (fotos próprias, Wikimedia Commons, acervos com licença)
   │   Commons: python -m src.fetch_commons_dataset data/sources/commons_links.txt
   │            → data/raw/images/ + data/sources/commons_metadata.csv (autor, URL, licença)
   ▼  data/raw/                         arquivos originais, qualquer formato/tamanho, subpastas livres
 python -m src prepare
   │   EXIF corrigido · RGB · lado maior ≤ 1024 px · JPEG q92 · dedup por SHA-1 · ids psa_0001...
   ▼  data/images/psa_XXXX.jpg          + data/metadata.csv (licença vem do Commons; outras fontes: à mão)
 python -m src label
   │   Claude (claude-opus-5-5) + glossário portuário · 3 legendas/imagem · objetos · fora_de_dominio
   ▼  data/labels.jsonl
 python -m src split
   │   70/15/15 com semente fixa · exclui fora_de_dominio · estável ao adicionar imagens
   ▼  data/splits.json
```

Tudo é retomável: rode `prepare` e `label` de novo depois de colocar mais imagens em
`data/raw/` — só o que é novo é processado, e o `split` distribui as imagens novas sem mexer no
conjunto de teste existente.

---

## 2. Coleta das imagens

### 2.1 O que coletar

O objetivo é cobrir a variedade visual do porto (ver
[`dominio-porto-salvador.md`](dominio-porto-salvador.md), seção 4.5). Uma distribuição
equilibrada entre:

| categoria de cena | exemplos |
|---|---|
| terminal de contêineres (Tecon) | navio porta-contêineres atracado, portêineres (STS), pátio de contêineres, transtêineres (RTG), reach stackers |
| granéis e carga geral | navio graneleiro, correia transportadora, silos, armazéns, blocos de granito |
| terminal de passageiros | navio de cruzeiro atracado, Elevador Lacerda / Mercado Modelo ao fundo |
| navegação e manobra | rebocadores, lanchas de apoio, navio no canal de acesso / Baía de Todos os Santos |
| infraestrutura | cais, berços, defensas, cabeços de amarração, quebra-mar |
| operação e pessoas | trabalhadores com EPI, caminhões no gate, operação de pátio |
| vistas gerais | panorâmicas do porto com a Cidade Baixa, vistas aéreas |

Varie também **ponto de vista** (nível do chão, elevado, aéreo), **distância** (detalhe × panorâmica),
**horário e clima** (dia, pôr do sol, nublado) e **época** — isso reduz vieses no modelo e torna a
avaliação mais honesta.

### 2.2 Quantas imagens

- **Mínimo útil:** ~300 imagens (≈ 45 no teste com o split 70/15/15).
- **Recomendado:** **≥ 700 imagens**, para ter **≥ 100 imagens de teste** — abaixo disso o CIDEr é
  instável (o IDF é calculado sobre o próprio conjunto de teste).
- Cada imagem gera 3 exemplos de treino (uma por legenda, `captions_per_image: all`).

### 2.3 Fontes e licença

Use apenas imagens cuja licença permita **uso acadêmico e redistribuição** (se o dataset for
publicado). Fontes típicas:

- **Fotos próprias** — melhor opção: licença sua (sugestão: CC BY 4.0) e controle total do conteúdo.
- **Wikimedia Commons** — categorias do Porto de Salvador; cada arquivo tem licença própria
  (CC BY, CC BY-SA, domínio público). Registre autor e licença exatamente como na página do arquivo.
- **Acervos institucionais e imprensa** (CODEBA, Wilson Sons, Agência Gov) — só com licença
  explícita ou autorização por escrito; na dúvida, não inclua.
- **Imagens de satélite / aéreas abertas** — respeite os termos do provedor.

Evite rostos identificáveis em primeiro plano e placas legíveis de veículos; se aparecerem,
prefira outra foto ou anote em `observacoes`.

### 2.4 Proveniência: `data/metadata.csv`

Criado e atualizado pelo `prepare` — é o manifesto do pipeline (uma linha por imagem
normalizada). As colunas `fonte`, `url` e `licenca` são preenchidas **automaticamente** para
imagens baixadas com `src/fetch_commons_dataset.py`, a partir do
`data/sources/commons_metadata.csv` (casando pelo nome do arquivo em `data/raw/`). Para fotos
próprias e outros acervos, **preencha-as à mão** — sem elas o dataset não é publicável nem
citável. O `prepare` nunca sobrescreve um valor já preenchido e lista as imagens sem licença.

> Não confunda os dois CSVs: `data/sources/commons_metadata.csv` é a saída bruta do download
> (uma linha por arquivo do Commons, com autor, termos de uso, SHA-256...);
> `data/metadata.csv` é o manifesto com os ids `psa_XXXX` que o resto do pipeline usa.

| coluna | preenchida por | conteúdo |
|---|---|---|
| `image_id` | `prepare` | `psa_0001`, `psa_0002`, ... (`prepare.prefix`) |
| `sha1` | `prepare` | hash do arquivo original (deduplicação) |
| `arquivo_origem` | `prepare` | caminho relativo em `data/raw/` |
| `largura`, `altura` | `prepare` | após o redimensionamento |
| `data_coleta` | `prepare` | data do processamento |
| `fonte` | `prepare` (Commons) ou **você** | autor / acervo (ex: "Fulano de Tal, Wikimedia Commons") |
| `url` | `prepare` (Commons) ou **você** | página de origem |
| `licenca` | `prepare` (Commons) ou **você** | ex: `CC BY-SA 4.0`, `CC BY 4.0`, `domínio público`, `própria` |
| `observacoes` | **você** | qualquer ressalva |

---

## 3. Legendas (rótulos)

Geradas pelo **Claude** (`labeling.model`, padrão `claude-opus-5-5`) com o *system prompt* de
[`src/prompts.py`](../src/prompts.py): glossário portuário (embarcações, partes do navio,
infraestrutura, equipamentos, carga, áreas, pessoas e **ações**) e regras de escrita. O foco é
técnico: **a operação em curso, a identificação dos objetos e a relação espacial entre eles**.
As imagens vêm de vários portos, então o prompt não assume Salvador.

1. descrever **só o que é visível** — nunca inventar carga, destino ou intenção;
2. **não citar nomes próprios nem textos legíveis** (navio, armador, terminal, códigos de
   contêiner) — só o tipo do objeto; o que foi omitido vai para `observacao`;
3. termo técnico quando houver certeza visual; termo genérico quando não houver
   ("guindaste portuário" em vez de "portêiner");
4. verbo técnico para a ação; "carregamento"/"descarga" só se o sentido for visível;
5. **não descrever** estado de conservação (ferrugem, sujeira), estética, clima, céu ou luz;
   cor só para distinguir dois objetos do mesmo tipo;
6. quantidades e tamanhos (20/40 pés) só com certeza visual;
7. uma frase de 12 a 30 palavras, com sujeito, ação e local;
8. 3 legendas com focos fixos: (1) operação principal e equipamento que a executa;
   (2) objetos técnicos e sua relação no espaço; (3) visão geral da cena operacional;
9. não começar com "Uma imagem de...";
10. `objetos`: todos os elementos técnicos visíveis, termos do glossário, singular, sem cor;
11. marcar `fora_de_dominio` se a imagem não for portuária (essas imagens saem do split).

O modelo gold (Gemini) recebe o glossário e as regras comuns (1–7) como *system prompt* e a
mesma instrução de tarefa das variantes Qwen (`model.instruction`). Cada linha do
`labels.jsonl` grava `prompt_versao`; todas devem ter a mesma versão.

Problema, discussão, decisões e impactos dessa especificação:
[`prompt-rotulagem.md`](prompt-rotulagem.md).

A saída é estruturada (JSON Schema), então todas as linhas têm o mesmo formato. Comece com
`--set labeling.limit=5` para conferir qualidade e custo antes de rotular tudo.

> **Limitação:** as referências são de um LLM, não de anotadores humanos. As métricas medem
> proximidade ao estilo e ao vocabulário do rotulador. Para quantificar isso, revise à mão uma
> amostra do teste (ex: 50 imagens) e reporte a concordância.

---

## 4. Formatos dos arquivos

### `data/labels.jsonl` — uma linha por imagem

```json
{"image_id": "psa_0001",
 "modelo": "claude-opus-5-5",
 "legendas": ["Navio porta-contêineres atracado no cais enquanto dois portêineres ...", "...", "..."],
 "objetos": ["navio porta-contêineres", "portêiner", "contêiner", "cais"],
 "fora_de_dominio": false,
 "observacao": "",
 "rotulado_em": "2026-09-26T10:00:00"}
```

### `data/splits.json`

```json
{"train": ["psa_0003", "..."], "val": ["..."], "test": ["..."],
 "seed": 42, "proporcoes": {"train": 0.7, "val": 0.15, "test": 0.15},
 "atualizado_em": "2026-09-26T10:05:00"}
```

### `results/preds_<variante>.jsonl` — uma linha por imagem de teste

```json
{"image_id": "psa_0042", "variante": "lora", "modelo": "Qwen/Qwen3-VL-2B-Instruct+lora",
 "legenda": "Navio porta-contêineres atracado no cais com portêineres ...", "segundos": 0.41}
```

`data/sample/` traz um exemplo mínimo desses formatos (só texto, sem imagens) para testar a avaliação.

---

## 5. Versionamento e publicação

| o quê | no git? | por quê |
|---|---|---|
| `data/raw/` | não | originais pesados (~300 MB, arquivos de até 20 MB); recuperáveis pelas URLs de `data/sources/` |
| `data/images/` | **sim** | normalizadas (≤ 1024 px, ~18 MB): preservam o dataset se as fontes saírem do ar; licença de cada uma em `metadata.csv` |
| `data/metadata.csv` | **sim** | proveniência e licença |
| `data/sources/` | **sim** | lista de links e metadados brutos do Commons |
| `data/labels.jsonl`, `data/splits.json` | **sim** | definem o experimento (o teste precisa ser reproduzível) |
| `outputs/`, `exports/` | não | modelos (GBs) |

Para publicar o dataset (ex: Zenodo ou Hugging Face Datasets): imagens + `metadata.csv` +
`labels.jsonl` + `splits.json` + este documento, com a licença mais restritiva entre as imagens
(ex: se houver CC BY-SA, o conjunto herda o SA).

---

## 6. Datasheet (preencher)

- ☐ **Período de coleta:**
- ☐ **Número de imagens** (total / por split / descartadas como fora de domínio):
- ☐ **Distribuição por categoria de cena** (tabela da seção 2.1):
- ☐ **Fontes e licenças** (contagem por licença, a partir do `metadata.csv`):
- ☐ **Rotulador** (modelo e data) e amostra revisada à mão:
- ☐ **Vieses conhecidos** (ex: excesso de fotos do Tecon, poucas noturnas):
- ☐ **Usos previstos e não previstos:** avaliação acadêmica de captioning no domínio portuário;
  não usar para vigilância ou identificação de pessoas.
