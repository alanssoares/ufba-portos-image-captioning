"""Prompts de dominio: rotulagem (Claude) e system prompt do modelo gold (Gemini).

Foco das legendas: a OPERACAO em curso, a IDENTIFICACAO TECNICA dos objetos e a
RELACAO ESPACIAL entre eles. Estado de conservacao, estetica, clima e luz ficam de fora.

Quem recebe o que:
  - rotulador (Claude): SYSTEM_PROMPT (glossario + regras comuns + regras de rotulagem);
  - gold (Gemini): GOLD_SYSTEM_PROMPT (glossario + regras comuns) se
    gold_model.use_domain_prompt, mais a MESMA instrucao de tarefa das variantes Qwen
    (model.instruction, na config) — a unica diferenca entre gold e Qwen e o glossario.

PROMPT_VERSAO identifica o texto do prompt de rotulagem e e gravado em cada linha do
labels.jsonl: todas as referencias de um experimento devem ter a mesma versao.
"""
import hashlib

GLOSSARIO = """\
Vocabulario tecnico portuario (use estes termos quando o elemento aparecer):
- Embarcacoes: navio porta-conteineres, navio graneleiro, navio-tanque, navio de cruzeiro,
  navio ro-ro, rebocador, lancha de apoio, barcaca.
- Partes do navio: casco, costado, conves, porao, tampa de escotilha, ponte de comando,
  proa, popa, guindaste de bordo.
- Infraestrutura: cais, berco de atracacao, defensa, cabeco de amarracao, duque d'Alba,
  quebra-mar, canal de acesso, bacia de evolucao.
- Equipamentos: portainer (guindaste de cais / STS), spreader, transtainer (RTG),
  guindaste movel, reach stacker, empilhadeira, caminhao, carreta porta-conteiner,
  correia transportadora, tremonha, carregador de navios, braco de carregamento.
- Carga: conteiner (20 ou 40 pes), conteiner refrigerado, conteiner-tanque, flat rack,
  carga geral, carga de projeto, granel solido, granel liquido; pecas de peacao
  (twistlock, barra de peacao).
- Areas: patio de conteineres, armazem, silo, tanque, gate, terminal de conteineres,
  terminal de graneis.
- Pessoas: trabalhador portuario, EPI, capacete, colete refletivo.
- Acoes: icamento, movimentacao, carregamento (embarque), descarga (desembarque),
  empilhamento, peacao, atracacao, desatracacao, amarracao, manobra, reboque,
  transporte no patio, inspecao.

As imagens vem de varios portos do mundo (entre eles o Porto de Salvador, BA). Nao assuma
o porto, a cidade nem o pais.\
"""

PRIORIDADES = """\
O que descrever, em ordem de prioridade:
  a) a operacao ou acao em curso e o equipamento ou pessoa que a executa;
  b) a identificacao tecnica das embarcacoes, equipamentos e cargas visiveis;
  c) a relacao espacial entre eles (sobre o conves, junto ao cais, no patio, a bordo).
O cenario so entra como localizacao operacional (cais, conves, patio, canal).\
"""

# Regras 1-7: valem para toda legenda do experimento (referencias e gold).
REGRAS_COMUNS = """\
1. Descreva APENAS o que e visivel. Nunca invente carga, destino ou intencao.
2. Nao cite nomes proprios nem textos legiveis (navio, armador, terminal, empresa, codigo de
   conteiner): use so o tipo do objeto ("navio porta-conteineres", nao o nome dele).
3. Use o termo tecnico do glossario quando tiver certeza visual; sem certeza, use o termo
   generico (ex: "guindaste portuario" em vez de "portainer").
4. Acao: use o verbo tecnico (ica, movimenta, empilha, reboca, atraca). So diga
   "carregamento" ou "descarga" se o sentido da operacao for visivel; senao, "movimentacao".
   Em cena sem acao, descreva a situacao operacional (atracado, empilhados, em espera).
5. NAO descreva: estado de conservacao (ferrugem, sujeira, desgaste), juizos esteticos, clima,
   ceu, luz ou hora do dia. Cor so quando for necessaria para distinguir dois objetos do
   mesmo tipo na cena.
6. Quantidades e tamanhos (ex: "dois portaineres", "conteiner de 40 pes") so com certeza visual.
7. Uma frase por legenda, entre 12 e 30 palavras, com sujeito, acao e local. Nao comece com
   "Uma imagem de", "Uma foto de" nem mencione a propria fotografia.\
"""

# Regras 8-10: so para o rotulador (3 legendas + campos estruturados).
REGRAS_ROTULAGEM = """\
8. As 3 legendas descrevem a mesma cena com focos diferentes:
   1a) a operacao principal e o equipamento que a executa;
   2a) os objetos tecnicos presentes e como se relacionam no espaco;
   3a) a visao geral da cena operacional (onde ocorre e demais elementos ou pessoas envolvidos).
9. Em `objetos`, liste todos os elementos tecnicos visiveis com os termos do glossario:
   minusculas, singular, sem cor.
10. Se a imagem nao for de ambiente portuario, diga isso em `observacao` e marque
    `fora_de_dominio` como verdadeiro. Use `observacao` tambem para registrar textos legiveis
    omitidos ou duvidas de identificacao.\
"""

_INTRO = """\
Voce anota imagens para um dataset academico de image captioning no dominio portuario.
Escreva sempre em portugues do Brasil.\
"""

SYSTEM_PROMPT = f"""\
{_INTRO}

{GLOSSARIO}

{PRIORIDADES}

Regras para cada legenda:
{REGRAS_COMUNS}
{REGRAS_ROTULAGEM}

Estas legendas serao usadas como referencia para treinar e avaliar outros modelos —
prefira ser conservador a ser especifico demais.\
"""

USER_PROMPT = "Gere as legendas de referencia para esta imagem."

GOLD_SYSTEM_PROMPT = f"""\
Voce descreve imagens do dominio portuario em portugues do Brasil.

{GLOSSARIO}

{PRIORIDADES}

Regras para a legenda:
{REGRAS_COMUNS}\
"""

# Acrescentado a instrucao de tarefa so no gold: formato da resposta da API, nao conteudo.
GOLD_FORMATO = "Responda apenas com a legenda, sem aspas."


def versao(*textos: str) -> str:
    """Hash curto e estavel de um conjunto de textos de prompt."""
    h = hashlib.sha256()
    for t in textos:
        h.update(t.replace("\r\n", "\n").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


PROMPT_VERSAO = versao(SYSTEM_PROMPT, USER_PROMPT)
