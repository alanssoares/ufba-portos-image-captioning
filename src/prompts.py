"""Prompts de dominio: rotulagem (Claude) e system prompt do modelo gold (Gemini).

Foco das legendas: a OPERACAO em curso, a IDENTIFICACAO TECNICA dos objetos e a
RELACAO ESPACIAL entre eles. Estado de conservacao, estetica, clima e luz ficam de fora.

Quem recebe o que:
  - rotulador (Claude): SYSTEM_PROMPT (glossario + regras comuns + regras de rotulagem);
  - gold (Gemini): GOLD_SYSTEM_PROMPT (glossario + regras comuns) se
    gold_model.use_domain_prompt, mais a MESMA instrucao de tarefa das variantes Qwen
    (model.instruction, na config) — a unica diferenca entre gold e Qwen e o glossario.

O TEXTO DOS PROMPTS E ACENTUADO de proposito: o modelo copia a grafia do glossario, e as
referencias estao em portugues acentuado ("contêiner", "portêiner"). Um glossario sem acento
fazia o gold escrever "conteiner"/"portainer", palavras diferentes para BLEU/CIDEr.

PROMPT_VERSAO identifica o texto do prompt de rotulagem e e gravado em cada linha do
labels.jsonl: todas as referencias de um experimento devem ter versoes compativeis
(VERSOES_COMPATIVEIS).
"""
import hashlib

GLOSSARIO = """\
Vocabulário técnico portuário (use estes termos quando o elemento aparecer):
- Embarcações: navio porta-contêineres, navio graneleiro, navio-tanque, navio de cruzeiro,
  navio ro-ro, rebocador, lancha de apoio, barcaça.
- Partes do navio: casco, costado, convés, porão, tampa de escotilha, ponte de comando,
  proa, popa, guindaste de bordo.
- Infraestrutura: cais, berço de atracação, defensa, cabeço de amarração, duque d'Alba,
  quebra-mar, canal de acesso, bacia de evolução.
- Equipamentos: portêiner (guindaste de cais / STS), spreader, transtêiner (RTG),
  guindaste móvel, reach stacker, empilhadeira, caminhão, carreta porta-contêiner,
  correia transportadora, tremonha, carregador de navios, braço de carregamento.
- Carga: contêiner (20 ou 40 pés), contêiner refrigerado, contêiner-tanque, flat rack,
  carga geral, carga de projeto, granel sólido, granel líquido; peças de peação
  (twistlock, barra de peação).
- Áreas: pátio de contêineres, armazém, silo, tanque, gate, terminal de contêineres,
  terminal de granéis.
- Pessoas: trabalhador portuário, EPI, capacete, colete refletivo.
- Ações: içamento, movimentação, carregamento (embarque), descarga (desembarque),
  empilhamento, peação, atracação, desatracação, amarração, manobra, reboque,
  transporte no pátio, inspeção.

As imagens vêm de vários portos do mundo (entre eles o Porto de Salvador, BA). Não assuma
o porto, a cidade nem o país.\
"""

PRIORIDADES = """\
O que descrever, em ordem de prioridade:
  a) a operação ou ação em curso e o equipamento ou pessoa que a executa;
  b) a identificação técnica das embarcações, equipamentos e cargas visíveis;
  c) a relação espacial entre eles (sobre o convés, junto ao cais, no pátio, a bordo).
O cenário só entra como localização operacional (cais, convés, pátio, canal).\
"""

# Regras 1-7: valem para toda legenda do experimento (referencias e gold).
REGRAS_COMUNS = """\
1. Descreva APENAS o que é visível. Nunca invente carga, destino ou intenção.
2. Não cite nomes próprios nem textos legíveis (navio, armador, terminal, empresa, código de
   contêiner): use só o tipo do objeto ("navio porta-contêineres", não o nome dele).
3. Use o termo técnico do glossário quando tiver certeza visual; sem certeza, use o termo
   genérico (ex: "guindaste portuário" em vez de "portêiner").
4. Ação: use o verbo técnico (iça, movimenta, empilha, reboca, atraca). Só diga
   "carregamento" ou "descarga" se o sentido da operação for visível; senão, "movimentação".
   Em cena sem ação, descreva a situação operacional (atracado, empilhados, em espera).
5. NÃO descreva: estado de conservação (ferrugem, sujeira, desgaste), juízos estéticos, clima,
   céu, luz ou hora do dia. Cor só quando for necessária para distinguir dois objetos do
   mesmo tipo na cena.
6. Quantidades e tamanhos (ex: "dois portêineres", "contêiner de 40 pés") só com certeza visual.
7. Uma frase por legenda, entre 12 e 30 palavras, com sujeito, ação e local. Não comece com
   "Uma imagem de", "Uma foto de" nem mencione a própria fotografia.\
"""

# Regras 8-10: so para o rotulador (3 legendas + campos estruturados).
REGRAS_ROTULAGEM = """\
8. As 3 legendas descrevem a mesma cena com focos diferentes:
   1a) a operação principal e o equipamento que a executa;
   2a) os objetos técnicos presentes e como se relacionam no espaço;
   3a) a visão geral da cena operacional (onde ocorre e demais elementos ou pessoas envolvidos).
9. Em `objetos`, liste todos os elementos técnicos visíveis com os termos do glossário:
   minúsculas, singular, sem cor.
10. Se a imagem não for de ambiente portuário, diga isso em `observacao` e marque
    `fora_de_dominio` como verdadeiro. Use `observacao` também para registrar textos legíveis
    omitidos ou dúvidas de identificação.\
"""

_INTRO = """\
Você anota imagens para um dataset acadêmico de image captioning no domínio portuário.
Escreva sempre em português do Brasil.\
"""

SYSTEM_PROMPT = f"""\
{_INTRO}

{GLOSSARIO}

{PRIORIDADES}

Regras para cada legenda:
{REGRAS_COMUNS}
{REGRAS_ROTULAGEM}

Estas legendas serão usadas como referência para treinar e avaliar outros modelos —
prefira ser conservador a ser específico demais.\
"""

USER_PROMPT = "Gere as legendas de referência para esta imagem."

GOLD_SYSTEM_PROMPT = f"""\
Você descreve imagens do domínio portuário em português do Brasil.

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

# Versoes anteriores com as MESMAS regras e so diferenca de grafia. Referencias geradas
# com elas continuam validas (nao precisam de --redo).
#   09101c626775 -> texto sem acentos e "portainer"/"transtainer"; as legendas geradas com
#                   ela ja usam a grafia acentuada ("contêiner", "portêiner").
VERSOES_EQUIVALENTES = {"09101c626775"}
VERSOES_COMPATIVEIS = {PROMPT_VERSAO} | VERSOES_EQUIVALENTES
