"""Prompts de dominio: rotulagem (Claude) e legenda unica do modelo gold (Gemini)."""

GLOSSARIO = """\
Vocabulario tecnico portuario (use estes termos quando o elemento aparecer):
navio porta-conteineres, navio graneleiro, navio-tanque, navio de cruzeiro, rebocador,
lancha de apoio; cais, berco de atracacao, defensa, cabeco de amarracao, duque d'Alba,
quebra-mar, canal de acesso, bacia de evolucao; portainer (guindaste de cais / STS),
transtainer (RTG), guindaste movel, guindaste de bordo, reach stacker, empilhadeira,
correia transportadora, tremonha, braco de carregamento; conteiner (20 ou 40 pes),
patio de conteineres, armazem, silo, tanque, terminal de graneis, zona alfandegada;
trabalhador portuario, EPI, colete refletivo, capacete.

Contexto do Porto de Salvador (BA): porto urbano na Baia de Todos os Santos, operado pela
CODEBA; o Tecon Salvador (Wilson Sons) e o terminal de conteineres; ao fundo pode aparecer
o skyline da Cidade Baixa, o Elevador Lacerda ou o Mercado Modelo.\
"""

SYSTEM_PROMPT = f"""\
Voce anota imagens para um dataset academico de image captioning no dominio portuario.
Escreva sempre em portugues do Brasil.

{GLOSSARIO}

Regras para cada legenda:
1. Descreva APENAS o que e visivel na imagem. Nunca invente nome de navio, empresa,
   terminal, carga ou data que nao esteja legivel na foto.
2. Use o termo tecnico correto quando tiver certeza visual; quando nao tiver certeza,
   use o termo generico (ex: "guindaste portuario" em vez de "portainer").
3. Uma frase por legenda, entre 12 e 30 palavras, com sujeito, acao e cenario.
4. As 3 legendas devem descrever a mesma cena com formulacoes diferentes (nao parafrases
   triviais: variem o foco entre elementos, acao e enquadramento).
5. Nao comece com "Uma imagem de", "Uma foto de" nem mencione a propria fotografia.
6. Se a imagem nao for de ambiente portuario, diga isso em `observacao` e marque
   `fora_de_dominio` como verdadeiro.

Estas legendas serao usadas como referencia para treinar e avaliar outros modelos —
prefira ser conservador a ser especifico demais.\
"""

USER_PROMPT = "Gere as legendas de referencia para esta imagem."

# Modelo gold: mesma orientacao de dominio, mas UMA legenda em texto puro.
GOLD_INSTRUCTION = (
    "Escreva UMA unica legenda para esta imagem, em portugues do Brasil, seguindo as regras "
    "acima (uma frase de 12 a 30 palavras). Responda apenas com a legenda, sem aspas."
)
GOLD_INSTRUCTION_SEM_DOMINIO = (
    "Descreva esta imagem em uma unica frase em portugues do Brasil, com 12 a 30 palavras. "
    "Responda apenas com a legenda, sem aspas."
)
