from src.labeling import versoes_desatualizadas
import re

from src.prompts import (GLOSSARIO, GOLD_SYSTEM_PROMPT, PROMPT_VERSAO, REGRAS_COMUNS, REGRAS_ROTULAGEM,
                         SYSTEM_PROMPT, USER_PROMPT, VERSOES_COMPATIVEIS, versao)


def test_versao_estavel_e_sensivel_ao_texto():
    assert PROMPT_VERSAO == versao(SYSTEM_PROMPT, USER_PROMPT)
    assert versao("a\r\nb") == versao("a\nb")              # CRLF do Windows nao muda a versao
    assert versao(SYSTEM_PROMPT + " ", USER_PROMPT) != PROMPT_VERSAO
    assert len(PROMPT_VERSAO) == 12


def test_gold_recebe_regras_comuns_mas_nao_as_de_rotulagem():
    assert REGRAS_COMUNS in GOLD_SYSTEM_PROMPT and REGRAS_COMUNS in SYSTEM_PROMPT
    assert REGRAS_ROTULAGEM in SYSTEM_PROMPT
    assert REGRAS_ROTULAGEM not in GOLD_SYSTEM_PROMPT       # gold escreve 1 legenda, sem JSON


def test_versoes_desatualizadas():
    rows = [{"image_id": "a", "prompt_versao": PROMPT_VERSAO},
            {"image_id": "b", "prompt_versao": "antiga"},
            {"image_id": "c"},
            {"image_id": "d", "prompt_versao": "09101c626775"}]   # so ortografica: compativel
    assert versoes_desatualizadas(rows) == ["b", "c"]
    assert "09101c626775" in VERSOES_COMPATIVEIS


def test_prompt_acentuado_como_as_referencias():
    """O modelo copia a grafia do glossario; sem acento, o gold escrevia 'conteiner'/'portainer'."""
    sem_acento = re.compile(r"\b(conteiner\w*|porta-conteiner\w*|portainer\w*|transtainer\w*|conves|porao|"
                            r"peacao|icamento|patio|portuario|tecnico|visivel|operacao|nao)\b", re.I)
    for texto in (SYSTEM_PROMPT, GOLD_SYSTEM_PROMPT, USER_PROMPT):
        assert not sem_acento.search(texto), sem_acento.search(texto).group()
    for termo in ("contêiner", "porta-contêineres", "portêiner", "transtêiner", "convés", "pátio", "peação"):
        assert termo in GLOSSARIO


def test_instrucao_do_slm_espelha_as_regras():
    from src.config import load_config
    instrucao = " ".join(str(load_config().model.instruction).split()).lower()
    for trecho in ("12 a 30 palavras", "nomes próprios", "conservação", "clima"):
        assert trecho in instrucao
