"""Quantizacao 4 bits: encoder de visao, mergers e lm_head nunca podem ser quantizados."""
import re

import torch.nn as nn

from src.vlm import SKIP_PADRAO, bnb_skip_patterns, modulos_quantizados_indevidos

# Nomes reais do Qwen3-VL (named_modules) — o erro original foi nos mergers.
VISAO = [
    "model.visual.blocks.0.attn.qkv",
    "model.visual.merger.linear_fc1",
    "model.visual.merger.linear_fc2",
    "model.visual.deepstack_merger_list.0.linear_fc1",
    "model.visual.deepstack_merger_list.2.linear_fc2",
]
LM_HEAD = ["lm_head"]
LLM = ["model.language_model.layers.0.self_attn.q_proj", "model.language_model.layers.27.mlp.down_proj"]


def pula_v5(nome, padroes):
    """Copia de transformers 5.x: quantizers_utils.should_convert_module (negada)."""
    return any(re.match(f"{k}\\.", nome) or re.match(f"{k}", nome) or nome.endswith(k) for k in padroes)


def pula_v4(nome, padroes):
    """transformers 4.x: substring no caminho do modulo."""
    return any(k in nome for k in padroes)


def test_padrao_antigo_falhava_no_transformers_5():
    assert not pula_v5("model.visual.merger.linear_fc1", SKIP_PADRAO)   # o bug do Colab


def test_padroes_pulam_visao_e_lm_head_nas_duas_versoes():
    padroes = bnb_skip_patterns(SKIP_PADRAO)
    for pula in (pula_v4, pula_v5):
        assert all(pula(n, padroes) for n in VISAO + LM_HEAD), pula.__name__
        assert not any(pula(n, padroes) for n in LLM), pula.__name__   # o LLM continua quantizado


def test_padroes_sem_duplicatas():
    p = bnb_skip_patterns(["visual", "visual", "lm_head"])
    assert len(p) == len(set(p)) == 4


class Fake4bit(nn.Linear):
    pass


def test_detecta_modulo_de_visao_quantizado():
    m = nn.Module()
    m.visual = nn.Module()
    m.visual.merger = Fake4bit(2, 2)        # quantizado indevidamente
    m.visual.ok = nn.Linear(2, 2)
    m.layers = Fake4bit(2, 2)               # LLM quantizado: esperado
    assert modulos_quantizados_indevidos(m, SKIP_PADRAO, classe_4bit=Fake4bit) == ["visual.merger"]
