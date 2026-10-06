"""Metricas de captioning adaptadas para legendas em PT-BR.

Decisoes que valem registrar na monografia:

* Tokenizacao propria em vez do PTBTokenizer do `pycocoevalcap`: o tokenizador
  original chama o Stanford CoreNLP (JAR + Java) e foi feito para o ingles.
  Aqui: NFC, minusculas, hifen vira espaco, pontuacao removida, acentos mantidos.
* BLEU / ROUGE-L / CIDEr vem do `pycocoevalcap` (implementacoes puras em Python),
  alimentadas com o texto ja tokenizado por nos.
* METEOR e SPICE ficaram de fora: dependem de JAR Java (e de WordNet em ingles,
  no caso do METEOR) — em PT-BR o ganho nao compensa o atrito.
* CIDEr calcula o IDF a partir do proprio conjunto avaliado; com poucas imagens
  (< ~100) o valor e instavel. Reporte sempre o N junto do score.
* BERTScore usa BERTimbau; CLIPScore usa o encoder de texto multilingue do CLIP.
"""
from __future__ import annotations

import contextlib
import io
import re
import unicodedata
from pathlib import Path

_PONTUACAO = re.compile(r"[^\w\s]", re.UNICODE)
_ESPACOS = re.compile(r"\s+")

BERTIMBAU = "neuralmind/bert-base-portuguese-cased"
CLIP_IMAGEM = "clip-ViT-B-32"
CLIP_TEXTO_MULTILINGUE = "sentence-transformers/clip-ViT-B-32-multilingual-v1"


def tokenize_pt(texto: str) -> list[str]:
    t = unicodedata.normalize("NFC", texto).lower().replace("-", " ")
    t = _PONTUACAO.sub(" ", t)
    return _ESPACOS.sub(" ", t).strip().split()


def normalizar(texto: str) -> str:
    return " ".join(tokenize_pt(texto))


def metricas_ngrama(refs: dict[str, list[str]], hyps: dict[str, str]) -> dict[str, float]:
    """BLEU-1..4, ROUGE-L e CIDEr. As chaves de refs e hyps devem coincidir."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    gts = {k: [normalizar(r) for r in refs[k]] for k in hyps}
    res = {k: [normalizar(v)] for k, v in hyps.items()}

    # O Bleu do pycocoevalcap imprime estatisticas internas em stdout (verbose fixo).
    with contextlib.redirect_stdout(io.StringIO()):
        bleu, _ = Bleu(4).compute_score(gts, res)
        rouge, _ = Rouge().compute_score(gts, res)
        cider, _ = Cider().compute_score(gts, res)
    return {
        "BLEU-1": float(bleu[0]),
        "BLEU-2": float(bleu[1]),
        "BLEU-3": float(bleu[2]),
        "BLEU-4": float(bleu[3]),
        "ROUGE-L": float(rouge),
        "CIDEr": float(cider),
    }


def bertscore_pt(
    refs: dict[str, list[str]],
    hyps: dict[str, str],
    model_type: str = BERTIMBAU,
    num_layers: int = 9,
    device: str | None = None,
) -> dict[str, float]:
    """BERTScore multi-referencia com BERTimbau (fica o melhor par entre as refs)."""
    from bert_score import BERTScorer

    chaves = list(hyps)
    cands = [hyps[k] for k in chaves]
    referencias = [refs[k] for k in chaves]

    scorer = BERTScorer(
        model_type=model_type,
        num_layers=num_layers,
        device=device,
        rescale_with_baseline=False,
    )
    # O tokenizador do BERTimbau nao declara model_max_length (vira ~1e30) e o `tokenizers`
    # novo estoura ao truncar com esse valor. BERT-base aceita no maximo 512 tokens.
    tok = scorer._tokenizer
    if tok.model_max_length > 512:
        tok.model_max_length = 512

    P, R, F1 = scorer.score(cands, referencias, verbose=False)
    return {
        "BERTScore-P": float(P.mean()),
        "BERTScore-R": float(R.mean()),
        "BERTScore-F1": float(F1.mean()),
    }


def clipscore_pt(
    imagens: dict[str, Path],
    hyps: dict[str, str],
    refs: dict[str, list[str]] | None = None,
    device: str | None = None,
) -> dict[str, float]:
    """CLIPScore = 2.5 * max(cos(imagem, legenda), 0) — nao usa referencia.

    RefCLIPScore = media harmonica entre o CLIPScore e a maior similaridade entre
    a legenda gerada e as referencias humanas (so e calculado se `refs` vier).
    O encoder de texto e a versao multilingue destilada, alinhada ao mesmo espaco
    do encoder de imagem CLIP ViT-B/32.
    """
    import numpy as np
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    chaves = [k for k in hyps if k in imagens]
    modelo_img = SentenceTransformer(CLIP_IMAGEM, device=device)
    modelo_txt = SentenceTransformer(CLIP_TEXTO_MULTILINGUE, device=device)

    fotos = [Image.open(imagens[k]).convert("RGB") for k in chaves]
    emb_img = modelo_img.encode(fotos, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    emb_txt = modelo_txt.encode([hyps[k] for k in chaves], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    for foto in fotos:
        foto.close()

    cos = np.sum(emb_img * emb_txt, axis=1)
    clip_s = 2.5 * np.clip(cos, 0, None)
    saida = {"CLIPScore": float(clip_s.mean())}

    if refs:
        melhores = []
        for i, k in enumerate(chaves):
            emb_refs = modelo_txt.encode(refs[k], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            melhores.append(float(np.max(emb_refs @ emb_txt[i])))
        melhores = np.clip(np.array(melhores), 0, None)
        denominador = clip_s + melhores
        harmonica = np.where(denominador > 0, 2 * clip_s * melhores / np.maximum(denominador, 1e-8), 0.0)
        saida["RefCLIPScore"] = float(harmonica.mean())
    return saida


def diversidade(hyps: dict[str, str]) -> dict[str, float]:
    """Distinct-1/2 e tamanho medio — detecta modelo que repete sempre a mesma frase."""
    tokens = [tokenize_pt(t) for t in hyps.values()]
    todos = [tok for seq in tokens for tok in seq]
    bigramas = [tuple(seq[i : i + 2]) for seq in tokens for i in range(len(seq) - 1)]
    return {
        "Distinct-1": len(set(todos)) / len(todos) if todos else 0.0,
        "Distinct-2": len(set(bigramas)) / len(bigramas) if bigramas else 0.0,
        "Vocabulario": float(len(set(todos))),
        "Tam.medio": sum(len(s) for s in tokens) / len(tokens) if tokens else 0.0,
    }
