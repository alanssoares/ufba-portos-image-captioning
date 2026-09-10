"""App de revisao humana das legendas (Streamlit).

    python -m streamlit run src/review_app.py

Fluxo: le data/drafts.jsonl (rascunhos do Claude), o revisor corrige/reescreve e
salva em data/captions.jsonl, que e o ground truth usado pela avaliacao.
"""
from __future__ import annotations

import datetime as dt

import streamlit as st

from src.common import CAPTIONS_JSONL, DRAFTS_JSONL, IMAGES_DIR, index_by, list_images, read_jsonl, write_jsonl

N_LEGENDAS = 3

st.set_page_config(page_title="Revisao de legendas — Porto de Salvador", layout="wide")


def carregar():
    return index_by(read_jsonl(DRAFTS_JSONL)), index_by(read_jsonl(CAPTIONS_JSONL))


def salvar(linha: dict) -> None:
    final = index_by(read_jsonl(CAPTIONS_JSONL))
    final[linha["image_id"]] = linha
    write_jsonl(CAPTIONS_JSONL, [final[k] for k in sorted(final)])


imagens = list_images(IMAGES_DIR)
if not imagens:
    st.error(f"Nenhuma imagem em {IMAGES_DIR}. Rode `python -m src.prepare_images` antes.")
    st.stop()

rascunhos, finais = carregar()
st.session_state.setdefault("idx", 0)
st.session_state.setdefault("revisor", "")

with st.sidebar:
    st.header("Progresso")
    revisadas = sum(1 for p in imagens if p.stem in finais)
    st.progress(revisadas / len(imagens))
    st.write(f"**{revisadas} / {len(imagens)}** revisadas")
    st.session_state["revisor"] = st.text_input("Revisor(a)", st.session_state["revisor"])

    rotulos = [f"{'OK ' if p.stem in finais else '   '} {p.stem}" for p in imagens]
    escolha = st.selectbox("Imagem", range(len(imagens)), index=st.session_state["idx"], format_func=lambda i: rotulos[i])
    if escolha != st.session_state["idx"]:
        st.session_state["idx"] = escolha
        st.rerun()

    if st.button("Pular para a proxima nao revisada", use_container_width=True):
        proxima = next((i for i, p in enumerate(imagens) if p.stem not in finais), None)
        if proxima is not None:
            st.session_state["idx"] = proxima
            st.rerun()

idx = st.session_state["idx"]
path = imagens[idx]
image_id = path.stem
rascunho = rascunhos.get(image_id, {})
final = finais.get(image_id, {})
base = final.get("legendas") or rascunho.get("legendas") or []

col_img, col_form = st.columns([1, 1])
with col_img:
    st.image(str(path), caption=image_id, use_container_width=True)
    if rascunho.get("observacao"):
        st.caption(f"Nota do rascunho: {rascunho['observacao']}")
    if rascunho.get("fora_de_dominio"):
        st.warning("O modelo marcou esta imagem como fora do dominio portuario.")

with col_form:
    st.subheader(f"{image_id}  ({idx + 1}/{len(imagens)})")
    with st.form("revisao", clear_on_submit=False):
        legendas = [
            st.text_area(f"Legenda {i + 1}", value=base[i] if i < len(base) else "", height=80, key=f"leg_{image_id}_{i}")
            for i in range(N_LEGENDAS)
        ]
        objetos = st.text_input(
            "Objetos visiveis (separados por virgula)",
            value=", ".join(final.get("objetos") or rascunho.get("objetos") or []),
            key=f"obj_{image_id}",
        )
        observacao = st.text_input("Observacao do revisor", value=final.get("observacao", ""), key=f"obs_{image_id}")
        descartar = st.checkbox("Descartar (fora do dominio / imagem ruim)", value=final.get("descartada", False))
        c1, c2 = st.columns(2)
        salvar_btn = c1.form_submit_button("Salvar", use_container_width=True)
        salvar_prox = c2.form_submit_button("Salvar e proxima", type="primary", use_container_width=True)

    if salvar_btn or salvar_prox:
        limpas = [t.strip() for t in legendas if t.strip()]
        if not limpas and not descartar:
            st.error("Escreva ao menos uma legenda ou marque como descartada.")
        else:
            salvar(
                {
                    "image_id": image_id,
                    "legendas": limpas,
                    "objetos": [o.strip().lower() for o in objetos.split(",") if o.strip()],
                    "observacao": observacao.strip(),
                    "descartada": bool(descartar),
                    "revisor": st.session_state["revisor"],
                    "revisado_em": dt.datetime.now().isoformat(timespec="seconds"),
                    "origem_rascunho": rascunho.get("modelo", ""),
                }
            )
            if salvar_prox and idx + 1 < len(imagens):
                st.session_state["idx"] = idx + 1
            st.rerun()
