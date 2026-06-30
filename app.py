"""
Interface Streamlit — Sistema de Recomendação Híbrido (TCC PUCSP).

Combina Collaborative Filtering (KNN item-based) com Zero-Shot Learning
via BERTimbau para sugerir locais urbanos na região PUC-SP.
"""

from __future__ import annotations

import contextlib
import io

import streamlit as st

from tcc_recommender.data.download import download_movielens
from tcc_recommender.data.loader import load_movielens, prepare_movie_texts
from tcc_recommender.data.places_puc import LUGARES_PUC, load_puc_places
from tcc_recommender.models import BertEmbedder, CollaborativeFilteringRecommender
from tcc_recommender.recommendation import generate_embeddings, hybrid_recommend

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Recomendação Híbrida de Locais",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS leve para cards de resultado
st.markdown(
    """
    <style>
    .place-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3d 100%);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #7c3aed;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .place-card h4 { margin: 0 0 0.4rem 0; color: #f1f5f9; }
    .place-card .score { color: #a78bfa; font-weight: 600; font-size: 0.95rem; }
    .place-card .desc { color: #94a3b8; font-size: 0.88rem; line-height: 1.45; }
    .rank-badge {
        display: inline-block;
        background: #7c3aed;
        color: white;
        border-radius: 50%;
        width: 28px; height: 28px;
        text-align: center;
        line-height: 28px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Funções de carregamento com cache (evitam reprocessamento a cada clique)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Carregando BERTimbau e treinando KNN...")
def initialize_system() -> dict:
    """
    Pipeline completo de setup: dados, embeddings e modelo colaborativo.

    Executado uma vez por sessão graças ao @st.cache_resource.
    """
    download_movielens()
    ratings, movies = load_movielens()
    load_puc_places()

    # Suprime logs verbosos do KNN durante o treinamento
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        embedder = BertEmbedder()
        movie_texts = prepare_movie_texts(movies)
        movie_embeddings = embedder.encode_batch(movie_texts)
        place_embeddings = generate_embeddings(embedder, LUGARES_PUC)

        cf = CollaborativeFilteringRecommender()
        cf.fit(ratings)
        rmse = cf.evaluate_rmse()

    return {
        "embedder": embedder,
        "cf": cf,
        "ratings": ratings,
        "movies": movies,
        "movie_embeddings": movie_embeddings,
        "place_embeddings": place_embeddings,
        "places": LUGARES_PUC,
        "rmse": rmse,
        "n_places": len(LUGARES_PUC),
        "n_users": ratings["userId"].nunique(),
    }


def render_place_card(rank: int, nome: str, score: float, desc: str, alpha: float) -> None:
    """Renderiza um card HTML estilizado para um local recomendado."""
    snippet = desc if len(desc) <= 160 else desc[:157] + "..."
    st.markdown(
        f"""
        <div class="place-card">
            <h4><span class="rank-badge">{rank}</span>{nome}</h4>
            <p class="score">Score híbrido: {score:.3f} &nbsp;·&nbsp; α = {alpha:.2f}</p>
            <p class="desc">{snippet}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar — Setup e controles administrativos
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/map-pin.png", width=72)
    st.title("⚙️ Configurações")
    st.caption("Sistema Híbrido · TCC PUCSP")

    st.divider()

    st.subheader("Treinamento & Dados")
    train_clicked = st.button(
        "🔄 Retreinar Modelo / Carregar Dados",
        type="primary",
        use_container_width=True,
        help="Baixa o MovieLens (se necessário), carrega o BERTimbau e treina o KNN.",
    )

    system = st.session_state.get("system")

    if train_clicked or system is None:
        with st.spinner("Inicializando modelos e embeddings..."):
            system = initialize_system()
            st.session_state["system"] = system
        st.success("Sistema pronto para recomendar!")

    if system:
        st.metric("Locais catalogados", system["n_places"])
        st.metric("Usuários (MovieLens)", f"{system['n_users']:,}")
        st.metric("RMSE (KNN)", f"{system['rmse']:.4f}")

    st.divider()

    with st.expander("⚙️ Configurações avançadas", expanded=False):
        st.subheader("Peso da Recomendação")
        alpha = st.slider(
            "α — balanceamento híbrido",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help=(
                "0.0 = 100% BERTimbau (texto/vibe) · "
                "1.0 = 100% KNN (histórico colaborativo)"
            ),
        )

        col_bert, col_knn = st.columns(2)
        col_bert.caption(f"📝 Texto: {(1 - alpha) * 100:.0f}%")
        col_knn.caption(f"👥 KNN: {alpha * 100:.0f}%")


# ---------------------------------------------------------------------------
# Área principal
# ---------------------------------------------------------------------------
st.title("📍 Descubra seu próximo rolê")
st.markdown("##### Recomendação Híbrida — BERTimbau + Collaborative Filtering (KNN)")

st.markdown(
    "**Como funciona**\n\n"
    "1. Descreva sua *vibe* no campo principal.\n"
    "2. (Opcional) Informe um ID de usuário para ativar o KNN.\n"
    "3. Clique em **Buscar Locais**."
)

if not st.session_state.get("system"):
    st.info("👈 Use a barra lateral para **Retreinar Modelo / Carregar Dados** e começar.")
    st.stop()

system = st.session_state["system"]

# Inputs do usuário
st.markdown("---")
col_input, col_user = st.columns([3, 1])

with col_input:
    vibe = st.text_area(
        "Qual é a sua vibe agora?",
        placeholder=(
            'Ex: "Quero um bar agitado com música ao vivo" ou '
            '"Um café silencioso para ler e focar"'
        ),
        height=90,
        label_visibility="visible",
    )

with col_user:
    user_id_raw = st.text_input(
        "ID do Usuário (opcional)",
        placeholder="Ex: 42",
        help="Histórico colaborativo via MovieLens 100K (usuários 1–943).",
    )
    user_id_input: int | None = None
    if user_id_raw.strip().isdigit():
        uid = int(user_id_raw.strip())
        if 1 <= uid <= 943:
            user_id_input = uid
        else:
            st.caption("⚠️ ID deve estar entre 1 e 943.")

search = st.button("🔍 Buscar Locais", type="primary", use_container_width=False)

# ---------------------------------------------------------------------------
# Geração e exibição de recomendações
# ---------------------------------------------------------------------------
if search:
    if not vibe or not vibe.strip():
        st.warning("Descreva sua vibe antes de buscar — o BERTimbau precisa de um texto de entrada.")
    else:
        with st.spinner("Processando embeddings e analisando a vizinhança..."):
            recommendations = hybrid_recommend(
                places=system["places"],
                query=vibe.strip(),
                embedder=system["embedder"],
                place_embeddings=system["place_embeddings"],
                alpha=alpha,
                top_k=1,
                user_id=user_id_input,
                cf=system["cf"],
                ratings=system["ratings"],
                movies=system["movies"],
                movie_embeddings=system["movie_embeddings"],
            )

        st.success("Local recomendado com sucesso!")

        if user_id_input and recommendations[0]["score_knn"] is None:
            st.warning(
                f"Usuário **{user_id_input}** sem histórico suficiente — "
                "resultados baseados apenas no BERTimbau (α efetivo = 0)."
            )

        st.markdown("### 🏆 Sua recomendação")

        # Layout em duas colunas: cards à esquerda, detalhes expandíveis à direita
        col_cards, col_details = st.columns([1, 1])

        with col_cards:
            for rec in recommendations:
                render_place_card(
                    rank=rec["rank"],
                    nome=rec["nome"],
                    score=rec["score"],
                    desc=rec["desc"],
                    alpha=rec["alpha"],
                )

        with col_details:
            for rec in recommendations:
                with st.expander(f"#{rec['rank']} · {rec['nome']}", expanded=(rec["rank"] == 1)):
                    st.markdown(f"**Score híbrido:** `{rec['score']:.4f}`")
                    st.markdown(f"**Score BERT (texto):** `{rec['score_bert']:.4f}`")
                    if rec["score_knn"] is not None:
                        st.markdown(f"**Score KNN (histórico):** `{rec['score_knn']:.4f}`")
                    st.markdown("**Descrição completa**")
                    st.write(rec["desc"])

else:
    st.markdown("---")
    st.markdown("#### 💡 Exemplos de vibe para testar")
    examples = [
        "Quero um bar agitado com música ao vivo e ambiente descontraído",
        "Café silencioso com Wi-Fi para estudar e tomar um café especial",
        "Lugar romântico para jantar italiano com massas frescas",
        "Balada underground com rock e clima alternativo",
        "Padaria acolhedora para um brunch de domingo",
    ]
    for example in examples:
        st.markdown(f"- *{example}*")
