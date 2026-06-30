import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from tcc_recommender.models.embeddings import BertEmbedder


def recommend_by_preference(
    places: list[dict],
    query: str,
    embedder: BertEmbedder,
) -> dict:
    """Recomenda o lugar mais similar a uma descrição de preferência."""
    embeddings = np.array([embedder.encode(p["desc"]) for p in places])
    preference = embedder.encode(query)
    sims = cosine_similarity([preference], embeddings)[0]
    idx = int(np.argmax(sims))

    result = {
        "nome": places[idx]["nome"],
        "desc": places[idx]["desc"],
        "similaridade": float(sims[idx]),
    }

    print(f"Recomendação sugerida: {result['nome']}")
    print(f"Descrição: {result['desc']}")
    print(f"Similaridade: {result['similaridade']:.3f}")

    return result
