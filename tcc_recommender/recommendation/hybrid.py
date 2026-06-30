"""Fusão híbrida entre Collaborative Filtering (KNN) e Zero-Shot (BERTimbau)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from tcc_recommender.models.collaborative import CollaborativeFilteringRecommender
from tcc_recommender.models.embeddings import BertEmbedder


def generate_embeddings(embedder: BertEmbedder, places: list[dict]) -> np.ndarray:
    """Gera embeddings BERTimbau para a lista de locais."""
    texts = [place["desc"] for place in places]
    return embedder.encode_batch(texts)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normaliza scores para [0, 1] via min-max."""
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.ones_like(scores)
    return (scores - lo) / (hi - lo)


def get_bert_scores(
    query: str,
    place_embeddings: np.ndarray,
    embedder: BertEmbedder,
) -> np.ndarray:
    """Similaridade semântica entre a vibe do usuário e cada local."""
    query_emb = embedder.encode(query)
    return cosine_similarity([query_emb], place_embeddings)[0]


def get_knn_recommendations(
    user_id: int,
    cf: CollaborativeFilteringRecommender,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    movie_embeddings: np.ndarray,
    place_embeddings: np.ndarray,
    rating_threshold: float = 4.0,
) -> np.ndarray | None:
    """
    Perfil colaborativo via histórico KNN do usuário.

    Combina filmes bem avaliados (proxy de preferências) com predições CF
    para produzir scores de similaridade contra os locais urbanos.
    """
    user_ratings = ratings[ratings["userId"] == user_id]
    if user_ratings.empty:
        return None

    movie_id_to_idx = {mid: idx for idx, mid in enumerate(movies["movieId"])}

    profile_vectors: list[np.ndarray] = []
    profile_weights: list[float] = []

    for _, row in user_ratings.iterrows():
        movie_id = int(row["movieId"])
        if movie_id not in movie_id_to_idx:
            continue

        idx = movie_id_to_idx[movie_id]
        observed = float(row["rating"])

        if cf.algo is not None:
            predicted = cf.algo.predict(user_id, movie_id).est
            weight = max(observed, predicted)
        else:
            weight = observed

        if observed >= rating_threshold:
            profile_vectors.append(movie_embeddings[idx] * weight)
            profile_weights.append(weight)

    if not profile_vectors:
        liked = user_ratings[user_ratings["rating"] >= rating_threshold]
        for _, row in liked.iterrows():
            movie_id = int(row["movieId"])
            if movie_id not in movie_id_to_idx:
                continue
            idx = movie_id_to_idx[movie_id]
            weight = float(row["rating"])
            profile_vectors.append(movie_embeddings[idx] * weight)
            profile_weights.append(weight)

    if not profile_vectors:
        return None

    user_profile = np.average(profile_vectors, axis=0, weights=profile_weights)
    return cosine_similarity([user_profile], place_embeddings)[0]


def hybrid_recommend(
    places: list[dict],
    query: str,
    embedder: BertEmbedder,
    place_embeddings: np.ndarray,
    alpha: float,
    top_k: int = 1,
    user_id: int | None = None,
    cf: CollaborativeFilteringRecommender | None = None,
    ratings: pd.DataFrame | None = None,
    movies: pd.DataFrame | None = None,
    movie_embeddings: np.ndarray | None = None,
) -> list[dict]:
    """
    Recomendação híbrida: score = (1 - α) · BERT + α · KNN.

    α = 0.0 → 100% similaridade textual (Zero-Shot)
    α = 1.0 → 100% perfil colaborativo (histórico KNN)
    """
    bert_raw = get_bert_scores(query, place_embeddings, embedder)
    bert_norm = _normalize_scores(bert_raw)

    knn_norm = np.zeros(len(places))
    used_knn = False

    if user_id is not None and cf is not None and ratings is not None and movies is not None:
        knn_raw = get_knn_recommendations(
            user_id, cf, ratings, movies, movie_embeddings, place_embeddings
        )
        if knn_raw is not None:
            knn_norm = _normalize_scores(knn_raw)
            used_knn = True

    effective_alpha = alpha if used_knn else 0.0
    final_scores = (1.0 - effective_alpha) * bert_norm + effective_alpha * knn_norm

    top_indices = np.argsort(final_scores)[-top_k:][::-1]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        results.append(
            {
                "rank": rank,
                "nome": places[idx]["nome"],
                "desc": places[idx]["desc"],
                "score": float(final_scores[idx]),
                "score_bert": float(bert_norm[idx]),
                "score_knn": float(knn_norm[idx]) if used_knn else None,
                "alpha": effective_alpha,
            }
        )

    return results
