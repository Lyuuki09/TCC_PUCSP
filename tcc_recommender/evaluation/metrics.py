import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from tcc_recommender.config import PRECISION_K, RATING_THRESHOLD


def precision_at_k_zero_shot(
    testset,
    movie_embeddings: np.ndarray,
    movies_df: pd.DataFrame,
    k: int = PRECISION_K,
    thresh: float = RATING_THRESHOLD,
) -> float:
    """Calcula Precision@K para recomendações zero-shot baseadas em embeddings."""
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(movies_df["movieId"])}
    hits = total = 0

    for uid, iid, true_r in testset:
        if iid not in movie_id_to_idx or true_r < thresh:
            continue
        liked_idx = [
            movie_id_to_idx[i]
            for u, i, r in testset
            if u == uid and r >= thresh and i in movie_id_to_idx
        ]
        if not liked_idx:
            continue
        user_emb = np.mean(movie_embeddings[liked_idx], axis=0)
        sims = cosine_similarity([user_emb], movie_embeddings)[0]
        top_k_idx = np.argsort(sims)[-k:][::-1]
        top_k_ids = movies_df.iloc[top_k_idx]["movieId"].values

        if iid in top_k_ids:
            hits += 1
        total += 1

    return hits / total if total > 0 else 0.0
