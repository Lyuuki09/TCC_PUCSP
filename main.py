"""Pipeline completo do sistema de recomendação híbrido."""

from tcc_recommender.config import PRECISION_K
from tcc_recommender.data import (
    download_movielens,
    load_movielens,
    load_puc_places,
    prepare_movie_texts,
)
from tcc_recommender.data.places_puc import DEMO_LUGARES
from tcc_recommender.evaluation import precision_at_k_zero_shot, simulate_ab_test
from tcc_recommender.models import BertEmbedder, CollaborativeFilteringRecommender
from tcc_recommender.recommendation import recommend_by_preference
from tcc_recommender.visualization import plot_metrics_comparison


def run_pipeline() -> dict:
    download_movielens()
    ratings, movies = load_movielens()
    load_puc_places()

    cf = CollaborativeFilteringRecommender()
    cf.fit(ratings)
    rmse = cf.evaluate_rmse()

    embedder = BertEmbedder()
    movie_texts = prepare_movie_texts(movies)
    print(f"Processando embeddings para {len(movie_texts)} itens...")
    movie_embeddings = embedder.encode_batch(movie_texts)
    print(f"Dimensão final dos embeddings: {movie_embeddings.shape}")

    prec_zero = precision_at_k_zero_shot(
        cf.testset, movie_embeddings, movies, k=PRECISION_K
    )
    print(f"Precision@5 (Zero-Shot): {prec_zero:.4f}")
    print(f"RMSE (CF): {rmse:.4f}")

    plot_metrics_comparison(rmse, prec_zero)

    recommend_by_preference(
        DEMO_LUGARES,
        "Bares descontraídos, música ao vivo e ambiente para conhecer gente nova",
        embedder,
    )

    simulate_ab_test(ratings, user_id=1)

    return {"rmse": rmse, "precision_at_5": prec_zero}


if __name__ == "__main__":
    run_pipeline()
