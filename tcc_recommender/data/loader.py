from pathlib import Path

import pandas as pd

from tcc_recommender.config import DATA_DIR


def load_movielens(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega ratings e filmes do MovieLens 100K (100% dos dados)."""
    data_dir = data_dir or DATA_DIR

    ratings = pd.read_csv(
        data_dir / "u.data",
        sep="\t",
        names=["userId", "movieId", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        data_dir / "u.item",
        sep="|",
        encoding="latin-1",
        usecols=[0, 1, 2],
        names=["movieId", "title", "genres"],
    )
    movies = movies[movies["movieId"].isin(ratings["movieId"].unique())]

    print("Dataset carregado com 100% de amostragem.")
    print(f"Usuários únicos: {ratings['userId'].nunique():,}")
    print(f"Lugares (itens MovieLens) únicos: {movies.shape[0]:,}")

    return ratings, movies


def prepare_movie_texts(movies: pd.DataFrame) -> list[str]:
    """Converte gêneros dos filmes em textos para embedding."""
    return (
        movies["genres"]
        .fillna("")
        .str.replace("|", " ", regex=False)
        .astype(str)
        .tolist()
    )
