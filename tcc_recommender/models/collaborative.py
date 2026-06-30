import pandas as pd
from surprise import Dataset, KNNBasic, Reader, accuracy
from surprise.model_selection import train_test_split

from tcc_recommender.config import KNN_K, RANDOM_STATE, TEST_SIZE


class CollaborativeFilteringRecommender:
    """KNN item-based com similaridade de cosseno via scikit-surprise."""

    def __init__(self, k: int = KNN_K):
        self.k = k
        self.algo = None
        self.trainset = None
        self.testset = None
        self.rmse = None

    def fit(self, ratings: pd.DataFrame) -> "CollaborativeFilteringRecommender":
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
        self.trainset, self.testset = train_test_split(
            data, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        sim_options = {"name": "cosine", "user_based": False}
        self.algo = KNNBasic(k=self.k, sim_options=sim_options, verbose=True)
        self.algo.fit(self.trainset)
        return self

    def evaluate_rmse(self) -> float:
        predictions = self.algo.test(self.testset)
        self.rmse = accuracy.rmse(predictions)
        print(f"RMSE - Collaborative Filtering: {self.rmse:.4f}")
        return self.rmse
