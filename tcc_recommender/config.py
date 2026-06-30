from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "ml-100k"
MOVIELENS_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"

BERT_MODEL = "neuralmind/bert-base-portuguese-cased"
KNN_K = 50
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 64
MAX_LENGTH = 128
PRECISION_K = 5
RATING_THRESHOLD = 4.0
