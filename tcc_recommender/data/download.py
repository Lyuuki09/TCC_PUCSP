import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from tcc_recommender.config import DATA_DIR, MOVIELENS_URL, PROJECT_ROOT


def download_movielens(dest_dir: Path | None = None) -> Path:
    """Baixa e extrai o dataset MovieLens 100K se ainda não existir."""
    dest_dir = dest_dir or DATA_DIR
    data_file = dest_dir / "u.data"

    if data_file.exists():
        print(f"Dataset já disponível em {dest_dir}")
        return dest_dir

    zip_path = PROJECT_ROOT / "ml-100k.zip"
    print("Baixando MovieLens 100K...")
    urlretrieve(MOVIELENS_URL, zip_path)

    print("Extraindo arquivos...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(PROJECT_ROOT)

    return dest_dir
