"""Atualiza o notebook para usar os módulos Python do projeto."""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "TCC-PUC.ipynb"

CODE_REPLACEMENTS = {
    "# Configuração do ambiente\nimport pandas as pd\nimport numpy as np\nimport torch\nfrom surprise import Dataset, Reader, KNNBasic\nfrom surprise.model_selection import train_test_split\nfrom surprise import accuracy\nfrom transformers import AutoTokenizer, AutoModel\nfrom sklearn.metrics.pairwise import cosine_similarity\n\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # prcura uma placa de video, se n tem, vai CPU normalmente\nprint(f\"Dispositivo utilizado: {device}\")": (
        "from tcc_recommender.models import BertEmbedder\n\n"
        "embedder = BertEmbedder()"
    ),
    '# Downkiad da base de Filmes\n\n!powershell -Command "Invoke-WebRequest -Uri \'http://files.grouplens.org/datasets/movielens/ml-100k.zip\' -OutFile \'ml-100k.zip\'"\n!powershell -Command "Expand-Archive -Path \'ml-100k.zip\' -DestinationPath \'.\' -Force"': (
        "from tcc_recommender.data import download_movielens\n\n"
        "download_movielens()"
    ),
}

# Cell 8 starts with load - replace entire cell if it contains lugares_puc
LOAD_CELL_MARKER = "# --- CARREGAMENTO TOTAL DA BASE ---"
LOAD_CELL_REPLACEMENT = """from tcc_recommender.data import load_movielens, load_puc_places

ratings, movies = load_movielens()
df_puc_places = load_puc_places()"""

CF_CELL_MARKER = "# Preparação dos dados para Surprise"
CF_CELL_REPLACEMENT = """from tcc_recommender.models import CollaborativeFilteringRecommender

cf = CollaborativeFilteringRecommender()
cf.fit(ratings)
rmse_knn = cf.evaluate_rmse()
testset = cf.testset"""

BERT_CELL_MARKER = "# Atualização para BERTimbau"
BERT_CELL_REPLACEMENT = "# Modelo carregado na célula de configuração (BertEmbedder)\n# encode() e encode_batch() disponíveis via embedder"

EMBED_CELL_MARKER = "# Geração de embeddings"
EMBED_CELL_REPLACEMENT = """from tcc_recommender.data import prepare_movie_texts

movie_texts = prepare_movie_texts(movies)
print(f"Processando embeddings para {len(movie_texts)} itens...")
movie_embeddings = embedder.encode_batch(movie_texts)
print(f"Dimensão final dos embeddings: {movie_embeddings.shape}")"""

BATCH_CELL_MARKER = "def get_embeddings_batch"
BATCH_CELL_REPLACEMENT = "# Processamento em batch disponível em embedder.encode_batch()"

METRICS_CELL_MARKER = "def precision_at_k_zero_shot"
METRICS_CELL_REPLACEMENT = """from tcc_recommender.evaluation import precision_at_k_zero_shot
from tcc_recommender.config import PRECISION_K

prec_zero = precision_at_k_zero_shot(testset, movie_embeddings, movies, k=PRECISION_K)
print(f"Precision@5 (Zero-Shot): {prec_zero:.4f}")
print(f"RMSE (CF): {rmse_knn:.4f}")"""

PLOT_CELL_MARKER = "## 4.12 Visualização de Desempenho"
PLOT_CELL_REPLACEMENT = """from tcc_recommender.visualization import plot_metrics_comparison

plot_metrics_comparison(rmse_knn, prec_zero)"""

DEMO_CELL_MARKER = 'lugares = [\n    {"nome": "Café Arte & Som"'
DEMO_CELL_REPLACEMENT = """from tcc_recommender.data.places_puc import DEMO_LUGARES
from tcc_recommender.recommendation import recommend_by_preference

recommend_by_preference(
    DEMO_LUGARES,
    "Bares descontraídos, música ao vivo e ambiente para conhecer gente nova",
    embedder,
)"""

AB_CELL_MARKER = "## 4.11 Simulador de Teste A/B"
AB_CELL_REPLACEMENT = """from tcc_recommender.evaluation import simulate_ab_test

simulate_ab_test(ratings, user_id=1)"""


def replace_cell_source(source: str) -> str | None:
    for old, new in CODE_REPLACEMENTS.items():
        if old in source:
            return new

    markers = [
        (LOAD_CELL_MARKER, LOAD_CELL_REPLACEMENT),
        (CF_CELL_MARKER, CF_CELL_REPLACEMENT),
        (BERT_CELL_MARKER, BERT_CELL_REPLACEMENT),
        (EMBED_CELL_MARKER, EMBED_CELL_REPLACEMENT),
        (BATCH_CELL_MARKER, BATCH_CELL_REPLACEMENT),
        (METRICS_CELL_MARKER, METRICS_CELL_REPLACEMENT),
        (PLOT_CELL_MARKER, PLOT_CELL_REPLACEMENT),
        (DEMO_CELL_MARKER, DEMO_CELL_REPLACEMENT),
        (AB_CELL_MARKER, AB_CELL_REPLACEMENT),
    ]
    for marker, replacement in markers:
        if marker in source:
            return replacement
    return None


def slim_markdown(source: str) -> str | None:
    pointer = "> Documentação completa desta etapa: [README.md](README.md)\n\n"
    long_sections = [
        "Neste capítulo é apresentado",
        "A escolha por uma arquitetura híbrida",
        "- **scikit-surprise**",
        "**Justificativa:** Utilizamos apenas 10%",
        "**Justificativa técnica:** Optamos pelo algoritmo",
        "**Justifica**tiva: O uso de mean pooling",
        "**Justificativa das métricas:**",
        "O sistema desenvolvido demonstra viabilidade",
    ]
    for section in long_sections:
        if section in source:
            title_line = source.split("\n")[0]
            return f"{title_line}\n\n{pointer}"
    return None


def main() -> None:
    with open(NOTEBOOK, encoding="utf-8") as f:
        nb = json.load(f)

    new_cells = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if src.strip().startswith("# tokenizer = AutoTokenizer"):
                continue
            replacement = replace_cell_source(src)
            if replacement is not None:
                cell["source"] = [replacement]
                cell["outputs"] = []
                cell["execution_count"] = None
        elif cell["cell_type"] == "markdown":
            src = "".join(cell["source"])
            if "<img src=" in src or "flowchart" in src.lower():
                cell["source"] = [
                    "## **4.9 Arquitetura do Sistema**\n\n"
                    "> Diagrama completo no [README.md](README.md#etapa-8--arquitetura-do-sistema).\n"
                ]
                cell["outputs"] = []
            else:
                slim = slim_markdown(src)
                if slim:
                    cell["source"] = [slim]
        new_cells.append(cell)

    nb["cells"] = new_cells

    with open(NOTEBOOK, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)

    print(f"Notebook atualizado: {NOTEBOOK}")


if __name__ == "__main__":
    main()
