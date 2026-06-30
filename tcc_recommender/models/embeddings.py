import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from tcc_recommender.config import BATCH_SIZE, BERT_MODEL, MAX_LENGTH


class BertEmbedder:
    """Gera embeddings semânticos com BERTimbau (mean pooling)."""

    def __init__(self, model_name: str = BERT_MODEL, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        print(f"Dispositivo utilizado: {self.device}")

    def encode(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    def encode_batch(self, texts: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            all_emb.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
        return np.vstack(all_emb)
