import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from rank_bm25 import BM25Okapi

from .schemas import IndexerConfig


class Indexer:
    def __init__(self, config: IndexerConfig = None):
        cfg = config or IndexerConfig()
        self.config = cfg
        self.model = SentenceTransformer(cfg.model_name, device=cfg.device)
        if cfg.precision == "bfloat16":
            self.model = self.model.half()
        self.chunks: List[str] = []
        self.embeddings: np.ndarray = np.array([])
        self.bm25: BM25Okapi = None

    def index(self, chunks: List[str]):
        if not chunks:
            return

        batches = [chunks[i:i + self.config.batch_size] for i in range(0, len(chunks), self.config.batch_size)]
        new_embeddings = np.vstack([
            self.model.encode(batch, convert_to_numpy=True) for batch in batches
        ])

        if self.embeddings.size == 0:
            self.embeddings = new_embeddings
            self.chunks = list(chunks)
        else:
            self.embeddings = np.vstack((self.embeddings, new_embeddings))
            self.chunks.extend(chunks)

        tokenized_corpus = [doc.lower().split() for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
