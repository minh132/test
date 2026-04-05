import numpy as np
from typing import List, Dict
from .indexer import Indexer
from .schemas import RetrieverConfig, RetrievedChunk
from .utils import cosine_similarity


class Retriever:
    def __init__(self, indexer: Indexer, config: RetrieverConfig = None):
        self.indexer = indexer
        self.config = config or RetrieverConfig()

    def retrieve(self, query: str, top_k: int = None, use_hybrid: bool = None, rrf_k: int = None) -> List[Dict]:
        top_k = top_k if top_k is not None else self.config.top_k
        use_hybrid = use_hybrid if use_hybrid is not None else self.config.use_hybrid
        rrf_k = rrf_k if rrf_k is not None else self.config.rrf_k
        block_size = self.config.block_size

        if not self.indexer.chunks:
            return []

        query_embedding = self.indexer.model.encode(query, convert_to_numpy=True)

        # Compute cosine similarity in blocks to avoid OOM on large corpora
        n = len(self.indexer.embeddings)
        dense_scores = np.empty(n, dtype=np.float32)
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            dense_scores[start:end] = cosine_similarity(query_embedding, self.indexer.embeddings[start:end])

        if use_hybrid and self.indexer.bm25 is not None:
            tokenized_query = query.lower().split()
            bm25_scores = np.array(self.indexer.bm25.get_scores(tokenized_query))

            dense_ranks = np.argsort(np.argsort(-dense_scores)) + 1
            bm25_ranks = np.argsort(np.argsort(-bm25_scores)) + 1

            k = min(rrf_k, len(self.indexer.chunks))
            final_scores = 1.0 / (k + dense_ranks) + 1.0 / (k + bm25_ranks)
        else:
            final_scores = dense_scores

        top_indices = np.argsort(final_scores)[::-1][:top_k]
        return [{"chunk": self.indexer.chunks[i], "score": float(final_scores[i])} for i in top_indices]
