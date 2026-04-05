import numpy as np
from typing import List, Dict
from .indexer import Indexer
from .utils import cosine_similarity

class Retriever:
    """
    Retriever class: given a query, returns the top-k most relevant chunks.
    Advanced feature: Hybrid Search using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def retrieve(self, query: str, top_k: int = 3, use_hybrid: bool = True, rrf_k: int = 60) -> List[Dict]:
        """
        Retrieves top_k chunks for the given query.
        If use_hybrid is True, combines Dense and BM25 rankings using RRF.

        RRF score: sum of 1 / (rrf_k + rank) across all ranking lists,
        where rank is 1-indexed (rank 1 = best).
        """
        if not self.indexer.chunks:
            return []

        # Dense scores
        query_embedding = self.indexer.model.encode(query, convert_to_numpy=True)
        dense_scores = cosine_similarity(query_embedding, self.indexer.embeddings)

        if use_hybrid and self.indexer.bm25 is not None:
            # BM25 scores
            tokenized_query = query.lower().split()
            bm25_scores = np.array(self.indexer.bm25.get_scores(tokenized_query))

            # Convert scores to 1-indexed ranks (rank 1 = highest score)
            dense_ranks = np.argsort(np.argsort(-dense_scores)) + 1
            bm25_ranks  = np.argsort(np.argsort(-bm25_scores))  + 1

            # Use corpus size as k when index is smaller than default k
            k = min(rrf_k, len(self.indexer.chunks))

            # Reciprocal Rank Fusion
            final_scores = 1.0 / (k + dense_ranks) + 1.0 / (k + bm25_ranks)
        else:
            final_scores = dense_scores

        top_indices = np.argsort(final_scores)[::-1][:top_k]

        return [{"chunk": self.indexer.chunks[i], "score": float(final_scores[i])} for i in top_indices]
