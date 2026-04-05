import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from rank_bm25 import BM25Okapi

class Indexer:
    """
    Indexer class: accepts a list of text chunks, embeds them, and stores embeddings in memory.
    Also builds a BM25 index for hybrid search.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks: List[str] = []
        self.embeddings: np.ndarray = np.array([])
        self.bm25: BM25Okapi = None

    def index(self, chunks: List[str], batch_size: int = 32):
        """Embeds text chunks in batches, stores them, and updates BM25 index."""
        if not chunks:
            return

        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        new_embeddings = np.vstack([
            self.model.encode(batch, convert_to_numpy=True) for batch in batches
        ])

        if self.embeddings.size == 0:
            self.embeddings = new_embeddings
            self.chunks = list(chunks)
        else:
            self.embeddings = np.vstack((self.embeddings, new_embeddings))
            self.chunks.extend(chunks)
            
        # Rebuild BM25 index with all chunks
        tokenized_corpus = [doc.lower().split() for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
