import hashlib
import sys
import os
from unittest.mock import patch

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RAG_DIR = os.path.abspath(os.path.join(_HERE, ".."))
_SRC_DIR = os.path.abspath(os.path.join(_HERE, "..", "src"))

for _p in (_RAG_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EMBED_DIM = 16


class _MockSentenceTransformer:
    def __init__(self, model_name="all-MiniLM-L6-v2", device="cpu"):
        pass

    def encode(self, input, convert_to_numpy=True, **kwargs):
        if isinstance(input, list):
            return np.stack([self._embed(t) for t in input]).astype("float32")
        return self._embed(input).astype("float32")

    @staticmethod
    def _embed(text: str) -> np.ndarray:
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.RandomState(seed)
        v = rng.rand(EMBED_DIM).astype("float32")
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v


_patcher = patch("sentence_transformers.SentenceTransformer", _MockSentenceTransformer)
_patcher.start()
