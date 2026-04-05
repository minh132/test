from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class IndexerConfig:
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    precision: str = "float32"
    batch_size: int = 32


@dataclass
class RetrieverConfig:
    top_k: int = 3
    use_hybrid: bool = True
    rrf_k: int = 60
    block_size: int = 64


@dataclass
class PipelineConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    timeout: float = 30.0


@dataclass
class RetrievedChunk:
    chunk: str
    score: float


@dataclass
class RAGResponse:
    answer: str
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
