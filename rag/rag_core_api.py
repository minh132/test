"""
Internal RAG core service.

Handles indexing, retrieval, and answer generation.
The user-facing gateway (api.py) routes requests here.
"""

import time
import logging
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pythonjsonlogger import jsonlogger

load_dotenv()

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rag_core import Indexer, Retriever, RAGPipeline

logger = logging.getLogger("rag-core-api")
_handler = logging.StreamHandler()
_handler.setFormatter(jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)

app = FastAPI(title="RAG Core API")

_indexer = Indexer()
_retriever = Retriever(_indexer)
_pipeline = RAGPipeline(_retriever)


class IndexRequest(BaseModel):
    chunks: List[str] = Field(..., min_length=1)


class IndexResponse(BaseModel):
    status: str
    doc_count: int


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1)
    use_hybrid: bool = True


class RetrieveResponse(BaseModel):
    chunks: List[Dict[str, Any]]


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1)
    chunks: List[Dict[str, Any]]


class GenerateResponse(BaseModel):
    answer: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    indexed_documents: int


@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest):
    logger.info("index request", extra={"chunk_count": len(request.chunks)})
    _indexer.index(request.chunks)
    return IndexResponse(status="success", doc_count=len(_indexer.chunks))


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_chunks(request: RetrieveRequest):
    chunks = _retriever.retrieve(request.query, top_k=request.top_k, use_hybrid=request.use_hybrid)
    return RetrieveResponse(chunks=chunks)


@app.post("/generate", response_model=GenerateResponse)
async def generate_answer(request: GenerateRequest):
    start = time.time()
    try:
        answer = await _pipeline.generate_answer(request.query, retrieved_items=request.chunks)
    except Exception:
        logger.error("generation failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Generation failed.")
    return GenerateResponse(answer=answer, latency_ms=(time.time() - start) * 1000)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", indexed_documents=len(_indexer.chunks))
