"""
User-facing gateway API.

Accepts requests from clients and proxies them to the internal RAG core service.
Set RAG_CORE_URL to point at the rag_core_api service (default: http://localhost:8001).
"""

import os
import logging
from typing import List, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from pythonjsonlogger import jsonlogger
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("rag-gateway")
_handler = logging.StreamHandler()
_handler.setFormatter(jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)

RAG_CORE_URL = os.environ.get("RAG_CORE_URL", "http://localhost:8001")

app = FastAPI(title="RAG Gateway API")


class IndexRequest(BaseModel):
    chunks: List[str] = Field(..., min_length=1)


class IndexResponse(BaseModel):
    status: str
    doc_count: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1)


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: List[Dict[str, Any]]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    indexed_documents: int


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"message": "Invalid input parameters", "details": exc.errors()},
    )


async def _post(path: str, payload: dict) -> dict:
    async with httpx.AsyncClient(base_url=RAG_CORE_URL, timeout=60.0) as client:
        resp = await client.post(path, json=payload)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Core service error: {resp.text}")
        return resp.json()


@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest):
    logger.info("index request", extra={"chunk_count": len(request.chunks)})
    data = await _post("/index", {"chunks": request.chunks})
    return IndexResponse(**data)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    logger.info("query request", extra={"question": request.question})
    retrieved = await _post("/retrieve", {"query": request.question, "top_k": request.top_k})
    generated = await _post("/generate", {"query": request.question, "chunks": retrieved["chunks"]})
    return QueryResponse(
        answer=generated["answer"],
        retrieved_chunks=retrieved["chunks"],
        latency_ms=generated["latency_ms"],
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    async with httpx.AsyncClient(base_url=RAG_CORE_URL, timeout=5.0) as client:
        try:
            resp = await client.get("/health")
            data = resp.json()
        except Exception:
            raise HTTPException(status_code=503, detail="Core service unavailable.")
    return HealthResponse(**data)
