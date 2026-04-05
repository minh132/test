# Exercise 1 & 4 — Mini RAG Pipeline + Containerised Inference API

Exercise 1 builds the core RAG pipeline. Exercise 4 wraps it in a production-ready FastAPI service with Docker support.

---

## Exercise 1 — Mini RAG Pipeline

A minimal RAG system built with `sentence-transformers`, `numpy`, `rank_bm25`, and `openai`.

### Project layout

```
rag/
├── src/
│   └── rag_core/
│       ├── chunker.py
│       ├── indexer.py
│       ├── ingestion.py
│       ├── pipeline.py
│       ├── prompt_template.py
│       ├── retriever.py
│       ├── schemas.py
│       └── utils.py
├── tests/
│   └── conftest.py
├── assets/           ← PDF documents
├── api.py            ← user-facing gateway
├── rag_core_api.py   ← internal RAG core service
├── demo.py
├── Dockerfile
├── Dockerfile-CUDA
└── pyproject.toml
```

### Architecture

| Class | Responsibility |
|---|---|
| `Indexer` | Embeds text chunks with `sentence-transformers` and stores them in a NumPy array. Also builds a BM25 index. Configurable via `IndexerConfig` (device, precision, quantize_embeddings). |
| `Retriever` | Returns the top-k most relevant chunks using **hybrid BM25 + dense cosine similarity**. Batches cosine similarity in `block_size=64` blocks to avoid OOM. Configurable via `RetrieverConfig`. |
| `RAGPipeline` | Chains retriever → prompt construction → OpenAI LLM call → answer. Configurable via `PipelineConfig` (temperature, top_p, max_tokens, model). |
| `RecursiveChunker` | Splits raw text using `RecursiveCharacterTextSplitter`. |
| `PDFIngester` | Extracts text from PDF files using PyMuPDF. |

### Advanced Retrieval Feature — Hybrid BM25 + Dense Search

#### Why BM25?

Dense embedding retrieval excels at semantic similarity but struggles with exact keyword matches. BM25 is a classical sparse retrieval function that scores documents on term frequency and inverse document frequency, strong where dense retrieval is weak.

#### Fusion — Reciprocal Rank Fusion (RRF)

```
RRF_score(d) = 1 / (k + rank_dense(d)) + 1 / (k + rank_bm25(d))
```

`k = 60` (smoothing constant), ranks are 1-indexed. RRF operates on ranks rather than raw scores, avoiding the normalisation sensitivity of weighted sum approaches.

The theoretical maximum score occurs when a document ranks 1st in both lists:

```
max = 1/(60+1) + 1/(60+1) ≈ 0.033
```

So a score of ~0.033 means the chunk topped both the dense and BM25 rankings — it is the best possible result, not a low one. The narrow range `(0, 0.033]` is intentional: RRF trades absolute magnitude for stability across queries and corpus sizes.

#### Block-wise cosine similarity

The retriever computes cosine similarity in blocks of `block_size=64` rows to avoid allocating the full `[n_chunks × embed_dim]` intermediate matrix, preventing OOM errors on CPU with large corpora.

### Setup

```bash
cd rag/
pip install .
# or with uv:
uv sync
```

Create a `.env` file in `rag/`:

```
OPENAI_API_KEY=sk-...
```

### Run the demo

```bash
cd rag/
python demo.py
```

Ingests 5 PDFs from `assets/`, indexes all chunks, then answers 5 domain-specific questions.

### Sample output

```
Total chunks from PDFs: 71
Sample chunk: 'Long RAG Document 1 — AI System Tradeoffs\n& Retrieval-Augmented Generation\nSection 1: Tradeoffs in Modern AI Systems\nMod'

Question 1: What are the four stages of a typical RAG pipeline, and which stage is considered the weakest link and why?
  Chunk 1 (score=0.033): 'generation on that context.
A typical RAG pipeline consists of four stages: (1) document ingestion and chunking, (2)
embedding computation and index construction, (3) query-time retrieval, and (4)
con...'
  Chunk 2 (score=0.030): 'Long RAG Document 2 — RAG Quality Factors
& Distributed System Challenges
Section 1: Quality Factors in RAG Systems...'
  Chunk 3 (score=0.030): 'between chunk granularity and query intent — each requiring a different fix.
Section 4: Practical Deployment Considerations...'
Answer 1: The four stages of a typical RAG pipeline are: (1) document ingestion and chunking,
(2) embedding computation and index construction, (3) query-time retrieval, and (4)
context-conditioned generation. The context does not specify which stage is the weakest link.

Question 2: How does the CAP theorem impact AI pipelines built on eventually consistent storage, and what practical SLA should teams define to manage index staleness?
  Chunk 1 (score=0.033): 'Distributed systems underpin modern AI infrastructure: training clusters, feature stores, model
serving fleets, and document indexes all operate as distributed services. Each introduces
consistency tr...'
  Chunk 2 (score=0.032): 'may compute gradients based on model parameters that have already been updated by other
workers, potentially slowing convergence or causing instability at large batch counts...'
  Chunk 3 (score=0.030): 'Read-your-writes consistency is a common middle ground: a client is guaranteed to see its
own writes reflected immediately, while other clients may still observe stale data...'
Answer 2: The CAP theorem implies that distributed systems cannot simultaneously guarantee
consistency, availability, and partition tolerance. To manage index staleness, teams should
define a maximum acceptable staleness window — e.g. ≤ 1 hour for news corpora and
≤ 24 hours for reference documentation.

Test completed successfully!
```

---

## Exercise 4 — Containerised Inference API

Two-service FastAPI architecture:

| Service | File | Port | Responsibility |
|---|---|---|---|
| RAG Core | `rag_core_api.py` | 8001 | Indexing, retrieval, generation |
| Gateway | `api.py` | 8000 | Receives user requests, calls core |

### Endpoints

#### RAG Core API (port 8001)

| Method | Path | Description |
|---|---|---|
| `POST` | `/index` | Index a list of text chunks |
| `POST` | `/retrieve` | Retrieve top-k chunks for a query |
| `POST` | `/generate` | Generate answer from pre-retrieved chunks |
| `GET` | `/health` | Service status |

#### Gateway API (port 8000)

| Method | Path | Description |
|---|---|---|
| `POST` | `/index` | Proxy to core `/index` |
| `POST` | `/query` | Retrieve + generate via core, return combined response |
| `GET` | `/health` | Proxy to core `/health` |

### Run locally

```bash
cd rag/
# Terminal 1 — core service
uvicorn rag_core_api:app --host 0.0.0.0 --port 8001

# Terminal 2 — gateway
RAG_CORE_URL=http://localhost:8001 uvicorn api:app --host 0.0.0.0 --port 8000
```

### Run with Docker (CPU)

```bash
cd rag/
docker build -t rag-core .
docker run -p 8001:8001 --env OPENAI_API_KEY=sk-... rag-core
```

### Run with Docker (CUDA/GPU)

```bash
cd rag/
docker build -f Dockerfile-CUDA -t rag-core-cuda .
docker run --gpus all -p 8001:8001 --env OPENAI_API_KEY=sk-... rag-core-cuda
```

### curl examples

#### POST /index (gateway)

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"chunks": ["The Eiffel Tower is in Paris.", "Python was created in 1991."]}'
```

```json
{"status": "success", "doc_count": 2}
```

#### POST /query (gateway)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Where is the Eiffel Tower?", "top_k": 2}'
```

```json
{
  "answer": "The Eiffel Tower is located in Paris.",
  "retrieved_chunks": [{"chunk": "The Eiffel Tower is in Paris.", "score": 0.923}],
  "latency_ms": 312.5
}
```
