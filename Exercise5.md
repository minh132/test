# Exercise 5 — Latency Profiling

The `/query` endpoint has a p99 latency of 4 seconds. Below are the three most likely bottlenecks specific to a RAG pipeline, how to diagnose each, and a concrete fix for each.

---



## Bottleneck 1 — Oversized context window slowing LLM generation

**What it is in RAG context:**
The retrieved chunks are concatenated into a single context string and passed to the LLM. When `top_k` is high or chunk sizes are large, the total input token count grows significantly. LLM generation latency scales with both input length (prefill cost) and output length (decoding cost) — a prompt with 3,000 tokens takes measurably longer to process than one with 500 tokens, and at p99 this compounds with model queue time on the provider side. The current implementation joins all retrieved chunks with no truncation, so a large corpus with high `top_k` can silently push prompts to thousands of tokens.

**Diagnose:**
Log the token count of each prompt before sending it to the LLM. Use the `tiktoken` library to count tokens accurately and track them as a metric in your observability stack. Correlate `prompt_tokens` against `latency_ms` in Grafana or Datadog — a clear linear relationship confirms the context size is the driver of slow requests.

**Fix:**
Reduce the effective context size by applying a re-ranker after retrieval. Instead of passing all `top_k` chunks to the LLM, use a cross-encoder model (e.g. `jinaai/jina-reranker-v2-base-multilingual`) to re-score the retrieved chunks and keep only the top 2–3 most relevant ones. This keeps prompt size small and consistent regardless of `top_k`, while improving answer quality because the chunks passed to the LLM are more precisely relevant than what cosine similarity alone selects.

---

## Bottleneck 2 — Cosine similarity search implemented from scratch

**What it is in RAG context:**
The `cosine_similarity` function computes a dot product over the entire embedding matrix on every query. With `N` indexed chunks of dimension `D`, this is a dense O(N × D) floating-point operation running in NumPy on a single thread. At small scale this is fast, but as the index grows after many `/index` calls, compute time scales linearly — a corpus of 50,000 chunks with 384-dimensional embeddings requires tens of milliseconds per query on CPU.

**Diagnose:**
Profile `retrieve()` with `py-spy` attached to the live uvicorn process without restarting it and generate a flame graph. Wide bars in `cosine_similarity` or `np.dot` confirm the bottleneck. Alternatively, add per-stage timing inside `retrieve()` to log `dense_ms` and `bm25_ms` separately and observe how they grow as the index size increases.

**Fix:**
Replace the in-process NumPy index with a dedicated vector database such as **Qdrant** or **Milvus**. Both store embeddings on disk, build an HNSW index for ANN search, and expose a query API — offloading the O(N × D) work entirely out of the Python process. `Qdrant` runs as a single Docker container and has a Python client that maps directly onto the current `retrieve()` interface; `Milvus` suits larger deployments with distributed scaling. At 10k+ chunks either option reduces dense search latency from tens of milliseconds to low single-digit milliseconds, and the index persists across service restarts without re-embedding.

---

## Bottleneck 3 — BM25 scoring running in pure Python on every query

**What it is in RAG context:**
The hybrid retriever calls `BM25Okapi.get_scores()` from the `rank_bm25` library on every `/query` request. Internally this iterates over every document in the tokenized corpus in CPython, computing term-frequency lookups in a Python loop with no SIMD or vectorisation. The time complexity is O(corpus_size × query_tokens) in interpreted Python — at 10k+ chunks with a 10-token query this can take 50–200ms, making BM25 the dominant retrieval cost and pushing p99 well past the dense similarity step.

**Diagnose:**
Add per-stage timing inside `retrieve()` and log `dense_ms` and `bm25_ms` separately:

```python
t0 = time.perf_counter()
bm25_scores = np.array(self.indexer.bm25.get_scores(tokenized_query))
bm25_ms = (time.perf_counter() - t0) * 1000
```

If `bm25_ms` grows linearly with corpus size and exceeds `dense_ms` at scale, `rank_bm25` is the bottleneck.

**Fix:**
Use a vector database that natively supports hybrid search — **Qdrant** and **Milvus** both combine dense ANN search with sparse vector scoring (BM25/SPLADE) in a single query, eliminating the need to run BM25 in-process entirely. Qdrant's sparse vector support accepts a pre-tokenized query and pre-built sparse index, then executes the scoring in optimised Rust; Milvus offers equivalent functionality via its sparse float vector field type. This consolidates Bottleneck 2 and Bottleneck 3 into a single infrastructure fix: one vector DB replaces both the NumPy cosine search and the `rank_bm25` scoring loop.
