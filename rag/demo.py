import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
from rag_core import Indexer, Retriever, RAGPipeline, RecursiveChunker, PDFIngester

load_dotenv()


async def main():
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")

    recursive_chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)
    ingester = PDFIngester()
    pdf_chunks = ingester.ingest_and_chunk(assets_dir, recursive_chunker)
    print(f"Total chunks from PDFs: {len(pdf_chunks)}")
    if pdf_chunks:
        print(f"Sample chunk: {pdf_chunks[0][:120]!r}\n")

    indexer = Indexer()
    indexer.index(pdf_chunks)

    retriever = Retriever(indexer)
    rag = RAGPipeline(retriever)

    questions = [
        "What are the four stages of a typical RAG pipeline, and which stage is considered the weakest link and why?",
        "How does the CAP theorem impact AI pipelines built on eventually consistent storage, and what practical SLA should teams define to manage index staleness?",
        "What is the 'embedding space conflation problem' in multi-hop reasoning, and what retrieval strategies are proposed to overcome it?",
        "Why is using a single automated metric insufficient for evaluating RAG systems, and what does RAGAS offer as an alternative?",
        "What new failure modes does Agentic RAG introduce compared to traditional RAG, and what safeguards are recommended to address them?",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\nQuestion {i}: {q}")
        chunks = retriever.retrieve(q, top_k=3)
        for j, c in enumerate(chunks, 1):
            print(f"  Chunk {j} (score={c['score']:.3f}): {c['chunk'][:200]!r}")
        ans = await rag.generate_answer(q, top_k=3)
        print(f"Answer {i}: {ans}")

    print("\nTest completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
