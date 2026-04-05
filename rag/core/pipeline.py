import os
import openai
from .retriever import Retriever

class RAGPipeline:
    """
    RAGPipeline: chains retriever → prompt construction → async LLM call → answer.
    """
    def __init__(self, retriever: Retriever, openai_api_key: str = None):
        self.retriever = retriever
        key = openai_api_key or os.environ.get("OPENAI_API_KEY", "dummy-key-for-testing")
        self.client = openai.AsyncOpenAI(api_key=key)

    async def generate_answer(
        self,
        query: str,
        top_k: int = 3,
        use_hybrid: bool = True,
        retrieved_items=None,
    ) -> str:
        """Runs the RAG pipeline.

        If *retrieved_items* is provided, the retrieval step is skipped and the
        supplied chunks are used directly, avoiding a redundant embedding call.
        """
        if retrieved_items is None:
            retrieved_items = self.retriever.retrieve(query, top_k=top_k, use_hybrid=use_hybrid)

        if not retrieved_items:
            return "I don't have enough context to answer that."

        context = "\n---\n".join([item["chunk"] for item in retrieved_items])

        prompt = f"""You are a helpful assistant. Use the following context to answer the user's question. If the answer is not in the context, say "I don't know based on the provided context."

Context:
{context}

Question:
{query}

Answer:"""

        response = await self.client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            timeout=30.0,
        )

        return response.choices[0].message.content
