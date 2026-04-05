import os
import openai
from typing import List, Dict, Optional
from .retriever import Retriever
from .schemas import PipelineConfig, RAGResponse, RetrievedChunk
from .prompt_template import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT


class RAGPipeline:
    def __init__(self, retriever: Retriever, config: PipelineConfig = None, openai_api_key: str = None):
        self.retriever = retriever
        self.config = config or PipelineConfig()
        key = openai_api_key or os.environ.get("OPENAI_API_KEY", "dummy-key-for-testing")
        self.client = openai.AsyncOpenAI(api_key=key)

    async def generate_answer(
        self,
        query: str,
        top_k: int = 3,
        use_hybrid: bool = True,
        retrieved_items: Optional[List[Dict]] = None,
    ) -> str:
        if retrieved_items is None:
            retrieved_items = self.retriever.retrieve(query, top_k=top_k, use_hybrid=use_hybrid)

        if not retrieved_items:
            return "I don't have enough context to answer that."

        context = "\n---\n".join([item["chunk"] for item in retrieved_items])
        prompt = RAG_USER_PROMPT.format(context=context, question=query)

        kwargs = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "timeout": self.config.timeout,
        }
        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens

        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
