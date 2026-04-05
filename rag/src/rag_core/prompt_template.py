RAG_SYSTEM_PROMPT = "You are a helpful assistant."

RAG_USER_PROMPT = """\
Use the following context to answer the question. \
If the answer is not in the context, say "I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:"""
