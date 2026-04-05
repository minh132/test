"""Web search tool using Tavily API."""

import os
from tavily import TavilyClient


def web_search(query: str) -> str:
    """Perform a web search using Tavily API and return combined snippets."""
    try:
        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        if not tavily_api_key:
            return "Error: TAVILY_API_KEY not set."

        client = TavilyClient(api_key=tavily_api_key)
        response = client.search(query, search_depth="basic", max_results=3)

        context = " ".join(
            [result["content"] for result in response.get("results", [])]
        )
        return context if context else "No relevant information found."
    except Exception as e:
        return f"Error performing web search: {e}"
