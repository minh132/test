"""Tool registry: OpenAI function calling schemas and dispatch map."""

from .web_search import web_search
from .calculator import calculator

# OpenAI function calling schemas
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for real-time information. "
                "Use this to look up facts, statistics, current data, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a mathematical expression and return the numeric result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

# Maps function name → callable(args_dict) → str
TOOL_DISPATCH = {
    "web_search": lambda args: web_search(args.get("query", "")),
    "calculator": lambda args: calculator(args.get("expression", "")),
}
