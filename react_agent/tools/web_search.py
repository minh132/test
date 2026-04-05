SEARCH_DB = {
    "gdp france 2023": "France's GDP in 2023 was approximately $2.78 trillion USD (World Bank estimate).",
    "france gdp 2023": "France's GDP in 2023 was approximately $2.78 trillion USD (World Bank estimate).",
    "gdp of france": "France's GDP in 2023 was approximately $2.78 trillion USD (World Bank estimate).",
    "france 2023 gdp usd": "France's GDP in 2023 was approximately $2.78 trillion USD (World Bank estimate).",
    "timsort worst case": "Timsort's worst-case time complexity is O(n log n).",
    "python sort complexity": "Python's built-in sort uses Timsort. Worst-case time complexity is O(n log n).",
}


def web_search(query: str) -> str:
    key = query.lower().strip()
    for k, v in SEARCH_DB.items():
        if k in key or key in k:
            return v
    return f"No results found for: {query}"
