"""
A free, no-API-key-needed web search tool using DuckDuckGo.

Note: the old `duckduckgo_search` PyPI package was renamed to `ddgs`.
We use the new `ddgs` package here.
"""

from crewai.tools import tool
from ddgs import DDGS


@tool("DuckDuckGo Search Tool")
def duckduckgo_search_tool(query: str) -> str:
    """
    Searches the web using DuckDuckGo for a given query and returns the
    top results (title, short summary, and source link).

    Use this tool whenever you need current facts, news, statistics, or
    any information you are not 100% sure about. Call it multiple times
    with different, specific queries to cover a topic well.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as e:
        return f"Search failed for query '{query}': {e}"

    if not results:
        return f"No results found for query: '{query}'."

    formatted_results = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "No title")
        body = r.get("body", "")
        link = r.get("href", "")
        formatted_results.append(f"{i}. {title}\n   {body}\n   Source: {link}")

    return "\n\n".join(formatted_results)
