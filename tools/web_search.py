"""
tools/web_search.py
A small web-search tool for the ML researcher agent.

Backend is swappable (same idea as the LLM provider abstraction): DuckDuckGo now
(free, no API key), Tavily later (better quality, needs a key) — callers don't care
which. Returns a compact list of {title, snippet, url}; never raw HTML pages.

NOTE: requires real internet. If you're behind a restricted network it will fail —
that's an environment issue, not a code bug.
"""
from dataclasses import dataclass


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str

    def as_line(self) -> str:
        """Compact one-liner for feeding into an LLM prompt. Title + a short, clean
        snippet — truncated so one giant/malformed blob can't dominate the block."""
        snip = " ".join(self.snippet.split())      # collapse newlines/runs of spaces
        if len(snip) > 200:
            snip = snip[:200] + "..."
        title = " ".join(self.title.split())[:120]
        return f"- {title}: {snip}\n  ({self.url})"


class SearchError(Exception):
    """Raised when search fails (network blocked, rate limited, no results)."""


def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Run a web search and return up to max_results compact results.
    Raises SearchError on failure so callers can degrade gracefully."""
    try:
        from ddgs import DDGS
    except ImportError as e:
        raise SearchError("ddgs not installed — run: pip install ddgs") from e

    try:
        with DDGS() as ddg:
            raw = list(ddg.text(query, max_results=max_results))
    except Exception as e:
        raise SearchError(f"search failed: {type(e).__name__}: {e}") from e

    if not raw:
        raise SearchError("no results returned")

    results = []
    for r in raw:
        results.append(SearchResult(
            title=r.get("title", "").strip(),
            snippet=r.get("body", "").strip(),     # ddgs calls the snippet 'body'
            url=r.get("href", "").strip(),
        ))
    return results


def search_summary(query: str, max_results: int = 5) -> str:
    """Convenience: run a search and return a single text block for an LLM prompt.
    Returns an honest 'no results' line instead of raising, so an agent can continue."""
    try:
        results = web_search(query, max_results=max_results)
    except SearchError as e:
        return f"[search unavailable: {e}]"
    return "\n".join(r.as_line() for r in results)
