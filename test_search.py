"""
test_search.py  --  standalone test for the web search tool.

Run this on YOUR machine (it needs real internet):
    python test_search.py
    python test_search.py "your own query here"

It tells you clearly whether search works, and shows sample results. This is
isolated on purpose — we verify the search tool ALONE before wiring it into the
ML researcher agent.
"""
import sys
from tools.web_search import web_search, search_summary, SearchError


def main():
    query = " ".join(sys.argv[1:]) or "telecom customer churn prediction best machine learning model"
    print(f"Query: {query}")
    print("-" * 60)

    # 1. raw results
    try:
        results = web_search(query, max_results=5)
    except SearchError as e:
        print(f"SEARCH FAILED: {e}")
        print("\nIf this says 'no results' or a network error, check your internet/VPN.")
        print("If it says 'ddgs not installed', run: pip install ddgs")
        return

    print(f"OK — got {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.title}")
        print(f"   {r.snippet[:120]}")
        print(f"   {r.url}\n")

    # 2. the summary form the researcher agent will actually use
    print("-" * 60)
    print("Summary block (what gets fed to the LLM):\n")
    print(search_summary(query, max_results=3))

    print("\n" + "=" * 60)
    print("SEARCH TOOL: WORKING" if results else "SEARCH TOOL: NO RESULTS")


if __name__ == "__main__":
    main()
