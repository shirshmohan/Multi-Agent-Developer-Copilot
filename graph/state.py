"""
graph/state.py
The shared State that travels through every node in the graph.

This is the single most important object in a LangGraph system. Each node reads
it and returns updates to it. LangGraph merges those updates and passes the new
state to the next node. Everything the system "remembers" mid-request lives here.
"""
from typing import TypedDict, Any


class AppState(TypedDict, total=False):
    # --- input ---
    request: str                 # the user's natural-language request

    # --- supervisor's decision ---
    route: str                   # which worker the supervisor chose: 'sql' | 'done' (more later)
    reason: str                  # WHY it routed there — inspectable, per the explicit-routing lesson

    # --- worker outputs (each worker writes its slice) ---
    sql_result: dict[str, Any]   # the SQLResult, as a dict
    data_result: dict[str, Any]  # the Data agent's result (cells, summary, notebook path)

    # --- final ---
    answer: str                  # the composed answer (the synthesizer will fill this)

    # --- QA / verification ---
    qa_ok: bool                  # did the last result pass verification?
    qa_feedback: str             # specific problems found, fed back to the agent on retry
    qa_retries: int              # how many regeneration attempts so far
    qa_exhausted: bool           # ran out of retries; result accepted but flagged unverified

    # --- control / safety ---
    steps: int                   # circuit breaker: how many supervisor hops so far
