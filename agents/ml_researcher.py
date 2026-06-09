"""
agents/ml_researcher.py
The ML researcher. Turns LOOKING into ADVICE.

It does NOT touch the data. It takes information ABOUT the problem (the task, the
column names, and the EDA inferences the data agent produced) and combines two
sources of knowledge — what the DATA shows and what the FIELD recommends — into:
  - guidance: structured instructions the data agent can act on
  - theory:   a plain-English explanation of WHY, shown to the user

Design: a fixed 3-step pipeline (generate queries -> search -> synthesize), not a
free tool-calling loop. Fewer moving parts, far more reliable, easy to test.
Resilient: if web search is unavailable, it still advises from EDA + general ML
knowledge rather than failing.
"""
import json
from dataclasses import dataclass, field
from llm import get_llm
from tools.web_search import search_summary


@dataclass
class ResearchResult:
    guidance: str = ""                          # -> fed into the data agent's modeling phase
    theory: str = ""                            # -> shown to the user (terminal)
    queries: list[str] = field(default_factory=list)
    search_used: bool = False                   # did web search actually return anything?


QUERY_SYS = """You plan web research for a machine-learning problem.
Given the task, the columns, and the EDA findings, output 2-3 SHORT web search
queries (3-7 words each) that would find current best practices: which models work
well, how to handle the data's characteristics, and which metric to optimize.
Respond with ONLY a JSON array of query strings, nothing else."""

SYNTH_SYS = """You are a senior ML researcher advising a data scientist.
You are given: the task, the columns, the EDA findings, and snippets from web
research. Combine what the DATA shows with what the FIELD recommends.

Produce a JSON object with these keys:
- "models": list of 2-4 specific models to try (e.g. "LogisticRegression with class_weight='balanced'")
- "preprocessing": list of concrete preprocessing steps the data suggests
- "metric": the single most appropriate evaluation metric and WHY (one sentence)
- "optimizations": list of tuning/optimization steps (e.g. "RandomizedSearchCV on n_estimators, max_depth")
- "theory": a 3-5 sentence plain-English explanation a human can read, tying the
  data's characteristics to the recommended approach. Cite the reasoning, not URLs.

Respond with ONLY the JSON object, no preamble."""


def _llm_json(llm, system: str, user: str, fallback):
    """Call the LLM expecting JSON; parse it, with a safe fallback on any failure."""
    resp = llm.generate([{"role": "system", "content": system},
                          {"role": "user", "content": user}])
    text = (resp.text or "").strip()
    # strip markdown fences if the model added them
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip() if "```" in text else text
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return fallback


def run_ml_researcher(problem: str, columns: list[str], eda_summary: str,
                      provider: str | None = None) -> ResearchResult:
    """Research how best to model this problem. Returns guidance + theory."""
    llm = get_llm(provider)
    context = (f"Task: {problem}\n"
               f"Columns: {columns}\n"
               f"EDA findings:\n{eda_summary}")

    # STEP 1 — decide what to search for
    queries = _llm_json(llm, QUERY_SYS, context, fallback=[f"{problem} best model"])
    if not isinstance(queries, list) or not queries:
        queries = [f"{problem} best machine learning model"]
    queries = [str(q) for q in queries[:3]]

    # STEP 2 — run the searches (deterministic; degrades gracefully if unavailable)
    blocks, search_used = [], False
    for q in queries:
        summary = search_summary(q, max_results=4)
        if not summary.startswith("[search unavailable"):
            search_used = True
        blocks.append(f"Query: {q}\n{summary}")
    research_text = "\n\n".join(blocks)

    # STEP 3 — synthesize EDA + research into structured guidance
    synth = _llm_json(
        llm, SYNTH_SYS,
        f"{context}\n\nWeb research results:\n{research_text}",
        fallback={})

    # assemble the guidance block the data agent will receive
    g = []
    if synth.get("models"):
        g.append("Models to try: " + "; ".join(synth["models"]))
    if synth.get("preprocessing"):
        g.append("Preprocessing: " + "; ".join(synth["preprocessing"]))
    if synth.get("metric"):
        g.append("Primary metric: " + synth["metric"])
    if synth.get("optimizations"):
        g.append("Optimizations: " + "; ".join(synth["optimizations"]))
    guidance = "\n".join(g) if g else "No specific guidance produced; use standard practice."

    return ResearchResult(
        guidance=guidance,
        theory=synth.get("theory", "(no theory produced)"),
        queries=queries,
        search_used=search_used,
    )
