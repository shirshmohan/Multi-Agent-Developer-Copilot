"""
graph/qa.py
The QA / verification node. Catches results that are obviously broken BEFORE the
supervisor accepts them. This is the net for errors grounding can't prevent
(like a wrong rate formula).

DESIGN: mostly deterministic checks (cheap, can't be jailbroken), matching the
Phase 9 philosophy. We only ask the LLM to re-examine when a deterministic check
trips — we don't pay for an LLM call on every healthy result.
"""
from graph.state import AppState

MAX_RETRIES = 1   # how many times we'll bounce a bad result back for regeneration


def _problems(sql_result: dict) -> list[str]:
    """Deterministic sanity checks. Returns a list of problems (empty = looks fine)."""
    problems = []

    if sql_result.get("error"):
        problems.append(f"query errored: {sql_result['error']}")
        return problems

    rows = sql_result.get("rows")
    if rows is None:
        problems.append("no rows object returned")
        return problems

    if len(rows) == 0:
        problems.append("query returned zero rows — likely a filter/casing mismatch")

    # check for all-NULL single-value results (the exact bug we just hit)
    if rows and len(rows) == 1:
        only = rows[0]
        if all(v is None for v in only.values()):
            problems.append("the single result row is entirely NULL — formula or filter is wrong")

    # sanity-check anything that looks like a rate/percentage
    for row in rows[:5]:
        for key, val in row.items():
            if val is None:
                continue
            k = key.lower()
            if ("rate" in k or "pct" in k or "percent" in k) and isinstance(val, (int, float)):
                if "pct" in k or "percent" in k:
                    if not (0 <= float(val) <= 100):
                        problems.append(f"{key}={val} out of 0-100 range for a percentage")
                else:  # a 'rate' should usually be a proportion 0..1
                    if not (0 <= float(val) <= 1.0001):
                        problems.append(f"{key}={val} out of 0-1 range for a rate")

    return problems


def qa_node(state: AppState) -> dict:
    """Inspect the latest SQL result. If it's broken and we have retries left,
    set a flag + feedback so the supervisor re-routes to sql with guidance."""
    sql_result = state.get("sql_result", {})
    problems = _problems(sql_result)
    retries = state.get("qa_retries", 0)

    if not problems:
        return {"qa_ok": True, "qa_feedback": ""}

    if retries >= MAX_RETRIES:
        # out of retries — accept it but mark it as unverified so the answer is honest
        return {"qa_ok": False, "qa_feedback": "; ".join(problems),
                "qa_exhausted": True}

    # ask for a retry, passing the specific problems as feedback to the SQL agent
    return {"qa_ok": False, "qa_feedback": "; ".join(problems),
            "qa_retries": retries + 1}
