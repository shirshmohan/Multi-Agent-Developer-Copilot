"""
graph/supervisor.py
The supervisor graph, built with the low-level StateGraph API so every routing
decision is visible (not hidden inside a create_supervisor helper).

Flow:
    START -> supervisor -> (conditional) -> sql -> supervisor -> ... -> END

The supervisor classifies the request, routes to a worker, the worker runs and
writes its result back to state, control returns to the supervisor, which decides
whether more work is needed or we're done. A step counter caps the loop.
"""
from langgraph.graph import StateGraph, START, END
from llm import get_llm
from agents.sql_agent import run_sql_agent
from agents.data_agent import run_data_agent
from tools.notebook_writer import write_notebook
from graph.state import AppState
from graph.qa import qa_node
from graph.synthesizer import synthesizer_node

MAX_STEPS = 5   # circuit breaker — the "most expensive mistake" guard. Never omit this.

# The supervisor decides a route via a tool call, giving us a clean structured choice.
ROUTER_TOOL = [{
    "type": "function",
    "function": {
        "name": "route",
        "description": "Choose which specialist should handle the request, or finish.",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "enum": ["sql", "data", "done"],   # grows as we add workers (mongo...)
                    "description": "'sql' for database questions; 'data' for analysis/EDA/ML on data; 'done' when fully answered.",
                },
                "reason": {"type": "string", "description": "one sentence: why this destination"},
            },
            "required": ["destination", "reason"],
        },
    },
}]

SUPERVISOR_SYS = """You are the supervisor of a telecom data team.
Look at the request and the work done so far, then route to the right specialist.

Specialists available:
- sql: answers questions about the telecom database (customers, churn, billing, etc.)
- data: does data science IN A NOTEBOOK — EDA, charts, preprocessing, training ML models.
        If a request needs data FROM the database first AND THEN analysis/modeling,
        route to 'sql' first; once the data is fetched, route to 'data'.

If the request has already been fully answered by prior work, route to 'done'.
Route to exactly one destination by calling the route tool."""


def supervisor_node(state: AppState) -> dict:
    """Classify + route. Writes 'route' and 'reason' to state. Increments the step counter."""
    steps = state.get("steps", 0) + 1
    if steps > MAX_STEPS:                              # circuit breaker fires
        return {"route": "done", "reason": "max steps reached", "steps": steps}

    llm = get_llm()
    # summarize what's been done — INCLUDING whether each step actually succeeded.
    # (Reporting only "ran" let the supervisor advance on a FAILED sql_result.)
    done = []
    if "sql_result" in state:
        sql = state["sql_result"]
        if sql.get("error"):
            done.append(f"SQL agent FAILED: {sql['error']}. The data was NOT fetched — "
                        f"route to 'sql' to try again, do not proceed to analysis.")
        else:
            n = len(sql.get("rows") or [])
            done.append(f"SQL agent succeeded: fetched {n} rows.")
    if "data_result" in state:
        dr = state["data_result"]
        if dr.get("error"):
            done.append(f"Data agent had trouble: {dr['error']}.")
        else:
            done.append("Data agent succeeded (analysis/notebook complete).")
    done_so_far = "; ".join(done) if done else "Nothing yet."
    messages = [
        {"role": "system", "content": SUPERVISOR_SYS},
        {"role": "user", "content": f"Request: {state['request']}\n\nWork so far: {done_so_far}"},
    ]
    resp = llm.generate(messages, tools=ROUTER_TOOL)
    if not resp.wants_tool:                            # model didn't route -> be safe, finish
        return {"route": "done", "reason": "supervisor gave no route", "steps": steps}

    args = resp.tool_calls[0].arguments
    return {"route": args["destination"], "reason": args.get("reason", ""), "steps": steps}


def sql_node(state: AppState) -> dict:
    """Wrap the existing SQL agent as a graph node. Writes its result into state.
    On a QA retry, passes the feedback so the agent knows what to fix."""
    feedback = state.get("qa_feedback") if not state.get("qa_ok", True) else None
    result = run_sql_agent(state["request"], feedback=feedback)
    return {"sql_result": result.__dict__}             # store the SQLResult as a dict


def _handoff_prelude(state: AppState) -> str | None:
    """SQL -> Data handoff that respects 'never move the data through the LLM'.
    Instead of serializing rows into the prelude, we pass the QUERY and let the
    data agent's kernel load the data DIRECTLY from Postgres via read_sql.
    The rows never touch any LLM context — only the query string does."""
    sql = state.get("sql_result")
    if not sql or sql.get("error") or not sql.get("sql"):
        return None
    # The LIMIT was added by the guard to protect the LLM CONTEXT from huge results.
    # But the kernel loads data directly (bypassing the LLM), so for training we want
    # the FULL dataset. Strip a trailing "LIMIT n" for the kernel load only.
    import re
    query = re.sub(r"\s+limit\s+\d+\s*;?\s*$", "", sql["sql"], flags=re.IGNORECASE).strip()
    query = query.replace("'", "''")   # escape single quotes for embedding
    # The kernel connects to Postgres itself (as the limited agent_user role) and
    # loads df. This is the database->kernel pipe; the data bypasses the LLM entirely.
    return (
        "import pandas as pd, psycopg2\n"
        "_conn = psycopg2.connect(host='localhost', port=5433, dbname='telecom', "
        "user='agent_user', password='agent_pw')\n"
        f"df = pd.read_sql('{query}', _conn)\n"
        "_conn.close()\n"
        "print('Loaded df directly from Postgres:', df.shape)"
    )


def data_node(state: AppState) -> dict:
    """Wrap the Data agent as a graph node. If SQL ran first, hand off its rows as `df`.
    Runs the data-science loop, writes the notebook, stores the result."""
    prelude = _handoff_prelude(state)

    # If the request clearly needed DB data but SQL did not succeed, do NOT run the
    # data agent on nothing (it would invent data). Report the blockage instead.
    sql = state.get("sql_result")
    if sql is not None and (sql.get("error") or not sql.get("rows")) and prelude is None:
        return {"data_result": {
            "summary": "Could not run analysis: the required data was not fetched "
                       "(the SQL step did not return usable data).",
            "n_cells": 0, "notebook": None, "had_data_handoff": False,
            "error": "no input data",
        }}

    # CRITICAL: if we handed off data, TELL the agent it exists. Otherwise the agent
    # assumes it must fetch data itself, fails (no DB access), and invents fake data.
    task = state["request"]
    if prelude is not None:
        sql = state["sql_result"]
        cols = list(sql["rows"][0].keys()) if sql.get("rows") else []
        task = (f"The data is ALREADY LOADED in the kernel as a pandas DataFrame named `df`, "
                f"loaded directly from the database (the FULL result set, columns: {cols}). "
                f"DO NOT create, download, or invent any data — use the existing `df`. "
                f"Start by inspecting it with df.shape and df.head().\n\n"
                f"Task: {state['request']}")

    result = run_data_agent(task, prelude=prelude)
    # write the notebook artifact
    nb_path = write_notebook(state["request"], result.cells, result.summary,
                             "data_agent_session.ipynb")
    # inspect what the agent actually did, so the synthesizer can report it honestly
    phases = list(dict.fromkeys(c.phase for c in result.cells if c.phase))
    all_code = "\n".join(c.code for c in result.cells)
    trained = ".fit(" in all_code and "train_test_split" in all_code
    return {"data_result": {
        "summary": result.summary,
        "n_cells": len(result.cells),
        "phases": phases,
        "trained_model": trained,
        "notebook": nb_path,
        "had_data_handoff": prelude is not None,
        "error": result.error,
    }}


def route_decision(state: AppState) -> str:
    """Conditional edge: read state['route'], return the name of the next node.
    'done' goes to the synthesizer (compose the answer) before ending."""
    dest = state.get("route", "done")
    return dest if dest in ("sql", "data") else "synthesizer"


def qa_decision(state: AppState) -> str:
    """Conditional edge after QA: retry the worker, or hand control back to supervisor.
    If QA failed AND retries remain, go back to 'sql'. Otherwise back to 'supervisor'."""
    if not state.get("qa_ok", True) and not state.get("qa_exhausted", False):
        return "sql"                                   # bounce back for a corrected attempt
    return "supervisor"                                # passed (or out of retries) -> reassess


def build_graph():
    """Assemble nodes + edges into a compiled, runnable graph."""
    g = StateGraph(AppState)

    g.add_node("supervisor", supervisor_node)
    g.add_node("sql", sql_node)
    g.add_node("data", data_node)                      # the flagship data-science worker
    g.add_node("qa", qa_node)                          # verification node
    g.add_node("synthesizer", synthesizer_node)        # final answer composer

    g.add_edge(START, "supervisor")                    # entry: always start at the supervisor
    g.add_conditional_edges("supervisor", route_decision,
                            {"sql": "sql", "data": "data", "synthesizer": "synthesizer"})
    g.add_edge("sql", "qa")                            # SQL results go through QA
    g.add_conditional_edges("qa", qa_decision, {"sql": "sql", "supervisor": "supervisor"})
    g.add_edge("data", "supervisor")                   # after data work, reassess
    g.add_edge("synthesizer", END)                     # synthesizer is the last stop

    return g.compile()


# a module-level compiled graph for convenience
app_graph = build_graph()
