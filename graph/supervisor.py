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
from db_access.csv_export import export_query_to_csv
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


def _handoff_prelude(state: AppState) -> tuple[str, dict] | tuple[None, None]:
    """SQL -> Data handoff via CSV. The SQL agent's query result is exported to a
    full CSV file (no row cap), then the data agent's kernel reads that file.
    Returns (prelude_code, csv_metadata). The rows live only in the file + kernel,
    never in any LLM context."""
    sql = state.get("sql_result")
    if not sql or sql.get("error") or not sql.get("sql"):
        return None, None
    try:
        meta = export_query_to_csv(sql["sql"], name="ml_dataset")   # full result -> CSV
    except Exception as e:
        return None, {"error": f"CSV export failed: {e}"}

    # prelude: read the CSV into the kernel. df.info()/describe() come next (the agent's job).
    prelude = (
        "import pandas as pd\n"
        f"df = pd.read_csv(r'{meta['path']}')\n"
        "print('Loaded df from CSV:', df.shape)"
    )
    return prelude, meta


def data_node(state: AppState) -> dict:
    """Wrap the Data agent as a graph node. SQL result is handed off as a CSV the
    agent reads into its kernel. Runs the loop, writes the notebook, stores results."""
    prelude, meta = _handoff_prelude(state)

    # If the request needed DB data but export failed / no data, don't fabricate.
    sql = state.get("sql_result")
    if sql is not None and (sql.get("error") or not sql.get("sql")) and prelude is None:
        return {"data_result": {
            "summary": "Could not run analysis: the required data was not fetched.",
            "n_cells": 0, "notebook": None, "had_data_handoff": False,
            "error": "no input data",
        }}

    task = state["request"]
    if prelude is not None:
        wants_model = any(w in state["request"].lower() for w in
                          ("train", "model", "predict", "classif", "regress"))
        extra = ""
        if wants_model:
            extra = (" Train and COMPARE multiple models (logistic regression, random "
                     "forest, gradient boosting), then tune the best with "
                     "RandomizedSearchCV, and report the final accuracy and "
                     "classification report.")
        task = (f"The data is ALREADY LOADED in the kernel as a pandas DataFrame `df`, "
                f"read from a CSV ({meta['rows']} rows, columns: {meta['columns']}). "
                f"DO NOT create, download, or invent data — use the existing `df`. "
                f"Begin by understanding it: df.shape, df.info(), df.describe(), and the "
                f"target distribution. Decide preprocessing from what you observe.{extra}\n\n"
                f"Task: {state['request']}")

    result = run_data_agent(task, prelude=prelude)
    # write the notebook artifact
    nb_path = write_notebook(state["request"], result.cells, result.summary,
                             "data_agent_session.ipynb")
    # inspect what the agent actually did, so the synthesizer can report it honestly
    phases = list(dict.fromkeys(c.phase for c in result.cells if c.phase))
    all_code = "\n".join(c.code for c in result.cells)
    trained = ".fit(" in all_code and "train_test_split" in all_code
    got_stuck = result.error is not None       # hit max cells / couldn't recover

    import os
    csv_path = os.path.abspath(os.path.join("exports", "ml_dataset.csv")) \
        if prelude is not None else None

    data_result = {
        "summary": result.summary,
        "n_cells": len(result.cells),
        "phases": phases,
        "trained_model": trained,
        "dataset_rows": meta.get("rows") if meta else None,   # the TRUE full-dataset size
        "notebook": nb_path,
        "csv": csv_path,                       # so YOU can open/edit the data
        "had_data_handoff": prelude is not None,
        "error": result.error,
    }
    if got_stuck:
        # PRAGMATIC COLLABORATION: agent couldn't finish. Hand control to the human
        # with the artifacts, instead of silently failing or faking success.
        data_result["handoff_to_user"] = (
            f"The agent got stuck and stopped. You can take over: the data is at "
            f"{csv_path} and the notebook-so-far is at {os.path.abspath(nb_path)}. "
            f"Open the notebook, fix the failing step, and continue manually.")
    return {"data_result": data_result}


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
