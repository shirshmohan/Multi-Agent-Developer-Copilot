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
from graph.state import AppState
from graph.qa import qa_node

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
                    "enum": ["sql", "done"],          # grows as we add workers (mongo, data...)
                    "description": "'sql' for database questions; 'done' when the request is fully answered.",
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

If the request has already been fully answered by prior work, route to 'done'.
Route to exactly one destination by calling the route tool."""


def supervisor_node(state: AppState) -> dict:
    """Classify + route. Writes 'route' and 'reason' to state. Increments the step counter."""
    steps = state.get("steps", 0) + 1
    if steps > MAX_STEPS:                              # circuit breaker fires
        return {"route": "done", "reason": "max steps reached", "steps": steps}

    llm = get_llm()
    # summarize what's been done so the supervisor doesn't re-route the same work
    done_so_far = "Nothing yet." if "sql_result" not in state else \
        f"SQL agent already ran. Result keys: {list(state['sql_result'].keys())}"
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


def route_decision(state: AppState) -> str:
    """Conditional edge: read state['route'], return the name of the next node.
    THIS is the routing logic LangGraph makes explicit instead of buried in if/else."""
    dest = state.get("route", "done")
    return dest if dest in ("sql",) else END           # 'done' -> END; otherwise the worker node


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
    g.add_node("qa", qa_node)                          # verification node

    g.add_edge(START, "supervisor")                    # entry: always start at the supervisor
    g.add_conditional_edges("supervisor", route_decision, {"sql": "sql", END: END})
    g.add_edge("sql", "qa")                            # every worker result goes through QA
    g.add_conditional_edges("qa", qa_decision, {"sql": "sql", "supervisor": "supervisor"})

    return g.compile()


# a module-level compiled graph for convenience
app_graph = build_graph()
