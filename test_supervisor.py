"""
test_supervisor.py  --  run the REAL supervisor graph end to end.

Needs your OPENAI_API_KEY set and the databases running (port 5433).

Usage:
  python test_supervisor.py "how many customers churned?"
  python test_supervisor.py            # built-in demo request
"""
import sys
from graph.supervisor import build_graph


def run(request: str):
    graph = build_graph()
    print(f"\nREQUEST: {request}\n" + "-" * 50)

    # stream() shows each node firing — you SEE the routing happen
    for event in graph.stream({"request": request, "steps": 0}):
        for node_name, update in event.items():
            print(f"[node: {node_name}]")
            if not isinstance(update, dict):      # some nodes/updates may be None — skip safely
                continue
            if "route" in update:
                print(f"   routed -> {update['route']}  ({update.get('reason','')})")
            if "qa_ok" in update:
                status = "passed" if update["qa_ok"] else f"FAILED: {update.get('qa_feedback','')}"
                print(f"   qa -> {status}")
            if "sql_result" in update:
                r = update["sql_result"]
                if r.get("error"):
                    print(f"   sql error: {r['error']}")
                else:
                    print(f"   sql: {r.get('sql')}")
                    print(f"   rows: {r.get('rows')}")
            if "answer" in update:
                print(f"\n=== FINAL ANSWER ===\n{update['answer']}")


if __name__ == "__main__":
    req = " ".join(sys.argv[1:]) or "what is the churn rate for month-to-month contracts?"
    run(req)
