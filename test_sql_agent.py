"""
test_sql_agent.py  --  drive the SQL agent from the command line.

Usage:
  python test_sql_agent.py "how many customers churned?"
  python test_sql_agent.py "average monthly charges per city"
  python test_sql_agent.py            # runs a built-in demo set incl. a guard test
"""
import sys
from agents.sql_agent import run_sql_agent
from db_access.sql_guard import guard, UnsafeSQLError


def show(result):
    print("\nQ:", result.question)
    if result.sql:
        print("SQL:", result.sql)
    if result.explanation:
        print("WHY:", result.explanation)
    if result.error:
        print("ERROR:", result.error)
    elif result.rows is not None:
        print(f"ROWS ({len(result.rows)}):")
        for r in result.rows[:10]:
            print("  ", r)


def guard_selftest():
    # prove the guard blocks danger WITHOUT any model or DB involved
    print("\n--- guard self-test ---")
    for bad in ["DROP TABLE customers", "SELECT 1; DROP TABLE cities", "DELETE FROM billing"]:
        try:
            guard(bad)
            print(f"  FAIL (not blocked): {bad}")
        except UnsafeSQLError as e:
            print(f"  blocked OK: {bad!r} -> {e}")


if __name__ == "__main__":
    guard_selftest()
    if len(sys.argv) > 1:
        show(run_sql_agent(" ".join(sys.argv[1:])))
    else:
        for q in [
            "how many customers churned versus stayed?",
            "what is the average monthly charge per city?",
            "which contract type has the highest churn rate?",
        ]:
            show(run_sql_agent(q))
