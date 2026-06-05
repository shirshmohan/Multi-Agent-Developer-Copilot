"""
db_access/sql_guard.py
The statement-level safety layer. DETERMINISTIC code, not an LLM.

This is the embryo of your Phase 9 risk service. It is cheap, fast, and cannot
be jailbroken by a clever prompt — because it never asks a model anything.

Defense-in-depth reminder: this is ONE of three layers.
  1. this guard (blocks dangerous SQL in our code)        <- here
  2. the agent_user DB role (database refuses writes)     <- schema.sql
  3. workspace isolation (writes go to agent_workspace)   <- later
Any single layer failing should not be catastrophic.
"""
import re

# Verbs that must never reach the source data via the SQL agent.
FORBIDDEN = re.compile(
    r"\b(DROP|DELETE|TRUNCATE|UPDATE|INSERT|ALTER|CREATE|GRANT|REVOKE|COPY)\b",
    re.IGNORECASE,
)


class UnsafeSQLError(Exception):
    """Raised when SQL fails a safety check. The agent reports this, never executes."""


def guard(sql: str) -> str:
    """Validate SQL is a safe single SELECT. Returns the cleaned SQL or raises.
    Called BEFORE any execution. If it raises, nothing touches the database."""
    cleaned = sql.strip().rstrip(";").strip()

    if not cleaned:
        raise UnsafeSQLError("Empty query.")

    # 1. must START as a SELECT (or a WITH ... SELECT CTE)
    head = cleaned.lstrip("(").lower()
    if not (head.startswith("select") or head.startswith("with")):
        raise UnsafeSQLError("Only SELECT queries are allowed.")

    # 2. no forbidden verbs anywhere (catches sneaky multi-clause attempts)
    if FORBIDDEN.search(cleaned):
        raise UnsafeSQLError("Query contains a forbidden keyword.")

    # 3. no statement stacking — one query only (block 'SELECT ...; DROP ...')
    if ";" in cleaned:
        raise UnsafeSQLError("Multiple statements are not allowed.")

    # 4. force a row cap if the model forgot one (cheap defense vs huge scans)
    if not re.search(r"\blimit\b", cleaned, re.IGNORECASE):
        cleaned += " LIMIT 100"

    return cleaned
