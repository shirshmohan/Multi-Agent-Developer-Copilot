"""
agents/sql_agent.py
The first real worker. English question -> validated SQL -> results.

Flow:
  1. fetch the live schema (context engineering)
  2. ask the LLM for SQL (provider-agnostic via get_llm)
  3. guard the SQL (deterministic safety) BEFORE executing
  4. execute as the limited agent_user role
  5. return a structured result the synthesizer / UI can use
"""
from dataclasses import dataclass
from llm import get_llm
from db_access.postgres import run_select, get_schema_text
from db_access.sql_guard import guard, UnsafeSQLError


SYSTEM_TEMPLATE = """You are a PostgreSQL expert for a telecom database.
Given a question, produce ONE valid PostgreSQL SELECT query that answers it.

Rules:
- SELECT queries only. Never modify data.
- Use only the tables and columns in the schema below.
- Use JOINs via the listed foreign keys when the answer spans tables.
- Keep results reasonable; prefer aggregates over dumping every row.

SCHEMA:
{schema}
"""

# The model returns SQL through this tool, giving us a clean structured string.
SQL_TOOL = [{
    "type": "function",
    "function": {
        "name": "emit_sql",
        "description": "Return the single PostgreSQL SELECT query that answers the question.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "a single SELECT statement"},
                "explanation": {"type": "string", "description": "one sentence on what it does"},
            },
            "required": ["sql"],
        },
    },
}]


@dataclass
class SQLResult:
    """Structured output. Either rows (success) or an error string (handled failure)."""
    question: str
    sql: str | None = None
    explanation: str | None = None
    rows: list[dict] | None = None
    error: str | None = None


def run_sql_agent(question: str, provider: str | None = None, feedback: str | None = None) -> SQLResult:
    llm = get_llm(provider)                              # provider-agnostic
    schema = get_schema_text()                           # 1. live schema context
    user_content = question
    if feedback:                                         # a QA retry: tell the model what was wrong
        user_content += (f"\n\nYour previous attempt had these problems: {feedback}. "
                         f"Fix them. Re-check column values and the calculation logic.")
    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(schema=schema)},
        {"role": "user", "content": user_content},
    ]

    # 2. ask for SQL via tool call
    resp = llm.generate(messages, tools=SQL_TOOL)
    if not resp.wants_tool:
        return SQLResult(question, error=f"Model did not produce SQL. Said: {resp.text}")

    call = resp.tool_calls[0]
    raw_sql = call.arguments.get("sql", "")
    explanation = call.arguments.get("explanation")

    # 3. GUARD before executing — this can raise, and that's the safe path
    try:
        safe_sql = guard(raw_sql)
    except UnsafeSQLError as e:
        return SQLResult(question, sql=raw_sql, error=f"Blocked by guard: {e}")

    # 4. execute as the limited role
    try:
        rows = run_select(safe_sql)
    except Exception as e:
        return SQLResult(question, sql=safe_sql, error=f"Execution error: {e}")

    # 5. structured success
    return SQLResult(question, sql=safe_sql, explanation=explanation, rows=rows)
