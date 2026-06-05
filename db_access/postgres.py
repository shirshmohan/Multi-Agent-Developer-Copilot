"""
db_access/postgres.py
All Postgres access for the agents. TWO critical design choices:

1. Agents connect as `agent_user` (the LIMITED role), NEVER as admin.
   This is the real security wall — the database itself refuses writes to
   the source tables, regardless of what SQL the model generates.

2. We can hand the model the live schema, so it writes SQL against YOUR
   actual tables instead of hallucinating column names.
"""
import psycopg2
from psycopg2.extras import RealDictCursor

# NOTE: agent_user / agent_pw — the locked-down role from schema.sql.
# Port 5432 (change to 5433 here if you remapped the port earlier).
AGENT_DB = dict(
    host="localhost", port=5433, dbname="telecom",
    user="agent_user", password="agent_pw",
)


def run_select(sql: str, limit: int = 100) -> list[dict]:
    """Execute a (validated) SELECT as the limited role. Returns rows as dicts.
    The guard runs BEFORE this is ever called — this function trusts that it passed."""
    conn = psycopg2.connect(**AGENT_DB)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchmany(limit)        # cap rows returned — never fetch unbounded
            return [dict(r) for r in rows]
    finally:
        conn.close()                            # always release the connection


def get_schema_text() -> str:
    """Introspect the public schema and render it as compact text for the prompt.
    This is the 'context engineering' — we give the model exactly what it needs to
    write correct SQL: table names, columns, types, and foreign keys."""
    conn = psycopg2.connect(**AGENT_DB)
    try:
        with conn.cursor() as cur:
            # columns per table
            cur.execute("""
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position;
            """)
            cols = cur.fetchall()
            # foreign keys, so the model knows how tables join
            cur.execute("""
                SELECT
                    tc.table_name, kcu.column_name,
                    ccu.table_name AS ref_table, ccu.column_name AS ref_column
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage ccu
                    ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_schema = 'public';
            """)
            fks = cur.fetchall()
    finally:
        conn.close()

    # render tables
    tables: dict[str, list[str]] = {}
    for table, col, dtype in cols:
        tables.setdefault(table, []).append(f"{col} ({dtype})")
    lines = []
    for table, columns in tables.items():
        lines.append(f"TABLE {table}: " + ", ".join(columns))
    # render FKs
    if fks:
        lines.append("FOREIGN KEYS:")
        for table, col, ref_t, ref_c in fks:
            lines.append(f"  {table}.{col} -> {ref_t}.{ref_c}")

    # GROUNDING: append real distinct values for low-cardinality text columns.
    # This is what stops the model guessing 'Month-to-Month' instead of 'Month-to-month'.
    enum_lines = get_column_values(cols)
    if enum_lines:
        lines.append("ENUM-LIKE COLUMN VALUES (use these EXACT strings):")
        lines.extend(enum_lines)

    return "\n".join(lines)


def get_column_values(cols, max_distinct: int = 12) -> list[str]:
    """For each text column, if it has <= max_distinct distinct values, list them.
    These exact strings go in the prompt so the model never guesses casing/spelling."""
    text_cols = [(t, c) for (t, c, dtype) in cols if dtype in ("text", "character varying")]
    out = []
    conn = psycopg2.connect(**AGENT_DB)
    try:
        with conn.cursor() as cur:
            for table, col in text_cols:
                # count distinct first; skip high-cardinality columns (e.g. customer_id)
                cur.execute(
                    f'SELECT COUNT(DISTINCT "{col}") FROM "{table}";')
                n = cur.fetchone()[0]
                if n == 0 or n > max_distinct:
                    continue
                cur.execute(
                    f'SELECT DISTINCT "{col}" FROM "{table}" '
                    f'WHERE "{col}" IS NOT NULL ORDER BY 1;')
                vals = [str(r[0]) for r in cur.fetchall()]
                out.append(f"  {table}.{col}: " + " | ".join(vals))
    finally:
        conn.close()
    return out
