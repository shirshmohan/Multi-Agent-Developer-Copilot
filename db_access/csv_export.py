"""
db_access/csv_export.py
Export a full SQL query result straight to a CSV file.

WHY this exists: the SQL guard adds LIMIT 100 to protect the LLM CONTEXT from huge
results. But for ML/analysis we want the FULL dataset — and a CSV file never touches
the LLM context, so there's no reason to cap it. This is the data-plane handoff:
SQL writes a file, the data agent reads the file. Rows never pass through any prompt.
"""
import re
import os
import pandas as pd
import psycopg2

# Same limited role as everywhere else — read-only on source data.
AGENT_DB = dict(host="localhost", port=5433, dbname="telecom",
                user="agent_user", password="agent_pw")

EXPORT_DIR = "exports"   # CSVs land here; you can open/edit them yourself


def export_query_to_csv(sql: str, name: str = "query_result") -> dict:
    """Run a SELECT (full result, no LIMIT) and write it to exports/<name>.csv.
    Returns metadata ONLY (path, shape, columns) — never the rows themselves."""
    # strip any trailing LIMIT the guard added — the file isn't context-limited
    clean = re.sub(r"\s+limit\s+\d+\s*;?\s*$", "", sql.strip(), flags=re.IGNORECASE).rstrip(";")

    os.makedirs(EXPORT_DIR, exist_ok=True)
    path = os.path.join(EXPORT_DIR, f"{name}.csv")

    conn = psycopg2.connect(**AGENT_DB)
    try:
        # Use a cursor directly (pd.read_sql on a raw psycopg2 conn warns about SQLAlchemy).
        with conn.cursor() as cur:
            cur.execute(clean)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=cols)
    finally:
        conn.close()

    df.to_csv(path, index=False)
    # return METADATA only — this is what's safe to put in front of the LLM
    return {
        "path": os.path.abspath(path),
        "rows": len(df),
        "columns": list(df.columns),
    }
