"""
graph/synthesizer.py
The final node. Turns raw worker results into one clean, grounded answer.

HARD RULE baked into the prompt: state ONLY what the data shows. No invented
numbers, no outside knowledge, no embellishment. A synthesizer that hallucinates
at the last step is worse than none — it undoes the QA work upstream.
"""
from llm import get_llm
from graph.state import AppState

SYNTH_SYS = """You write the final answer for a data assistant.
You are given the user's question and the raw results gathered by specialist agents.

Rules:
- Answer ONLY using the provided results. Do not invent or estimate any numbers.
- Some results are labeled "VERIFIED" or "VERIFIED FACT" — these are derived from the
  actual code/data and OVERRIDE any "UNVERIFIED self-report". If a self-report claims
  something a verified fact contradicts, believe the VERIFIED fact and report honestly.
- Do not add outside knowledge, benchmarks, or industry comparisons.
- Be concise and direct. State the finding, with the actual figures from the data.
- If a result looks unverified or empty, say so plainly rather than guessing.
- Round long decimals sensibly (e.g. 0.427 -> 42.7%)."""


def _format_results(state: AppState) -> str:
    """Collect whatever workers produced into a compact text block for the prompt."""
    parts = []
    sql = state.get("sql_result")
    if sql:
        if sql.get("error"):
            parts.append(f"SQL agent error: {sql['error']}")
        else:
            rows = sql.get("rows") or []
            parts.append(f"SQL query: {sql.get('sql')}")
            # This 'rows' is only a 100-row PREVIEW (the guard's LIMIT). The full dataset
            # was exported to CSV — the real count is in data_result, not here.
            parts.append(f"SQL query ran successfully (preview of {len(rows)} rows shown; "
                         f"full result was exported for analysis — see dataset size below).")
            if rows:
                parts.append(f"Columns: {list(rows[0].keys())}")
            if state.get("qa_exhausted"):
                parts.append("NOTE: this result did not pass verification — flag it as unverified.")

    data = state.get("data_result")
    if data:
        if data.get("error"):
            parts.append(f"Data agent could not complete: {data['error']}")
        else:
            parts.append(f"Data agent summary (UNVERIFIED self-report): {data.get('summary')}")
            parts.append(f"Data agent ran {data.get('n_cells', 0)} notebook cells "
                         f"across phases: {', '.join(data.get('phases', [])) or 'n/a'}.")
            if data.get("dataset_rows"):
                parts.append(f"The analysis used the FULL dataset of {data['dataset_rows']} rows "
                             f"(not the 100-row preview).")
            if data.get("trained_model"):
                parts.append("VERIFIED: a model was actually trained (model.fit was called).")
            else:
                parts.append("VERIFIED FACT: NO model was trained (no model.fit in the code). "
                             "Do NOT claim a model was trained. Say analysis was done but "
                             "model training did not complete.")
            if data.get("notebook"):
                parts.append(f"Notebook saved to: {data['notebook']}")
            if data.get("csv"):
                parts.append(f"The dataset CSV is at: {data['csv']}")
            if data.get("handoff_to_user"):
                parts.append(f"IMPORTANT — TELL THE USER: {data['handoff_to_user']}")
    return "\n".join(parts) if parts else "No results were gathered."


def synthesizer_node(state: AppState) -> dict:
    """Compose the final answer from accumulated state. Writes state['answer']."""
    llm = get_llm()
    messages = [
        {"role": "system", "content": SYNTH_SYS},
        {"role": "user", "content":
            f"Question: {state['request']}\n\nResults gathered:\n{_format_results(state)}"},
    ]
    resp = llm.generate(messages)              # plain text generation, no tools
    return {"answer": resp.text or "(no answer produced)"}
