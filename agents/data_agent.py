"""
agents/data_agent.py
The flagship worker. An ITERATIVE loop: generate a code cell, run it in the
kernel, read the output back, decide the next step, repeat until done.

This is what lets ONE agent move through preprocessing -> EDA -> feature
engineering -> training -> evaluation as phases, each informed by the last.
The model reacts to real data, not a fixed script.

Safety: max_cells caps the loop (an agent running arbitrary code must not run
forever). The kernel is a separate process; later we sandbox it in Docker.
"""
from dataclasses import dataclass, field
from llm import get_llm
from tools.kernel import JupyterKernel, CellOutput

MAX_CELLS = 25   # circuit breaker. ML workflows need room: clean+EDA+features+train+eval,
                 # PLUS spare cells for the agent to recover from errors it hits.

SYSTEM = """You are a senior data scientist working in a live Jupyter kernel.
You complete the user's task by writing and running Python code ONE cell at a time.

How you work:
- Call run_code to execute a cell. You will see its output (text, tables, errors,
  and whether a chart was produced) before deciding the next cell.
- NEVER invent, download, or randomly generate data. If a DataFrame `df` is already
  loaded in the kernel, use it. If you are unsure what data exists, inspect it first
  with df.head() and df.info() — do not fabricate a dataset.
- State persists between cells (variables, imports, DataFrames stay alive). Once you
  have loaded or created data, DO NOT re-create it — reuse the existing variables.
- Each cell must make NEW progress. Move forward through phases; never repeat a
  step you have already completed.
- Work in clear phases: load/inspect data, clean it, explore (EDA with charts),
  engineer features if useful, train a model, evaluate it.
- When training a model, follow this ORDER strictly and verify each step's output
  before the next:
    1. Encode categorical/text columns to numbers (e.g. pd.get_dummies). The target
       'churn' is boolean — convert it to int (0/1).
    2. Define X (all feature columns) and y (the target) explicitly. Drop ID columns
       like customer_id from X.
    3. Print X.shape and y.shape to CONFIRM they exist before splitting.
    4. train_test_split, then fit the model, then evaluate (accuracy + a report).
  Never call train_test_split or model.fit before X and y are defined and confirmed.
- Read each output and adapt. If a cell errors, READ the error and fix the actual
  cause in the next cell — do not blindly repeat the same failing code.
- Use matplotlib for charts (they are captured automatically). seaborn is NOT
  guaranteed to be installed — prefer matplotlib.
- When the task is fully done, call finish with a short summary. Do NOT keep going.

Available libraries: pandas, numpy, matplotlib, scikit-learn.
"""

RUN_CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "run_code",
        "description": "Execute one Python cell in the live Jupyter kernel and see its output.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "the Python code for this cell"},
                "phase": {"type": "string",
                          "description": "current phase, e.g. 'load', 'clean', 'eda', 'features', 'train', 'eval'"},
            },
            "required": ["code"],
        },
    },
}
FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Call when the task is fully complete.",
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string", "description": "what was accomplished"}},
            "required": ["summary"],
        },
    },
}


@dataclass
class Cell:
    """One executed step: the code, the phase label, and the captured output."""
    code: str
    phase: str
    output: CellOutput


@dataclass
class DataResult:
    summary: str = ""
    cells: list[Cell] = field(default_factory=list)   # full history -> becomes the .ipynb
    error: str | None = None


def run_data_agent(task: str, kernel: JupyterKernel | None = None,
                   provider: str | None = None, prelude: str | None = None) -> DataResult:
    """Run the iterative data-science loop for `task`.
    `kernel` lets a caller pass a pre-seeded kernel (e.g. with a df already loaded);
    `prelude` is optional setup code run silently before the agent starts."""
    own_kernel = kernel is None
    k = kernel or JupyterKernel()
    llm = get_llm(provider)
    cells: list[Cell] = []

    try:
        if prelude:
            k.run(prelude)                              # silent setup, not shown to the model

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": task},
        ]

        for _ in range(MAX_CELLS):
            resp = llm.generate(messages, tools=[RUN_CODE_TOOL, FINISH_TOOL])

            if not resp.wants_tool:                     # model talked instead of acting -> nudge once
                messages.append({"role": "assistant", "content": resp.text or ""})
                messages.append({"role": "user", "content":
                                 "Use run_code to make progress, or finish if done."})
                continue

            call = resp.tool_calls[0]
            if call.name == "finish":
                # VERIFY the agent actually did what the task implied before accepting "done".
                # The agent will claim success it didn't achieve — don't take its word.
                all_code = "\n".join(c.code for c in cells)
                wants_model = any(w in task.lower() for w in
                                  ("train", "model", "predict", "classif", "regress"))
                did_train = ".fit(" in all_code
                if wants_model and not did_train and len(cells) < MAX_CELLS - 2:
                    # reject the premature finish, tell it exactly what's missing
                    messages.append({"role": "assistant", "content": "(tried to finish)"})
                    messages.append({"role": "user", "content":
                        "You have NOT trained a model yet — there is no model.fit() in your "
                        "work. The task requires training a model. Do not finish. Define X and "
                        "y, split the data, fit a model, and evaluate it (print accuracy)."})
                    continue
                return DataResult(summary=call.arguments.get("summary", "done"), cells=cells)

            # run_code: execute the cell, record it, feed the summary back
            code = call.arguments.get("code", "")
            phase = call.arguments.get("phase", "")
            out = k.run(code)
            cells.append(Cell(code=code, phase=phase, output=out))

            # Feed BACK the actual code + its output, so the model sees its own prior
            # work and builds on it instead of repeating (e.g. re-creating the DataFrame).
            messages.append({"role": "assistant",
                             "content": f"I ran this cell (phase={phase}):\n```python\n{code}\n```"})
            if not out.ok:                              # a cell errored — make recovery the priority
                feedback = (f"The cell FAILED with an error:\n{out.summary_for_model()}\n\n"
                            f"Read the error carefully and fix the ROOT CAUSE in your next cell. "
                            f"Do not repeat the same code. If a variable is undefined, define it first.")
            else:
                feedback = (f"Cell output:\n{out.summary_for_model()}\n\n"
                            f"Continue to the NEXT step. Do not repeat work already done.")
            messages.append({"role": "user", "content": feedback})

        return DataResult(summary="Stopped: hit max cells.", cells=cells,
                          error="max cells reached")
    finally:
        if own_kernel:
            k.shutdown()                                # only shut down a kernel we created
