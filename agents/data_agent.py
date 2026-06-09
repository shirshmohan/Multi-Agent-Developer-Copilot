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

MAX_CELLS = 35   # richer pipeline (EDA + multi-model + RandomizedSearchCV + eval) plus
                 # headroom to recover from errors. Each cell is one LLM round-trip.

SYSTEM = """You are a senior ML engineer working in a live Jupyter kernel.
You complete the user's task by writing and running Python code ONE cell at a time.

How you work:
- Call run_code to execute a cell. You will see its output (text, tables, errors,
  and whether a chart was produced) before deciding the next cell.
- NEVER invent, download, or randomly generate data. A DataFrame `df` is already
  loaded. Use it. If unsure what it contains, inspect with df.head()/df.info() first.
- State persists between cells. Reuse variables; never re-create data you already have.
- Each cell must make NEW progress. Read each output and adapt. If a cell errors,
  read the error and fix the ROOT CAUSE in the next cell — never repeat failing code.
- Use matplotlib for charts (captured automatically). seaborn may not be installed.

WHEN THE TASK INVOLVES TRAINING A MODEL, follow this EXACT pipeline. Do each step as
its own cell and CONFIRM its output before moving on. Do not skip or merge steps:

  STEP 1 — Understand: df.shape, df.info(), and the target's value_counts.
  STEP 2 — Clean & encode: convert booleans to int; one-hot encode categorical
           columns with pd.get_dummies. DROP identifier columns (e.g. customer_id)
           — they are not features.
  STEP 3 — DEFINE X AND y EXPLICITLY. This step is MANDATORY and the most commonly
           skipped — do NOT skip it:
               y = df_encoded['churn'].astype(int)
               X = df_encoded.drop(columns=['churn'])
           Then PRINT X.shape and y.shape to confirm both exist.
  STEP 4 — Split: train_test_split(X, y, test_size=0.2, random_state=42, stratify=y).
           Print the train/test shapes.
  STEP 5 — Train MULTIPLE models and compare. Fit at least: LogisticRegression
           (max_iter=1000), RandomForestClassifier, and GradientBoostingClassifier.
           Print each model's test accuracy so they can be compared.
  STEP 6 — Tune the BEST model with RandomizedSearchCV (a small param grid, cv=3,
           n_iter=5). Print the best params and best cross-validated score.
  STEP 7 — Evaluate the tuned model: print accuracy and classification_report on
           the test set. If imbalanced, note it.

NEVER call train_test_split or model.fit before X and y are defined and printed.
Only call finish AFTER a model has actually been trained and evaluated (a real
model.fit has run without error). Do not claim success you have not achieved.

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


def _names_exist(kernel: JupyterKernel, names: list[str]) -> bool:
    """Probe the live kernel: are ALL these variable names actually defined?
    This is how we VERIFY state instead of trusting the model's claims.
    Uses globals() — dir() inside print() runs in a different scope and misses them."""
    check = "print(all(n in globals() for n in %r))" % names
    out = kernel.run(check)
    return "True" in out.stdout


def _blocked_output(reason: str) -> CellOutput:
    """A CellOutput representing a cell we refused to run (so the notebook shows why)."""
    co = CellOutput()
    co.ok = False
    co.error = f"[blocked by guard] {reason}"
    return co


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
            out = k.run(prelude)                        # setup (e.g. load df from CSV)
            # RECORD it as the notebook's first cell, so the notebook is REPRODUCIBLE:
            # a human can open it and "Run All" — df gets created by a visible cell,
            # not by hidden setup. Without this, cell 1 references an undefined df.
            cells.append(Cell(code=prelude, phase="setup", output=out))

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

            # DETERMINISTIC GUARD: the model repeatedly runs steps before their inputs
            # exist (split before X/y; fit before the split). Telling it harder in the
            # prompt doesn't work, so VERIFY kernel state and intercept doomed cells.
            block_msg = None
            if "train_test_split(" in code and not _names_exist(k, ["X", "y"]):
                block_msg = (
                    "STOP. I checked the kernel: `X` and `y` do NOT exist yet. You cannot "
                    "split before defining them. Your NEXT cell must be exactly:\n"
                    "    y = df_encoded['churn'].astype(int)\n"
                    "    X = df_encoded.drop(columns=['churn'])\n"
                    "    print(X.shape, y.shape)\n"
                    "(adjust the DataFrame name if yours differs). Define X and y now.")
            elif ".fit(" in code and not _names_exist(k, ["X_train", "y_train"]):
                block_msg = (
                    "STOP. I checked the kernel: `X_train`/`y_train` do NOT exist yet. You "
                    "cannot fit a model before splitting. If X and y exist, your NEXT cell "
                    "must be:\n"
                    "    from sklearn.model_selection import train_test_split\n"
                    "    X_train, X_test, y_train, y_test = train_test_split("
                    "X, y, test_size=0.2, random_state=42, stratify=y)\n"
                    "    print(X_train.shape, X_test.shape)\n"
                    "If X and y don't exist yet, define those first.")
            if block_msg:
                cells.append(Cell(code=code, phase=phase,
                                  output=_blocked_output(block_msg.split(chr(10))[0])))
                messages.append({"role": "assistant",
                                 "content": "(blocked: prerequisite not defined)"})
                messages.append({"role": "user", "content": block_msg})
                continue

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
