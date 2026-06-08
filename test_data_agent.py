"""
test_data_agent.py  --  run the Data agent and SAVE the session as a real .ipynb.

Usage:
  python test_data_agent.py "your data-science task here"
  python test_data_agent.py            # uses a default task

Needs OPENAI_API_KEY set and the kernel deps installed.
"""
import sys
from agents.data_agent import run_data_agent
from tools.notebook_writer import write_notebook

DEFAULT = ("Create a pandas DataFrame of 20 random numbers, show summary statistics, "
           "and plot a histogram.")


def main():
    task = " ".join(sys.argv[1:]) or DEFAULT
    print(f"TASK: {task}\n" + "-" * 50)

    result = run_data_agent(task)

    # show what happened in the terminal
    for i, c in enumerate(result.cells, 1):
        print(f"\n[cell {i}] phase={c.phase}")
        print(f"  code: {c.code[:70].replace(chr(10),' ')}...")
        print(f"  out:  {c.output.summary_for_model()[:80].replace(chr(10),' ')}")
    print(f"\nSUMMARY: {result.summary}")

    # THE PART THAT WAS MISSING: write the notebook file
    path = write_notebook(task, result.cells, result.summary, "data_agent_session.ipynb")
    print(f"\nNotebook written to: {path}")
    print("Open it in Jupyter or VS Code to see the cells and embedded charts.")


if __name__ == "__main__":
    main()
