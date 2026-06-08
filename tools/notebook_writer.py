"""
tools/notebook_writer.py
Turn the Data agent's cell history into a real .ipynb file.

An .ipynb is just JSON: a list of cells, each with a type, source, and outputs.
We translate our Cell objects into that schema, embedding captured charts as
real image outputs so they render when you open the notebook.

No nbformat dependency needed — we build the (well-documented) v4 structure directly.
"""
import json
import uuid
from pathlib import Path


def _cell_id() -> str:
    return uuid.uuid4().hex[:8]


def _code_cell(code: str, output) -> dict:
    """Build one notebook code cell with its outputs (stdout, result text, images)."""
    outputs = []
    if output.stdout.strip():
        outputs.append({
            "output_type": "stream", "name": "stdout",
            "text": output.stdout.splitlines(keepends=True),
        })
    if output.result.strip():
        outputs.append({
            "output_type": "execute_result", "execution_count": None,
            "data": {"text/plain": output.result.splitlines(keepends=True)},
            "metadata": {},
        })
    for img_b64 in output.images:                       # embed each chart as a real image output
        outputs.append({
            "output_type": "display_data",
            "data": {"image/png": img_b64},
            "metadata": {},
        })
    if output.error:
        outputs.append({
            "output_type": "error", "ename": "Error", "evalue": "",
            "traceback": output.error.splitlines(),
        })
    return {
        "cell_type": "code", "execution_count": None, "id": _cell_id(),
        "metadata": {}, "source": code.splitlines(keepends=True),
        "outputs": outputs,
    }


def _markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "id": _cell_id(), "metadata": {},
            "source": text.splitlines(keepends=True)}


def write_notebook(task: str, cells: list, summary: str, path: str) -> str:
    """Write the full session to `path` as a .ipynb. Returns the path."""
    nb_cells = [_markdown_cell(f"# Data Agent Session\n\n**Task:** {task}")]

    last_phase = None
    for c in cells:
        if c.phase and c.phase != last_phase:           # a header when the phase changes
            nb_cells.append(_markdown_cell(f"## Phase: {c.phase}"))
            last_phase = c.phase
        nb_cells.append(_code_cell(c.code, c.output))

    if summary:
        nb_cells.append(_markdown_cell(f"## Summary\n\n{summary}"))

    notebook = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    Path(path).write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return path
