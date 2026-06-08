"""
tools/kernel.py
Owns a live Jupyter kernel. Runs code in it and captures rich outputs.

WHY a real kernel instead of exec():
  - isolation: the kernel is a SEPARATE process, not our own
  - state: variables/imports/DataFrames PERSIST between run() calls (like notebook cells)
  - rich output: we get tables, tracebacks, AND images (charts as base64 PNG) back,
    which is what makes the chart-feedback loop possible.

This wraps jupyter_client — the same library the notebook UI uses underneath.
"""
import queue
from dataclasses import dataclass, field
from jupyter_client import KernelManager


@dataclass
class CellOutput:
    """Everything one code execution produced — mirrors what a notebook cell shows."""
    stdout: str = ""                       # printed text
    result: str = ""                       # the value of the last expression (repr)
    images: list[str] = field(default_factory=list)  # base64 PNGs (matplotlib charts)
    error: str = ""                        # traceback text, if the cell raised
    ok: bool = True                        # did it run without error?

    def summary_for_model(self) -> str:
        """A compact TEXT view to feed back to the LLM. Images are noted, not inlined
        (we don't send the raw base64 to the model — too big; we say a chart was made)."""
        parts = []
        if self.stdout.strip():
            parts.append(f"stdout:\n{self.stdout.strip()}")
        if self.result.strip():
            parts.append(f"result:\n{self.result.strip()}")
        if self.images:
            parts.append(f"[{len(self.images)} chart(s) produced]")
        if self.error:
            parts.append(f"ERROR:\n{self.error.strip()}")
        return "\n".join(parts) if parts else "(no output)"


class JupyterKernel:
    """A single long-lived kernel. Start it, run() repeatedly (state persists), shutdown()."""

    def __init__(self):
        self._km = KernelManager()
        self._km.start_kernel()
        self._kc = self._km.client()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=30)
        # Configure matplotlib to emit charts as inline PNGs (display_data messages).
        # Without this, plt.show() produces no capturable image.
        self.run("%matplotlib inline\nimport matplotlib;matplotlib.use('module://matplotlib_inline.backend_inline')")

    def run(self, code: str, timeout: int = 60) -> CellOutput:
        """Execute code in the kernel. Blocks until the cell finishes. State persists."""
        msg_id = self._kc.execute(code)
        out = CellOutput()

        while True:
            try:
                msg = self._kc.get_iopub_msg(timeout=timeout)
            except queue.Empty:
                out.error = f"Execution timed out after {timeout}s"
                out.ok = False
                break

            # only care about messages replying to OUR execute request
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            mtype = msg["msg_type"]
            content = msg["content"]

            if mtype == "stream":                       # print() output
                out.stdout += content.get("text", "")
            elif mtype == "execute_result":             # last-expression value
                out.result += content["data"].get("text/plain", "")
            elif mtype == "display_data":               # plots, rich display
                data = content["data"]
                if "image/png" in data:
                    out.images.append(data["image/png"])  # base64 PNG
                elif "text/plain" in data:
                    out.result += data["text/plain"]
            elif mtype == "error":                      # exception
                out.error = "\n".join(content["traceback"])
                out.ok = False
            elif mtype == "status" and content["execution_state"] == "idle":
                break                                   # cell finished

        return out

    def shutdown(self):
        """Always call this — a leaked kernel is a leaked process."""
        try:
            self._kc.stop_channels()
            self._km.shutdown_kernel(now=True)
        except Exception:
            pass
