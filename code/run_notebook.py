"""Turn a readable percent-cell script into a notebook and execute each cell.

Uses the current Python interpreter; saves outputs after every cell so long
experiments retain completed outputs if a later cell fails.
"""

import contextlib
import io
import re
import sys
import traceback
from pathlib import Path
import nbformat

source = Path(sys.argv[1]).resolve()
target = source.parents[1] / "notebooks" / (source.stem + ".ipynb")
nb = nbformat.v4.new_notebook()
for section in re.split(r"^# %%", source.read_text(encoding="utf-8"), flags=re.M):
    if not section.strip():
        continue
    if section.startswith(" [markdown]"):
        text = "\n".join(re.sub(r"^# ?", "", line) for line in section.splitlines()[1:])
        nb.cells.append(nbformat.v4.new_markdown_cell(text))
    else:
        nb.cells.append(nbformat.v4.new_code_cell(section.strip()))
nb.metadata.kernelspec = {
    "display_name": "Python (PA4 Intel XPU)",
    "language": "python",
    "name": "python3",
}
namespace = {"__name__": "__main__", "__file__": str(source)}
sys.path.insert(0, str(source.parent))
nbformat.write(nb, target)


class Tee(io.StringIO):
    def write(self, s):
        sys.__stdout__.write(s)
        sys.__stdout__.flush()
        return super().write(s)


count = 0
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    count += 1
    cell.execution_count = count
    capture = Tee()
    print(f"\nExecuting {source.stem} cell {count}", flush=True)
    try:
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            exec(compile(cell.source, str(source), "exec"), namespace)
        cell.outputs = [
            nbformat.v4.new_output("stream", name="stdout", text=capture.getvalue())
        ]
    except Exception as exc:
        cell.outputs = [
            nbformat.v4.new_output("stream", name="stdout", text=capture.getvalue()),
            nbformat.v4.new_output(
                "error",
                ename=type(exc).__name__,
                evalue=str(exc),
                traceback=traceback.format_exc().splitlines(),
            ),
        ]
        nbformat.write(nb, target)
        raise
    nbformat.write(nb, target)
