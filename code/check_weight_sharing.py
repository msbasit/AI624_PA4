"""Re-run the assignment's path-sharing verification cell, not training."""

import contextlib
import io
import json
import re
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "code" / "task4.py").read_text()
cells = [
    s.strip()
    for s in re.split(r"^# %%", source, flags=re.M)
    if s.strip() and not s.startswith(" [markdown]")
]
namespace = {"__name__": "__main__"}
exec(cells[0], namespace)
capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    exec(cells[1], namespace)
result = json.loads((ROOT / "results" / "task4.json").read_text())
result["checks"].update(namespace["checks"])
(ROOT / "results" / "task4.json").write_text(json.dumps(result, indent=2))
path = ROOT / "notebooks" / "task4.ipynb"
nb = nbformat.read(path, as_version=4)
code_cells = [c for c in nb.cells if c.cell_type == "code"]
code_cells[1].source = cells[1]
code_cells[1].outputs = [
    nbformat.v4.new_output("stream", name="stdout", text=capture.getvalue())
]
nbformat.write(nb, path)
print(capture.getvalue())
