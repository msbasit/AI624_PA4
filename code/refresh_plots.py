"""Replot from the saved measurements without rerunning experiments."""

from common import *

t3 = json.loads((ROOT / "results" / "task3.json").read_text())
t4 = json.loads((ROOT / "results" / "task4.json").read_text())
namespace = dict(
    globals(),
    EDGES=t3["edges"],
    OPS=t3["operations"],
    encoding=t3["encoding"],
    probabilities=t3["probabilities"],
    search=t3["search"],
    scratch_run=t3["scratch"],
)
source = (ROOT / "code" / "task3.py").read_text()
exec(source[source.rfind("# %%\n") + 5 :], namespace)
namespace = dict(
    globals(),
    OPS=["conv3x3", "conv5x5", "dwsep3x3", "identity"],
    training=t4["training"],
    scratch_runs=t4["scratch"],
    adaptation=t4["adaptation"],
    ranks=t4["ranks"],
    counts=np.array(t4["counts"]),
)
source = (ROOT / "code" / "task4.py").read_text()
exec(source[source.rfind("# %%\n") + 5 :], namespace)
print("Regenerated T3 and T4 plots from saved results.")
