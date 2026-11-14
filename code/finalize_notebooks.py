"""Embed the saved plots and concise analysis in each executed notebook."""

import base64
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
plots = {
    1: ["t1_training", "t1_hint"],
    2: ["t2_loss"],
    3: ["t3_training", "t3_probabilities", "t3_graphs"],
    4: ["t4_training", "t4_sampling", "t4_adaptation"],
    5: ["t5_search"],
}
setup = """from pathlib import Path
import sys
project_root = Path.cwd()
if project_root.name == "notebooks":
    project_root = project_root.parent
sys.path.insert(0, str(project_root / "code"))"""
for task, names in plots.items():
    path = ROOT / "notebooks" / f"task{task}.ipynb"
    nb = nbformat.read(path, as_version=4)
    # Unexecuted path-only setup cell makes a fresh interactive kernel portable.
    if not any("project_root = Path.cwd()" in c.source for c in nb.cells):
        nb.cells.insert(1, nbformat.v4.new_code_cell(setup))
    nb.cells = [c for c in nb.cells if not c.metadata.get("report_figures")]
    if task in [3, 4]:
        # Keep plot-only source aligned with the reviewed graph/label layout.
        plot_source = (
            (ROOT / "code" / f"task{task}.py").read_text().rsplit("# %%", 1)[-1].strip()
        )
        marker = "t3_graphs.png" if task == 3 else "t4_sampling.png"
        for cell in nb.cells:
            if cell.cell_type == "code" and marker in cell.source:
                cell.source = plot_source
    if task in [1, 3]:
        note = nbformat.v4.new_markdown_cell(
            "**Latency note:** Timings printed in the training cells are provisional. The final report and analysis below use the controlled comparison in `reprofile.ipynb`, which reloads the saved checkpoints and times them together after training."
        )
        note.metadata["report_figures"] = True
        nb.cells.insert(2, note)
    output = []
    for name in names:
        content = base64.b64encode(
            (ROOT / "figures" / f"{name}.png").read_bytes()
        ).decode()
        output.append(
            nbformat.v4.new_output(
                "display_data",
                data={"image/png": content, "text/plain": name},
                metadata={},
            )
        )
    cell = nbformat.v4.new_code_cell(
        "from IPython.display import Image, display\n"
        + "\n".join(
            f'display(Image(filename=str(project_root / "figures" / "{name}.png")))'
            for name in names
        ),
        outputs=output,
    )
    cell.metadata["report_figures"] = True
    nb.cells.append(cell)
    # The exact task analysis from the final report also remains visible inline.
    import re

    report = (ROOT / "results.md").read_text(encoding="utf-8")
    sections = re.findall(rf"## T{task}\.\d:.*?(?=\n## |\Z)", report, flags=re.S)
    analysis = nbformat.v4.new_markdown_cell(
        ("# Analysis\n\n" + "\n\n".join(sections)).replace("(figures/", "(../figures/")
    )
    analysis.metadata["report_figures"] = True
    nb.cells.append(analysis)
    # Formatting only: preserve the executed computations and saved outputs.
    try:
        import black

        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.source = black.format_str(
                    cell.source, mode=black.Mode(line_length=88)
                )
    except ImportError:
        pass
    nbformat.validate(nb)
    assert not any(
        o.output_type == "error"
        for c in nb.cells
        if c.cell_type == "code"
        for o in c.outputs
    )
    nbformat.write(nb, path)
    print(path.name, "cells:", len(nb.cells))

path = ROOT / "notebooks" / "reprofile.ipynb"
nb = nbformat.read(path, as_version=4)
if not any("project_root = Path.cwd()" in c.source for c in nb.cells):
    nb.cells.insert(1, nbformat.v4.new_code_cell(setup))
for cell in nb.cells:
    if cell.cell_type == "code":
        try:
            import black

            cell.source = black.format_str(cell.source, mode=black.Mode(line_length=88))
        except ImportError:
            pass
nbformat.validate(nb)
nbformat.write(nb, path)
