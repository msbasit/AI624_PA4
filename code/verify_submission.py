"""Check assignment invariants and deliverable consistency after execution."""

import json
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / "results" / f"{name}.json").read_text())


checks = {}
for task, seed in [(3, 43), (4, 44)]:
    split = read(f"cifar_split_{seed}")
    assert len(split["train"]) == 5000 and len(split["validation"]) == 1000
    assert not set(split["train"]) & set(split["validation"])
    checks[f"T{task}_split_sizes_disjoint"] = True
t1 = read("task1")
t2 = read("task2")
t3 = read("task3")
t4 = read("task4")
t5 = read("task5")
assert len({r["metrics"]["parameters"] for r in t1["runs"].values()}) == 1
assert (
    t1["runs"]["direct_hint"]["updates"] == t1["runs"]["adapted_hint"]["updates"] == 64
)
assert all(len(r["metrics"]["timings_ms"]) == 20 for r in t1["runs"].values())
checks["T1_equal_deployment_parameters_and_hint_budgets"] = True
assert t2["sampled"] == 1000 and t2["training_examples"] > 0
records = json.loads((ROOT / "data" / "teacher_rationales.json").read_text())
assert len(records) == 1000
assert sum(r["valid"] for r in records) == t2["accepted"]
assert all(r["prediction"] == r["reference"] for r in records if r["valid"])
for label in ["original", "finetuned"]:
    predictions = read(f"t2_{label}_predictions")
    assert len(predictions) == 300
    assert sum(r["set"] == "gsm8k" for r in predictions) == 200
    assert sum(r["set"] == "synthetic" for r in predictions) == 100
    for domain in ["gsm8k", "synthetic"]:
        subset = [r for r in predictions if r["set"] == domain]
        measured = 100 * sum(r["correct"] for r in subset) / len(subset)
        assert abs(measured - t2[label][domain]) < 1e-8
checks["T2_teacher_filter_and_held_out_counts"] = True
assert len(t3["encoding"]) == 5
assert all(
    t3["checks"][key]
    for key in [
        "weight_step_alpha_unchanged",
        "weight_step_weights_changed",
        "alpha_step_weights_unchanged",
        "alpha_step_alpha_changed",
    ]
)
assert t3["search"]["weight_updates"] == t3["search"]["alpha_updates"] == 237
assert t3["scratch"]["updates"] == 395
assert all(len(v["timings_ms"]) == 20 for v in t3["comparison"].values())
checks["T3_search_separation_and_budgets"] = True
assert len(t4["adaptation"]) == len({tuple(r["path"]) for r in t4["adaptation"]}) == 12
assert all(sum(row) == 395 for row in t4["counts"])
assert t4["checks"]["inactive_parameters_unchanged"]
assert all(set(r["accuracy"]) == {"0", "10", "30"} for r in t4["adaptation"])
assert all(r["updates"] == 395 for r in t4["scratch"].values())
checks["T4_path_sampling_independence_and_budgets"] = True
for method, runs in t5["runs"].items():
    assert len(runs) == 3
    for seed, history in runs.items():
        assert len(history) == 100
        assert all(
            a["best_ever"] <= b["best_ever"] for a, b in zip(history, history[1:])
        )
        if method == "aging_evolution":
            assert all(
                r["population_size"] == min(r["evaluation"], 10) for r in history
            )
checks["T5_equal_budgets_and_population_checks"] = True
report = (ROOT / "results.md").read_text(encoding="utf-8")
assert "Muhammad Sami Ul Basit" in report and "25280025" in report
assert "**Student ID:**" not in report
for task in range(1, 6):
    nb = nbformat.read(ROOT / "notebooks" / f"task{task}.ipynb", as_version=4)
    nbformat.validate(nb)
    assert len([c for c in nb.cells if c.cell_type == "code"]) >= 4
    assert any(c.outputs for c in nb.cells if c.cell_type == "code")
    assert not any(
        o.output_type == "error"
        for c in nb.cells
        if c.cell_type == "code"
        for o in c.outputs
    )
checks["all_notebooks_have_visible_outputs_and_no_errors"] = True
(ROOT / "results" / "verification.json").write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))
