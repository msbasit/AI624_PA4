# %% [markdown]
# # Final inference timing: synchronized, interleaved comparisons
# Initial per-run latency varied considerably on the shared integrated GPU.
# Re-measure saved models together after all training. Each model receives ten
# warm-ups and twenty timed batch-one forwards; loading is outside timing.

# %%
import ast
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision.models import efficientnet_b0, efficientnet_b7, EfficientNet_B7_Weights
import common
from common import ROOT, DEVICE, sync, save_json

t1 = json.loads((ROOT / "results" / "task1.json").read_text())
t3 = json.loads((ROOT / "results" / "task3.json").read_text())
original = {
    "task1": {k: v["metrics"]["timings_ms"] for k, v in t1["runs"].items()},
    "teacher": t1["teacher"]["timings_ms"],
    "task3": {k: v["timings_ms"] for k, v in t3["comparison"].items()},
}
save_json("initial_latency_samples", original)

# %% [markdown]
# ## Load trained CNNs and the frozen teacher

# %%
cnn = {}
for name in t1["runs"]:
    net = efficientnet_b0(weights=None)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, 100)
    net.load_state_dict(
        torch.load(
            ROOT / "checkpoints" / f"t1_{name}.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    cnn[name] = net.to(DEVICE).eval()
cnn["teacher"] = (
    efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT).to(DEVICE).eval()
)

# %% [markdown]
# ## Timed round-robin measurements
# A seeded random order in each round spreads transient system effects across
# models. Report median and retain all twenty measurements, without removing outliers.


# %%
@torch.no_grad()
def measure_group(models, shape):
    x = torch.randn(shape, device=DEVICE)
    for _ in range(10):
        for model in models.values():
            model(x)
    sync()
    rng = np.random.default_rng(42)
    samples = {name: [] for name in models}
    for _ in range(20):
        for name in rng.permutation(list(models)):
            sync()
            start = time.perf_counter()
            models[name](x)
            sync()
            samples[name].append(1000 * (time.perf_counter() - start))
    return {
        name: {"latency_ms": float(np.median(v)), "timings_ms": v}
        for name, v in samples.items()
    }


cnn_times = measure_group(cnn, (1, 3, 224, 224))
for name, record in cnn_times.items():
    (t1["teacher"] if name == "teacher" else t1["runs"][name]["metrics"]).update(record)
save_json("task1", t1)
print({k: v["latency_ms"] for k, v in cnn_times.items()})
del cnn, net
torch.xpu.empty_cache()

# %% [markdown]
# ## Load the continuous and discrete DARTS networks without retraining

# %%
namespace = dict(vars(common))
namespace.update(EDGES=t3["edges"], encoding=t3["encoding"])
tree = ast.parse((ROOT / "code" / "task3.py").read_text())
definitions = ast.Module(
    body=[n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))],
    type_ignores=[],
)
exec(compile(definitions, "task3_definitions", "exec"), namespace)
Supernet, Discrete = namespace["Supernet"], namespace["Discrete"]
continuous = Supernet().to(DEVICE)
continuous.load_state_dict(
    torch.load(
        ROOT / "checkpoints" / "t3_supernet.pt", map_location=DEVICE, weights_only=True
    )
)
inherited = Discrete(continuous).to(DEVICE)
scratch = Discrete(Supernet()).to(DEVICE)
scratch.load_state_dict(
    torch.load(
        ROOT / "checkpoints" / "t3_scratch.pt", map_location=DEVICE, weights_only=True
    )
)
nas = {
    name: net.eval()
    for name, net in [
        ("continuous", continuous),
        ("inherited", inherited),
        ("scratch", scratch),
    ]
}
nas_times = measure_group(nas, (1, 3, 32, 32))
for name, record in nas_times.items():
    t3["comparison"][name].update(record)
save_json("task3", t3)
save_json(
    "final_latency_protocol",
    {
        "warmups_per_model": 10,
        "timed_forwards_per_model": 20,
        "batch_size": 1,
        "synchronized": True,
        "order": "Seed-42 shuffled round robin within each task",
        "precision": "FP32",
        "cnn": cnn_times,
        "darts": nas_times,
    },
)
print({k: v["latency_ms"] for k, v in nas_times.items()})
