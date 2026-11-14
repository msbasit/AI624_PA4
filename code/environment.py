"""Save the actual hardware and package versions used for the experiments."""

import importlib.metadata
import json
import platform
import sys
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1]
packages = [
    "torch",
    "torchvision",
    "transformers",
    "datasets",
    "peft",
    "numpy",
    "scipy",
    "matplotlib",
    "nbformat",
    "nats-bench",
    "reportlab",
    "pymupdf",
    "requests",
    "ipython",
    "ipykernel",
    "black",
]
versions = {name: importlib.metadata.version(name) for name in packages}
result = {
    "python": sys.version,
    "platform": platform.platform(),
    "gpu": torch.xpu.get_device_name(),
    "xpu_available": torch.xpu.is_available(),
    "versions": versions,
    "cpu_threads": 8,
    "precision": {
        "T1": "FP32; cached hint tensors stored FP16",
        "T2": "BF16 backbone; PEFT adapter parameters FP32",
        "T3": "FP32",
        "T4": "FP32",
    },
    "latency_protocol": "Batch 1, 10 warmups, 20 timed forwards, XPU synchronized, median, excludes loading",
    "timing_note": "Training-loop wall times include epoch validation; downloads and data preparation excluded.",
}
(root / "results" / "environment.json").write_text(json.dumps(result, indent=2))
(root / "requirements.txt").write_text(
    "# Install the XPU build of torch and torchvision first; see README.\n"
    + "\n".join(
        f"{n}=={v}" for n, v in versions.items() if n not in ["torch", "torchvision"]
    )
    + "\n"
)
print(json.dumps(result, indent=2))
