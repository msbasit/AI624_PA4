"""Reduce the official benchmark to validation lookup values for reproducibility."""

import json
from pathlib import Path
from nats_bench import create

ROOT = Path(__file__).resolve().parents[1]
print("Loading official benchmark metadata...", flush=True)
api = create(
    str(ROOT / "data" / "NATS-tss-v1_0-3ffb9.pickle.pbz2"),
    "tss",
    fast_mode=False,
    verbose=False,
)
lookup = {}
for i in range(len(api)):
    lookup[api.arch(i)] = api.get_more_info(
        i, "cifar10-valid", hp="200", is_random=False
    )["valid-accuracy"]
    if i % 1000 == 0:
        print("Extracted", i, flush=True)
(ROOT / "data" / "nats_validation.json").write_text(
    json.dumps(lookup), encoding="utf-8"
)
print("Saved", len(lookup), "official benchmark validation scores.", flush=True)
