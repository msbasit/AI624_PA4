"""Download a fixed systematic ImageNet-100 subset using the public viewer API.

Only the selected thumbnails are fetched. Their source row indices are saved.
The source already resizes the short side to 160 pixels; training resizes to 224.
"""

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import requests
import torch
from PIL import Image
from torchvision.models import EfficientNet_B7_Weights

ROOT = Path(__file__).resolve().parents[1]


def request(url, **kwargs):
    for attempt in range(6):
        try:
            r = requests.get(url, timeout=90, **kwargs)
            r.raise_for_status()
            return r
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2**attempt)


def row(split, index):
    return request(
        "https://datasets-server.huggingface.co/rows",
        params={
            "dataset": "clane9/imagenet-100",
            "config": "default",
            "split": split,
            "offset": int(index),
            "length": 1,
        },
    ).json()


metadata = row("train", 0)
names = metadata["features"][1]["type"]["names"]
categories = EfficientNet_B7_Weights.DEFAULT.meta["categories"]
mapping = []
for name in names:
    choices = [
        i for i, c in enumerate(categories) if c == name or c == name.split(",")[0]
    ]
    assert len(choices) == 1, (name, choices)
    mapping.append(choices[0])
(ROOT / "results" / "imagenet_class_mapping.json").write_text(
    json.dumps(
        [
            {"label": i, "name": n, "imagenet1k_index": m}
            for i, (n, m) in enumerate(zip(names, mapping))
        ],
        indent=2,
    )
)
for split, n, total in [("train", 500, 126689), ("validation", 200, 5000)]:
    path = ROOT / "data" / f"imagenet100_{split}.pt"
    if (
        path.exists()
        and len(torch.load(path, weights_only=True)["labels"].unique()) >= 90
    ):
        continue
    # Systematic blocks span the entire class-ordered split instead of selecting
    # only a few contiguous blocks. Cache responses and respect rate limits.
    rows = []
    starts = np.linspace(0, total - n // 100, 100, dtype=int)
    for label, start in enumerate(starts):
        cached = ROOT / "tmp" / f"imagenet_rows_{split}_{label}.json"
        if cached.exists():
            response = json.loads(cached.read_text())
        else:
            response = request(
                "https://datasets-server.huggingface.co/rows",
                params={
                    "dataset": "clane9/imagenet-100",
                    "config": "default",
                    "split": split,
                    "offset": int(start),
                    "length": n // 100,
                },
            ).json()
            cached.write_text(json.dumps(response))
            time.sleep(3)
        rows.extend(response["rows"])
        if label % 10 == 0:
            print(split, "class", label, flush=True)
    indices = [r["row_idx"] for r in rows]

    def get(record):
        data = record["row"]
        im = (
            Image.open(io.BytesIO(request(data["image"]["src"]).content))
            .convert("RGB")
            .resize((224, 224))
        )
        return torch.from_numpy(np.array(im)).permute(2, 0, 1), data["label"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(get, rows))
    torch.save(
        {
            "images": torch.stack([x for x, y in records]),
            "labels": torch.tensor([y for x, y in records]),
            "indices": indices,
        },
        path,
    )
    print(split, len(records), "classes", len(set(y for x, y in records)), flush=True)

from balance_imagenet import balance

balance()
