"""Fill the two short-class gaps in the systematic ImageNet subset."""

import io
import json
from pathlib import Path
import numpy as np
import requests
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def balance():
    path = ROOT / "data" / "imagenet100_train.pt"
    data = torch.load(path, weights_only=True)
    labels = data["labels"]
    rows = [
        r
        for p in (ROOT / "tmp").glob("imagenet_rows_train_*.json")
        for r in json.loads(p.read_text())["rows"]
    ]
    for missing in sorted(set(range(100)) - set(labels.tolist())):
        lower = max(r["row_idx"] for r in rows if r["row"]["label"] < missing)
        upper = min(r["row_idx"] for r in rows if r["row"]["label"] > missing)
        offset = (lower + upper) // 2
        r = requests.get(
            "https://datasets-server.huggingface.co/rows",
            params={
                "dataset": "clane9/imagenet-100",
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": 5,
            },
            timeout=90,
        )
        r.raise_for_status()
        selected = r.json()["rows"]
        assert all(row["row"]["label"] == missing for row in selected)
        for row in selected:
            counts = torch.bincount(labels, minlength=100)
            donor = int(torch.nonzero(counts > 5)[0])
            index = int(torch.nonzero(labels == donor)[-1])
            response = requests.get(row["row"]["image"]["src"], timeout=90)
            response.raise_for_status()
            image = (
                Image.open(io.BytesIO(response.content))
                .convert("RGB")
                .resize((224, 224))
            )
            data["images"][index] = torch.from_numpy(np.array(image)).permute(2, 0, 1)
            labels[index] = missing
            data["indices"][index] = row["row_idx"]
        print(
            "Filled class",
            missing,
            "from source rows",
            offset,
            "through",
            offset + 4,
            flush=True,
        )
    assert torch.equal(torch.bincount(labels, minlength=100), torch.full((100,), 5))
    torch.save(data, path)
    print("Training subset: exactly five images per class.")


if __name__ == "__main__":
    balance()
