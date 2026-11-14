"""Shared reproducibility, data, measurement and plotting utilities."""

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("xpu" if torch.xpu.is_available() else "cpu")
torch.set_num_threads(8)


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "xpu":
        torch.xpu.manual_seed_all(seed)


def sync():
    if DEVICE.type == "xpu":
        torch.xpu.synchronize()


def save_json(name, obj):
    path = ROOT / "results" / (name + ".json")
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def cifar_data(seed):
    """Fixed disjoint stratified 500/class train and 100/class validation."""
    from datasets import load_dataset

    cache = ROOT / "data" / "cifar_train.pt"
    if cache.exists():
        images, labels = torch.load(cache, weights_only=True)
    else:
        ds = load_dataset("uoft-cs/cifar10", split="train")
        images = torch.from_numpy(np.stack([np.array(row["img"]) for row in ds]))
        labels = torch.tensor(ds["label"])
        torch.save((images, labels), cache)
    rng = np.random.default_rng(seed)
    train_ids, val_ids = [], []
    for c in range(10):
        ids = rng.permutation(np.flatnonzero(labels.numpy() == c))
        train_ids.extend(ids[:500].tolist())
        val_ids.extend(ids[500:600].tolist())
    save_json(f"cifar_split_{seed}", {"train": train_ids, "validation": val_ids})
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.247, 0.2435, 0.2616]).view(1, 3, 1, 1)

    def subset(ids):
        x = images[ids].permute(0, 3, 1, 2).float() / 255
        return TensorDataset((x - mean) / std, labels[ids])

    return subset(train_ids), subset(val_ids)


def loader(ds, shuffle=False, seed=42, batch=64):
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.no_grad()
def evaluate(model, batches, forward=None):
    model.eval()
    loss, correct, count = 0.0, 0, 0
    for x, y in batches:
        x, y = x.to(DEVICE), y.to(DEVICE)
        z = model(x) if forward is None else forward(x)
        loss += F.cross_entropy(z, y, reduction="sum").item()
        correct += (z.argmax(1) == y).sum().item()
        count += len(y)
    return {"loss": loss / count, "accuracy": 100 * correct / count}


def train_epoch(model, batches, optimizer, forward=None):
    model.train()
    total, correct, count = 0.0, 0, 0
    for x, y in batches:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        z = model(x) if forward is None else forward(x)
        loss = F.cross_entropy(z, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        correct += (z.argmax(1) == y).sum().item()
        count += len(y)
    return {"loss": total / count, "accuracy": 100 * correct / count}


def fit(model, train, val, epochs=5, seed=42):
    batches = loader(train, True, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    sync()
    start = time.perf_counter()
    for epoch in range(epochs):
        tr = train_epoch(model, batches, optimizer)
        va = evaluate(model, loader(val))
        history.append({"epoch": epoch + 1, "train": tr, "validation": va})
        print(history[-1], flush=True)
    sync()
    return {
        "history": history,
        "seconds": time.perf_counter() - start,
        "updates": epochs * len(batches),
    }


@torch.no_grad()
def profile(model, shape=(1, 3, 32, 32)):
    """Measured FP32 latency; analytical Conv/Linear MACs (other ops excluded)."""
    model.eval()
    x = torch.randn(shape, device=DEVICE)
    macs = [0]
    handles = []

    def hook(layer, args, out):
        if isinstance(layer, nn.Conv2d):
            macs[0] += (
                out.numel()
                * (layer.in_channels // layer.groups)
                * np.prod(layer.kernel_size)
            )
        elif isinstance(layer, nn.Linear):
            macs[0] += out.numel() * layer.in_features

    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            handles.append(layer.register_forward_hook(hook))
    model(x)
    for h in handles:
        h.remove()
    for _ in range(10):
        model(x)
    sync()
    timings = []
    for _ in range(20):
        start = time.perf_counter()
        model(x)
        sync()
        timings.append(1000 * (time.perf_counter() - start))
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "macs": int(macs[0]),
        "size_mib": sum(
            t.numel() * t.element_size() for t in model.state_dict().values()
        )
        / 2**20,
        "latency_ms": float(np.median(timings)),
        "timings_ms": timings,
    }


def curves(runs, name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for label, run in runs.items():
        h = run["history"]
        axes[0].plot(
            [r["epoch"] for r in h], [r["train"]["loss"] for r in h], label=label
        )
        axes[1].plot(
            [r["epoch"] for r in h],
            [r["validation"]["accuracy"] for r in h],
            label=label,
        )
    axes[0].set(xlabel="Epoch", ylabel="Training loss")
    axes[1].set(xlabel="Epoch", ylabel="Validation accuracy (%)")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"{name}.png", dpi=150)
    plt.close(fig)
