# %% [markdown]
# # T4: Uniform weight-sharing NAS
# Four blocks each choose one of four operations, so $|\mathcal S|=4^4=256$.
# GroupNorm avoids path-dependent BatchNorm running statistics.

# %%
from common import *
from scipy.stats import kendalltau, rankdata

seed_all(44)
train, val = cifar_data(44)
OPS = ["conv3x3", "conv5x5", "dwsep3x3", "identity"]

# %% [markdown]
# ## Supernet, path encoding and independent extraction
# A path is a tuple of four operation indices. Extraction deep-copies only the
# selected operations, plus the shared stem and head.


# %%
def operation(index):
    if index == 3:
        return nn.Identity()
    if index == 2:
        conv = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False),
            nn.Conv2d(16, 16, 1, bias=False),
        )
    else:
        k = 3 if index == 0 else 5
        conv = nn.Conv2d(16, 16, k, padding=k // 2, bias=False)
    return nn.Sequential(conv, nn.GroupNorm(4, 16), nn.ReLU())


class Supernet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.ReLU()
        )
        self.blocks = nn.ModuleList(
            [nn.ModuleList([operation(i) for i in range(4)]) for _ in range(4)]
        )
        self.head = nn.Linear(16, 10)

    def forward(self, x, path):
        x = self.stem(x)
        for block, index in zip(self.blocks, path):
            x = block[index](x)
        return self.head(x.mean((2, 3)))


class Subnet(nn.Module):
    def __init__(self, source, path):
        super().__init__()
        self.stem = copy.deepcopy(source.stem)
        self.blocks = nn.ModuleList(
            [copy.deepcopy(block[i]) for block, i in zip(source.blocks, path)]
        )
        self.head = copy.deepcopy(source.head)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x.mean((2, 3)))


model = Supernet().to(DEVICE)
rng = random.Random(44)
path_a = (0, 1, 2, 3)
path_b = (0, 2, 2, 1)
x, y = next(iter(loader(train)))
x, y = x.to(DEVICE), y.to(DEVICE)
checks = {
    "paths": [path_a, path_b],
    "active_operations": [[OPS[i] for i in p] for p in [path_a, path_b]],
    "output_shapes": [list(model(x, p).shape) for p in [path_a, path_b]],
    "shared_blocks": [i for i, (a, b) in enumerate(zip(path_a, path_b)) if a == b],
    "identity_parameters": sum(p.numel() for p in operation(3).parameters()),
    "mappings": [
        [f"blocks.{j}.{i} -> subnet.blocks.{j}" for j, i in enumerate(p)]
        for p in [path_a, path_b]
    ],
}
extracted = Subnet(model, path_a).to(DEVICE)
checks["extraction_matches"] = torch.allclose(extracted(x), model(x, path_a))
checks["extraction_independent_storage"] = (
    extracted.stem[0].weight.data_ptr() != model.stem[0].weight.data_ptr()
)
checks["shared_operation_parameter_storage"] = {
    str(j): [p.data_ptr() for p in model.blocks[j][a].parameters()]
    == [p.data_ptr() for p in model.blocks[j][b].parameters()]
    for j, (a, b) in enumerate(zip(path_a, path_b))
}
print(checks)

# %% [markdown]
# ## Train for five epochs, sampling one path per minibatch
# Setting gradients to None prevents inactive operations from being updated by
# optimizer momentum. We check actual parameter values after an optimizer step.

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
counts = np.zeros((4, 4), dtype=int)
history = []
batches = loader(train, True, 44)
sync()
start = time.perf_counter()
for epoch in range(5):
    model.train()
    total = correct = n = 0
    for step, (x, y) in enumerate(batches):
        path = tuple(rng.randrange(4) for _ in range(4))
        for j, i in enumerate(path):
            counts[j, i] += 1
        before = (
            {k: p.detach().clone() for k, p in model.named_parameters()}
            if epoch == step == 0
            else None
        )
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        z = model(x, path)
        loss = F.cross_entropy(z, y)
        loss.backward()
        optimizer.step()
        if before is not None:
            inactive = []
            active = []
            for key, p in model.named_parameters():
                parts = key.split(".")
                is_active = parts[0] != "blocks" or int(parts[2]) == path[int(parts[1])]
                (active if is_active else inactive).append(
                    not torch.equal(before[key], p)
                )
            checks["inactive_parameters_unchanged"] = not any(inactive)
            checks["active_parameters_changed"] = any(active)
            checks["checked_path"] = path
        total += loss.item() * len(y)
        correct += (z.argmax(1) == y).sum().item()
        n += len(y)
    history.append(
        {
            "epoch": epoch + 1,
            "train": {"loss": total / n, "accuracy": 100 * correct / n},
            "validation": evaluate(model, loader(val), lambda x: model(x, path_a)),
        }
    )
    print(history[-1], flush=True)
sync()
training = {
    "history": history,
    "seconds": time.perf_counter() - start,
    "updates": 5 * len(batches),
}
torch.save(model.state_dict(), ROOT / "checkpoints" / "t4_supernet.pt")

# %% [markdown]
# ## Twelve independent adaptation runs: 0, 10 and 30 updates
# All models start from the same checkpoint. Each gets the same deterministic
# minibatch order and optimizer; step 30 continues from step 10.

# %%
import itertools

paths = rng.sample(list(itertools.product(range(4), repeat=4)), 12)
adaptation = []
for number, path in enumerate(paths):
    net = Subnet(model, path).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    scores = {"0": evaluate(net, loader(val))["accuracy"]}
    sync()
    start = time.perf_counter()
    for step, (x, y) in enumerate(loader(train, True, 45), start=1):
        net.train()
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(net(x), y).backward()
        opt.step()
        if step in [10, 30]:
            scores[str(step)] = evaluate(net, loader(val))["accuracy"]
        if step == 30:
            break
    sync()
    adaptation.append(
        {"path": path, "accuracy": scores, "seconds": time.perf_counter() - start}
    )
    torch.save(net.state_dict(), ROOT / "checkpoints" / f"t4_adapt_{number+1}.pt")
    print(number + 1, adaptation[-1], flush=True)
scores = {s: [r["accuracy"][s] for r in adaptation] for s in ["0", "10", "30"]}
tau = {
    f"{a}_{b}": float(kendalltau(scores[a], scores[b]).statistic)
    for a, b in [("0", "10"), ("0", "30"), ("10", "30")]
}
ranks = {
    s: rankdata(-np.array(v), method="average").tolist() for s, v in scores.items()
}

# %% [markdown]
# ## Train the best adapted path and a different random path from scratch
# Both receive five epochs, Adam at $10^{-3}$, identical seed and data order.

# %%
best = int(np.argmax(scores["30"]))
other = rng.choice([i for i in range(12) if i != best])
scratch_runs = {}
for index in [best, other]:
    seed_all(44)
    net = Subnet(Supernet(), paths[index]).to(DEVICE)
    run = fit(net, train, val, 5, 44)
    run["accuracy"] = evaluate(net, loader(val))["accuracy"]
    scratch_runs[str(index + 1)] = run
    torch.save(net.state_dict(), ROOT / "checkpoints" / f"t4_scratch_{index+1}.pt")
result = {
    "checks": checks,
    "training": training,
    "counts": counts.tolist(),
    "adaptation": adaptation,
    "tau": tau,
    "ranks": ranks,
    "best_index": best + 1,
    "random_index": other + 1,
    "scratch": scratch_runs,
}
save_json("task4", result)
print("Kendall tau-b:", tau, "Selected paths:", best + 1, other + 1)

# %%
curves(
    {
        "Supernet (fixed validation path)": training,
        **{f"Scratch path {k}": v for k, v in scratch_runs.items()},
    },
    "t4_training",
)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for i, row in enumerate(adaptation):
    color = plt.get_cmap("tab20")(i)
    axes[0].plot(
        [0, 10, 30],
        [row["accuracy"][s] for s in ["0", "10", "30"]],
        marker="o",
        color=color,
        label=str(i + 1),
    )
    axes[1].plot(
        [0, 10, 30], [ranks[s][i] for s in ["0", "10", "30"]], marker="o", color=color
    )
axes[0].set(xlabel="Adaptation updates", ylabel="Validation accuracy (%)")
axes[0].legend(ncol=3, fontsize=7)
axes[1].set(xlabel="Adaptation updates", ylabel="Rank (1 = best)")
axes[1].invert_yaxis()
fig.tight_layout()
fig.savefig(ROOT / "figures" / "t4_adaptation.png", dpi=150)
plt.close(fig)
fig, ax = plt.subplots(figsize=(6, 3))
im = ax.imshow(counts, cmap="Blues")
ax.set(
    xticks=range(4),
    xticklabels=OPS,
    yticks=range(4),
    yticklabels=range(1, 5),
    xlabel="Operation",
    ylabel="Block",
)
for j in range(4):
    for i in range(4):
        ax.text(i, j, str(counts[j, i]), ha="center", va="center")
fig.colorbar(im, ax=ax, label="Sampled minibatches")
ax.tick_params(axis="x", labelsize=8)
plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "t4_sampling.png", dpi=150)
plt.close(fig)
