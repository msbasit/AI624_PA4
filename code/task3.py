# %% [markdown]
# # T3: First-order differentiable architecture search
# Six edges connect all pairs $i<j$ in a four-node DAG. Each edge computes
# $\bar o(x)=\sum_o\mathrm{softmax}(\alpha)_o o(x)$, and each node sums its inputs.
# The two cells share $\alpha\in\mathbb{R}^{6\times5}$ but have separate weights.

# %%
from common import *

seed_all(43)
train, val = cifar_data(43)
EDGES = [(i, j) for j in range(1, 4) for i in range(j)]
OPS = ["none", "skip", "conv1x1", "conv3x3", "avgpool3x3"]

# %% [markdown]
# ## Operations and continuous network
# GroupNorm has no running statistics, so freezing weights during architecture
# updates also avoids any hidden BatchNorm-buffer updates.


# %%
class Zero(nn.Module):
    def forward(self, x):
        return x * 0


def operation(index):
    if index == 0:
        return Zero()
    if index == 1:
        return nn.Identity()
    if index == 4:
        return nn.AvgPool2d(3, 1, 1, count_include_pad=False)
    kernel = 1 if index == 2 else 3
    return nn.Sequential(
        nn.Conv2d(16, 16, kernel, padding=kernel // 2, bias=False),
        nn.GroupNorm(4, 16),
        nn.ReLU(),
    )


class Cell(nn.Module):
    def __init__(self):
        super().__init__()
        self.edges = nn.ModuleList(
            [nn.ModuleList([operation(k) for k in range(5)]) for _ in EDGES]
        )

    def forward(self, x, prob):
        nodes = [x]
        for j in range(1, 4):
            terms = []
            for edge, (i, dest) in enumerate(EDGES):
                if dest == j:
                    terms.append(
                        sum(
                            p * op(nodes[i])
                            for p, op in zip(prob[edge], self.edges[edge])
                        )
                    )
            nodes.append(sum(terms))
        return nodes[3]


class Supernet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        self.cells = nn.ModuleList([Cell(), Cell()])
        self.head = nn.Linear(16, 10)
        self.alpha = nn.Parameter(torch.zeros(6, 5))

    def forward(self, x):
        x = self.stem(x)
        for cell in self.cells:
            x = cell(x, self.alpha.softmax(-1))
        return self.head(x.mean((2, 3)))


model = Supernet().to(DEVICE)
weights = [p for n, p in model.named_parameters() if n != "alpha"]
x, y = next(iter(loader(train)))
x, y = x.to(DEVICE), y.to(DEVICE)
before = model(x).detach()
F.cross_entropy(model(x), y).backward()
checks = {
    "output_shape": list(before.shape),
    "alpha_shape": list(model.alpha.shape),
    "probability_sums": model.alpha.softmax(-1).sum(-1).tolist(),
    "alpha_gradient_nonzero": bool(model.alpha.grad.abs().sum() > 0),
    "separate_cell_weights": model.cells[0].edges[0][2][0].weight.data_ptr()
    != model.cells[1].edges[0][2][0].weight.data_ptr(),
}
with torch.no_grad():
    model.alpha[:, 1] = 2
checks["alpha_changes_output"] = not torch.allclose(before, model(x))
with torch.no_grad():
    model.alpha.zero_()
model.zero_grad(set_to_none=True)
print(checks)

# %% [markdown]
# ## Alternating first-order updates: three epochs
# Training batches update only $w$. Validation batches update only $\alpha$,
# without unrolling or differentiating through the weight optimizer.

# %%
wopt = torch.optim.Adam(weights, lr=0.001)
aopt = torch.optim.Adam([model.alpha], lr=0.003)
history = []
probabilities = [model.alpha.softmax(-1).detach().cpu().tolist()]
tr_loader = loader(train, True, 43)
va_loader = loader(val, True, 44)
sync()
start = time.perf_counter()
for epoch in range(3):
    model.train()
    va_iter = iter(va_loader)
    total = correct = count = 0
    for step, (x, y) in enumerate(tr_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        alpha_before = model.alpha.detach().clone()
        w_before = [p.detach().clone() for p in weights] if epoch == step == 0 else None
        model.alpha.requires_grad_(False)
        wopt.zero_grad(set_to_none=True)
        z = model(x)
        loss = F.cross_entropy(z, y)
        loss.backward()
        wopt.step()
        if epoch == step == 0:
            checks["weight_step_alpha_unchanged"] = torch.equal(
                alpha_before, model.alpha
            )
            checks["weight_step_weights_changed"] = any(
                not torch.equal(a, b) for a, b in zip(w_before, weights)
            )
        total += loss.item() * len(y)
        correct += (z.argmax(1) == y).sum().item()
        count += len(y)
        try:
            vx, vy = next(va_iter)
        except StopIteration:
            va_iter = iter(va_loader)
            vx, vy = next(va_iter)
        for p in weights:
            p.requires_grad_(False)
        model.alpha.requires_grad_(True)
        w_before = [p.detach().clone() for p in weights] if epoch == step == 0 else None
        aopt.zero_grad(set_to_none=True)
        F.cross_entropy(model(vx.to(DEVICE)), vy.to(DEVICE)).backward()
        aopt.step()
        if epoch == step == 0:
            checks["alpha_step_weights_unchanged"] = all(
                torch.equal(a, b) for a, b in zip(w_before, weights)
            )
            checks["alpha_step_alpha_changed"] = not torch.equal(
                alpha_before, model.alpha
            )
        for p in weights:
            p.requires_grad_(True)
    history.append(
        {
            "epoch": epoch + 1,
            "train": {"loss": total / count, "accuracy": 100 * correct / count},
            "validation": evaluate(model, loader(val)),
        }
    )
    probabilities.append(model.alpha.softmax(-1).detach().cpu().tolist())
    print(history[-1], flush=True)
sync()
search = {
    "history": history,
    "seconds": time.perf_counter() - start,
    "weight_updates": 3 * len(tr_loader),
    "alpha_updates": 3 * len(tr_loader),
}
torch.save(model.state_dict(), ROOT / "checkpoints" / "t3_supernet.pt")

# %% [markdown]
# ## Discretize the same checkpoint, then train a fresh copy
# For each node retain its two strongest incoming edges (one for node 1),
# ranked by maximum non-zero probability. No adaptation precedes evaluation.

# %%
prob = model.alpha.softmax(-1).detach().cpu()
encoding = []
for j in range(1, 4):
    incoming = [e for e, (i, d) in enumerate(EDGES) if d == j]
    retained = sorted(incoming, key=lambda e: float(prob[e, 1:].max()), reverse=True)[
        :2
    ]
    encoding.extend((e, int(prob[e, 1:].argmax()) + 1) for e in retained)
assert len(encoding) == 5


class Discrete(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.stem = copy.deepcopy(source.stem)
        self.cells = nn.ModuleList(
            [
                nn.ModuleDict(
                    {str(e): copy.deepcopy(c.edges[e][o]) for e, o in encoding}
                )
                for c in source.cells
            ]
        )
        self.head = copy.deepcopy(source.head)

    def forward(self, x):
        x = self.stem(x)
        for cell in self.cells:
            nodes = [x]
            for j in range(1, 4):
                nodes.append(
                    sum(
                        cell[str(e)](nodes[EDGES[e][0]])
                        for e, o in encoding
                        if EDGES[e][1] == j
                    )
                )
            x = nodes[3]
        return self.head(x.mean((2, 3)))


inherited = Discrete(model).to(DEVICE)
comparison = {}
for name, net in [("continuous", model), ("inherited", inherited)]:
    comparison[name] = {**evaluate(net, loader(val)), **profile(net)}
seed_all(43)
scratch = Discrete(Supernet()).to(DEVICE)
scratch_run = fit(scratch, train, val, 5, 43)
comparison["scratch"] = {**evaluate(scratch, loader(val)), **profile(scratch)}
torch.save(scratch.state_dict(), ROOT / "checkpoints" / "t3_scratch.pt")
result = {
    "checks": checks,
    "search": search,
    "probabilities": probabilities,
    "edges": EDGES,
    "operations": OPS,
    "encoding": encoding,
    "comparison": comparison,
    "scratch": scratch_run,
}
save_json("task3", result)
print(comparison)

# %% [markdown]
# ## Curves and cell graphs

# %%
curves({"Search": search, "Discrete scratch": scratch_run}, "t3_training")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, e in zip(axes, [0, 5]):
    for k, op in enumerate(OPS):
        ax.plot(range(4), np.array(probabilities)[:, e, k], label=op)
    ax.set(
        title=f"Edge {EDGES[e]}", xlabel="Search epoch", ylabel="Operation probability"
    )
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "t3_probabilities.png", dpi=150)
plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
positions = {0: (0, 0), 1: (1, 1), 2: (2, 1), 3: (3, 0)}
for ax, title, graph in [
    (axes[0], "Continuous cell", [(e, None) for e in range(6)]),
    (axes[1], "Extracted cell", encoding),
]:
    for e, o in graph:
        i, j = EDGES[e]
        a, b = positions[i], positions[j]
        ax.annotate(
            "",
            xy=b,
            xytext=a,
            arrowprops={
                "arrowstyle": "->",
                "connectionstyle": "arc3,rad=0.12",
                "shrinkA": 12,
                "shrinkB": 12,
            },
        )
        ax.text(
            (a[0] + b[0]) / 2,
            (a[1] + b[1]) / 2 - 0.12,
            "mixed" if o is None else OPS[o],
            fontsize=7,
        )
    for n, (a, b) in positions.items():
        ax.text(
            a,
            b,
            str(n),
            bbox={"boxstyle": "circle", "facecolor": "lightblue"},
            ha="center",
            va="center",
        )
    ax.set(xlim=(-0.3, 3.3), ylim=(-0.6, 1.4), title=title)
    ax.axis("off")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "t3_graphs.png", dpi=150)
plt.close(fig)
