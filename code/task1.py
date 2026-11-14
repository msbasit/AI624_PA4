# %% [markdown]
# # T1: EfficientNet knowledge distillation on ImageNet-100
# Identical random B0 initialization and minibatch order are used for CE and KD.
# The frozen B7 teacher uses the recorded 100-class ImageNet-1K mapping.
# $L_{KD}=(1-\lambda)L_{CE}+\lambda T^2 KL(p_T(T)\Vert p_S(T))$,
# with $T=3$, $\lambda=0.5$. Hint variants add $0.05L_{hint}$.

# %%
from common import *
from torchvision.models import efficientnet_b0, efficientnet_b7, EfficientNet_B7_Weights

seed_all(42)
mapping = json.loads((ROOT / "results" / "imagenet_class_mapping.json").read_text())
class_indices = torch.tensor([r["imagenet1k_index"] for r in mapping], device=DEVICE)


def data(split):
    raw = torch.load(ROOT / "data" / f"imagenet100_{split}.pt", weights_only=True)
    images = raw["images"].float() / 255
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return TensorDataset((images - mean) / std, raw["labels"])


train, val = data("train"), data("validation")
print("Training/validation:", len(train), len(val), "Device:", DEVICE)
teacher = efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT).to(DEVICE).eval()
teacher.requires_grad_(False)

# %% [markdown]
# ## Cache deterministic frozen teacher outputs and verify feature shapes
# The input transform is fixed resize plus normalization. Caching therefore
# reuses the same teacher targets while avoiding repeated B7 inference. Logits
# remain FP32; hint maps use FP16 storage with small quantization error.

# %%
features = {}
hooks = [
    teacher.features[1].register_forward_hook(
        lambda m, a, o: features.update(direct=o)
    ),
    teacher.features[5].register_forward_hook(
        lambda m, a, o: features.update(adapted=o)
    ),
]


@torch.no_grad()
def cache_teacher(ds, split):
    path = ROOT / "data" / f"teacher_{split}.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    zs, ds_features, as_features = [], [], []
    sync()
    start = time.perf_counter()
    for x, y in loader(ds, batch=8):
        z = teacher(x.to(DEVICE))[:, class_indices]
        zs.append(z.cpu())
        if split == "train":
            ds_features.append(features["direct"].cpu().half())
            as_features.append(features["adapted"].cpu().half())
    sync()
    result = {"logits": torch.cat(zs), "seconds": time.perf_counter() - start}
    if split == "train":
        result.update(direct=torch.cat(ds_features), adapted=torch.cat(as_features))
    torch.save(result, path)
    return result


targets = cache_teacher(train, "train")
val_targets = cache_teacher(val, "validation")
for h in hooks:
    h.remove()
teacher_profile = profile(teacher, (1, 3, 224, 224))


def accuracy(logits, labels):
    top = logits.topk(5, dim=1).indices
    return {
        "top1": 100 * (top[:, 0] == labels).float().mean().item(),
        "top5": 100 * (top == labels[:, None]).any(1).float().mean().item(),
    }


teacher_result = {**accuracy(val_targets["logits"], val.tensors[1]), **teacher_profile}
del teacher
torch.xpu.empty_cache()


def student():
    net = efficientnet_b0(weights=None)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, 100)
    return net.to(DEVICE)


initial = student()
initial_state = copy.deepcopy(initial.state_dict())
del initial
print("Teacher:", teacher_result)
print("Cached feature shapes:", targets["direct"].shape, targets["adapted"].shape)

# %% [markdown]
# ## Loss, validation and training helpers
# Hint tensors are cached in FP16 for storage and converted to FP32 for MSE.
# Only the student and (when present) the adapter receive gradients.


# %%
def kd_loss(z, t, y):
    ce = F.cross_entropy(z, y)
    kl = F.kl_div(
        F.log_softmax(z / 3, dim=1), F.softmax(t / 3, dim=1), reduction="batchmean"
    )
    return 0.5 * ce + 0.5 * 9 * kl


@torch.no_grad()
def assess(net):
    net.eval()
    zs = []
    total = 0
    for x, y in loader(val, batch=16):
        z = net(x.to(DEVICE))
        zs.append(z.cpu())
        total += F.cross_entropy(z, y.to(DEVICE), reduction="sum").item()
    acc = accuracy(torch.cat(zs), val.tensors[1])
    acc["loss"] = total / len(val)
    acc["accuracy"] = acc["top1"]
    return acc


def train_variant(name, state, epochs, hint=None):
    seed_all(42)
    net = student()
    net.load_state_dict(state)
    adapter = nn.Conv2d(80, 224, 1).to(DEVICE) if hint == "adapted" else None
    captured = {}
    handle = None
    if hint:
        layer = net.features[0] if hint == "direct" else net.features[4]
        handle = layer.register_forward_hook(lambda m, a, o: captured.update(h=o))
    parameters = list(net.parameters()) + (
        list(adapter.parameters()) if adapter else []
    )
    optimizer = torch.optim.Adam(parameters, lr=0.001)
    # Adapter construction must not shift the student's dropout RNG sequence.
    seed_all(42)
    ids = TensorDataset(torch.arange(len(train)))
    batches = loader(ids, True, 42, batch=16)
    history = []
    verification = {}
    sync()
    start = time.perf_counter()
    for epoch in range(epochs):
        net.train()
        total = ce_total = hint_total = correct = n = 0
        for step, (idx,) in enumerate(batches):
            x = train.tensors[0][idx].to(DEVICE)
            y = train.tensors[1][idx].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            z = net(x)
            ce = F.cross_entropy(z, y)
            loss = (
                ce
                if name == "independent"
                else kd_loss(z, targets["logits"][idx].to(DEVICE), y)
            )
            hl = torch.tensor(0.0, device=DEVICE)
            if hint:
                hs = captured["h"]
                ht = targets[hint][idx].to(DEVICE).float()
                projected = adapter(hs) if adapter else hs
                assert projected.shape == ht.shape
                hl = F.mse_loss(projected, ht)
                loss = loss + 0.05 * hl
                if epoch == step == 0:
                    verification.update(
                        student_shape=list(hs.shape),
                        teacher_shape=list(ht.shape),
                        after_adapter_shape=list(projected.shape),
                    )
            loss.backward()
            if epoch == step == 0 and hint:
                verification.update(
                    student_gradient_nonzero=any(
                        p.grad is not None and bool(p.grad.abs().sum() > 0)
                        for p in net.parameters()
                    ),
                    teacher_frozen=True,
                    teacher_targets_require_grad=targets[hint].requires_grad,
                    adapter_gradient_nonzero=(
                        any(
                            p.grad is not None and bool(p.grad.abs().sum() > 0)
                            for p in adapter.parameters()
                        )
                        if adapter
                        else None
                    ),
                )
            optimizer.step()
            total += loss.item() * len(y)
            ce_total += ce.item() * len(y)
            hint_total += hl.item() * len(y)
            correct += (z.argmax(1) == y).sum().item()
            n += len(y)
        history.append(
            {
                "epoch": epoch + 1,
                "train": {
                    "loss": total / n,
                    "ce": ce_total / n,
                    "hint": hint_total / n,
                    "accuracy": 100 * correct / n,
                },
                "validation": assess(net),
            }
        )
        print(name, history[-1], flush=True)
    sync()
    elapsed = time.perf_counter() - start
    if handle:
        handle.remove()
    torch.save(net.state_dict(), ROOT / "checkpoints" / f"t1_{name}.pt")
    if adapter:
        torch.save(adapter.state_dict(), ROOT / "checkpoints" / "t1_adapter.pt")
    record = {
        "history": history,
        "seconds": elapsed,
        "updates": epochs * len(batches),
        "metrics": {**assess(net), **profile(net, (1, 3, 224, 224))},
        "verification": verification,
        "adapter_parameters": (
            sum(p.numel() for p in adapter.parameters()) if adapter else 0
        ),
    }
    state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    del net, adapter
    torch.xpu.empty_cache()
    return state, record


# %% [markdown]
# ## Independent and logit-KD students: three epochs each

# %%
_, ce_run = train_variant("independent", initial_state, 3)
kd_state, kd_run = train_variant("logit_kd", initial_state, 3)

# %% [markdown]
# ## Two separate hint experiments: two additional epochs each
# Direct: $h_S,h_T\in\mathbb R^{B\times32\times112\times112}$.
# Adapted: $r_\theta:\mathbb R^{80}\rightarrow\mathbb R^{224}$ is a 1x1 convolution.
# Both branches restart from the same logit-KD checkpoint; the adapter is discarded for deployment.

# %%
_, direct_run = train_variant("direct_hint", kd_state, 2, "direct")
_, adapted_run = train_variant("adapted_hint", kd_state, 2, "adapted")
runs = {
    "independent": ce_run,
    "logit_kd": kd_run,
    "direct_hint": direct_run,
    "adapted_hint": adapted_run,
}
save_json(
    "task1",
    {
        "teacher": teacher_result,
        "teacher_cache_seconds": targets["seconds"] + val_targets["seconds"],
        "train_examples": len(train),
        "validation_examples": len(val),
        "train_classes": len(train.tensors[1].unique()),
        "validation_classes": len(val.tensors[1].unique()),
        "runs": runs,
    },
)
curves(runs, "t1_training")
fig, ax = plt.subplots(figsize=(6, 3))
for name in ["direct_hint", "adapted_hint"]:
    ax.plot(
        [1, 2],
        [h["train"]["hint"] for h in runs[name]["history"]],
        marker="o",
        label=name,
    )
ax.set(xlabel="Additional epoch", ylabel="Mean squared hint loss")
ax.legend()
fig.tight_layout()
fig.savefig(ROOT / "figures" / "t1_hint.png", dpi=150)
plt.close(fig)
for name, run in runs.items():
    print(name, run["metrics"])
