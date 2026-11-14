# %% [markdown]
# # T5: Random search versus aging evolution
# The topology space has $5^6=15,625$ cells. All scores are benchmark lookups:
# CIFAR-10 validation (`cifar10-valid`), horizon 200, averaged across benchmark
# training seeds (`is_random=False`). Search seeds are 42, 43 and 44.

# %%
from common import *
from collections import deque

OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
EDGES = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]


def encode(arch):
    return "+".join(
        "|"
        + "|".join(f"{OPS[arch[e]]}~{i}" for e, (i, d) in enumerate(EDGES) if d == j)
        + "|"
        for j in range(1, 4)
    )


def random_arch(rng):
    return tuple(rng.randrange(5) for _ in range(6))


print("Space size:", 5**6, "Example:", encode((0, 1, 2, 3, 4, 1)))

# %% [markdown]
# ## Load a compact validation-only lookup extracted from official NATS-Bench
# The conversion script discards all unused datasets and training curves.

# %%
lookup = json.loads((ROOT / "data" / "nats_validation.json").read_text())


def score(arch):
    return lookup[encode(arch)]


# %% [markdown]
# ## Random search and FIFO aging evolution
# Population initialization counts toward the 100 evaluations. Tournament size
# is three, population size ten, and mutation always changes exactly one edge.
# Duplicates are permitted and still consume an evaluation in both methods.


# %%
def search(seed, evolution=False):
    rng = random.Random(seed)
    population = deque()
    history = []
    best = -float("inf")
    for evaluation in range(100):
        parent = None
        changed = None
        if not evolution or len(population) < 10:
            arch = random_arch(rng)
        else:
            parent = max(
                rng.sample(list(population), 3), key=lambda member: member["accuracy"]
            )
            child = list(parent["architecture"])
            changed = rng.randrange(6)
            child[changed] = rng.choice([o for o in range(5) if o != child[changed]])
            arch = tuple(child)
            assert sum(a != b for a, b in zip(arch, parent["architecture"])) == 1
        value = score(arch)
        best = max(best, value)
        record = {
            "evaluation": evaluation + 1,
            "architecture": arch,
            "encoding": encode(arch),
            "accuracy": value,
            "best_ever": best,
        }
        if evolution:
            population.append(record)
            if len(population) > 10:
                population.popleft()
            assert len(population) == min(evaluation + 1, 10)
            record.update(
                population_size=len(population),
                population_best=max(m["accuracy"] for m in population),
                parent_evaluation=parent["evaluation"] if parent else None,
                mutated_edge=changed,
            )
        history.append(record)
    return history


runs = {
    method: {
        str(seed): search(seed, method == "aging_evolution") for seed in [42, 43, 44]
    }
    for method in ["random", "aging_evolution"]
}
summary = {}
for method, seed_runs in runs.items():
    values = [h[-1]["best_ever"] for h in seed_runs.values()]
    summary[method] = {
        "best": max(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "per_seed": values,
        "evaluations_per_run": 100,
        "runs": 3,
    }
save_json(
    "task5",
    {
        "runs": runs,
        "summary": summary,
        "metric": "cifar10-valid/valid-accuracy",
        "hp": "200",
        "benchmark_seeds": "mean",
        "space_size": 5**6,
    },
)
print(summary)

# %% [markdown]
# ## Search curves and population aging

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for method, seed_runs in runs.items():
    for seed, h in seed_runs.items():
        axes[0].plot(
            range(1, 101), [r["best_ever"] for r in h], label=f"{method}, {seed}"
        )
for seed, h in runs["aging_evolution"].items():
    axes[1].plot(
        range(1, 101), [r["population_best"] for r in h], label=f"Population, {seed}"
    )
    axes[1].plot(
        range(1, 101),
        [r["best_ever"] for r in h],
        linestyle="--",
        alpha=0.6,
        label=f"Ever, {seed}",
    )
for ax in axes:
    ax.set(
        xlabel="Architecture evaluations", ylabel="Benchmark validation accuracy (%)"
    )
    ax.legend(fontsize=6)
    ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "t5_search.png", dpi=150)
plt.close(fig)
