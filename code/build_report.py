"""Fill the supplied report sections using saved measurements, never estimates."""

import json
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / "results" / f"{name}.json").read_text())


def fmt(x):
    return f"{x:,.2f}" if isinstance(x, (float, np.floating)) else str(x)


def table(headers, rows):
    return (
        "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
            ]
            + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
        )
        + "\n"
    )


def figure(name, caption):
    return f"\n![{caption}](figures/{name}.png)\n\n{caption}\n"


t1, t2, t3, t4, t5 = [read(f"task{i}") for i in range(1, 6)]
env = read("environment")
sections = {}
runs = t1["runs"]
labels = {
    "independent": "Independent",
    "logit_kd": "Logit KD",
    "direct_hint": "Direct hint",
    "adapted_hint": "Adapted hint",
}
ce, kd = runs["independent"]["metrics"], runs["logit_kd"]["metrics"]
sections["T1.1"] = table(
    ["Model", "Top-1 (%)", "Top-5 (%)", "Parameters", "Median latency (ms)"],
    [
        [
            "EfficientNet-B7 teacher",
            t1["teacher"]["top1"],
            t1["teacher"]["top5"],
            t1["teacher"]["parameters"],
            t1["teacher"]["latency_ms"],
        ]
    ]
    + [
        [
            labels[k],
            runs[k]["metrics"]["top1"],
            runs[k]["metrics"]["top5"],
            runs[k]["metrics"]["parameters"],
            runs[k]["metrics"]["latency_ms"],
        ]
        for k in ["independent", "logit_kd"]
    ],
)
sections["T1.1"] += rf"""
Both B0 models began with identical weights and used seed 42, Adam (learning rate 0.001), batch size 16, three epochs and {runs['independent']['updates']} updates each. The fixed subset contains {t1['train_examples']} training and {t1['validation_examples']} validation images, covering {t1['train_classes']} and {t1['validation_classes']} classes respectively; no test set was used for model selection. Systematic blocks span the class-ordered source, with the two short-class gaps filled from their midpoint rows to give exactly five training images per class. Image indices are stored with the subset tensors; the 100-label mapping is in `results/imagenet_class_mapping.json`. The source is `clane9/imagenet-100`, whose images are already reduced to a 160-pixel short side; we resize to 224 x 224 and apply ImageNet normalization. No random augmentation is used. Teacher logits are cached in FP32; hint maps use FP16 storage and are converted to FP32 for MSE, introducing small storage quantization error.

$$L_{{KD}}=(1-\lambda)L_{{CE}}+\lambda T^2\,D_{{KL}}(p_T(T)\Vert p_S(T)),\quad T=3,\quad\lambda=0.5.$$

KD changed final Top-1 accuracy by {kd['top1']-ce['top1']:+.2f} percentage points and Top-5 by {kd['top5']-ce['top5']:+.2f} points. Independent training CE fell from {runs['independent']['history'][0]['train']['ce']:.3f} to {runs['independent']['history'][-1]['train']['ce']:.3f}; KD training CE fell from {runs['logit_kd']['history'][0]['train']['ce']:.3f} to {runs['logit_kd']['history'][-1]['train']['ce']:.3f}. However, final validation CE was {ce['loss']:.3f} and {kd['loss']:.3f}, respectively, and both Top-1 scores remained near the 1% chance level. Thus the students are substantially undertrained and the curves do not demonstrate stable validation convergence. The plotted objectives also differ: KD includes KL while the independent objective is CE; the JSON histories separately record CE. Both deployed students have exactly the same architecture and parameter count. Measured latency differences reflect runtime variability, not an extra KD operation at inference.
""" + figure(
    "t1_training",
    "T1 training objectives and validation Top-1 accuracy. Hint epochs are additional epochs after the common KD checkpoint.",
)
direct = runs["direct_hint"]
adapted = runs["adapted_hint"]
sections["T1.2"] = (
    rf"""Teacher `features[1]` and student `features[0]` both produced {direct['verification']['teacher_shape']} on the first batch. No adapter was used. Two additional epochs ({direct['updates']} updates) from the logit-KD checkpoint gave Top-1 {direct['metrics']['top1']:.2f}% and Top-5 {direct['metrics']['top5']:.2f}%. Mean hint losses by epoch were {', '.join(f"{h['train']['hint']:.6f}" for h in direct['history'])}.

$$L=L_{{KD}}+0.05L_{{hint}},\qquad L_{{hint}}=\mathrm{{MSE}}(h_S,h_T).$$

Matching batch, channel and spatial dimensions permits elementwise MSE. Equal shapes do not imply equal information: independently learned channels need not represent the same patterns, scales or semantics. The teacher layer also includes an MBConv stage while the student layer is the stem.
"""
    + figure(
        "t1_hint", "Direct and adapted feature MSE during the two additional epochs."
    )
)
sections[
    "T1.3"
] = rf"""The student feature shape was {adapted['verification']['student_shape']}; the teacher shape was {adapted['verification']['teacher_shape']}. A trainable 1 x 1 convolution changed the student shape to {adapted['verification']['after_adapter_shape']}. Its parameter count is 80 x 224 + 224 = {adapted['adapter_parameters']:,}, including bias. Checks confirmed nonzero gradients in both the student and adapter, with detached targets from a frozen teacher.

$$L_{{hint}}=\mathrm{{MSE}}(r_\theta(h_S),h_T).$$

The adapter is required because 80 student channels cannot be compared elementwise with 224 teacher channels. It learns a channel projection at each spatial location; both maps already have 14 x 14 spatial resolution. This run started from a separate copy of the same KD checkpoint. After {adapted['updates']} updates, Top-1 was {adapted['metrics']['top1']:.2f}% and Top-5 {adapted['metrics']['top5']:.2f}%. Hint losses were {', '.join(f"{h['train']['hint']:.6f}" for h in adapted['history'])}. The adapter is removed for deployment.
"""
best = max(runs, key=lambda k: runs[k]["metrics"]["top1"])
sections["T1.4"] = table(
    [
        "Configuration",
        "Top-1 (%)",
        "Top-5 (%)",
        "Cumulative training loop (s)",
        "Deployed parameters",
        "Latency (ms)",
    ],
    [
        [
            labels[k],
            v["metrics"]["top1"],
            v["metrics"]["top5"],
            v["seconds"] + (runs["logit_kd"]["seconds"] if "hint" in k else 0),
            v["metrics"]["parameters"],
            v["metrics"]["latency_ms"],
        ]
        for k, v in runs.items()
    ],
)
sections[
    "T1.4"
] += rf"""The highest final Top-1 result was {labels[best]} ({runs[best]['metrics']['top1']:.2f}%). Direct and adapted hints added {direct['seconds']:.2f} s and {adapted['seconds']:.2f} s, respectively, beyond KD training. Hint variants have two more epochs, so this is not an equal-total-budget comparison. Teacher target generation took {t1['teacher_cache_seconds']:.2f} s (shared preprocessing cost, excluded from the table). Caching is possible because inputs and teacher weights are fixed. Feature training adds feature storage, transfer and MSE computation; the adapted case also adds projection parameters and gradients. None of these additions changes the deployed B0 inference graph.
"""
sections[
    "T2.1"
] = r"""The complete held-out set is `data/synthetic_questions.json`; the generation prompt is `results/synthetic_prompt.txt`. The set uses 20 multi-step arithmetic problem families, each instantiated with five deterministic parameter settings (100 unique questions). Generation uses programmatic parameter variants rather than 100 independent API completions. The explicit teacher prompt was:

> Generate 100 original GSM8K-style word problems with concise worked solutions ending in #### <value>. Use diverse multi-step arithmetic situations, unambiguous quantities, and exactly checkable integer answers. Keep these questions separate from GSM8K training. Include a programmatic arithmetic check for every answer.

All answers were recomputed with exact integer arithmetic from the stated quantities, and the final marker, uniqueness and count were checked. The chosen quantities make divisions exact; no synthetic arithmetic error was found. For example, the first problem makes 20 x 12 = 240 rolls, sells 2, and divides the remaining 238 between two shops: 119 rolls per shop. The full checking record is in `results/synthetic_checks.json`.

The set resembles GSM8K in short everyday stories and multi-step arithmetic, but has only 20 underlying structures, cleaner wording, exclusively integer final values and less linguistic diversity. Repeated structures make its accuracy a limited measure of general reasoning. No synthetic question or solution enters LoRA training.
"""
sections["T2.2"] = table(
    ["Training examples", "Trainable parameters", "Trainable (%)", "Training loop (s)"],
    [
        [
            t2["training_examples"],
            t2["trainable_parameters"],
            t2["trainable_percent"],
            t2["training_seconds"],
        ]
    ],
)
sections[
    "T2.2"
] += rf"""A fixed seed-42 sample of 1,000 GSM8K training questions was answered by `{t2['teacher']}`. {t2['accepted']} outputs matched the reference final number; {1000-t2['accepted']} wrong or missing-answer outputs were excluded. An additional {t2['overlength_excluded']} accepted examples exceeded the 512-token budget and were excluded rather than truncating the rationale. The raw completions and accept/reject flags are in `data/teacher_rationales.json`. Matching the final answer does not establish that every intermediate reasoning step is correct.

For {t2['format_corrected']} accepted examples, a missing final-answer marker was corrected only because the last stated numerical value matched the GSM8K reference. Raw text is retained. Incorrect final numbers were never changed to reference values. The student is `{t2['student']}`. LoRA uses rank 8, alpha 16 (scale 2), dropout 0.05, and `q_proj`/`v_proj` targets. AdamW uses learning rate 0.0002; microbatch size 4 with accumulation over 4 microbatches gives effective batch size up to 16. One epoch used {t2['microbatches']} microbatches and {t2['optimizer_updates']} optimizer updates. The BF16 backbone is frozen; only LoRA parameters are trained. Prompt and padding tokens have label -100 and do not contribute to the objective. Teacher generation used SDPA attention in batches of 64. An Intel GPU resource failure during the initial student evaluation was resolved by switching both student evaluations to eager attention and batch size 8; no failed-run predictions were reused.

$$\Delta W=\frac{{\alpha}}{{r}}BA,\qquad L_{{SFT}}=-\sum_t\log p_\theta(y_t\mid x,y_{{<t}}).$$

The student learns to predict the teacher's calculation tokens and final answer, conditioned on the question and preceding completion tokens. This transfers a reasoning-output pattern using next-token supervision, without requiring teacher logits. The concise teacher prompt and all settings are in the notebook. A pilot prompt generated long responses that exceeded the token cap and responses such as `#### The total ... is 8 pounds`, which fail strict numerical extraction; those pilot records are retained separately and were not used for training.
""" + figure(
    "t2_loss", "Completion-only training cross-entropy for each microbatch."
)
sections["T2.3"] = table(
    ["Model", "GSM8K accuracy (%)", "Synthetic accuracy (%)"],
    [
        [name, t2[key]["gsm8k"], t2[key]["synthetic"]]
        for name, key in [
            ("Original student", "original"),
            ("Fine-tuned student", "finetuned"),
        ]
    ],
)
sections[
    "T2.3"
] += rf"""Both students used the same 200 GSM8K test questions and all 100 synthetic questions, greedy decoding and a {t2['max_new_tokens']}-token generation cap. Test questions were never used for model selection or training. Exact match extracts the last `####` numerical value, removes commas and compares normalized decimal values; missing markers count as incorrect. The fine-tuned model changed real-question accuracy by {t2['finetuned']['gsm8k']-t2['original']['gsm8k']:+.2f} points and synthetic accuracy by {t2['finetuned']['synthetic']-t2['original']['synthetic']:+.2f} points. The synthetic result applies to the limited family-based set described above, not unrestricted unseen arithmetic.

""" + (
    ROOT / "results" / "t2_manual_review_summary.md"
).read_text(
    encoding="utf-8"
)
sections["T3.1"] = figure(
    "t3_graphs",
    "Continuous four-node cell (six edges) and the extracted five-edge architecture; both networks contain two cells.",
)
sections[
    "T3.1"
] += r"""Each cell connects (0,1), (0,2), (1,2), (0,3), (1,3), (2,3). Candidates are none, skip, 1 x 1 convolution, 3 x 3 convolution, and 3 x 3 average pooling. Convolutional candidates use GroupNorm and ReLU. All operations preserve 16 channels and 32 x 32 resolution. The fixed 3 x 3 stem is followed by two searchable cells, global average pooling and a 10-class linear head.

$$p_{e,o}=\frac{e^{\alpha_{e,o}}}{\sum_{o'}e^{\alpha_{e,o'}}},\qquad x_j=\sum_{i<j}\sum_o p_{(i,j),o}\,o(x_i).$$

There is one shared architecture tensor of shape 6 x 5 (30 parameters). The cells own separate operation weights. Softmax and weighted sums are differentiable, so cross-entropy gradients reach the architecture parameters via the mixture coefficients. No discrete sampling occurs during continuous search.

""" + table(
    ["Verification", "Observed value"],
    [
        [k, v]
        for k, v in t3["checks"].items()
        if not k.startswith(("weight_step", "alpha_step"))
    ],
)
sections["T3.2"] = table(
    ["Optimizer verification", "Passed"],
    [
        [k, v]
        for k, v in t3["checks"].items()
        if k.startswith(("weight_step", "alpha_step"))
    ],
)
sections["T3.2"] += (
    rf"""The seed-43 CIFAR-10 split contains exactly 500 training and 100 validation images per class, with no overlap. Inputs use fixed normalization and no random augmentation. Batch size is 64. Adam updates network weights at learning rate 0.001 using training batches; a separate Adam updates alpha at 0.003 using validation batches. Weight gradients are disabled for the architecture step and alpha gradients are disabled for the weight step. GroupNorm has no mutable running statistics. The validation iterator cycles as needed.

Three epochs performed {t3['search']['weight_updates']} weight updates and {t3['search']['alpha_updates']} architecture updates, taking {t3['search']['seconds']:.2f} s including epoch validation. This is first-order optimization; it does not differentiate through the weight update.
"""
    + figure(
        "t3_training",
        "Training losses and validation accuracy for the search and fresh discrete network.",
    )
    + figure(
        "t3_probabilities",
        "Probability histories for edges (0,1) and (2,3), including the uniform initialization.",
    )
)
sections["T3.2"] += (
    "Final highest-probability operations by edge: "
    + ", ".join(
        f"{tuple(edge)}: {t3['operations'][int(np.argmax(p))]} ({max(p):.3f})"
        for edge, p in zip(t3["edges"], t3["probabilities"][-1])
    )
    + ".\n"
)
comp = t3["comparison"]
sections["T3.3"] = table(
    [
        "Model",
        "Validation accuracy (%)",
        "Parameters",
        "Conv/linear MACs",
        "Tensor size (MiB)",
        "Median latency (ms)",
    ],
    [
        [
            name,
            comp[key]["accuracy"],
            comp[key]["parameters"],
            comp[key]["macs"],
            comp[key]["size_mib"],
            comp[key]["latency_ms"],
        ]
        for name, key in [
            ("Continuous supernet", "continuous"),
            ("Discrete: inherited", "inherited"),
            ("Discrete: scratch", "scratch"),
        ]
    ],
)
sections["T3.3"] += (
    "Encoding: "
    + ", ".join(
        f"{tuple(t3['edges'][e])} = {t3['operations'][o]}" for e, o in t3["encoding"]
    )
    + ".\n\n"
)
sections[
    "T3.3"
] += rf"""For each node we ranked incoming edges by their largest non-none probability, retained up to two, and selected the corresponding non-none operation. The five-edge architecture is shown above. The continuous and inherited models were evaluated from the same saved search checkpoint with no intervening fine-tuning. Discretization changed validation accuracy by {comp['inherited']['accuracy']-comp['continuous']['accuracy']:+.2f} percentage points. Fresh initialization and five epochs ({t3['scratch']['updates']} updates) changed it by {comp['scratch']['accuracy']-comp['inherited']['accuracy']:+.2f} points relative to inherited weights. Scratch training took {t3['scratch']['seconds']:.2f} s.

Parameters and MACs are analytical counts; latency is measured. MACs count convolution and linear multiply-accumulates only, excluding pooling, normalization, activations, softmax and mixture arithmetic. Tensor size counts serialized state tensors, excluding file-container overhead. Inherited and scratch models have the same computational graph; their latency difference is measurement/runtime variability, not an architectural change.
"""
sections["T4.1"] = table(
    ["Path", "Active operations", "Output shape"],
    [
        [p, ops, shape]
        for p, ops, shape in zip(
            t4["checks"]["paths"],
            t4["checks"]["active_operations"],
            t4["checks"]["output_shapes"],
        )
    ],
)
sections["T4.1"] += "\n" + table(
    ["Verification", "Observed value"],
    [
        [k, v]
        for k, v in t4["checks"].items()
        if k not in ["paths", "active_operations", "output_shapes", "mappings"]
    ],
)
sections["T4.1"] += (
    "\nMappings: " + "; ".join(", ".join(m) for m in t4["checks"]["mappings"]) + ".\n\n"
)
sections[
    "T4.1"
] += r"""Indices 0, 1, 2 and 3 mean conv3x3, conv5x5, depthwise-separable conv3x3 and identity. The example paths share blocks 1 and 3 (one-based), plus the stem and head. Other selected operations use separate tensors. All paths preserve 16 channels and spatial size; the head returns ten logits. Identity has zero parameters. Extraction deep-copies only selected modules: outputs match the supernet path but storage is independent. A fixed stem and four blocks with GroupNorm give 256 paths. Weight sharing replaces hundreds of independent training runs with one pool of reusable operation weights, although the trained weights may favor paths that co-adapt well.
"""
sections["T4.2"] = (
    rf"""The seed-44 stratified split contains 5,000 training and 1,000 validation images from the official training split, disjoint within the task. Every minibatch samples one uniformly random operation per block. Only the selected operations, stem and head execute. Five epochs used Adam at 0.001, batch 64, {t4['training']['updates']} updates and {t4['training']['seconds']:.2f} s. Each block's sampling counts sum to {t4['training']['updates']}. Inactive parameters were compared before and after an actual optimizer step and remained unchanged; gradients were reset to None to avoid stale-gradient/momentum updates.

Inherited weights make evaluation inexpensive and place candidates on a shared training budget. However, paths receive unequal combinations of updates, operations interfere and co-adapt, and stand-alone optimization can alter rankings. Uniform operation sampling is not a learned ENAS controller. The validation curve uses one fixed path (0,1,2,3), while training loss averages different sampled paths.
"""
    + figure(
        "t4_training",
        "Supernet training and five-epoch scratch runs for the two selected paths.",
    )
    + figure("t4_sampling", "Observed operation sampling counts by block.")
)
sections["T4.3"] = table(
    ["Path number", "Encoding", "0 steps (%)", "10 steps (%)", "30 steps (%)"],
    [
        [i + 1, r["path"], r["accuracy"]["0"], r["accuracy"]["10"], r["accuracy"]["30"]]
        for i, r in enumerate(t4["adaptation"])
    ],
)
best0 = int(np.argmax([r["accuracy"]["0"] for r in t4["adaptation"]])) + 1
changed = [i + 1 for i in range(12) if t4["ranks"]["0"][i] != t4["ranks"]["30"][i]]
sections[
    "T4.3"
] += rf"""Each model was independently deep-copied from the same supernet checkpoint, then adapted with Adam at 0.001 on identical seeded training minibatches. Step 30 continues from step 10. No validation minibatch is used for adaptation. Across 12 models, adaptation required 360 optimizer updates and {sum(r['seconds'] for r in t4['adaptation']):.2f} s including staged validation. Average ranks handle ties; Kendall tau-b is used.

Kendall correlations: 0 versus 10 steps = {t4['tau']['0_10']:.4f}; 0 versus 30 = {t4['tau']['0_30']:.4f}; 10 versus 30 = {t4['tau']['10_30']:.4f}. Paths changing rank between 0 and 30 steps: {changed}. Best inherited path: {best0}; best after 30 steps: {t4['best_index']}. Thus the best inherited path {'remained' if best0==t4['best_index'] else 'did not remain'} best after adaptation. Full ranks are recorded in `results/task4.json`.
""" + figure(
    "t4_adaptation",
    "Validation accuracy trajectories and rankings for the 12 independent subnetworks.",
)
selected = [t4["best_index"], t4["random_index"]]
sections["T4.4"] = table(
    ["Path", "Inherited (%)", "Adapted 30 steps (%)", "Scratch (%)"],
    [
        [
            ("Best adapted: " if k == selected[0] else "Random: ") + str(k),
            t4["adaptation"][k - 1]["accuracy"]["0"],
            t4["adaptation"][k - 1]["accuracy"]["30"],
            t4["scratch"][str(k)]["accuracy"],
        ]
        for k in selected
    ],
)
order = {
    s: np.sign(
        t4["adaptation"][selected[0] - 1]["accuracy"][s]
        - t4["adaptation"][selected[1] - 1]["accuracy"][s]
    )
    for s in ["0", "30"]
}
scratch_order = np.sign(
    t4["scratch"][str(selected[0])]["accuracy"]
    - t4["scratch"][str(selected[1])]["accuracy"]
)
sections[
    "T4.4"
] += rf"""Both fresh networks used seed 44, the same minibatch order, optimizer, learning rate and five-epoch budget (395 updates each). Inherited ranking agrees with scratch ranking: {bool(order['0']==scratch_order)}. Adapted ranking agrees with scratch ranking: {bool(order['30']==scratch_order)}. This is a two-architecture comparison only; it cannot establish reliable rank prediction across the whole search space. Scratch wall times were {t4['scratch'][str(selected[0])]['seconds']:.2f} s and {t4['scratch'][str(selected[1])]['seconds']:.2f} s.
"""
sections["T5.1"] = (
    r"""The space has six edges and five choices per edge:

$$|\mathcal S|=5^6=15{,}625.$$

Example encoding: `|none~0|+|skip_connect~0|nor_conv_1x1~1|+|nor_conv_3x3~0|avg_pool_3x3~1|skip_connect~2|`. Each edge is sampled uniformly and independently. Each run evaluates 100 architectures; duplicates are allowed and count against the budget. Scores are official NATS-Bench topology-space CIFAR-10 validation lookups (`cifar10-valid`, `valid-accuracy`, hp='200', mean over available benchmark training seeds with `is_random=False`). They are not new GPU training measurements. `data/nats_validation.json` contains the compact 15,625-entry lookup derived from the official benchmark file.

"""
    + table(
        ["Search seed", "Random best (%)", "Evolution best (%)"],
        [
            [
                seed,
                t5["summary"]["random"]["per_seed"][i],
                t5["summary"]["aging_evolution"]["per_seed"][i],
            ]
            for i, seed in enumerate([42, 43, 44])
        ],
    )
    + figure(
        "t5_search",
        "Best-so-far validation accuracy versus evaluation count; evolution population-best curves can decrease.",
    )
)
sections["T5.2"] = r"""Pseudocode:

```text
population = FIFO queue; best_ever = -infinity
for evaluation = 1,...,100:
    if population has fewer than 10 members: sample a random cell
    else:
        sample 3 distinct population members uniformly
        choose the one with highest validation accuracy
        change one random edge to one of its other 4 operations
    look up validation accuracy; update best_ever
    append the new cell; remove the oldest if size exceeds 10
    record population_best and best_ever
```

Assertions verify population size equals min(evaluations, 10) and every mutation changes exactly one edge. The ten initialization evaluations are included in the 100-evaluation budget. All three runs passed. Full traces, parent evaluation IDs and mutated edges are in `results/task5.json`. The current population's best can fall when its strongest member ages out, even if it remains globally best. The best-ever statistic cannot decrease because it is the running maximum across all evaluated candidates.
"""
sections["T5.3"] = table(
    [
        "Method",
        "Best validation (%)",
        "Mean best (%)",
        "Std. dev. (pp)",
        "Evaluations per run",
    ],
    [
        [label, *[t5["summary"][key][k] for k in ["best", "mean", "std"]], 100]
        for label, key in [
            ("Random search", "random"),
            ("Aging evolution", "aging_evolution"),
        ]
    ],
)
sections[
    "T5.3"
] += rf"""Statistics summarize the final best-so-far score across three search seeds (42, 43, 44); standard deviation uses ddof=0. Each method used 300 evaluations in total. Aging evolution's mean was {t5['summary']['aging_evolution']['mean']-t5['summary']['random']['mean']:.3f} percentage points higher under the same per-run budget. This supports evolution in these three runs; the small sample and shared benchmark do not establish universal superiority. Every lookup uses the same metric and training horizon, with no CIFAR-10 test accuracy used to guide search.
"""
header = rf"""# AI624 Assignment 4: Results

**Name:** Muhammad Sami Ul Basit  
**Roll No:** 25280025

## Experimental setup

All training and model inference ran on {env['gpu']} using PyTorch {env['versions']['torch']}. CNN experiments used FP32; the language models used BF16. The NPU was not needed. Python/package versions are recorded in `results/environment.json`; training budgets and seeds appear under each task. Timings are measured wall time, including validation within training loops but excluding downloads. Latency uses batch size 1, ten warm-ups and twenty synchronized timed forwards, reporting the median and excluding data loading. Initial per-training-run latency varied substantially on this integrated GPU; final report values come from a separate post-training, seeded shuffled round-robin comparison of the saved models in `notebooks/reprofile.ipynb`. All raw samples are retained. Results are single-run measurements except the three-seed T5 searches. There is no minimum target accuracy in the assignment, and short training budgets limit the conclusions.

"""
template = (ROOT / "code" / "results_template.md").read_text(encoding="utf-8")
body = header
for block in re.split(r"(?=^## )", template, flags=re.M):
    match = re.match(r"## (T\d\.\d): ([^\n]+)", block)
    if not match:
        continue
    key, title = match.groups()
    q = re.search(r"\*\*Q1\.\*\* (.*)", block)
    body += (
        f"## {key}: {title}\n\n**Q1.** {q.group(1)}\n\n**Answer:**\n\n{sections[key]}\n"
    )
body += r"""## AI-use acknowledgement

OpenAI Codex assisted with implementing and debugging the experiments, developing synthetic problem families, analysing outputs, and drafting the report. Reported results were obtained from executed experiments and benchmark queries and checked against saved outputs. I implemented and reviewed the report, code, and notebook outputs. Qwen2.5-1.5B-Instruct generated the teacher rationales used in Task 2.

## Sources and implementation attribution

The task-specific KD losses, DARTS cells and search loop, path-sharing network, adaptation procedure, random search and aging evolution were implemented for this submission. No external repository's complete task implementation was copied. Standard model architectures, pretrained weights, datasets and LoRA components come from the following libraries/sources:

- EfficientNet models and weight sources: https://docs.pytorch.org/vision/stable/models/efficientnet.html
- ImageNet-100 data and source-resize disclosure: https://huggingface.co/datasets/clane9/imagenet-100
- CIFAR-10: https://huggingface.co/datasets/uoft-cs/cifar10
- GSM8K: https://huggingface.co/datasets/openai/gsm8k
- Student checkpoint: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- Teacher checkpoint: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- PEFT LoRA implementation: https://github.com/huggingface/peft
- DARTS method: https://arxiv.org/abs/1806.09055
- ENAS inspiration: https://arxiv.org/abs/1802.03268
- Official NATS-Bench metadata and API: https://github.com/D-X-Y/NATS-Bench

Public pretrained model weights are excluded from the ZIP; the sources above and code recreate them. Small trained CNN/NAS checkpoints and the student LoRA adapter are included.
"""
(ROOT / "results.md").write_text(body, encoding="utf-8")
print("Wrote completed results.md")
