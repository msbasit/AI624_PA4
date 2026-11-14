# %% [markdown]
# # T2: Reasoning distillation with completion-only LoRA
# Teacher: Qwen2.5-1.5B-Instruct. Student: Qwen2.5-0.5B-Instruct.
# A fixed sample of 1,000 GSM8K training questions is sent to the teacher.
# Teacher outputs with a wrong/missing final answer are excluded, never silently
# replaced with GSM8K reference rationales. Synthetic questions are held out.

# %%
from common import *
import re
from decimal import Decimal, InvalidOperation
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import make_synthetic

seed_all(42)
TEACHER = "Qwen/Qwen2.5-1.5B-Instruct"
STUDENT = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = "Solve the arithmetic problem using at most 3 short lines of calculations (under 80 words). No headings or restatement. Finish with a separate line containing #### and ONLY the number. Example format: 3 * 4 = 12. 12 - 2 = 10.\n#### 10"
gsm = load_dataset("openai/gsm8k", "main")
train_ids = (
    np.random.default_rng(42).choice(len(gsm["train"]), 1000, replace=False).tolist()
)
test_ids = (
    np.random.default_rng(42).choice(len(gsm["test"]), 200, replace=False).tolist()
)
synthetic = json.loads((ROOT / "data" / "synthetic_questions.json").read_text())
save_json("t2_splits", {"train": train_ids, "test": test_ids})


def final_answer(text):
    matches = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if not matches:
        return None
    try:
        return str(Decimal(matches[-1].replace(",", "")).normalize())
    except InvalidOperation:
        return None


def load_model(name):
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    # The 0.5B SDPA prefill hit an Intel driver resource limit at batch 32.
    # Eager attention and batch 8 keep before/after student evaluation identical.
    attention = "eager" if name == STUDENT else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, attn_implementation=attention
    ).to(DEVICE)
    return tokenizer, model


def prompt(tokenizer, question):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def generate(model, tokenizer, questions, max_new_tokens=192):
    model.eval()
    inputs = tokenizer(
        [prompt(tokenizer, q) for q in questions], return_tensors="pt", padding=True
    ).to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(
        outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )


# %% [markdown]
# ## Generate and verify teacher rationales
# The exact-match check uses only the final numerical answer. It does not prove
# every reasoning step is sound; the report explicitly discusses this limitation.

# %%
teacher_path = ROOT / "data" / "teacher_rationales.json"
records = json.loads(teacher_path.read_text()) if teacher_path.exists() else []
if len(records) < 1000:
    tokenizer, teacher = load_model(TEACHER)
    sync()
    start = time.perf_counter()
    for start_idx in range(len(records), 1000, 64):
        ids = train_ids[start_idx : start_idx + 64]
        rows = [gsm["train"][i] for i in ids]
        completions = generate(teacher, tokenizer, [r["question"] for r in rows])
        for index, row, completion in zip(ids, rows, completions):
            predicted = final_answer(completion)
            reference = final_answer(row["answer"])
            records.append(
                {
                    "gsm8k_index": index,
                    "question": row["question"],
                    "completion": completion,
                    "prediction": predicted,
                    "reference": reference,
                    "valid": predicted is not None and predicted == reference,
                }
            )
        teacher_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(
            "Teacher generated:",
            len(records),
            "accepted:",
            sum(r["valid"] for r in records),
            flush=True,
        )
    sync()
    save_json(
        "t2_teacher_generation",
        {"seconds": time.perf_counter() - start, "teacher": TEACHER, "prompt": SYSTEM},
    )
    del teacher
    torch.xpu.empty_cache()
# Repair answer formatting only when the last stated number independently
# matches the reference. Keep raw text for audit; never replace a wrong number.
for row in records:
    row.setdefault("raw_completion", row["completion"])
    if final_answer(row["completion"]) is None:
        numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", row["completion"])
        if (
            numbers
            and str(Decimal(numbers[-1].replace(",", "")).normalize())
            == row["reference"]
        ):
            row["completion"] += "\n#### " + format(Decimal(row["reference"]), "f")
            row["format_corrected"] = True
            row["prediction"] = row["reference"]
            row["valid"] = True
teacher_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
accepted = [r for r in records if r["valid"]]
assert accepted
print("Accepted teacher rationales:", len(accepted), "Excluded:", 1000 - len(accepted))

# %% [markdown]
# ## Evaluate the original student on the fixed held-out sets
# Exact match requires the final-answer marker; missing markers count as wrong.
# Greedy decoding and a 192-token cap are identical before and after training.

# %%
tokenizer, student = load_model(STUDENT)
evaluation = [
    {
        "set": "gsm8k",
        "id": i,
        "question": gsm["test"][i]["question"],
        "reference": final_answer(gsm["test"][i]["answer"]),
    }
    for i in test_ids
]
evaluation += [
    {
        "set": "synthetic",
        "id": r["id"],
        "question": r["question"],
        "reference": str(Decimal(r["answer"]).normalize()),
    }
    for r in synthetic
]


def evaluate_reasoning(model, label):
    path = ROOT / "results" / f"t2_{label}_predictions.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    for start in range(len(rows), len(evaluation), 8):
        batch = evaluation[start : start + 8]
        completions = generate(model, tokenizer, [r["question"] for r in batch])
        for row, text in zip(batch, completions):
            answer = final_answer(text)
            rows.append(
                {
                    **row,
                    "completion": text,
                    "prediction": answer,
                    "correct": answer is not None and answer == row["reference"],
                }
            )
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(label, "evaluated:", len(rows), flush=True)
    return {
        s: 100 * np.mean([r["correct"] for r in rows if r["set"] == s])
        for s in ["gsm8k", "synthetic"]
    }


baseline = evaluate_reasoning(student, "original")
print("Original:", baseline)

# %% [markdown]
# ## Tokenize with prompt masking and add LoRA
# Only teacher-completion tokens contribute to $L=-\sum_t\log p_\theta(y_t\mid x,y_{<t})$.
# LoRA learns $\Delta W=(\alpha/r)BA$ with $r=8$, $\alpha=16$ on query/value projections.

# %%
tokenizer.padding_side = "right"
config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
student = get_peft_model(student, config)
trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
total = sum(p.numel() for p in student.parameters())
examples = []
truncated = 0
for row in accepted:
    prefix = tokenizer(prompt(tokenizer, row["question"]), add_special_tokens=False)[
        "input_ids"
    ]
    completion = tokenizer(
        row["completion"] + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"]
    # Keep whole reasoning traces: discard sequences over the recorded length budget.
    if len(prefix) + len(completion) > 512:
        truncated += 1
        continue
    examples.append(
        {"input_ids": prefix + completion, "labels": [-100] * len(prefix) + completion}
    )


def collate(rows):
    length = max(len(r["input_ids"]) for r in rows)
    return {
        "input_ids": torch.tensor(
            [
                r["input_ids"]
                + [tokenizer.pad_token_id] * (length - len(r["input_ids"]))
                for r in rows
            ],
            device=DEVICE,
        ),
        "attention_mask": torch.tensor(
            [
                [1] * len(r["input_ids"]) + [0] * (length - len(r["input_ids"]))
                for r in rows
            ],
            device=DEVICE,
        ),
        "labels": torch.tensor(
            [r["labels"] + [-100] * (length - len(r["labels"])) for r in rows],
            device=DEVICE,
        ),
    }


batches = DataLoader(
    examples,
    batch_size=4,
    shuffle=True,
    collate_fn=collate,
    generator=torch.Generator().manual_seed(42),
)
optimizer = torch.optim.AdamW(
    [p for p in student.parameters() if p.requires_grad], lr=2e-4
)
print(
    "Training examples:",
    len(examples),
    "Trainable:",
    trainable,
    "Percent:",
    100 * trainable / total,
)

# %% [markdown]
# ## One epoch of completion-only supervised fine-tuning
# Microbatch 4, gradient accumulation 4 (effective batch up to 16), AdamW.

# %%
student.train()
student.config.use_cache = False
losses = []
updates = 0
optimizer.zero_grad(set_to_none=True)
sync()
start = time.perf_counter()
for step, batch in enumerate(batches, 1):
    loss = student(**batch).loss
    losses.append(float(loss.detach()))
    group_size = min(4, len(batches) - ((step - 1) // 4) * 4)
    (loss / group_size).backward()
    if step % 4 == 0 or step == len(batches):
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updates += 1
    if step % 20 == 0:
        print("Microbatch:", step, "Loss:", losses[-1], flush=True)
sync()
training_seconds = time.perf_counter() - start
student.save_pretrained(ROOT / "checkpoints" / "t2_lora")
tokenizer.save_pretrained(ROOT / "checkpoints" / "t2_lora")
student.config.use_cache = True

# %% [markdown]
# ## Evaluate the fine-tuned model and save comparable results

# %%
tokenizer.padding_side = "left"
finetuned = evaluate_reasoning(student, "finetuned")
result = {
    "teacher": TEACHER,
    "student": STUDENT,
    "sampled": 1000,
    "accepted": len(accepted),
    "format_corrected": sum(r.get("format_corrected", False) for r in records),
    "training_examples": len(examples),
    "overlength_excluded": truncated,
    "trainable_parameters": trainable,
    "total_parameters": total,
    "trainable_percent": 100 * trainable / total,
    "training_seconds": training_seconds,
    "optimizer_updates": updates,
    "microbatches": len(batches),
    "losses": losses,
    "original": baseline,
    "finetuned": finetuned,
    "seed": 42,
    "max_new_tokens": 192,
}
save_json("task2", result)
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(range(1, len(losses) + 1), losses, alpha=0.6)
ax.set(xlabel="Training microbatch", ylabel="Completion cross-entropy")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "t2_loss.png", dpi=150)
plt.close(fig)
print(result)
