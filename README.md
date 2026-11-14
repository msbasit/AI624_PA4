# AI624 PA4 submission

**Muhammad Sami Ul Basit — Roll No: 25280025**

The submission contains five executed task notebooks plus a final profiling notebook, the completed Markdown/PDF report, figures, machine-readable measurements, small trained checkpoints, and the LoRA adapter. Public pretrained models and the full CIFAR-10/NATS-Bench archives are not included.

## Read the deliverables

- `results.md` and `results.pdf`: the same report, organized by the supplied task IDs.
- `notebooks/task1.ipynb` through `task5.ipynb`: separated explanatory, implementation, verification, experiment and plotting cells with saved outputs.
- `figures/`: all report plots.
- `results/`: measured metrics, exact split indices, class mapping, predictions, checks and search traces.
- `checkpoints/`: trained B0 students, NAS networks and the Qwen LoRA adapter.
- `data/synthetic_questions.json`: 100 held-out synthetic questions and worked answers.
- `data/teacher_rationales.json`: 1,000 teacher generations, numerical verification, formatting corrections and acceptance flags.
- `data/nats_validation.json`: compact official NATS-Bench validation lookup; these are benchmark values, not measurements made on this computer.

## Environment

The experiments use Python 3.12 and the existing Intel XPU environment. Exact versions and the GPU name are saved in `results/environment.json`. Training/inference ran on the Intel Arc 140T GPU. T1/T3/T4 use FP32; T2 uses a BF16 backbone and trainable LoRA parameters. The NPU is not used.

Use the installed Intel-GPU Python/Jupyter environment. If recreating it, install the matching XPU builds of PyTorch and torchvision from the official XPU package index before installing the remaining packages:

```powershell
python -m pip install torch==2.14.1+xpu torchvision==0.29.1+xpu --index-url https://download.pytorch.org/whl/xpu
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.xpu.is_available()); print(torch.xpu.get_device_name())"
```

Do not replace a working XPU installation with CPU-only PyTorch. Select that same environment as the notebook kernel. Open notebooks from this extracted project directory or its `notebooks` subdirectory. Internet is needed only for public source downloads not included in the ZIP. Setting `HF_HUB_DISABLE_XET=1` avoids the stalled Xet transfer encountered on this machine.

## Reproduce experiments

Run commands from the project root. The percent-cell scripts contain the same implementation as the notebooks. `run_notebook.py` executes cells using the current Python interpreter and saves real cell outputs after each step.

```powershell
$env:HF_HUB_DISABLE_XET='1'
python code/environment.py
python code/prepare_data.py
python code/run_notebook.py code/task1.py
python code/run_notebook.py code/task2.py
python code/check_synthetic.py
python code/run_notebook.py code/task3.py
python code/run_notebook.py code/task4.py
python code/run_notebook.py code/task5.py
python code/run_notebook.py code/reprofile.py
```

Run GPU tasks sequentially, especially during latency profiling. The included ImageNet subset tensors allow T1 to run without refetching images. `prepare_data.py` records exact source row indices and the class mapping and caches requests to respect public API limits. Teacher features are computed once and reused because preprocessing is deterministic; the large feature cache is excluded from the submission.

Final report latencies come from `notebooks/reprofile.ipynb`, which measures saved models in shuffled round-robin order after training. This replaces the noticeably variable provisional timings printed in the original training cells. Every model still receives ten warm-ups and twenty timed forwards, with all samples retained.

T2 resumes teacher generation from its saved JSON and uses cached evaluation predictions. To perform a genuinely new T2 evaluation, move the corresponding `results/t2_original_predictions.json` and `results/t2_finetuned_predictions.json` to a different directory first. To regenerate teacher solutions, move `data/teacher_rationales.json` too. Never reuse prediction caches after changing prompts, checkpoints or decoding settings. Teacher responses are filtered by reference-answer agreement; missing markers are corrected only when the last stated numerical value matches the reference, with original text retained. This does not certify intermediate reasoning.

T3 and T4 download CIFAR-10 on first use. They each construct their own fixed disjoint stratified 5,000/1,000 split from the official training set. Test data are not used for architecture selection. All scratch comparisons preserve settings and data order. Full budgets, optimizer updates and seeds are reported under each task.

T5 normally needs only the included compact lookup. To rebuild it from the official source:

```powershell
python code/download_nats.py
python code/extract_nats.py
```

The original metadata download is approximately 1 GB and loading it can require tens of GB of RAM. It contains no downloaded pretrained weights. `extract_nats.py` uses `cifar10-valid`, hp `200`, and averages available benchmark training seeds consistently for every architecture.

## Report and notebook export

```powershell
python code/build_report.py
python code/render_report.py
python code/finalize_notebooks.py
python code/package_submission.py
```

The report generator uses saved results and `results/t2_manual_review_summary.md`; it does not invent values or automatically reassess reasoning. Regenerating predictions requires reviewing the new solutions and updating the review files before rebuilding the report. PDF equations are rendered from the report's LaTeX expressions. The packaging script checks required files and excludes large public source archives, model caches, temporary files, teacher feature caches, the submission manifest and internal PDF-layout checks.

## Sources and attribution

The algorithms were implemented for this assignment; torchvision supplies EfficientNet, Transformers supplies Qwen, and PEFT supplies LoRA. Public pretrained sources are `torchvision` EfficientNet-B7 default weights, `Qwen/Qwen2.5-0.5B-Instruct`, and `Qwen/Qwen2.5-1.5B-Instruct`. Dataset sources are `clane9/imagenet-100`, `uoft-cs/cifar10`, `openai/gsm8k`, and official [NATS-Bench](https://github.com/D-X-Y/NATS-Bench). The report contains the full source links.

The synthetic dataset is generated from 20 problem families, not 100 independently sampled prompts. Its reduced diversity and the small CNN training subset limit generalization claims. See the report for the observed results and reasoning errors.

The report's AI-use acknowledgement describes assistance from OpenAI Codex and the Qwen teacher model, along with the author's review of the report, code, and notebook outputs.
