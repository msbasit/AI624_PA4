"""Create the one-file LMS submission and verify its expected contents."""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
files = []
# Internal document-layout checks are not assignment evidence.
excluded = {"results/pdf_checks.json"}
for directory in ["code", "notebooks", "figures", "results", "checkpoints"]:
    files.extend(
        p
        for p in (ROOT / directory).rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.relative_to(ROOT).as_posix() not in excluded
    )
files.extend(
    ROOT / name
    for name in [
        "results.md",
        "results.pdf",
        "README.md",
        "requirements.txt",
    ]
)
files.extend(
    ROOT / "data" / name
    for name in [
        "synthetic_questions.json",
        "teacher_rationales.json",
        "nats_validation.json",
        "imagenet100_train.pt",
        "imagenet100_validation.pt",
    ]
)
for p in files:
    assert p.exists(), p
path = ROOT / "PA4_Muhammad_Sami_Ul_Basit_25280025.zip"
with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for p in files:
        archive.write(p, p.relative_to(ROOT))
with zipfile.ZipFile(path) as archive:
    assert archive.testzip() is None
    assert all(f"notebooks/task{i}.ipynb" in archive.namelist() for i in range(1, 6))
    assert "submission_manifest.json" not in archive.namelist()
    assert "AI624_PA4_results.md" not in archive.namelist()
    assert not excluded.intersection(archive.namelist())
print(
    "Verified ZIP:",
    path,
    "MiB:",
    round(path.stat().st_size / 2**20, 2),
    "files:",
    len(files),
)
