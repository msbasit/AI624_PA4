"""Independent exact-arithmetic checks for every synthetic question family."""

import json
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT / "data" / "synthetic_questions.json").read_text())
checks = []
for row in rows:
    a = 20 + 4 * row["variant"]
    b = 12 + 4 * row["variant"]
    c = 2 + 2 * row["variant"]
    # Recompute from story quantities, with exact rational arithmetic for division.
    answers = [
        F(a * b - c, 2),
        F(a * b - a + c, 2),
        a * b - c + b,
        F(3 * a * b, 4) + c,
        2 * (a * b + c),
        2 * (a * b - c),
        a * b + a * b - c,
        a * b + b * c - a,
        F(a * b * 100 + c * 100, 100),
        a * b - c * b,
        a + b + (a - c) + b,
        2 * b * a + c,
        F(a * b, 2) * c + F(a * b, 2) * (2 * c),
        (a + c) * b - b,
        3 * a * b - 3 * c,
        a * b + (2 * a * b - c),
        F(a * 4 * 75, 100) * b + c,
        a * b + c,
        a * b - b * c,
        a * (b - c),
    ]
    expected = answers[row["family"]]
    passed = F(row["answer"]) == expected and row["solution"].endswith(
        "#### " + row["answer"]
    )
    assert passed, row["id"]
    checks.append(
        {
            "id": row["id"],
            "expected": str(expected),
            "answer": row["answer"],
            "passed": passed,
        }
    )
assert len(rows) == len({r["question"] for r in rows}) == 100
(ROOT / "results" / "synthetic_checks.json").write_text(json.dumps(checks, indent=2))
print("100/100 exact-arithmetic, format and uniqueness checks passed.")
