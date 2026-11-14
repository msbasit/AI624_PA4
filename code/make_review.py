"""Assemble the explicitly inspected reasoning examples and reviewer notes.

Notes are authored after reading the raw completions, not inferred from the
exact-match score. The record includes both the original text and assessment.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTES = {
    ("finetuned", "synthetic", 0): (
        "correct",
        "Correctly computes 20*12=240, 240-2=238 and 238/2=119. Strict scoring still rejects it because the required marker is missing.",
    ),
    ("finetuned", "synthetic", 1): (
        "incorrect",
        "Halves 220 before accounting for the two boarding passengers, then miscalculates its expression (112, not 220). Correct answer is 111; the original reasoning was correct.",
    ),
    ("finetuned", "synthetic", 2): (
        "incorrect",
        "Evaluates 20*12+12 as 240 rather than 252, then subtracts 2. Correct answer is 250, not 238.",
    ),
    ("finetuned", "synthetic", 3): (
        "incorrect",
        "Computes 60 dead plants correctly but subtracts from 20 rows instead of 240 plants. Correct answer is 182, not -38.",
    ),
    ("finetuned", "synthetic", 4): (
        "incorrect",
        "Counts the 2-km walking segment only once on the round trip. Correct answer is 242+242=484, not 240+242=482.",
    ),
    ("finetuned", "synthetic", 5): (
        "incorrect",
        "Still treats boxes as pencils and omits the factor of 12. Correct revenue is 476, not 36.",
    ),
    ("finetuned", "synthetic", 6): (
        "incorrect",
        "Treats the two-liter leak as two liters per minute and also miscalculates its own expression (456, not 252). Correct initial-content calculation gives 478.",
    ),
    ("finetuned", "synthetic", 7): (
        "correct",
        "Both ticket revenues and the printing deduction are correct: 240+24-20=244, with a valid final marker.",
    ),
    ("finetuned", "synthetic", 8): (
        "incorrect",
        "Uses an inappropriate division/addition of laps and meters and miscalculates that expression. Correct value is 242 hundred-meter segments, not 10.",
    ),
    ("finetuned", "synthetic", 9): (
        "correct",
        "Correctly computes 240-2*12=216 and supplies the required marker. This fixes the original arithmetic error.",
    ),
    ("finetuned", "gsm8k", 439): (
        "incorrect",
        "Correctly computes the 12-minute drive-time difference, then discards it and uses 4+5=9. The needed lead time is 12+5=17.",
    ),
    ("finetuned", "gsm8k", 645): (
        "correct",
        "Correctly computes 8*2=16 kg per bag and 16*3=48 kg, with the required final marker.",
    ),
    ("finetuned", "gsm8k", 582): (
        "incorrect",
        "Adds the 30% discount amount to Herb's house value rather than adding Juan's 70% value. Correct combined value is 129200.",
    ),
    ("finetuned", "gsm8k", 116): (
        "incomplete",
        "The ratio equations and y=78 water calculation are correct, but the response truncates before x=42 sugar or a final answer.",
    ),
    ("finetuned", "gsm8k", 113): (
        "correct",
        "Correct calculation 2*15-5=25 and correct #### 25 marker. The original already had the same arithmetic answer but no marker.",
    ),
    ("finetuned", "gsm8k", 412): (
        "incorrect",
        "Substitutes 12 cookies per day for the stated two per night. Correct requirement is 30*2/12=5 dozen, not 30.",
    ),
    ("finetuned", "gsm8k", 583): (
        "incorrect",
        "Confuses cups and dishes, assigns Judy the friend's extra dishes, and omits the friend's cups. Correct total is 120, not 68.",
    ),
    ("finetuned", "gsm8k", 812): (
        "incorrect",
        "Fails to remove one flower from the first two groups and miscalculates its own expression (which equals 90, not 190). Correct answer is 79.",
    ),
    ("finetuned", "gsm8k", 296): (
        "incorrect",
        "Omits the second daily meal, inverts the consumption/duration division and mishandles week conversion. Correct answer is 3 weeks.",
    ),
    ("finetuned", "gsm8k", 757): (
        "correct",
        "All five size counts are correct (11,22,18,9,15), and the sum 75 and final marker are correct.",
    ),
    ("original", "synthetic", 0): (
        "incorrect",
        "Uses 20*12=216 instead of 240. The intended formula is appropriate, but the product and final 107 are wrong; correct answer is 119.",
    ),
    ("original", "synthetic", 1): (
        "correct",
        "All steps correctly give 111 remaining passengers. Strict exact-match fails only because #### is missing.",
    ),
    ("original", "synthetic", 2): (
        "incorrect",
        "Omits the $2 purchase and evaluates 20*12+12 as 240. Correct balance is 250.",
    ),
    ("original", "synthetic", 3): (
        "incomplete",
        "Correctly computes 180 surviving plants but fails to add the two replacements. It gives a correctly formatted but wrong final answer, 180 instead of 182.",
    ),
    ("original", "synthetic", 4): (
        "incorrect",
        "States 52 without supporting calculation. Correct round-trip distance is 2*(20*12+2)=484.",
    ),
    ("original", "synthetic", 5): (
        "incorrect",
        "Treats boxes as individual pencils and ignores 12 pencils per box. Correct revenue is 2*(20*12-2)=476.",
    ),
    ("original", "synthetic", 6): (
        "incorrect",
        "Invents a per-minute leakage rate and calls initial contents capacity, then subtracts the water added. With the intended initial-content reading, 240+20*12-2=478 liters.",
    ),
    ("original", "synthetic", 7): (
        "incorrect",
        "The expression 12*20+2*12-20 is correct, but evaluates it as 240 rather than 244.",
    ),
    ("original", "synthetic", 8): (
        "incorrect",
        "Adds laps to extra meters without converting units. Correct distance is (12*2000+200)/100=242 segments.",
    ),
    ("original", "synthetic", 9): (
        "incorrect",
        "Correct expression 240-2*12, incorrect arithmetic result 168 instead of 216.",
    ),
    ("original", "gsm8k", 439): (
        "incorrect",
        "Adds the two drive times into Wayne's time. Required lead time is 16 - 4 + 5 = 17 minutes, not 1. Missing final marker.",
    ),
    ("original", "gsm8k", 645): (
        "incorrect",
        "Omits 2 kg per portion: 8 * 2 * 3 = 48 kg, not 24. Missing marker.",
    ),
    ("original", "gsm8k", 582): (
        "incorrect",
        "The expression 76000 + 76000 * 0.7 is appropriate but equals 129200, not 125200. Arithmetic error and missing marker.",
    ),
    ("original", "gsm8k", 116): (
        "incorrect",
        "Treats the combined total of 120 as sugar alone. Correct share is 120 * 7 / (7 + 13) = 42. Response also ends unfinished.",
    ),
    ("original", "gsm8k", 113): (
        "correct",
        "Correctly computes 2 * 15 - 5 = 25. Exact-match fails solely because the required #### marker is absent.",
    ),
    ("original", "gsm8k", 412): (
        "incomplete",
        "Correctly computes 60 cookies but does not convert to 60 / 12 = 5 dozen. Missing final marker.",
    ),
    ("original", "gsm8k", 583): (
        "incorrect",
        "Claims 12 + 2 * 12 = 48 and mishandles the friend's dishes. Correct total is 12 + 24 + 40 + 44 = 120.",
    ),
    ("original", "gsm8k", 812): (
        "incorrect",
        "Drops one flower in total instead of one of each type and treats all flowers as having 22 petals. Correct total is 2*5 + 3*6 + 4*4 + 5*7 = 79.",
    ),
    ("original", "gsm8k", 296): (
        "incorrect",
        "Omits that rice is eaten twice daily and mixes daily/weekly units. Correct duration is 42 / (5 * 0.2 * 2 * 7) = 3 weeks. Response unfinished.",
    ),
    ("original", "gsm8k", 757): (
        "incorrect",
        "Large should be 18/2=9 and extra-large 9+6=15. It instead uses 12 and 24; its displayed sum is also arithmetically wrong. Correct total is 75.",
    ),
}


def main():
    records = []
    for model in ["original", "finetuned"]:
        predictions = json.loads(
            (ROOT / "results" / f"t2_{model}_predictions.json").read_text()
        )
        for domain in ["gsm8k", "synthetic"]:
            selected = [r for r in predictions if r["set"] == domain][:10]
            assert len(selected) == 10
            for row in selected:
                key = (model, domain, row["id"])
                assert key in NOTES, f"Missing explicit inspection: {key}"
                verdict, note = NOTES[key]
                records.append(
                    {
                        **row,
                        "model": model,
                        "reasoning_verdict": verdict,
                        "review_note": note,
                    }
                )
    assert len(records) == 40
    (ROOT / "results" / "t2_manual_review.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    print("Saved 40 inspected solutions: 20 per model, 10 from each evaluation set.")
    predictions = {
        m: json.loads((ROOT / "results" / f"t2_{m}_predictions.json").read_text())
        for m in ["original", "finetuned"]
    }
    assert all(len(rows) == 300 for rows in predictions.values())
    compliance = {
        m: {
            s: sum(r["prediction"] is not None for r in rows if r["set"] == s)
            for s in ["gsm8k", "synthetic"]
        }
        for m, rows in predictions.items()
    }
    summary = f"""The review covers the same 20 questions for each model (10 GSM8K and 10 synthetic), giving 40 inspected solutions. The complete questions, raw outputs, references and individual reasoning assessments are in `results/t2_manual_review.json`. This is a targeted qualitative sample, not an estimate of the error rate across the whole dataset.

Formatting is a major confound: the original student produced a parseable required marker on {compliance['original']['gsm8k']}/200 real and {compliance['original']['synthetic']}/100 synthetic questions, versus {compliance['finetuned']['gsm8k']}/200 and {compliance['finetuned']['synthetic']}/100 after fine-tuning. Thus the strict exact-match gain cannot be interpreted solely as improved arithmetic. Some answers without the marker were mathematically correct; no reference-based formatting repair was applied to student evaluation outputs.

Examples from the inspected outputs:

| Set / ID | Model | Observed solution | Assessment |
|---|---|---|---|
| GSM8K / 645 | Original | 8 kg/portion x 3 bags = 24 kg | Omits 2 kg per portion; wrong reasoning and no marker. |
| GSM8K / 645 | Fine-tuned | 8 x 2 = 16; 16 x 3 = 48; `#### 48` | Correct reasoning and final answer. |
| GSM8K / 113 | Both | 2 x 15 - 5 = 25 | Both get the arithmetic right; only the fine-tuned output has `#### 25`. |
| GSM8K / 757 | Fine-tuned | 11 + 22 + 18 + 9 + 15 = 75; `#### 75` | Correct count relationships and sum; original used wrong large/extra-large counts. |
| GSM8K / 582 | Fine-tuned | 76000 x 0.3 = 22800; 76000 + 22800 = 98800 | Adds the discount amount instead of the discounted house value; reference is 129200. |
| Synthetic / 7 | Original / fine-tuned | Original evaluates 12 x 20 + 2 x 12 - 20 as 240; fine-tuned computes 264 - 20 = 244 | A genuine arithmetic correction, with a valid `#### 244` after fine-tuning. |
| Synthetic / 0 | Fine-tuned | 240 - 2 = 238; 238 / 2 = 119 | Correct arithmetic, still rejected because it uses a boxed answer instead of the required marker. |
| Synthetic / 1 | Original / fine-tuned | Original correctly gets 111; fine-tuned gives `#### 220` | A regression in reasoning despite improved formatting. |

No correct numerical answer reached through demonstrably incorrect reasoning was found in these 40 inspected outputs. This does not rule out such cases elsewhere. The fine-tuned model also truncates a correctly started ratio solution (GSM8K 116) before reaching its final answer, showing that the 192-token cap can reduce measured accuracy. Transfer to synthetic questions is mixed: some multiplication/subtraction errors are corrected, but unit errors, omitted quantities and regressions remain. The lower-diversity synthetic set and answer-format sensitivity limit any claim of broad reasoning generalization.
"""
    (ROOT / "results" / "t2_manual_review_summary.md").write_text(
        summary, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
