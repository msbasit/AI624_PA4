The review covers the same 20 questions for each model (10 GSM8K and 10 synthetic), giving 40 inspected solutions. The complete questions, raw outputs, references and individual reasoning assessments are in `results/t2_manual_review.json`. This is a targeted qualitative sample, not an estimate of the error rate across the whole dataset.

Formatting is a major confound: the original student produced a parseable required marker on 0/200 real and 1/100 synthetic questions, versus 148/200 and 96/100 after fine-tuning. Thus the strict exact-match gain cannot be interpreted solely as improved arithmetic. Some answers without the marker were mathematically correct; no reference-based formatting repair was applied to student evaluation outputs.

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
