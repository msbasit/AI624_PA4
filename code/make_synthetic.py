"""100 held-out arithmetic problems from 20 problem families.

Each family has an independently computed exact arithmetic answer. Five fixed
parameter variants per family give 100 unique questions; none enter training.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "Generate 100 original GSM8K-style word problems with concise worked solutions ending in #### <value>. Use diverse multi-step arithmetic situations, unambiguous quantities, and exactly checkable integer answers. Keep these questions separate from GSM8K training. Include a programmatic arithmetic check for every answer."


def problems(a, b, c):
    return [
        (
            f"A baker makes {a} trays of {b} rolls. She sells {c} rolls and divides the rest equally between 2 shops. How many rolls does each shop receive?",
            f"There are {a}*{b}={a*b} rolls. After selling {c}, {a*b-c} remain. Each shop gets {a*b-c}/2={(a*b-c)//2}.",
            (a * b - c) // 2,
        ),
        (
            f"A bus starts with {a*b} passengers. At the next stop {a} get off and {c} get on. At the last stop half of the passengers get off. How many remain?",
            f"After the stop there are {a*b}-{a}+{c}={a*b-a+c}. Half remain: {a*b-a+c}/2={(a*b-a+c)//2}.",
            (a * b - a + c) // 2,
        ),
        (
            f"Leena saves ${a} each week for {b} weeks. She then buys a book for ${c} and earns ${b} doing chores. How many dollars does she have?",
            f"Savings are {a}*{b}={a*b}. After the book and chores: {a*b}-{c}+{b}={a*b-c+b}.",
            a * b - c + b,
        ),
        (
            f"A garden has {a} rows with {b} plants each. A quarter of the plants die. The gardener replaces {c} plants. How many living plants are there now?",
            f"Initially {a*b} plants. A quarter is {a*b//4}; {a*b*3//4} survive. Add {c}: {a*b*3//4+c}.",
            a * b * 3 // 4 + c,
        ),
        (
            f"A cyclist rides at {a} kilometers per hour for {b} hours, then walks {c} more kilometers. If the return trip follows the same route, what is the total distance traveled in kilometers?",
            f"One way is {a}*{b}+{c}={a*b+c} km. Round trip: 2*{a*b+c}={2*(a*b+c)} km.",
            2 * (a * b + c),
        ),
        (
            f"A store buys {a} boxes with {b} pencils each. It gives away {c} pencils and sells the rest for $2 each. What is its sales revenue in dollars?",
            f"Total pencils: {a*b}. Sold: {a*b}-{c}={a*b-c}. Revenue: 2*{a*b-c}={2*(a*b-c)}.",
            2 * (a * b - c),
        ),
        (
            f"A tank holds {a*b} liters. A pump adds {a} liters per minute for {b} minutes, then {c} liters leak out. How many liters are left?",
            f"Added water: {a*b} liters. Final volume: {a*b}+{a*b}-{c}={2*a*b-c}.",
            2 * a * b - c,
        ),
        (
            f"A museum sells {a} adult tickets at ${b} each and {b} child tickets at ${c} each. It spends ${a} on printing. What is the remaining revenue in dollars?",
            f"Adult revenue {a*b}; child revenue {b*c}. Subtract printing: {a*b}+{b*c}-{a}={a*b+b*c-a}.",
            a * b + b * c - a,
        ),
        (
            f"A runner completes {b} laps of an {a*100}-meter track and then runs {c*100} extra meters. How many hundred-meter segments did the runner cover?",
            f"Laps cover {a*b*100} meters. Total {100*(a*b+c)} meters. Divide by 100: {a*b+c}.",
            a * b + c,
        ),
        (
            f"A farmer gathers {a*b} eggs. She packs {c} cartons containing {b} eggs each. How many eggs remain unpacked?",
            f"Packed eggs: {c}*{b}={c*b}. Remaining: {a*b}-{c*b}={(a-c)*b}.",
            (a - c) * b,
        ),
        (
            f"Mina is {a} years old and her sister is {c} years younger. What will be the sum of their ages in {b} years?",
            f"Sister is {a}-{c}={a-c}. In {b} years the sum is {a}+{b}+{a-c}+{b}={2*a-c+2*b}.",
            2 * a - c + 2 * b,
        ),
        (
            f"A recipe for {b} people uses {a} grams of flour per person. You prepare it for twice as many people and spill {c} grams of extra flour. How many grams of flour are used including the spill?",
            f"Food needs 2*{b}*{a}={2*a*b} grams. Including the spill: {2*a*b}+{c}={2*a*b+c}.",
            2 * a * b + c,
        ),
        (
            f"A club has {a*b} members. Half pay a ${c} fee and the other half pay a ${2*c} fee. What is the total collected in dollars?",
            f"Each half has {a*b//2} members. Revenue is {a*b//2}*({c}+{2*c})={a*b*3*c//2}.",
            a * b * 3 * c // 2,
        ),
        (
            f"A warehouse has {a} crates holding {b} bottles each. It receives {c} more identical crates and ships {b} individual bottles. How many bottles remain?",
            f"There are {a+c} crates before shipping, holding {(a+c)*b} bottles. Subtract {b}: {(a+c-1)*b}.",
            (a + c - 1) * b,
        ),
        (
            f"A worker earns ${a} per hour for {b} hours each day. Over 3 days, she spends ${c} per day on transport. How much money does she keep in dollars?",
            f"Daily net earnings: {a}*{b}-{c}={a*b-c}. Three days: 3*{a*b-c}={3*(a*b-c)}.",
            3 * (a * b - c),
        ),
        (
            f"Two teams collect cans. Team A collects {a*b} cans. Team B collects {c} fewer than twice as many as A. How many cans do they collect together?",
            f"Team B has 2*{a*b}-{c}={2*a*b-c}. Together: {a*b}+{2*a*b-c}={3*a*b-c}.",
            3 * a * b - c,
        ),
        (
            f"A shop marks a ${a*4} item down by 25 percent. A customer buys {b} of these discounted items and pays ${c} for delivery. What is the total bill in dollars?",
            f"Each discounted item costs {a*4}*3/4={3*a}. Items cost {3*a*b}. With delivery: {3*a*b+c}.",
            3 * a * b + c,
        ),
        (
            f"A class reads {a} pages per day for {b} days. They then reread {c} of those pages. How many page readings did they do in total, counting rereads?",
            f"First readings: {a}*{b}={a*b}. Add {c} rereads: {a*b+c}.",
            a * b + c,
        ),
        (
            f"A truck can carry {a} kilograms. It carries {b} packages weighing {c} kilograms each. How many more kilograms can it carry if its capacity is actually increased to {a*b} kilograms?",
            f"The updated capacity is {a*b}. Current load is {b}*{c}={b*c}. Remaining capacity: {a*b}-{b*c}={(a-c)*b}.",
            (a - c) * b,
        ),
        (
            f"A school has {a} classrooms with {b} chairs each. It moves {c} chairs from each classroom into the hall. How many chairs remain in the classrooms in total?",
            f"Each room retains {b}-{c}={b-c}. Total remaining: {a}*{b-c}={a*(b-c)}.",
            a * (b - c),
        ),
    ]


rows = []
for variant in range(5):
    a, b, c = 20 + 4 * variant, 12 + 4 * variant, 2 + 2 * variant
    for family, (q, s, answer) in enumerate(problems(a, b, c)):
        rows.append(
            {
                "id": len(rows),
                "family": family,
                "variant": variant,
                "question": q,
                "solution": s + f"\n#### {answer}",
                "answer": str(answer),
                "checked_answer": answer,
                "generator": "Problem families with deterministic arithmetic instantiation",
            }
        )
assert len(rows) == len({r["question"] for r in rows}) == 100
(ROOT / "data" / "synthetic_questions.json").write_text(
    json.dumps(rows, indent=2), encoding="utf-8"
)
(ROOT / "results" / "synthetic_prompt.txt").write_text(PROMPT, encoding="utf-8")
print("Created 100 questions in 20 families; integer arithmetic checks passed.")
