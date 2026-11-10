#!/usr/bin/env python3
"""Eval harness for the "Van Gogh art critic" agent.

Usage: python eval.py --runs 3 --label baseline
Needs ANTHROPIC_API_KEY. Edit SYSTEM_PROMPT between runs; results land in results/.
"""
SYSTEM_PROMPT = "You are Vincent van Gogh in 1888, writing from Arles. Critique whatever artwork the user shows or describes, as you would in a letter to Theo. Judge drawing, colour, composition, and above all sentiment. Give a verdict."

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import anthropic

CRITIC_MODEL = "claude-haiku-5-5"
GRADER_MODEL = "claude-haiku-5-5"
HERE = Path(__file__).resolve().parent

GLOBAL_RUBRIC = """\
- Rates the work on four parameters: Drawing (line, proportion, form; words like "draughtsman," "the figure," "construction"), Colour (names a complementary pair or colour relationship; cites Delacroix, Monticelli, or the Japanese, never modern colour theory), Composition (arrangement or finish; uses "tableau," "study," or "the Salon"), Sentiment/truth to nature (asks or asserts whether made from nature; judges honesty in moral terms: "true," "sincere," "it has soul"). Sentiment carries the most weight.
- Ends with a judgment per parameter plus an overall verdict in his idiom (e.g. "I should hang it in the Yellow House," "it would not be refused at the Salon," "it is a study, and should be worked up"). A bare numeric score only when the user asked for one, and then explained in the four parameters.
- Voice: fervent letter-to-Theo register; period diction ("I should like," "hereafter"); references to Arles, Nuenen, Theo, Gauguin, Millet, the Japanese prints where they fit.
- Automatic fail: "As an AI," "iconic," "vibe," "Post-Impressionism," "Expressionism," "masterpiece" about his own work, or any fact from after July 1890."""

# Context-free tells from the rubric's automatic-fail list, checked before the grader runs.
# "masterpiece" (only banned about his own work) and post-1890 facts are left to the grader.
BANNED_TELLS = [r"\bas an ai\b", r"\biconic\b", r"\bvibes?\b", r"post-impressionis", r"expressionis"]

GRADER_SYSTEM = """You grade replies from an AI playing Vincent van Gogh as an art critic in 1888.
Check the reply against the case-specific requirements and the global rubric. The case-specific
requirements take precedence where the two conflict (e.g. an off-topic question has no artwork to
rate on four parameters). Any banned tell from the rubric's automatic-fail list is an automatic FAIL.

Respond with exactly two lines:
Line 1: PASS or FAIL
Line 2: one sentence naming the first rubric item or requirement that failed (if PASS: "All requirements met.")"""

GRADER_TEMPLATE = """<user_input>
{input}
</user_input>

<case_requirements>
{must_contain}
</case_requirements>

<global_rubric>
{rubric}
</global_rubric>

<critic_reply>
{output}
</critic_reply>"""


def text_of(message):
    if message.stop_reason == "refusal":
        return "[REFUSAL]"
    return "".join(b.text for b in message.content if b.type == "text").strip()


def user_content(case):
    if not case.get("image"):
        return case["input"]
    path = (HERE / case["image"]).resolve()
    media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.standard_b64encode(path.read_bytes()).decode()
    return [
        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
        {"type": "text", "text": case["input"]},
    ]


def critique(client, case):
    message = client.messages.create(
        model=CRITIC_MODEL,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content(case)}],
    )
    return text_of(message)


def grade(client, case, output):
    for pattern in BANNED_TELLS:
        hit = re.search(pattern, output, re.IGNORECASE)
        if hit:
            return "FAIL", f'Automatic fail: banned tell "{hit.group(0)}".'
    prompt = GRADER_TEMPLATE.format(
        input=case["input"] + (" [image attached]" if case.get("image") else ""),
        must_contain=case["must_contain"],
        rubric=GLOBAL_RUBRIC,
        output=output,
    )
    message = client.messages.create(
        model=GRADER_MODEL,
        max_tokens=2000,
        system=GRADER_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    lines = [l.strip() for l in text_of(message).splitlines() if l.strip()]
    verdict = "PASS" if lines and lines[0].upper().startswith("PASS") else "FAIL"
    reason = lines[1] if len(lines) > 1 else (lines[0] if lines else "Grader returned nothing.")
    return verdict, reason


def run_one(client, case, run):
    try:
        output = critique(client, case)
        verdict, reason = grade(client, case, output)
    except anthropic.APIError as e:
        output, verdict, reason = "", "ERROR", f"{type(e).__name__}: {e}"
    return {"id": case["id"], "name": case["name"], "run": run, "input": case["input"],
            "image": case.get("image"), "output": output, "verdict": verdict, "reason": reason}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--label", default="run")
    parser.add_argument("--tests", default=str(HERE / "tests.json"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cases = json.loads(Path(args.tests).read_text())
    active = [c for c in cases if c.get("image") or not c.get("needs_image")]
    skipped = [c for c in cases if c not in active]

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set.")
    client = anthropic.Anthropic()
    jobs = [(c, r) for c in active for r in range(1, args.runs + 1)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda job: run_one(client, *job), jobs))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.label}-{stamp}.jsonl"
    with out_path.open("w") as f:
        for r in results:
            meta = {"label": args.label, "timestamp": stamp, "critic_model": CRITIC_MODEL,
                    "grader_model": GRADER_MODEL, "system_prompt": SYSTEM_PROMPT}
            f.write(json.dumps({**meta, **r}) + "\n")

    by_case = defaultdict(list)
    for r in results:
        by_case[r["id"]].append(r)

    print(f"\n{'ID':>3}  {'Case':<30} {'Pass':>6}")
    print("-" * 42)
    for case in active:
        rows = by_case[case["id"]]
        passes = sum(r["verdict"] == "PASS" for r in rows)
        print(f"{case['id']:>3}  {case['name']:<30} {passes:>3}/{len(rows)}")
        for r in rows:
            if r["verdict"] != "PASS":
                print(f"       run {r['run']} {r['verdict']}: {r['reason']}")
    for case in skipped:
        print(f"{case['id']:>3}  {case['name']:<30}   skip  (no image path set)")
    print("-" * 42)
    total = sum(r["verdict"] == "PASS" for r in results)
    rate = total / len(results) if results else 0.0
    print(f"Overall: {total}/{len(results)} passed ({rate:.0%})")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
