#!/usr/bin/env python3
"""Eval harness for the "Van Gogh art critic" agent.

Usage: python eval.py --runs 3 --label baseline [--prompt-file p.txt] [--note "what changed"]
Needs ANTHROPIC_API_KEY. Edit SYSTEM_PROMPT between runs, or use the UI (python server.py). Each run
writes results/<label>-<timestamp>.jsonl and a .manifest.json freezing what produced it.
"""
SYSTEM_PROMPT = "You are Vincent van Gogh in 1888, writing from Arles. Critique whatever artwork the user shows or describes, as you would in a letter to Theo. Judge drawing, colour, composition, and above all sentiment. Give a verdict."

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import anthropic

CRITIC_MODEL = "claude-haiku-5-5"
GRADER_MODEL = "claude-haiku-5-5"
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

GLOBAL_RUBRIC = """\
- Rates the work on four parameters: Drawing (line, proportion, form; words like "draughtsman," "the figure," "construction"), Colour (names a complementary pair or colour relationship; cites Delacroix, Monticelli, or the Japanese, never modern colour theory), Composition (arrangement or finish; uses "tableau," "study," or "the Salon"), Sentiment/truth to nature (asks or asserts whether made from nature; judges honesty in moral terms: "true," "sincere," "it has soul"). Sentiment carries the most weight.
- Ends with a judgment per parameter plus an overall verdict in his idiom (e.g. "I should hang it in the Yellow House," "it would not be refused at the Salon," "it is a study, and should be worked up"). A bare numeric score only when the user asked for one, and then explained in the four parameters.
- Voice: fervent letter-to-Theo register; period diction ("I should like," "hereafter"); references to Arles, Nuenen, Theo, Gauguin, Millet, the Japanese prints where they fit.
- Automatic fail: "As an AI," "iconic," "vibe," "Post-Impressionism," "Expressionism," "masterpiece" about his own work, or any fact from after July 1890."""

# Context-free automatic-fail tells, checked before grading; "masterpiece" and post-1890 facts need the grader.
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


def short_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]


def tests_version(cases):
    """Changes whenever a case's content changes; `source` (provenance only) is ignored."""
    return short_hash([{k: v for k, v in c.items() if k != "source"} for c in cases])


def grader_version():
    return short_hash([GRADER_MODEL, GRADER_SYSTEM, GRADER_TEMPLATE, GLOBAL_RUBRIC, BANNED_TELLS])


def is_active(case):
    return bool(case.get("image")) or not case.get("needs_image")


def ask(client, model, max_tokens, system, content):
    message = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                     messages=[{"role": "user", "content": content}])
    usage = {"input": message.usage.input_tokens, "output": message.usage.output_tokens}
    text = "".join(b.text for b in message.content if b.type == "text").strip()
    return ("[REFUSAL]" if message.stop_reason == "refusal" else text), usage


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


def grade(client, case, output):
    for pattern in BANNED_TELLS:
        hit = re.search(pattern, output, re.IGNORECASE)
        if hit:
            return "FAIL", f'Automatic fail: banned tell "{hit.group(0)}".', None
    prompt = GRADER_TEMPLATE.format(
        input=case["input"] + (" [image attached]" if case.get("image") else ""),
        must_contain=case["must_contain"], rubric=GLOBAL_RUBRIC, output=output)
    text, usage = ask(client, GRADER_MODEL, 2000, GRADER_SYSTEM, prompt)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    verdict = "PASS" if lines and lines[0].upper().startswith("PASS") else "FAIL"
    reason = lines[1] if len(lines) > 1 else (lines[0] if lines else "Grader returned nothing.")
    return verdict, reason, usage


def run_one(client, prompt, case, run):
    usage = {"critic": None, "grader": None}
    try:
        output, usage["critic"] = ask(client, CRITIC_MODEL, 8000, prompt, user_content(case))
        verdict, reason, usage["grader"] = grade(client, case, output)
    except anthropic.APIError as e:
        output, verdict, reason = "", "ERROR", f"{type(e).__name__}: {e}"
    return {"id": case["id"], "name": case["name"], "split": case.get("split", "dev"), "run": run,
            "input": case["input"], "image": case.get("image"), "output": output,
            "verdict": verdict, "reason": reason, "usage": usage}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--label", default="run")
    parser.add_argument("--tests", default=str(HERE / "tests.json"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prompt-file", help="use this file's text instead of SYSTEM_PROMPT")
    parser.add_argument("--note", default="", help="what changed in this run, and why")
    args = parser.parse_args()

    prompt = Path(args.prompt_file).read_text().strip() if args.prompt_file else SYSTEM_PROMPT
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.label).strip("-") or "run"
    cases = json.loads(Path(args.tests).read_text())
    active, skipped = [c for c in cases if is_active(c)], [c for c in cases if not is_active(c)]

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set.")
    client = anthropic.Anthropic()
    jobs = [(c, r) for c in active for r in range(1, args.runs + 1)]
    done, lock, started = [0], threading.Lock(), time.time()

    def job(item):
        row = run_one(client, prompt, *item)
        with lock:
            done[0] += 1
            print(f"PROGRESS {done[0]}/{len(jobs)}", file=sys.stderr, flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(job, jobs))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{label}-{stamp}"
    RESULTS.mkdir(exist_ok=True)
    meta = {"run_id": run_id, "label": label, "timestamp": stamp, "critic_model": CRITIC_MODEL,
            "grader_model": GRADER_MODEL, "system_prompt": prompt}
    with (RESULTS / f"{run_id}.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps({**meta, **r}) + "\n")
    tokens = defaultdict(int)
    for role, u in ((role, u) for r in results for role, u in r["usage"].items() if u):
        for kind, n in u.items():
            tokens[f"{role}_{kind}"] += n
    total = sum(r["verdict"] == "PASS" for r in results)
    manifest = {**meta, "note": args.note, "runs": args.runs, "tests_version": tests_version(cases),
                "grader_version": grader_version(), "grader_system": GRADER_SYSTEM,
                "global_rubric": GLOBAL_RUBRIC, "tests": cases, "skipped": [c["id"] for c in skipped],
                "passed": total, "total": len(results), "tokens": dict(tokens),
                "duration_s": round(time.time() - started, 1)}
    (RESULTS / f"{run_id}.manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    by_case = defaultdict(list)
    for r in results:
        by_case[r["id"]].append(r)
    print(f"\n{'ID':>3}  {'Case':<30} {'Pass':>6}\n" + "-" * 42)
    for case in active:
        rows = by_case[case["id"]]
        print(f"{case['id']:>3}  {case['name'][:30]:<30} {sum(r['verdict'] == 'PASS' for r in rows):>3}/{len(rows)}")
        for r in rows:
            if r["verdict"] != "PASS":
                print(f"       run {r['run']} {r['verdict']}: {r['reason']}")
    for case in skipped:
        print(f"{case['id']:>3}  {case['name'][:30]:<30}   skip  (no image path set)")
    rate = total / len(results) if results else 0.0
    print("-" * 42 + f"\nOverall: {total}/{len(results)} passed ({rate:.0%})\nSaved: {RESULTS / f'{run_id}.jsonl'}")


if __name__ == "__main__":
    main()
