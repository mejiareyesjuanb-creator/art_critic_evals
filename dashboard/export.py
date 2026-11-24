"""Export every saved eval run as the two JSON datasets the hosted dashboard reads.

Run from the repo root:  python dashboard/export.py
Reads results/ directly (no server or API key needed) and writes dashboard/build/runs.json
and dashboard/build/results.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import server  # noqa: E402

OUT = Path(__file__).resolve().parent / "build"


def iso(stamp):
    d, t = stamp.split("-")
    return f"{d[:4]}-{d[4:6]}-{d[6:]}T{t[:2]}:{t[2:4]}:{t[4:]}Z"


def pct(x):
    return None if x is None else round(x * 100)


def split_pct(splits, name):
    s = splits.get(name)
    return pct(s["passed"] / s["total"]) if s and s["total"] else None


def main():
    runs, results = [], []
    for s in server.list_runs():
        d = server.run_detail(s["run_id"])
        runs.append({
            "run_id": s["run_id"], "label": s["label"], "run_at": iso(s["timestamp"]),
            "note": s["note"], "system_prompt": d["manifest"]["system_prompt"], "runs_per_case": s["runs"],
            "passed": s["passed"], "total": s["total"], "pass_pct": pct(s["rate"]),
            "ci_low_pct": pct(s["ci"][0]), "ci_high_pct": pct(s["ci"][1]),
            "solid_cases": s["solid_cases"], "case_count": s["case_count"], "errors": s["errors"],
            "dev_pass_pct": split_pct(s["splits"], "dev"), "holdout_pass_pct": split_pct(s["splits"], "holdout"),
            "reviewed": s["reviewed"], "agreement_pct": pct(s["agreement"]),
            "tests_version": s["tests_version"], "grader_version": s["grader_version"],
            "critic_model": s["critic_model"], "grader_model": s["grader_model"],
            "cost_usd": None if s["cost_usd"] is None else round(s["cost_usd"], 4),
        })
        for r in d["rows"]:
            case = d["cases"].get(str(r["id"]), {})
            review = d["reviews"].get(f"{r['id']}-{r['run']}", {})
            results.append({
                "result_id": f"{s['run_id']}:{r['id']}:{r['run']}", "run_id": s["run_id"], "run_label": s["label"],
                "case_id": r["id"], "case_name": r["name"], "split": r.get("split", "dev"), "run": r["run"],
                "verdict": r["verdict"], "passed": 1 if r["verdict"] == "PASS" else 0, "reason": r["reason"],
                "human_verdict": review.get("human"), "output": r["output"],
                "input": r["input"], "must_contain": case.get("must_contain", ""),
            })
    OUT.mkdir(exist_ok=True)
    (OUT / "runs.json").write_text(json.dumps(runs, ensure_ascii=False, indent=1))
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False))
    print(f"{len(runs)} runs, {len(results)} graded replies -> {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
