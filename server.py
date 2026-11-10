#!/usr/bin/env python3
"""Local web UI for the Van Gogh critic eval.

Run:  ANTHROPIC_API_KEY=... python server.py      then open http://localhost:8000
Standard library only. Listens on 127.0.0.1; the API key stays in this process's environment
and is passed to eval.py, never to the browser.
"""
import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import eval as ev

HERE = ev.HERE
TESTS = HERE / "tests.json"
RESULTS = ev.RESULTS
IMAGES = HERE / "images"
UI = HERE / "ui" / "index.html"
# $ per million tokens (input, output) for prompts under 100K tokens.
PRICES = {"claude-haiku-5-5": (0.10, 0.50)}
MAX_BODY = 12 * 1024 * 1024
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMPORT_ROWS = 500
IMAGE_TYPES = {"image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif", "image/webp": ".webp"}
RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
SPLITS = ("dev", "holdout")

job = {"running": False, "done": 0, "total": 0, "label": None, "run_id": None, "error": None}
job_lock = threading.Lock()
file_lock = threading.Lock()


class ApiError(Exception):
    pass


# ---------- runs ----------

def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def load_rows(run_id):
    path = RESULTS / f"{run_id}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wilson(k, n, z=1.96):
    """95% Wilson interval for a pass rate; honest about small samples."""
    if n == 0:
        return [0.0, 0.0]
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, centre - half), min(1.0, centre + half)]


def cost_usd(tokens, manifest):
    if not tokens:
        return None
    ci, co = PRICES.get(manifest.get("critic_model"), (0, 0))
    gi, go = PRICES.get(manifest.get("grader_model"), (0, 0))
    return (tokens.get("critic_input", 0) * ci + tokens.get("critic_output", 0) * co
            + tokens.get("grader_input", 0) * gi + tokens.get("grader_output", 0) * go) / 1e6


def case_hash(case):
    return ev.short_hash({k: v for k, v in case.items() if k != "source"})


def summarize(manifest, rows, reviews):
    n = len(rows)
    passed = sum(r["verdict"] == "PASS" for r in rows)
    per_case = {}
    for r in rows:
        c = per_case.setdefault(str(r["id"]), {"passed": 0, "total": 0})
        c["total"] += 1
        c["passed"] += r["verdict"] == "PASS"
    splits = {}
    for s in SPLITS:
        sub = [r for r in rows if r.get("split", "dev") == s]
        if sub:
            k = sum(r["verdict"] == "PASS" for r in sub)
            splits[s] = {"passed": k, "total": len(sub), "ci": wilson(k, len(sub))}
    reviewed = agree = adjusted = 0
    for r in rows:
        human = reviews.get(f"{r['id']}-{r['run']}", {}).get("human")
        if human:
            reviewed += 1
            agree += human == r["verdict"]
        adjusted += (human or r["verdict"]) == "PASS"
    return {
        "run_id": manifest["run_id"], "label": manifest["label"], "timestamp": manifest["timestamp"],
        "note": manifest.get("note", ""), "runs": manifest.get("runs"),
        "critic_model": manifest.get("critic_model"), "grader_model": manifest.get("grader_model"),
        "tests_version": manifest.get("tests_version"), "grader_version": manifest.get("grader_version"),
        "passed": passed, "total": n, "rate": passed / n if n else 0.0, "ci": wilson(passed, n),
        "errors": sum(r["verdict"] == "ERROR" for r in rows),
        "solid_cases": sum(c["passed"] == c["total"] for c in per_case.values()),
        "case_count": len(per_case), "splits": splits,
        "reviewed": reviewed, "agreement": agree / reviewed if reviewed else None,
        "human_adjusted_rate": adjusted / n if n else 0.0,
        "tokens": manifest.get("tokens"), "cost_usd": cost_usd(manifest.get("tokens"), manifest),
        "duration_s": manifest.get("duration_s"),
    }


def list_runs():
    runs = []
    for path in RESULTS.glob("*.manifest.json"):
        manifest = json.loads(path.read_text())
        run_id = manifest["run_id"]
        runs.append(summarize(manifest, load_rows(run_id), read_json(RESULTS / f"{run_id}.reviews.json", {})))
    return sorted(runs, key=lambda r: r["timestamp"])


def run_detail(run_id):
    if not RUN_ID.match(run_id) or not (RESULTS / f"{run_id}.manifest.json").exists():
        raise ApiError(f"No run named {run_id}.")
    manifest = json.loads((RESULTS / f"{run_id}.manifest.json").read_text())
    rows = load_rows(run_id)
    reviews = read_json(RESULTS / f"{run_id}.reviews.json", {})
    cases = {str(c["id"]): {**c, "hash": case_hash(c)} for c in manifest.get("tests", [])}
    return {"summary": summarize(manifest, rows, reviews), "manifest": {**manifest, "tests": None},
            "cases": cases, "rows": rows, "reviews": reviews}


# ---------- test cases ----------

def resolve_image(path):
    full = (HERE / path).resolve()
    if HERE not in full.parents or not full.is_file():
        raise ApiError(f"Image not found: {path}")
    return full


def save_image(data, ext):
    if len(data) > MAX_IMAGE_BYTES:
        raise ApiError("Image is larger than 5 MB.")
    IMAGES.mkdir(exist_ok=True)
    name = hashlib.sha256(data).hexdigest()[:16] + ext
    (IMAGES / name).write_bytes(data)
    return f"images/{name}"


def download_image(url):
    req = urllib.request.Request(url, headers={"User-Agent": "art-critic-evals"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            ctype = resp.headers.get_content_type()
            data = resp.read(MAX_IMAGE_BYTES + 1)
    except (urllib.error.URLError, ValueError) as e:
        raise ApiError(f"Could not download image {url}: {e}")
    if ctype not in IMAGE_TYPES:
        raise ApiError(f"{url} is not a PNG, JPEG, GIF or WebP image (got {ctype}).")
    return save_image(data, IMAGE_TYPES[ctype])


def clean_case(raw, require_id):
    """Validate one case and return it in tests.json shape (defaults left out so versions stay stable)."""
    case = {}
    if require_id:
        if not isinstance(raw.get("id"), int):
            raise ApiError("every case needs an integer id")
        case["id"] = raw["id"]
    for field in ("name", "input", "must_contain"):
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ApiError(f"missing {field}")
        case[field] = value.strip()
    image = raw.get("image") or None
    if image:
        if re.match(r"^https?://", image):
            image = download_image(image)
        resolve_image(image)
    case = {"id": case.get("id"), "name": case["name"], "input": case["input"], "image": image,
            "must_contain": case["must_contain"]}
    if not require_id:
        del case["id"]
    split = str(raw.get("split") or "dev").strip().lower()
    if split not in SPLITS:
        raise ApiError(f'split must be "dev" or "holdout", not "{split}"')
    if split != "dev":
        case["split"] = split
    if str(raw.get("needs_image", "")).strip().lower() in ("true", "yes", "1"):
        case["needs_image"] = True
    if raw.get("source"):
        case["source"] = str(raw["source"])
    return case


def save_tests(cases):
    if not isinstance(cases, list):
        raise ApiError("cases must be a list")
    cleaned, ids = [], set()
    for i, raw in enumerate(cases, 1):
        try:
            case = clean_case(raw, require_id=True)
        except ApiError as e:
            raise ApiError(f"Case {raw.get('id', i)}: {e}")
        if case["id"] in ids:
            raise ApiError(f"Duplicate case id {case['id']}")
        ids.add(case["id"])
        cleaned.append(case)
    with file_lock:
        TESTS.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False) + "\n")
    return cleaned


FIELD_ALIASES = {"name": "name", "title": "name", "input": "input", "prompt": "input",
                 "must_contain": "must_contain", "must": "must_contain", "criteria": "must_contain",
                 "expected": "must_contain", "image": "image", "split": "split", "needs_image": "needs_image"}
DOC_KEY = re.compile(r"^\s*(" + "|".join(FIELD_ALIASES) + r")\s*:\s*(.*)$", re.IGNORECASE)


def parse_doc(text):
    """A JSON array, or blocks separated by '---' lines with 'Name:', 'Input:', 'Must:' fields."""
    text = text.lstrip("﻿").strip()
    if text.startswith("["):
        return json.loads(text)
    cases = []
    for block in re.split(r"(?m)^\s*-{3,}\s*$", text):
        fields, current = {}, None
        for line in block.splitlines():
            m = DOC_KEY.match(line)
            if m:
                current = m.group(1).lower()
                fields[current] = m.group(2)
            elif current:
                fields[current] += "\n" + line
        if any(v.strip() for v in fields.values()):
            cases.append({k: v.strip() for k, v in fields.items()})
    return cases


def fetch_google(url, kind):
    want = "spreadsheets" if kind == "gsheet" else "document"
    m = re.search(rf"docs\.google\.com/{want}/d/([A-Za-z0-9_-]+)", url or "")
    if not m:
        raise ApiError(f"That doesn't look like a Google {'Sheet' if kind == 'gsheet' else 'Doc'} link.")
    if kind == "gsheet":
        gid = re.search(r"[#?&]gid=(\d+)", url)
        export = f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv"
        export += f"&gid={gid.group(1)}" if gid else ""
    else:
        export = f"https://docs.google.com/document/d/{m.group(1)}/export?format=txt"
    not_shared = ("Google didn't return the file. Share it as 'Anyone with the link can view', "
                  "or download it as CSV/JSON and upload that file instead.")
    try:
        with urllib.request.urlopen(urllib.request.Request(export, headers={"User-Agent": "art-critic-evals"}),
                                    timeout=20) as resp:
            if resp.headers.get_content_type() == "text/html":
                raise ApiError(not_shared)
            return resp.read(MAX_BODY).decode("utf-8-sig")
    except urllib.error.HTTPError:
        raise ApiError(not_shared)
    except urllib.error.URLError as e:
        raise ApiError(f"Could not reach Google: {e.reason}")


def preview_import(body):
    kind, content = body.get("kind"), body.get("content") or ""
    try:
        if kind == "json":
            rows = json.loads(content)
            rows = rows.get("cases", rows) if isinstance(rows, dict) else rows
        elif kind == "csv":
            rows = list(csv.DictReader(io.StringIO(content.lstrip("﻿"))))
        elif kind == "gsheet":
            rows = list(csv.DictReader(io.StringIO(fetch_google(body.get("url"), kind))))
        elif kind == "gdoc":
            rows = parse_doc(fetch_google(body.get("url"), kind))
        else:
            raise ApiError("Unknown import type.")
    except (json.JSONDecodeError, csv.Error) as e:
        raise ApiError(f"Could not parse the {kind.upper()}: {e}")
    if not isinstance(rows, list):
        raise ApiError("Expected a list of test cases.")
    if len(rows) > MAX_IMPORT_ROWS:
        raise ApiError(f"Too many rows ({len(rows)}); import at most {MAX_IMPORT_ROWS} at a time.")
    source = f"{kind}:{body.get('url') or body.get('filename') or 'pasted'}"
    existing = {c["input"].strip().lower() for c in read_json(TESTS, [])}
    cases, errors, warnings = [], [], []
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            errors.append(f"Row {i}: not an object")
            continue
        mapped = {FIELD_ALIASES[k]: v for k, v in
                  ((str(k).strip().lower().replace(" ", "_"), v) for k, v in row.items())
                  if k in FIELD_ALIASES and v not in (None, "")}
        try:
            case = clean_case({**mapped, "source": source}, require_id=False)
        except ApiError as e:
            errors.append(f"Row {i}: {e}")
            continue
        key = case["input"].lower()
        if key in existing:
            warnings.append(f"Row {i} ({case['name']}): same input as an existing case")
        existing.add(key)
        cases.append(case)
    return {"cases": cases, "errors": errors, "warnings": warnings, "source": source}


# ---------- running ----------

def run_job(cmd, prompt_path):
    try:
        proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, tail = [], []
        reader = threading.Thread(target=lambda: out.extend(proc.stdout), daemon=True)
        reader.start()
        for line in proc.stderr:
            m = re.match(r"PROGRESS (\d+)/(\d+)", line)
            if m:
                with job_lock:
                    job["done"], job["total"] = int(m.group(1)), int(m.group(2))
            else:
                tail = (tail + [line])[-20:]
        proc.wait()
        reader.join()
        saved = next((l for l in out if l.startswith("Saved: ")), None)
        with job_lock:
            if proc.returncode == 0 and saved:
                job["run_id"] = Path(saved[7:].strip()).name.removesuffix(".jsonl")
            else:
                job["error"] = "".join(tail[-8:]).strip() or f"eval.py exited with code {proc.returncode}"
    finally:
        os.unlink(prompt_path)
        with job_lock:
            job["running"] = False


def start_run(body):
    prompt = (body.get("prompt") or "").strip()
    runs, label = body.get("runs"), body.get("label") or "run"
    if not prompt:
        raise ApiError("The system prompt is empty.")
    if not isinstance(runs, int) or not 1 <= runs <= 10:
        raise ApiError("Runs must be a whole number from 1 to 10.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ApiError("ANTHROPIC_API_KEY isn't set in the server's environment. Restart the server with it set.")
    with job_lock:
        if job["running"]:
            raise ApiError("A run is already in progress.")
        job.update(running=True, done=0, total=0, label=label, run_id=None, error=None)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(prompt)
    cmd = [sys.executable, str(HERE / "eval.py"), "--runs", str(runs), "--label", label,
           "--prompt-file", f.name, "--note", str(body.get("note") or "")]
    threading.Thread(target=run_job, args=(cmd, f.name), daemon=True).start()
    return status()


def status():
    with job_lock:
        return dict(job)


def state():
    tests = read_json(TESTS, [])
    runs = list_runs()
    latest = runs[-1] if runs else None
    latest_prompt = None
    if latest:
        latest_prompt = json.loads((RESULTS / f"{latest['run_id']}.manifest.json").read_text())["system_prompt"]
    with_tokens = [r for r in runs if r["tokens"]]
    per_row = with_tokens[-1]["cost_usd"] / with_tokens[-1]["total"] if with_tokens else None
    return {"tests": tests, "tests_version": ev.tests_version(tests), "grader_version": ev.grader_version(),
            "critic_model": ev.CRITIC_MODEL, "grader_model": ev.GRADER_MODEL,
            "default_prompt": ev.SYSTEM_PROMPT, "latest_prompt": latest_prompt,
            "cost_per_case_run": per_row, "api_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "job": status()}


# ---------- http ----------

class Handler(BaseHTTPRequestHandler):
    allowed_hosts = {"localhost", "127.0.0.1"}

    def log_message(self, fmt, *args):
        pass

    def send(self, status, body, ctype):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj, status=200):
        self.send(status, json.dumps(obj).encode(), "application/json")

    def host_ok(self):
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        return self.allowed_hosts is None or host in self.allowed_hosts

    def do_GET(self):
        if not self.host_ok():
            return self.send_json({"error": "Host not allowed."}, 403)
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                return self.send(200, UI.read_bytes(), "text/html; charset=utf-8")
            if path.startswith("/images/") and re.match(r"^[0-9a-f]{16}\.(png|jpg|gif|webp)$", path[8:]):
                file = IMAGES / path[8:]
                if file.is_file():
                    ctype = next(t for t, e in IMAGE_TYPES.items() if e == file.suffix)
                    return self.send(200, file.read_bytes(), ctype)
            if path == "/api/state":
                return self.send_json(state())
            if path == "/api/runs":
                return self.send_json(list_runs())
            if path.startswith("/api/runs/"):
                return self.send_json(run_detail(path[len("/api/runs/"):]))
            if path == "/api/status":
                return self.send_json(status())
            self.send_json({"error": "Not found."}, 404)
        except ApiError as e:
            self.send_json({"error": str(e)}, 400)

    def do_POST(self):
        # JSON-only POSTs from an allowed Host: a page on another site can't send these without a CORS preflight.
        if not self.host_ok() or self.headers.get_content_type() != "application/json":
            return self.send_json({"error": "Forbidden."}, 403)
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self.send_json({"error": "Request too large."}, 413)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            path = urlparse(self.path).path
            if path == "/api/tests":
                cases = save_tests(body.get("cases"))
                return self.send_json({"tests": cases, "tests_version": ev.tests_version(cases)})
            if path == "/api/import":
                return self.send_json(preview_import(body))
            if path == "/api/image":
                m = re.match(r"^data:(image/[a-z]+);base64,(.+)$", body.get("data_url") or "", re.S)
                if not m or m.group(1) not in IMAGE_TYPES:
                    raise ApiError("Upload a PNG, JPEG, GIF or WebP image.")
                return self.send_json({"path": save_image(base64.b64decode(m.group(2)), IMAGE_TYPES[m.group(1)])})
            if path == "/api/run":
                return self.send_json(start_run(body))
            if path == "/api/review":
                return self.send_json(save_review(body))
            if path == "/api/note":
                return self.send_json(save_note(body))
            self.send_json({"error": "Not found."}, 404)
        except ApiError as e:
            self.send_json({"error": str(e)}, 400)
        except (json.JSONDecodeError, ValueError) as e:
            self.send_json({"error": f"Bad request: {e}"}, 400)


def save_review(body):
    run_id = str(body.get("run_id") or "")
    run_detail(run_id)  # validates the id
    key = f"{body.get('case_id')}-{body.get('run')}"
    human = body.get("human")
    if human not in ("PASS", "FAIL", None):
        raise ApiError("human must be PASS, FAIL or null")
    path = RESULTS / f"{run_id}.reviews.json"
    with file_lock:
        reviews = read_json(path, {})
        if human is None and not body.get("note"):
            reviews.pop(key, None)
        else:
            reviews[key] = {"human": human, "note": str(body.get("note") or "")}
        path.write_text(json.dumps(reviews, indent=2, ensure_ascii=False))
    return reviews


def save_note(body):
    run_id = str(body.get("run_id") or "")
    run_detail(run_id)
    path = RESULTS / f"{run_id}.manifest.json"
    with file_lock:
        manifest = json.loads(path.read_text())
        manifest["note"] = str(body.get("note") or "")
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return {"note": manifest["note"]}


def main():
    parser = argparse.ArgumentParser(description="Local web UI for the Van Gogh critic eval.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--allow-any-host", action="store_true",
                        help="accept requests whose Host isn't localhost (e.g. behind a preview proxy)")
    args = parser.parse_args()
    if args.allow_any_host:
        Handler.allowed_hosts = None
    RESULTS.mkdir(exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Eval UI on http://localhost:{args.port}  (API key {'set' if os.environ.get('ANTHROPIC_API_KEY') else 'NOT set'})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
