# Hosted dashboard

A read-only view of every eval run, published on claude.ai as a Data dashboard:
https://claude.ai/artifact/4ZQZu1iPuLgBKRzLc4y2y6

It shows the latest takeaway, the pass rate per run (with dev and holdout split out), a side-by-side
comparison of any two runs (prompt diff and per-case changes), and a reader for every graded reply.
It does not run evals: use the local UI (`server.py`) for that.

## Files

| Path | What it is |
|---|---|
| `index.html` | The dashboard page, exactly as published (a fragment the dashboard runtime renders, not a standalone page) |
| `export.py` | Builds the two datasets the page reads from `results/` |
| `preview.js` | Renders the page locally with a stand-in runtime and saves screenshots |
| `build/` | Export and preview output (git-ignored; rebuild any time) |

## Datasets

The page reads two datasets, both produced by `export.py`:

- **`runs`**: one row per eval run (label, time, prompt, pass rate and 95% interval, dev/holdout rates,
  cost, note, test set and grader versions).
- **`results`**: one row per graded reply (case, split, try number, the critic's reply, the grader's
  verdict and reason, and your own Pass/Fail review if you gave one).

## Updating it after new runs

The published copy does not update by itself. After a new run:

1. `python dashboard/export.py` (writes `dashboard/build/runs.json` and `results.json`; no API key needed).
2. Optionally check the page locally:
   `cd dashboard && npm install && node preview.js` (screenshots land in `dashboard/build/`).
3. Ask Claude in a session with this repo to "republish the dashboard". That uploads the two JSON files
   as new versions of the `runs` and `results` datasets and, if `index.html` changed, the page.

Editing the page: change `index.html`, run the preview, then republish. Commit `index.html` so the repo
and the published copy stay in step.
