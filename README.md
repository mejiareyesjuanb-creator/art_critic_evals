# Art critic evals

An eval harness and local web UI for a "Van Gogh art critic" agent: a critic model answers each
test case in character, a grader model checks the reply against the case's requirements and a
global rubric, and every run is saved so you can track and compare prompts over time.

## Setup

```
pip install anthropic
export ANTHROPIC_API_KEY=...        # your key; never commit it
```

## Use the UI

```
python server.py                    # then open http://localhost:8000
```

- **Runs** shows the headline pass grade, a pass-rate trend across runs, and every run with its
  versions, cost and note. Tick two runs to compare them.
- **Run detail** shows each case and run: the input, what the grader checked, the critic's full
  output, and the grader's reason. Mark your own Pass/Fail on any verdict to measure how far you
  can trust the grader.
- **Compare** puts two runs side by side: the change in pass rate, a word diff of the system prompts,
  per-case regressions and improvements, and both runs' outputs.
- **Edit & run** is where you change the system prompt, runs per case, label and note, edit or
  add test cases (with images), import cases, and click **Run eval**.

The server only listens on `127.0.0.1`, and your API key stays in its environment.

## Use the command line

```
python eval.py --runs 3 --label baseline
python eval.py --runs 3 --label try-structure --prompt-file prompt.txt --note "Ask for a per-parameter verdict"
```

## Files

| Path | What it is |
|---|---|
| `eval.py` | The harness: `SYSTEM_PROMPT`, `GLOBAL_RUBRIC`, the grader, the runner |
| `server.py` | The local UI server (standard library only) |
| `ui/index.html` | The UI |
| `tests.json` | The working test set |
| `images/` | Images attached to test cases (named by content hash) |
| `results/<run>.jsonl` | Every input, output, verdict, reason and token count of one run |
| `results/<run>.manifest.json` | The frozen record of what produced the run (see below) |
| `results/<run>.reviews.json` | Your Pass/Fail verdicts on that run's grader calls |

## Importing test cases

In **Edit & run → Import cases…**, from:

- **A JSON file**: a list of `{name, input, must_contain, image?, split?, needs_image?}`.
- **A CSV file or Google Sheet**: a header row with `name`, `input`, `must_contain` (required) and
  optional `image` (a URL or a path in this folder), `split` (`dev` or `holdout`) and `needs_image`.
  The UI has a downloadable template.
- **A Google Doc**: either a JSON list, or cases separated by a `---` line:

  ```
  Name: Rival
  Input: What do you think of Gauguin's Vision After the Sermon?
  Must: Personal, slightly wounded voice; credits the colour.
  ---
  Name: ...
  ```

A Google Sheet or Doc must be shared as **Anyone with the link can view**. For a private one,
download it as CSV or JSON and upload that file instead. The app never asks for your Google login.

Every import is checked and shown as a preview first. Rows with errors are left out and listed,
and inputs that duplicate an existing case are flagged. Imported cases get new IDs and a `source`
field recording where they came from. Image URLs are downloaded into `images/`. Cases are
**copied in**, so later edits to the Sheet or Doc don't change your test set; re-import to pick
them up. Nothing changes until you click **Save test set**.

## Best practices this follows

1. **Every run is a frozen record.** The manifest stores the system prompt, the full test set,
   the rubric, the grader instructions, the models, runs per case, token use and your note.
2. **Change one thing at a time.** The test set and the grader each have a version hash. Compare
   warns when the test set, grader, critic model, or the prompt as well as something else changed.
   Runs always use the *saved* test set, so you can't run with unsaved edits.
3. **Compare runs side by side**, with regressions listed first and a prompt diff.
4. **Read the outputs.** Every reply and grader reason is one click away.
5. **Check the grader.** Your Pass/Fail marks give an agreement rate and a pass rate with your
   verdicts applied.
6. **Don't over-read small changes.** Pass rates show `k/N` and a 95% interval, and Compare says
   when the intervals overlap.
7. **Show cost and progress.** Calls and estimated cost before you run, live progress during it,
   and tokens and cost after.
8. **Write down why.** Each run has a note.
9. **Keep the API key on the server.** It stays in the environment and never reaches the browser.
10. **Hold some cases back.** Mark cases `holdout` and their pass rate is reported separately, so
    you can check that prompt gains carry over to cases you didn't tune against.

The headline **pass grade** is the share of case runs that passed, with a band (Strong ≥ 90%,
Good ≥ 70%, Weak ≥ 40%, otherwise Failing). The bands are set in `GRADE_BANDS` at the top of
`ui/index.html`.
