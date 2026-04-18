# Paper-Worthiness Rubric

What counts as a paper-worthy story in the AIMM curator's judgment.
Read fresh on every AIMM cycle — edit anytime to tune the bar; no
redeploy needed.

A story is paper-worthy when it would plausibly belong in a case-study
write-up of the autonomous dev team. Not every shipped story clears
this bar, and that is fine: the rubric is a filter, not a gate.

---

## The four criteria

A story must satisfy **all four** to be considered paper-worthy.

### 1. Concrete

The work produces a specific, named artifact: a file, a metric, a
dashboard, a fixed bug, a new mechanism. Not "explored X" or
"investigated Y" — "shipped Z that does W."

If the story description could apply to ten different systems without
editing, it is too abstract. Paper-worthy stories name the exact file
path, the exact failure mode, or the exact number that moved.

### 2. Reproducible

The outcome can be re-derived from the commit, the test suite, and the
logs. Someone reading the case study next year should be able to point
to the evidence without asking "how did you measure that?"

Reproducibility does not require determinism — a flaky failure mode is
still reproducible if the conditions under which it fires are
documented. But anecdotes ("I saw this once") do not clear the bar.

### 3. Narrative-forward

The story has a *why* that is worth telling, not just a *what*. A
paper-worthy story answers: what surprised us, what did we learn, what
would we do differently, what does this imply about autonomous systems
in general?

Pure maintenance work (bump a dependency, rename a variable) is
valuable but not narrative-forward unless the renaming surfaces a
hidden coupling, or the dependency bump exposes an incompatibility
worth writing about.

### 4. Has an observable metric

There exists a number, a log line, a commit count, a test result, a
dashboard value, or a before/after comparison that a reader can point
to as evidence. "The bot feels more stable" is not a metric.
"Crashes-per-day dropped from 4.2 to 0.1 over the following week" is.

The metric does not have to be novel; it can be an existing counter
that moved. But without a number, the story is an opinion.

---

## Examples

### Paper-worthy

- **Stall-detector race.** (concrete: specific timing window in
  `aim/worker.py`; reproducible: log lines one second apart are in the
  commit history; narrative: "we found a bug by letting the system run,
  not by writing a test"; metric: 1 false-positive cancellation
  observed, fix verified by 0 false positives in the following 200
  executions.)
- **425 commits in 24 hours.** (concrete: named `git log` query;
  reproducible: run the query; narrative: threshold crossing that
  makes "autonomous dev team" operational; metric: 425.)

### Not paper-worthy (but still worth shipping)

- **Refactor the Discord command dispatcher.** (concrete ✓,
  reproducible ✓, narrative ✗ — internal cleanup with no surprising
  learning, metric ✗ — no user-visible number moved.)
- **Investigate why tests are slow.** (concrete ✗ — "investigate" is
  not an artifact; does not clear criterion 1 even if it succeeds.)

---

## How the curator uses this

On every AIMM cycle, the curator reads this file and the in-flight
story descriptions, then asks claude -p to score each story against
the four criteria (pass / fail / unclear per criterion). A story is
tagged `paper-worthy` only when all four criteria pass.

Failures on one or two criteria do not veto the story — they just keep
it out of the case-study pile. The story still ships if the executor
accepts it.

---

## Editing notes

- Re-numbering, renaming, or adding criteria takes effect on the next
  AIMM cycle.
- Changing the rubric does not retroactively rescore old stories.
- Keep examples current — stale examples confuse the scoring prompt.
