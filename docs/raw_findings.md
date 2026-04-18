# Raw Findings — Autonomous Dev Team on Claude Code

Live capture of observations, surprises, and system-level patterns discovered
while running TK (Technomancer) + FA (40Acres) autonomous dev teams.

Append new findings to the top. Each finding: date, headline, what happened,
why it matters, supporting evidence (commit/log pointer).

> File is gitignored — personal case-study material.

---

## 2026-04-18 — 425 commits in 24 hours, full branch-test-merge-deploy workflow

**What:** The two AIM daemons (TK + FA) shipped 425 commits to `main`
across both repos in one calendar day, each one going through the full
safe_update.py workflow: branch-from-main → claude -p edits → pytest →
merge to main → push origin → README regen → push public mirror.

**Why it matters:** This is the headline throughput number. At this
volume, "autonomous dev team" isn't aspirational — it's operational. The
thing to prove in the paper is that these commits are *quality*, not
just volume: tests pass, main stays green, no rollbacks. The shape of
the evidence is 425 clean merges with passing CI and no main-branch
reverts.

**Evidence:** `git log --oneline --since="yesterday" --until="today"
main | wc -l` on both repos. README auto-regen touches at end of every
merge, so the count is a fair lower bound (excludes failed/reverted).

---

## 2026-04-18 — Stalled-but-shipped 1-second race

**What:** At 04:16:19 the FA worker logged "Execution FA-116 stalled
for 600s, cancelling". **One second later** at 04:16:20 the same
worker logged "FA-116 completed successfully." The stall timer and the
actual claude -p completion event raced.

**Why it matters:** Stall-detection based on stdout-silence is a
probabilistic approximation, not a ground truth. TK-567 earlier in the
session fixed the *Jira-state* consequence (don't mark a story Failed
if it actually shipped to main), but the *warning noise* in the logs
is still there. Two research lessons: (a) signal-to-truth gaps are
real in autonomous systems and need post-hoc reconciliation, (b) the
"last stdout line" heuristic caught an edge case that only appears
when the subprocess terminates *during* the stall-check window.

**Evidence:** Log lines
`[FA-WRK] 2026-04-18 04:16:19 ... stalled for 600s, cancelling`
followed by `[FA-WRK] 2026-04-18 04:16:20 ... FA-116 completed
successfully`.

---

## 2026-04-18 — Orphan auto-commit stole concurrent work

**What:** A worker process was running TK-605 on branch
`2026-04-17-224604-TK-605`. Its AIM manager had been killed by the
operator (me) at 22:48, but the worker subprocess didn't know — it kept
running claude -p. Meanwhile the operator aborted the branch, created a
new feature branch (`2026-04-17-225002-story-format`), and started
editing files. At 22:50:50 the orphan worker's claude -p subprocess
finally exited, and the executor's auto-commit fallback ran —
picking up the *operator's in-progress files* (a new module +
splitter edits) and committing them to the new feature branch under a
`[TK-605] Auto-commit from claude -p session` label.

**Why it matters:** The autonomy boundary doesn't know about adjacent
working trees. The auto-commit fallback exists to handle
narrate-without-commit (a real failure mode where claude finishes
without running `git commit`); but without worktree isolation, it
scoops up anything dirty in the repo. This is exactly what the TK-630
parallel-execution epic is designed to fix — git worktree per slot.
It's a clean case study for *why* isolation matters and what happens
without it.

**Evidence:** Commit `d89c57b1bf349b27bb30b3d17b71cee0d19bfed8` titled
`[TK-605] Auto-commit from claude -p session` containing changes the
worker had no business touching.

---

## 2026-04-18 — Splitter recursive decomposition (no auto-veto)

**What:** The splitter used to auto-veto on splitter-children (a
failed story that was itself a split couldn't be re-split), to prevent
infinite decomposition. TK-566 removed that cap, replacing it with a
depth-counter alert at generation N. Tonight this paid off: TK-589
(hub-feedback) failed → split into TK-647/648/649/650 → TK-649 itself
failed → split (gen 2) into TK-651/652/653 → all three gen-2 children
shipped cleanly.

**Why it matters:** Recursive decomposition-until-it-ships is a
distinct autonomy pattern from "try once, fail, escalate." The system
treats its own failures as a prompt to decompose further. The
depth-cap alert means operator stays in the loop only when
decomposition *keeps* failing — not on every routine split.

**Evidence:** Split log `[Splitter] Split TK-649 into 3 stories (gen
2): TK-651, TK-652, TK-653`. All three reached Done.

---

## 2026-04-18 — Haiku-brain, Opus-worker split

**What:** The AIM brain (decision classification: should we assign
work, create work, or wait?) was moved from opus to haiku-4-5. The
worker (actual code generation, tests, commits) stays on opus. Brain
fires every ~30s cycle; worker fires per story.

**Why it matters:** The cost curve should show a step-down. Brain is a
high-frequency low-complexity task — classification into four buckets —
so haiku is adequate. Worker is low-frequency high-complexity — a
full PR — so opus earns its cost. This is the first deliberate
model-routing decision in the system, and it separates the two cost
drivers cleanly. The research thread is "where's the cheap part of
the task graph and can we recognize it automatically."

**Evidence:** `agent/anthropic_shim.py` + brain.py `--model
claude-haiku-4-5` flag on the claude -p call. TK-613 (story_model_usage
table) will let us quantify the savings once the daily rollup lands.

---

## 2026-04-18 — Narrate-without-commit as a pattern, not a bug

**What:** A recurring failure mode: claude -p finishes its session,
describes what it did, but *never ran `git commit`*. The file edits
are in the working tree but untracked by git. Without intervention,
safe_update.py would see "nothing to merge" and the work is effectively
lost.

**Fix:** Executor now runs an auto-commit fallback (a) after claude -p
exits, and (b) pre-merge (catches test-phase side effects like a
Django `db.sqlite3` being written during pytest).

**Why it matters:** LLM output is narrative by nature. Claude will
describe success even when a discrete required step was skipped. The
fix isn't to prompt-engineer harder; it's to *add the step as code*.
Quoted lesson from Jeremy: "if it's claude -p missing a step, we
change the workflow to include the step as code." That's an
autonomy-design principle worth naming in the paper.

---

## 2026-04-18 — Multi-project drift surfacing hardcoded coupling

**What:** Extending the system to manage a second project (40Acres
Django resume site, alongside Technomancer) surfaced a series of
hardcodes that had looked like features:

- `local_agent_dir = Path(__file__).parent.parent` — worked for TK,
  ran TK's tests against FA's repo.
- `_cleanup_orphaned_processes` in AIM manager killed peer AIM's
  worker PIDs because it didn't know about peer projects.
- sys.modules shim install order assumed one project's env was loaded
  at import time.
- `shlex.split` assumed POSIX escaping on Windows.

**Why it matters:** A "single-tenant" system feels generic until you
run a second tenant. Each tenant exposes another implicit assumption.
This is the autonomous-system version of the classic
"micro-service-ification" pain. Worth logging as a pattern: autonomy
systems accumulate implicit environment coupling faster than
hand-written ones because they're generated by LLMs that pattern-match
on the existing repo.

---

## Template for new entries

```
## YYYY-MM-DD — Headline (one line)

**What:** {factual description, 1-3 sentences}

**Why it matters:** {the system-level or research-angle insight this
exposes; 2-4 sentences}

**Evidence:** {commit SHA, log line, screenshot path, story key, etc.}
```
