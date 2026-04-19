# Raw Findings — Autonomous Dev Team on Claude Code

Live capture of observations, surprises, and system-level patterns discovered
while running TK (Technomancer) + FA (40Acres) autonomous dev teams.

Append new findings to the top. Each finding: date, headline, what happened,
why it matters, supporting evidence (commit/log pointer).

> File is gitignored — personal case-study material.

---

## 2026-04-19 — We burned through Anthropic's MAX 20x weekly quota in under 5 days

**What:** By day 5 of running Technomancer + 40Acres autonomous daemons,
the Anthropic Max 20x subscription (the highest non-enterprise tier
Anthropic sells) showed **94% of the weekly "All models" bucket
consumed**, with Sonnet untouched at 0%. Resets Friday 4am. That means
the two projects shipped ~500 commits on the full Max 20x budget plus
some. The *cap* is the constraint, not the cost per story.

**Why it matters:** This is its own headline. The closest competitor
pitch ("AI coding agent that writes code for you") implies a task-per-
day cadence. Technomancer's cadence is task-per-hour sustained for 5
days on the most expensive consumer plan available — and it ran out.
Two useful angles for the paper:

1. **Saturation is the autonomy test.** If an autonomous system doesn't
   hit the ceiling of whatever resource you give it, it's probably not
   being aggressive enough about finding work to do. Technomancer's
   brain generates new stories on idle cycles, the splitter
   recursively decomposes failures, AIMM proposes hypotheses, AIV
   scores every merge — every role burns compute by design. The
   resource cap surfaces as a natural back-pressure.

2. **Tiered model routing is the obvious mitigation.** The Sonnet
   bucket was at 0% because I gutted Ollama earlier in the week and
   forced everything through claude -p (Opus for workers, Haiku for
   brain). Re-introducing ollama for classification-shaped work (AIM
   brain decision, dedup judge, AIMM observer, AIMM suggester) and
   routing code generation to Sonnet instead of Opus is a 10× cost
   reduction without architectural change. That is: the system is
   dominated by Opus calls that don't need Opus-grade reasoning.

**The specific trigger:** at 94% weekly consumption, session-level
usage was still 28% on the current rolling window. So the weekly cap
is hit by *sustained* operation, not a burst. A less aggressive
system would never approach the weekly ceiling in a week; ours hit it
by day 5. Measure of how autonomous autonomy actually gets.

**Evidence:** Anthropic usage dashboard screenshot (not committed,
personal). Raw number: 94% of Max 20x weekly, day 5 of the reset
window, with daemons stopped for parts of days 3-4.

**Action taken:** AIMs stopped for the remainder of the week; test
suite pruned (`--reruns` removed, top-10 slow tests mocked). Ollama
routing plan in flight — re-introduce ollama as tier-0 for
classification tasks and save the Claude budget for code generation.

---

## 2026-04-18 — Competitive landscape: what exists vs what Technomancer is

**What:** Survey of adjacent commercial + open-source systems in the
autonomous coding space, with each compared point-by-point against
Technomancer's architecture. The concept of "an AI that writes code"
isn't novel in 2026; every item below proves pieces of it. What's
unusual is the shape of Technomancer's whole.

**Why it matters:** When pitching the project to a hiring manager, the
claim can't be "I built a coding agent" — plenty exist. The claim has to
be "I built a self-governing multi-role dev team on Claude Code." That
requires being able to articulate exactly where each comparable stops
and Technomancer continues. Product differentiation + real-world
comparison is what makes the idea resonate.

### Commercial systems (all solve parts; none the whole)

**Devin (Cognition AI)** — launched March 2024 as "the first autonomous
AI software engineer." Cloud-hosted SaaS, ~$500/month subscription
tier. Architecture: a single agent backed by a sandboxed VM (browser +
IDE + terminal). User assigns a task via natural language; Devin
plans, writes code, runs tests, reports back when done. Session-scoped:
the task is the unit of work. Differences from Technomancer: no ranked
backlog (user is the scheduler), no continuous pipeline (you don't
"set it and forget it" against a Jira board for 24/7 burn), no
multi-role separation (Devin's planning and execution are one agent
talking to itself), no post-merge self-audit (success = session's own
tests pass, not a separate validator). Also cloud-first vs
Technomancer's local-first deployment.

**Sweep AI** — GitHub app launched 2023. Workflow: label a GitHub
issue with `sweep:`, Sweep generates a PR targeting that issue,
human reviews and merges. Focus is bug fixes and small features
from natural-language descriptions. Differences from Technomancer:
single-shot per issue (if the PR is wrong, human handles it; no
recursive decomposition on failure), human-gated merge (no autonomous
ship), no continuous operation (each issue triggers one bot run),
GitHub-native (no Jira integration without scripting).

**GitHub Copilot Workspace + Agents** — Copilot Workspace introduced
late 2023; agents productized through 2024-2025. Inside GitHub: take
an issue, delegate to the agent, get a PR back. Good CI integration
(uses GitHub Actions for validation). Session-scoped — you hand off
one task and receive a PR. Differences from Technomancer: no always-on
daemon scanning a board, no multi-role, human approves and merges, no
self-audit layer.

**Cursor background agents** — IDE feature from 2024. User-triggered:
"run this agent on this task in the background while I keep coding."
Good for parallelizing work between yourself and a model. Differences
from Technomancer: user augmentation, not replacement — tasks come
from the user typing them, not from a ranked board. No continuous
loop. No multi-role. Single-task scoped per invocation.

**Claude Code** (what Technomancer runs on) — Anthropic's official CLI
coding agent. Interactive by default: user prompts → Claude edits →
user reviews. CLI flag `-p` provides non-interactive one-shot prompts,
which Technomancer uses as the hands of the AIW role. Claude Code
itself has no continuous loop, no state between invocations beyond
project CLAUDE.md. Technomancer's novelty is that it wraps Claude Code
in the orchestration layer Anthropic deliberately didn't ship.

### Open source / research

**OpenDevin / OpenHands** — active open-source Devin clone. Agent
architectures like CodeAct and an agent hub with role types. Single-
task model — runs a task in a VM, reports results. SWE-bench scores
published. Research-oriented. Differences from Technomancer:
task-oriented vs production-oriented — OpenDevin proves capability on
a benchmark; Technomancer proves sustained throughput in production
(425 commits/day). OpenDevin runs locally but isn't a 24/7 daemon
against a live board.

**SWE-Agent (Princeton)** — the research project that first won
SWE-bench leaderboards. Contribution: the Agent-Computer Interface
(ACI), a purpose-built tool API for code edits that beat general
shell-access agents. Single-task academic project. Differences from
Technomancer: SWE-Agent proves "can an LLM pass a benchmark," not
"can an LLM run a pipeline." Complementary research, not the same
product claim.

**Aider** — open-source pair-programming REPL. Chat-with-your-repo
interactive tool; great for "sit with me while I code." Differences
from Technomancer: Aider is a better human-in-the-loop tool;
Technomancer removes the human from the loop.

### The asymmetry that makes the paper pitch work

Every commercial system above is **task-oriented** — you hand it a
task, it returns. Technomancer is **pipeline-oriented** — it watches
a board, picks the next story by rank, ships it, logs the result, and
keeps going. Five concrete differences that don't exist anywhere else:

1. **Continuous pipeline** — always-on AIM/AIW daemons, not
   session-scoped. 425 commits/day isn't a benchmark run; it's a
   Tuesday output, measured against a production board.
2. **Multi-role team** — AIM (manager) + AIW (worker) + AIMM
   (researcher) + AIV (validator) with distinct jobs. Every commercial
   system above is a single agent, possibly with an internal critic.
3. **Rank-driven scheduling** — Jira priority rank IS the scheduler.
   No prompt engineering to pick what's next, no user session deciding
   the task. The board is the input; the merge is the output.
4. **Recursive self-repair** — the splitter decomposes a failed story
   into smaller children and re-queues them. Failure doesn't escalate
   to a human; it decomposes until it ships or hits a depth cap. No
   comparable system treats its own failures as input rather than
   exits.
5. **Post-merge self-audit (AIV)** — after every merge, AIV opens the
   page / hits the API / queries the DB and scores the shipped story
   on seven axes plus red flags. Nothing I found in the landscape
   does post-merge verification against the original story contract.
   Closest analogue is human code review, which none of the
   commercial systems run on their own output.

**The pitch line:** "I built a self-governing AI dev team on Claude
Code — four roles, continuous pipeline, Jira-ranked backlog,
self-repair on failure, self-audit on merge. 425 commits in one day."
No single adjacent product ships all of that; most ship one layer.

**Evidence:** Competitive scan conducted by Claude (Opus 4.7, 2026-04
knowledge cutoff). Product pages + changelogs + SWE-bench leaderboards
for each vendor.

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
