# Paper Themes

Editable priority list of research themes the AIMM curator uses when
scoring and generating stories. The curator reads this file fresh on
every cycle — edit it anytime; no redeploy needed.

`target_per_week` is the number of paper-worthy stories AIMM should aim
to ship under this theme in a rolling 7-day window. When the actual
count falls below target, the theme has a gap and gets prioritized in
the next crafting round.

Default target is 5 stories/week per theme.

---

## failure-mode discoveries
- target_per_week: 5

Stories that surface a real, previously-unknown failure mode of the
autonomous dev team or the systems it operates on, and ship a fix or a
diagnostic that makes the failure observable next time. "We found it
*because* we let it run, not because we predicted it" — the failure has
to be something a human reviewer wouldn't have caught in design.

Examples: stall detector races completion event; executor orphans a
Jira story when claude -p exits non-zero; auto-splitter loops on
ambiguous epics; worker wipes feature branch on cleanup.

Keywords: crash, stall, orphan, race, silent-failure, regression,
wedge, deadlock, data-loss, rollback.

---

## measurement + benchmarks
- target_per_week: 5

Stories that add an observable metric, dashboard, or benchmark to the
system, or that ship a measurement that changes what we know is true.
"If we can't see it, we don't believe it" — the story has to produce a
number, a chart, or a reproducible delta, not just a new feature.

Examples: per-phase story timings DB; cost-per-merged-commit trend;
Prometheus metrics for LLM latency distributions; regression corpus
pass-rate over model versions; tool-usage analytics by executor phase.

Keywords: metric, benchmark, dashboard, telemetry, cost-tracking,
profiling, observability, histogram, percentile, SLO.

---

## novel autonomy mechanisms
- target_per_week: 5

Stories that ship a new mechanism by which the team self-directs,
self-monitors, or self-improves without human input, beyond the
baseline claude-p-in-a-loop pattern. The mechanism should be nameable
and generalizable — something another team could lift into their own
system.

Examples: auto-splitter for failed stories; crash-triage auto-filing
Jira tickets; fallback orchestrator for API outages; knowledge-gap
resolver workflow; paper-worthiness scoring loop itself.

Keywords: self-healing, self-improving, auto-file, auto-split,
auto-resolve, feedback-loop, closed-loop, meta-level, reflection,
hypothesis-test.

---

## Editing notes

- Theme name is the markdown `## heading` text (lowercased, trimmed).
- `target_per_week` must be an integer on its own bullet line.
- The free-form paragraphs + keywords are for human readers and for
  AIMM's theme-matching prompts. They carry no format contract.
- Adding or removing a theme takes effect on the next AIMM cycle.
- Reducing a target doesn't retroactively cancel in-flight stories.
