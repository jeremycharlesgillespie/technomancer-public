# Incidents Log

A hand-editable record of bugs, outages, and unexpected behaviors encountered
during development. Used as source material for the PPTX incidents slide.

| Date | What broke | How detected | Fix commit |
|------|------------|--------------|------------|
| 2026-04-17 | splitter-auto-veto-on-children blocked recursive decomposition (TK-566) | Story never decomposed; AIM stalled on multi-step epics | TK-566 |
| 2026-04-17 | stalled-but-shipped false-negative clobbered main-branch shipped stories (TK-567) | Deployed stories overwritten by stale branch merges | TK-567 |
| 2026-04-17 | executor auto-commit orphan committed unrelated work under a dead story branch | Noticed stray commits on closed branch during git log review | manual cleanup |
| 2026-04-17 | AIM peer-cross-kill of worker PIDs between TK and FA managers | FA worker killed by TK manager; stories stopped executing | config isolation fix |
| 2026-04-17 | hardcoded Technomancer test paths ran on 40Acres project | 40Acres executor run failed with path-not-found errors | path parameterization |
