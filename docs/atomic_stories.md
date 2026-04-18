# Atomic Stories

A Jira story qualifies as **atomic** when, and only when, all four of the
following criteria hold. Every story generator — `aim/splitter.py`,
`agent/idea_generator.py`, the evergreen generator, and the brain
synthesizer — must check these before attaching the `atomic` label, and
the manager's slot scheduler uses the label to pick safe work.

## The four criteria

1. **Touches at most 2 files.** The `FILES:` section of the story names
   no more than two paths. New tests for the change count against this
   limit. Stories that sprawl across more files are not atomic and must
   be split first.

2. **No cross-module dependencies.** The work lives inside one
   subsystem. A change that requires a coordinated edit in an unrelated
   module (for example, touching both `agent/news_digest.py` and
   `idea_board/web.py` in a single story) is not atomic.

3. **Uses patterns that already exist in the codebase.** The story
   applies an established pattern — a helper, fixture, config key, or
   convention already in the repo — rather than introducing a new
   abstraction. "Add a new framework" stories are never atomic.

4. **Acceptance criteria are fully specified as test assertions.**
   Each bullet under `ACCEPTANCE CRITERIA:` names a concrete, testable
   outcome that a pytest assertion could check: a return value, a file
   existing, a status code, an importable symbol. Vague goals
   ("improve reliability", "clean up logging") are not atomic.

## The label

Generators and the scheduler share a single source of truth for the
label name: the `ATOMIC_LABEL` constant in
`local-agent/agent/story_format.py`. Import that constant instead of
hard-coding the string `"atomic"` so the value can never drift.

```python
from agent.story_format import ATOMIC_LABEL

labels.append(ATOMIC_LABEL)
```
