"""Shared story-description format for every generator.

Every place that writes a Jira story description (splitter, brain, evergreen,
idea_generator) references :data:`STORY_DESCRIPTION_FORMAT` so the layout
never drifts between sources. The format combines a Scrum user story, 5W
context, and acceptance criteria.
"""

# Label attached to stories that satisfy the atomic criteria defined in
# docs/atomic_stories.md. Every generator and scheduler imports this
# constant rather than hard-coding the string so the value never drifts.
ATOMIC_LABEL = "atomic"

STORY_DESCRIPTION_FORMAT = """\
Write the description using EXACTLY this layout, with a blank line between each
section (Scrum user story + 5W + acceptance criteria + files). All sections are
REQUIRED.

As a <role>, I want <capability>, so that <benefit>.

WHO: <who is affected or benefits — end user, bot operator, developer, etc.>
WHAT: <what to change — concrete functions, data structures, endpoints>
WHEN: <trigger or phase — on startup, on request, scheduled, on failure>
WHERE: <modules/paths/components touched>
WHY: <the problem this solves or motivation>

ACCEPTANCE CRITERIA:
- <testable check 1, e.g. pytest passes, endpoint returns 201, migration applies>
- <testable check 2>
- <testable check 3 if needed>

FILES: <comma-separated paths that will be edited or added>
"""


EPIC_DESCRIPTION_FORMAT = """\
Write the epic description as the full design brief. Use EXACTLY this layout,
with a blank line between sections. All sections are REQUIRED.

As a <role>, I want <capability>, so that <benefit>.

WHO: <who is affected or benefits>
WHAT: <the outcome delivered when all child stories ship>
WHEN: <when this matters — phase, trigger, deadline if any>
WHERE: <modules/areas touched>
WHY: <problem statement, motivation>

ARCHITECTURE: <design choices, components, interfaces, data flow>
TRADE-OFFS: <options considered and why this path was chosen>
CONSTRAINTS: <limits, invariants, non-goals>
FAILURE MODES: <what can go wrong and how we recover>

ACCEPTANCE CRITERIA (epic-level):
- <observable outcome 1>
- <observable outcome 2>
"""
