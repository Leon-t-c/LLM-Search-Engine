"""The evaluation harness: the golden set, the scoring core, and the runs.

`EVALUATOR_VERSION` is the version of *what the numbers mean*, not of the
code that computes them. It changes when a scored definition changes --
`end_to_end_correct` gaining Phase 3's semantic-equivalence judge, say --
and a stored run records the version it was scored under.

The rule it exists to enforce: a definition is never quietly edited. A new
definition is a new version, so two runs are comparable exactly when their
versions match, and no model can be favoured by a rule chosen after seeing
its results. A bug fix that makes the evaluator do what it already said it
did is not a new version; a change of mind about what counts as correct is.
"""

EVALUATOR_VERSION = "1"
