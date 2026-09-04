"""The closed set of reasons a question cannot be expressed at all.

Refusals are reserved for what the schema *structurally* cannot say. Ambiguity
is not here: it is answerable, and routes to needs_clarification with concrete
candidates. Aggregation is not here either: the IR carries it.
"""
import re
from enum import StrEnum


class RefusalReason(StrEnum):
    NEEDS_NESTED_LOGIC = "needs_nested_logic"
    JOIN_NOT_AVAILABLE = "join_not_available"
    FIELD_NOT_IN_SCHEMA = "field_not_in_schema"
    NOT_A_QUERY = "not_a_query"


#: An explicit parenthesised disjunction.
_PARENTHESISED_OR = re.compile(r"\([^)]*\bor\b[^)]*\)", re.IGNORECASE)
#: A conjunction somewhere in the question.
_CONJUNCTION = re.compile(r"\band\b", re.IGNORECASE)


def looks_nested(question: str) -> bool:
    """Cheap pre-check for a question needing BOTH a disjunction and a
    conjunction, which one combinator cannot express.

    Flattening `(A or B) and C` into `A or B or C` returns wrong rows that look
    plausible, so refusing is correct — where we are confident.

    **Precision is chosen over recall deliberately**, because the two costs are
    not symmetric the way they are usually assumed to be:

    - A false positive refuses a question that was answerable. The user gets
      nothing, gives up, and no example is logged. That loss is invisible.
    - A false negative spends one model call. The model is separately
      instructed that it cannot express grouped logic, so it is a second line
      of defence; when it flattens instead, that is a model error the
      evaluation harness counts and can act on.

    So this fires only on an explicit parenthesised disjunction *combined with*
    a conjunction. Two shapes it deliberately does not catch:

    - `(settled or withdrawn) widgets` — a plain disjunction, expressible with
      a single OR. Parentheses alone are not a nesting signal, and an earlier
      draft refused this.
    - `widgets that are active or retired, and shipped in 2024` — real nesting
      with no parentheses. An earlier draft caught it with a comma pattern that
      also refused ordinary phrasing: "filed on or before 2024-01-01, and
      settled" is an everyday date range, not boolean grouping. Refusing those
      is far worse than missing this.

    Widen this only once the evaluation harness shows how often the model fails
    to refuse the unparenthesised shape on its own.
    """
    return bool(_PARENTHESISED_OR.search(question) and _CONJUNCTION.search(question))
