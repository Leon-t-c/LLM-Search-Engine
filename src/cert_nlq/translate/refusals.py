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


#: An OR inside parentheses, or an ", and" closing a preceding "or" group —
#: both mean a single AND/OR combinator cannot express the question.
_PARENTHESISED_OR = re.compile(r"\([^)]*\bor\b[^)]*\)", re.IGNORECASE)
_OR_THEN_AND = re.compile(r"\bor\b.*,\s*and\b", re.IGNORECASE)


def looks_nested(question: str) -> bool:
    """Cheap pre-check for a question needing grouped boolean logic.

    Flattening `(A OR B) AND C` into `A OR B OR C` returns wrong rows that look
    plausible, so refusing is correct. This is a heuristic; the model is also
    instructed to refuse, and this catches the obvious cases before a call.
    """
    return bool(_PARENTHESISED_OR.search(question) or _OR_THEN_AND.search(question))
