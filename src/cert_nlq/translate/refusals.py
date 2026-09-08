"""The closed set of reasons a question cannot be expressed at all.

Refusals are reserved for what the schema *structurally* cannot say. Ambiguity
is not here: it is answerable, and routes to needs_clarification with concrete
candidates. Aggregation is not here either: the IR carries it.

`JOIN_NOT_AVAILABLE` is deliberately never constructed in this service, and
that is the design rather than an oversight: the router *drops* a join the
registry does not list instead of refusing, because an over-broad slice still
answers the question and a refusal does not. It stays in the set because the
set is a shared vocabulary — the host compiles the payload and can reach a
join this service cannot see. Do not go hunting for the code path; there is
none.
"""
from enum import StrEnum


class RefusalReason(StrEnum):
    JOIN_NOT_AVAILABLE = "join_not_available"
    #: A payload that could not be made to validate against the registry —
    #: an invented field, an illegal operator, a missing value. Structural.
    FIELD_NOT_IN_SCHEMA = "field_not_in_schema"
    #: A value that would not resolve, on a field with no closed vocabulary
    #: to offer back as candidates. The field is fine and the structure is
    #: fine; there is simply nothing to ask the user to choose between, so it
    #: refuses instead of clarifying. Distinct from the above because a money
    #: amount that will not parse is not a field missing from the schema, and
    #: reporting it as one sends the reader after the wrong bug.
    VALUE_NOT_UNDERSTOOD = "value_not_understood"
    NOT_A_QUERY = "not_a_query"
