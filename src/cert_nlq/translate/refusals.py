"""The closed set of reasons a question cannot be expressed at all.

Refusals are reserved for what the schema *structurally* cannot say. Ambiguity
is not here: it is answerable, and routes to needs_clarification with concrete
candidates. Aggregation is not here either: the IR carries it.
"""
from enum import StrEnum


class RefusalReason(StrEnum):
    JOIN_NOT_AVAILABLE = "join_not_available"
    FIELD_NOT_IN_SCHEMA = "field_not_in_schema"
    NOT_A_QUERY = "not_a_query"
