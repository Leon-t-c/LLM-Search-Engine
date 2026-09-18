"""The golden set: the questions, and what the service is supposed to say.

A gold case is a question paired with one expectation, and the expectation is
the *whole* answer the harness scores against -- not a hint, not a partial
one. Three shapes, because the service has three: `ok` carries the payload it
should return **after value resolution** (stored codes, not the words a user
typed), `clarify` names the field whose vocabulary should come back as
candidates, and `refusal` names the closed reason.

Two rules run this module, and both exist because a wrong gold case is worse
than a missing one -- it does not fail, it silently reports the wrong number
for every run afterwards:

1. An ok-payload is validated through `ir.validate.validate_payload`, the same
   gate the service itself must pass. A gold payload naming a field that does
   not exist, or an operator the field does not allow, could never be produced
   by a correct service, so scoring against it measures nothing.
2. Every problem in the file is collected and reported together. A loader that
   raises on the first one turns a review pass over a set of this size
   into one run per mistake.
"""
import json
import re
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..ir.payload import Payload
from ..ir.validate import PayloadError, validate_payload
from ..registry.models import Registry, RootSpec
from ..translate.refusals import RefusalReason
from ..translate.values import resolve_vocabulary

#: Bumped whenever `derive_slices` or `derive_difficulty` changes what it
#: says about an unchanged case. A stored run records it, so a later
#: re-slicing of the same rows is visibly a different ruler rather than a
#: quiet re-interpretation of old numbers.
SLICE_RUBRIC_VERSION = "1"

#: The derived slice axis. These OVERLAP by design -- a case can be
#: `multi-condition` and `join` and `coded-vocabulary` at once -- so a report
#: broken down by slice does *not* partition the set and must not be summed
#: as though it did. Every one of them is computed from the gold
#: payload/expectation by `derive_slices`; none is stored, because a stored
#: copy of a derived fact drifts the first time the derivation is corrected.
SLICES = frozenset({
    "simple-filter",
    "multi-condition",
    "nested-logic",
    "in-op",
    "join",
    "aggregation",
    "group-by",
    "having",
    "coded-vocabulary",
    "clarification",
    "refusal",
})

#: Word tokens of a question, for the clarify-reachability check below.
_WORD_RE = re.compile(r"[A-Za-z0-9&]+")

#: The closed set of hand tags. `tags` carries **only** these: the slice axis
#: is derived, so a slice tag written by hand is a second copy of a fact the
#: file already contains, and the two would disagree the first time a payload
#: was edited.
#:
#: * `review` -- the author was not confident; the owner has not confirmed it.
#: * `regression` -- a real failure this service has already had. Never
#:   reword one of these questions; the whitespace is part of the case.
#: * `ambiguous-wording` -- the *question* admits more than one honest
#:   reading. It cannot be derived from a payload (the payload records one
#:   reading; the ambiguity is in the English), so it is the one hand tag on
#:   the slice axis, and `derive_difficulty` reads it.
AUX_TAGS = frozenset({"review", "regression", "ambiguous-wording"})

#: Numbers, years and code-shaped literals in a question. The paraphrase
#: guard requires every one of these to survive a rewording verbatim -- see
#: `_check_paraphrases`.
#: A comma is part of a number only between digit groups -- `$5,000,000` is
#: one token, while the comma in "tax year 2024, most common first" is
#: punctuation. Matching greedily through it would demand the paraphrase
#: keep the comma too, failing a rewording that moved the clause.
_NUMBER_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_CODE_RE = re.compile(r"<[^<>\s]+>|\b[A-Za-z]+-\d+\b|\"[^\"]+\"|'[^']+'")

#: The slots a payload actually has. `Payload` ignores anything else -- the
#: service must stay tolerant on its wire, where an unexpected key is the host's
#: business and not a reason to fail a request. A *gold* payload is the
#: opposite: it is hand-written, nothing downstream reads a stray key, and a
#: typo'd `"groupby"` would validate perfectly and then never match anything,
#: reporting a translation failure that is really a typo in the ruler.
_PAYLOAD_KEYS = frozenset(Payload.model_fields)


#: Every model here forbids extra keys. The golden file is the foundation the
#: measurement stands on, and the failure mode it has to rule out is silence:
#: a mistyped `"notes"` on one of the three cases whose note exists to defend a
#: counter-intuitive expectation would simply vanish, and a `"field"` left
#: behind on a case edited from `clarify` to `ok` would look like it still said
#: something. A key nobody reads is a key nobody notices is missing.
_STRICT = ConfigDict(frozen=True, extra="forbid")


class ExpectOk(BaseModel):
    """The payload the service should return, resolved values and all.

    Held as a `dict` rather than a `Payload` on purpose. The harness scores
    exact match against what the service returned, and the comparison is a
    property of the gold file's own text: parsing into the model first would
    fill in defaults and normalise shapes, so a gold file that quietly said
    something other than what its author wrote would compare equal anyway.
    Validation happens separately, in `load_golden`, where its failures can be
    reported by case id.
    """

    model_config = _STRICT

    kind: Literal["ok"] = "ok"
    payload: dict


class ExpectClarify(BaseModel):
    """`needs_clarification`, on this field.

    The field, not the candidate list: the candidates are the field's whole
    stored vocabulary, which the registry already decides. What the gold case
    asserts is that the service asked about the *right* field -- that is the
    numerator of candidate recall, and the part a translation can get wrong.
    """

    model_config = _STRICT

    kind: Literal["clarify"] = "clarify"
    field: str


class ExpectRefusal(BaseModel):
    model_config = _STRICT

    kind: Literal["refusal"] = "refusal"
    reason: str


Expectation = Annotated[
    ExpectOk | ExpectClarify | ExpectRefusal, Field(discriminator="kind")
]


class GoldenCase(BaseModel):
    model_config = _STRICT

    id: str
    question: str
    tags: tuple[str, ...] = ()
    expect: Expectation
    #: Why this case expects what it expects, when the answer is not the one a
    #: reader would first assume. Carried on the case rather than left in a
    #: review document because the surprise belongs next to the line that
    #: surprises -- a reader who edits the expectation back to the "obvious"
    #: one has to delete the sentence saying why it is not.
    note: str | None = None
    #: Where this case came from. Required, with no default: provenance is
    #: the thing nobody remembers six weeks later, and the analysis actually
    #: turns on it -- Task 4b's primary comparison excludes paraphrases, and
    #: it can only do that if every case says what it is.
    #:
    #: * `hand-written` -- written for the set by a person.
    #: * `paraphrase` -- a rewording of another case; see `paraphrase_of`.
    #: * `correction` -- written in response to an observed wrong answer.
    #: * `live-capture` -- a question a real caller actually asked.
    source: Literal["hand-written", "paraphrase", "correction", "live-capture"]
    #: Overrides `derive_difficulty` for this one case. The rubric is
    #: structural and the questions are English, so it is sometimes wrong;
    #: the escape hatch costs a sentence, and that sentence is mandatory --
    #: an unexplained override is indistinguishable from a typo, and it
    #: silently moves a case between the buckets a report compares.
    difficulty_override: Literal["easy", "medium", "hard"] | None = None
    #: The root the question should route to, for the cases whose gold
    #: expectation carries no payload to read it off. Only clarify and
    #: refusal cases have anything to add here; on an ok case the payload
    #: already says it, so a second copy is legal only when it agrees (and
    #: is then worth nothing). Left `None` where there is no right answer --
    #: "hello" routes nowhere.
    gold_root: str | None = None
    #: The case this one rephrases. The family key for cluster bootstrap is
    #: `family(case)`: this id, or the case's own when it is an original.
    #: One level only -- a paraphrase of a paraphrase would make the family
    #: depend on which link you walked from.
    paraphrase_of: str | None = None

    @model_validator(mode="after")
    def _check_case_invariants(self) -> "GoldenCase":
        """The invariants that need no registry, enforced everywhere.

        On the model rather than in the loader because a `GoldenCase` built
        in code -- a test fixture, a runner's synthetic case -- is exactly
        as capable of asserting nonsense as a line in the file is.
        """
        if self.difficulty_override is not None and not self.note:
            raise ValueError(
                "difficulty_override needs a note saying why the rubric is "
                "wrong here"
            )
        if self.gold_root is not None and isinstance(self.expect, ExpectOk):
            payload_root = self.expect.payload.get("root")
            if self.gold_root != payload_root:
                raise ValueError(
                    f"gold_root {self.gold_root!r} disagrees with the gold "
                    f"payload's root {payload_root!r}; on an ok case the "
                    f"payload is the answer"
                )
        if (self.source == "paraphrase") != (self.paraphrase_of is not None):
            raise ValueError(
                "source 'paraphrase' and paraphrase_of go together: one "
                "without the other leaves the family key guessing"
            )
        if self.paraphrase_of == self.id:
            raise ValueError("paraphrase_of points at the case itself")
        return self


def family(case: GoldenCase) -> str:
    """The cluster key: the original's id, or the case's own.

    Every metric that must not count one question twice -- and the
    bootstrap, which resamples clusters rather than rows -- groups on this.
    """
    return case.paraphrase_of or case.id


def iter_condition_dicts(node):
    """Every leaf condition dict under a raw (dict-shaped) `where` node.

    Tolerant by construction: this walks the same shape whether it came off
    a gold line or off the wire -- "has `children`" means group, "has
    `field`" means condition -- rather than assuming a shape and raising
    when it is wrong. Lives here, with the gold model, because both the
    derivations below and `scoring` need it and only one copy can be right.
    """
    if not isinstance(node, dict):
        return
    if "children" in node:
        for child in node.get("children") or ():
            yield from iter_condition_dicts(child)
    elif "field" in node:
        yield node


def _where_depth(node) -> int:
    """Nesting depth of a `where` tree: 0 for nothing, 1 for a flat group.

    A leaf condition contributes no depth of its own -- the depth being
    counted is of *groups*, because that is what "the question needed mixed
    logic" means. So `AND[a, b]` is 1 and `AND[a, OR[b, c]]` is 2, which is
    exactly the `> 1` the nested-logic slice tests.
    """
    if not isinstance(node, dict):
        return 0
    if "children" not in node:
        return 0
    children = node.get("children") or ()
    return 1 + max((_where_depth(c) for c in children), default=0)


def _payload_fields(payload: dict) -> list[str]:
    """Every field key the payload names anywhere, conditions included."""
    keys = [
        c["field"]
        for c in iter_condition_dicts(payload.get("where"))
        if isinstance(c.get("field"), str)
    ]
    for slot in ("columns", "group_by"):
        keys += [k for k in (payload.get(slot) or ()) if isinstance(k, str)]
    keys += [
        a.get("field")
        for a in (payload.get("aggregate") or ())
        if isinstance(a, dict) and isinstance(a.get("field"), str)
        and a.get("field") != "*"
    ]
    sort = payload.get("sort")
    if isinstance(sort, dict) and isinstance(sort.get("field"), str):
        keys.append(sort["field"])
    return keys


def _tables(keys, root: RootSpec | None) -> set[str]:
    if root is None:
        return set()
    by_key = root.fields_by_key
    return {by_key[k].table for k in keys if k in by_key}


def _joined_tables(payload: dict, root: RootSpec | None) -> set[str]:
    """The tables this payload actually reaches beyond the root's own.

    Read from the fields rather than trusting the `join` slot alone: the
    slot is what the model was asked to declare, and a gold payload that
    names a joined field without the slot would still be a join case. Both
    are read, and the union is the answer.
    """
    if root is None:
        return {t for t in (payload.get("join") or ()) if isinstance(t, str)}
    available = {j.table for j in root.joins}
    declared = {
        t for t in (payload.get("join") or ()) if isinstance(t, str)
    }
    return (declared | _tables(_payload_fields(payload), root)) & available


def derive_slices(case: GoldenCase, registry: Registry) -> frozenset[str]:
    """Which slices this case belongs to, computed, never stored.

    Slices overlap on purpose: a question can be a multi-condition join with
    a coded value in it, and the point of the axis is to be able to ask
    "how does it do on joins" without first deciding that the case is *only*
    a join case. The consequence is that per-slice counts do not sum to the
    set size, and a report that adds them up is wrong.

    A clarify case is `clarification` and a refusal is `refusal`, and
    neither picks up any structural slice: there is no gold payload to read
    one off, and inferring structure from the *question* would be the model's
    job, not the ruler's.
    """
    expect = case.expect
    if isinstance(expect, ExpectRefusal):
        return frozenset({"refusal"})
    if isinstance(expect, ExpectClarify):
        return frozenset({"clarification"})

    payload = expect.payload
    root_name = payload.get("root")
    root = registry.root(root_name) if isinstance(root_name, str) else None
    conditions = list(iter_condition_dicts(payload.get("where")))
    joins = _joined_tables(payload, root)
    aggregate = payload.get("aggregate") or ()
    group_by = payload.get("group_by") or ()
    having = payload.get("having") or ()

    found = set()
    # "and nothing else": one condition is only a *simple* filter when the
    # payload does no other work -- a lone condition under a group_by is a
    # breakdown question wearing one filter, and calling it simple would
    # flatter the easiest slice with the hardest cases.
    only_a_filter = not (
        joins or aggregate or group_by or having
        or payload.get("columns") or payload.get("sort")
    )
    if len(conditions) == 1 and only_a_filter:
        found.add("simple-filter")
    if len(conditions) >= 2:
        found.add("multi-condition")
    if _where_depth(payload.get("where")) > 1:
        found.add("nested-logic")
    if any(c.get("op") == "in" for c in conditions):
        found.add("in-op")
    if joins:
        found.add("join")
    if aggregate:
        found.add("aggregation")
    if group_by:
        found.add("group-by")
    if having:
        found.add("having")
    if root is not None:
        by_key = root.fields_by_key
        if any(
            isinstance(c.get("field"), str)
            and c["field"] in by_key
            and by_key[c["field"]].vocabulary
            for c in conditions
        ):
            found.add("coded-vocabulary")
    return frozenset(found)


def _mixed_grain_aggregate(payload: dict, root: RootSpec | None) -> bool:
    """Does this payload aggregate across more than one grain?

    Two shapes count, and both are the same mistake waiting to happen:

    * the aggregated field and the `group_by` field come from different
      tables (`avg(cityproperty.fullval) by boro`) -- the reader has to hold
      two row-grains in their head at once; and
    * the aggregate runs over a one-to-many join, where attaching the table
      multiplies the root's rows and a `count` silently answers a different
      question than the one asked.
    """
    aggregate = payload.get("aggregate") or ()
    if not aggregate or root is None:
        return False
    agg_keys = [
        a.get("field")
        for a in aggregate
        if isinstance(a, dict) and isinstance(a.get("field"), str)
        and a.get("field") != "*"
    ]
    group_keys = [k for k in (payload.get("group_by") or ()) if isinstance(k, str)]
    if len(_tables(agg_keys + group_keys, root)) > 1:
        return True
    fanning = {j.table for j in root.joins if j.cardinality == "one-to-many"}
    return bool(_tables(agg_keys, root) & fanning)


def derive_difficulty(case: GoldenCase, registry: Registry) -> str:
    """`easy` / `medium` / `hard`, by the owner's rubric of 2026-09-18.

    hard = nested logic, or a mixed-grain aggregate, or two-plus joins, or a
    question hand-tagged `ambiguous-wording`; easy = exactly one condition
    and no join and no aggregate; everything else is medium.

    One rule sits outside the structural reading, because the structure
    cannot express it (owner ruling, 2026-09-18): a `not_a_query` refusal is
    **easy**. A greeting is not a medium task, and refusing one is not the
    same job as refusing a question about a table that nearly exists. Every
    other refusal reason, and every clarify case, stays `medium` -- "one
    condition" is something only an ok case has, and nothing structural
    separates a hard refusal from an ordinary one, so the honest default is
    the middle bucket with `difficulty_override` as the escape hatch.

    (`SLICE_RUBRIC_VERSION` is unchanged at "1": this rule landed before the
    first scored run, so there are no stored rows it could reinterpret.)
    """
    if case.difficulty_override is not None:
        return case.difficulty_override
    if "ambiguous-wording" in case.tags:
        return "hard"
    slices = derive_slices(case, registry)
    if "nested-logic" in slices:
        return "hard"
    if (
        isinstance(case.expect, ExpectRefusal)
        and case.expect.reason == RefusalReason.NOT_A_QUERY.value
    ):
        return "easy"
    if isinstance(case.expect, ExpectOk):
        payload = case.expect.payload
        root_name = payload.get("root")
        root = registry.root(root_name) if isinstance(root_name, str) else None
        if len(_joined_tables(payload, root)) >= 2:
            return "hard"
        if _mixed_grain_aggregate(payload, root):
            return "hard"
        conditions = list(iter_condition_dicts(payload.get("where")))
        if (
            len(conditions) == 1
            and not _joined_tables(payload, root)
            and not (payload.get("aggregate") or ())
        ):
            return "easy"
    return "medium"


class GoldenError(ValueError):
    """Everything wrong with a golden file, in one exception."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


def load_golden(path: str | Path, registry: Registry) -> list[GoldenCase]:
    """Read a JSONL golden file and check every case against the registry.

    One case per line; blank lines and `#` comment lines are skipped so the
    file can carry section headings for the human who reviews it.

    Raises `GoldenError` listing every problem found, each prefixed with the
    case id it belongs to -- or with the line number, when the case is too
    broken to have a usable id.
    """
    problems: list[str] = []
    cases: list[GoldenCase] = []
    first_seen: dict[str, int] = {}

    for lineno, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            problems.append(f"line {lineno}: not valid JSON ({exc.msg})")
            continue
        if not isinstance(raw, dict):
            problems.append(f"line {lineno}: expected one JSON object per line")
            continue
        label = raw.get("id") if isinstance(raw.get("id"), str) else f"line {lineno}"
        try:
            case = GoldenCase.model_validate(raw)
        except ValidationError as exc:
            problems.append(f"{label}: {_summarise(exc)}")
            continue
        if case.id in first_seen:
            # Reported, then checked anyway. A duplicate id is a problem with
            # the *file*; whatever else is wrong with this case is a problem
            # with the case, and swallowing the second so the reader can see
            # the first costs them another whole run to find it.
            problems.append(
                f"{case.id}: duplicate id (first seen on line {first_seen[case.id]})"
            )
        else:
            first_seen[case.id] = lineno
        problems += _check_tags(case)
        problems += _check_gold_root(case, registry)
        problems += _check_clarify_is_reachable(case, registry)
        problems += _check_expectation(case, registry)
        cases.append(case)

    problems += _check_paraphrases(cases)
    if problems:
        raise GoldenError(problems)
    return cases


def _summarise(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or '<case>'}: {err['msg']}"
        for err in exc.errors()
    )


def _check_tags(case: GoldenCase) -> list[str]:
    """`tags` carries aux tags and nothing else.

    The old one-slice-tag mandate is retired: slices are derived now, and a
    hand-written slice tag would be a second, stale copy of a fact the gold
    payload already states. Anything outside `AUX_TAGS` is rejected rather
    than ignored -- including the retired slice names, so a file half-way
    through the migration fails loudly instead of silently losing its
    tagging.
    """
    strays = sorted(set(case.tags) - AUX_TAGS)
    if strays:
        return [
            f"{case.id}: unknown tag(s) {strays} -- `tags` carries only "
            f"{', '.join(sorted(AUX_TAGS))}; slices are derived"
        ]
    return []


def _check_clarify_is_reachable(case: GoldenCase, registry: Registry) -> list[str]:
    """A clarify case's ambiguous word must not resolve on its own field.

    A gold `clarify` asserts that some value in the question fails to map
    onto the field's vocabulary. If the word actually resolves, the service
    correctly answers `ok` and this case scores it a failure for ever --
    the same class of unreachable gold as a clarify case with no anchor
    condition, moved from "nothing survives pruning" to "nothing failed".

    Checked against the *live* vocabulary through `resolve_vocabulary`, the
    same function the service resolves with, so this cannot drift from it.
    Every whole word and every adjacent word pair is tried, and **exact
    matches only** -- case-folded code, meaning or synonym, which is all
    `resolve_vocabulary` itself accepts. No fuzzy matching: a false block on
    a legitimate case costs more here than a miss, because the miss is
    visible the first time the case is scored and the false block stops the
    whole file loading.
    """
    if not isinstance(case.expect, ExpectClarify):
        return []
    # One-character tokens are dropped, and that is the one concession to
    # noise: several vocabularies code their entries as single letters
    # (`F` for Filed, `A` for Active), and English writes "a" constantly. A
    # lone letter in prose is not a user typing a stored code, and blocking
    # the file over the indefinite article would be exactly the false block
    # this check is supposed to be cheap enough to avoid.
    words = [w for w in _WORD_RE.findall(case.question) if len(w) > 1]
    phrases = words + [f"{a} {b}" for a, b in pairwise(words)]
    problems = []
    for root in registry.roots:
        spec = root.fields_by_key.get(case.expect.field)
        if spec is None:
            continue
        for phrase in phrases:
            code = resolve_vocabulary(spec, phrase)
            if code is not None:
                problems.append(
                    f"{case.id}: {phrase!r} resolves on {case.expect.field!r} "
                    f"to code {code!r}, so the service answers this question "
                    f"rather than asking about it -- a clarify case needs a "
                    f"value that genuinely does not resolve"
                )
                break
    return problems


def _check_gold_root(case: GoldenCase, registry: Registry) -> list[str]:
    """A declared `gold_root` has to name a root the registry offers.

    The ok-case agreement rule is on the model (it needs no registry); this
    is the half that does.
    """
    if case.gold_root is None:
        return []
    if registry.root(case.gold_root) is None:
        return [f"{case.id}: unknown gold_root {case.gold_root!r}"]
    return []


def _literals(question: str) -> list[str]:
    """The tokens a rewording is not allowed to lose.

    Numbers (so years, blocks, lots, amounts and thresholds), and the
    code-shaped literals a question quotes -- `C-102`, `<CLIENT_1>`, or
    anything in quotes. These are exactly the tokens whose loss changes
    which rows come back while leaving the sentence reading fine.
    """
    return _NUMBER_RE.findall(question) + _CODE_RE.findall(question)


#: A root carrying no fields, used only to borrow `scoring.canonical`'s
#: *structural* normalisation -- `value2: null` folded together with an
#: absent `value2`, `join: []` with no `join` at all, children sorted,
#: aggregate aliases positionalised -- without its registry-typed value
#: coercion. The coercion is the one part that must not apply here: it maps
#: `2024` and `2024.0` onto the same number, and the gold conventions treat
#: those as different facts (an int year, a float money amount). No spec is
#: found for any field, so every value is rendered as written.
_TYPELESS_ROOT = RootSpec(root="", label="", key=(), fields=())


def _expectation_text(case: GoldenCase) -> str:
    """One case's whole expectation, rendered for an exact comparison.

    Absence and emptiness are not differences -- a gold payload that spells
    out `"value2": null` says exactly what one that omits the key says --
    so an ok payload goes through the same canonical form the scorer uses,
    minus the type coercion (see `_TYPELESS_ROOT`). Clarify and refusal
    expectations have no payload and are rendered as they stand.
    """
    # Imported here, not at module scope: `scoring` imports this module, and
    # the normalisation is wanted in exactly one function.
    from .scoring import canonical

    dumped = case.expect.model_dump()
    payload = dumped.get("payload")
    if isinstance(payload, dict):
        dumped["payload"] = canonical(payload, _TYPELESS_ROOT)
    return json.dumps(dumped, sort_keys=True)


def _check_paraphrases(cases: list[GoldenCase]) -> list[str]:
    """`paraphrase_of` resolves, does not chain, and preserves the literals.

    The literal check is the cheap mechanical half of the scope-confirmation
    rule: "filed after 2024" and "filed in or after 2024" are different
    queries, and a paraphrase that inherits its original's gold payload
    while quietly moving a boundary reports the model wrong for being right.
    A rewording may reorder, resynonymise and change register; it may not
    drop the number. The other half -- did the *meaning* survive -- is a
    human read, which is why every paraphrase is born `review`-tagged.

    The expectation itself is checked outright: a paraphrase asserts the
    same answer as its original, or it is not a paraphrase. The comparison
    is over the canonical form of the gold payload (see `_expectation_text`)
    -- so writing `value2: null` where the original omitted it is not a
    difference, while `2024` against `2024.0` still is. A rewording that
    genuinely changes scope belongs in the set as a case of its own, with
    its own id and its own gold.
    """
    by_id = {case.id: case for case in cases}
    problems = []
    for case in cases:
        original_id = case.paraphrase_of
        if original_id is None:
            continue
        original = by_id.get(original_id)
        if original is None:
            problems.append(
                f"{case.id}: paraphrase_of {original_id!r} names no case in "
                f"this file"
            )
            continue
        if original.paraphrase_of is not None:
            problems.append(
                f"{case.id}: paraphrase_of {original_id!r} is itself a "
                f"paraphrase (of {original.paraphrase_of!r}); a family is one "
                f"level deep, so point at the original"
            )
            continue
        if _expectation_text(case) != _expectation_text(original):
            problems.append(
                f"{case.id}: expectation differs from its original "
                f"{original_id!r} -- a rewording that changes the answer is "
                f"not a paraphrase; make it a case of its own"
            )
        lost = [
            token
            for token in _literals(original.question)
            if token not in case.question
        ]
        if lost:
            problems.append(
                f"{case.id}: paraphrase of {original_id!r} drops {lost} from "
                f"the original question -- a number or code that moves is a "
                f"different query, not a rewording"
            )
    return problems


def _check_expectation(case: GoldenCase, registry: Registry) -> list[str]:
    expect = case.expect
    if isinstance(expect, ExpectRefusal):
        if expect.reason not in {reason.value for reason in RefusalReason}:
            return [f"{case.id}: unknown refusal reason {expect.reason!r}"]
        return []
    if isinstance(expect, ExpectClarify):
        # Any root: a clarification names a field, and which root the question
        # routes to is the service's decision, not the gold case's.
        specs = [
            root.fields_by_key[expect.field]
            for root in registry.roots
            if expect.field in root.fields_by_key
        ]
        if not specs:
            return [f"{case.id}: unknown clarification field {expect.field!r}"]
        # Existing is not enough. `needs_clarification` is only reachable
        # through a field with a closed vocabulary -- a value that will not
        # resolve on a field with nothing to offer back is *refused*, by
        # design. So a gold clarify case naming a vocabulary-less field asserts
        # an outcome the service cannot produce, and candidate recall, whose
        # numerator these cases are, would be measuring a constant.
        if not any(spec.vocabulary for spec in specs):
            return [
                f"{case.id}: clarification field {expect.field!r} has no "
                f"vocabulary, so it can never produce candidates"
            ]
        return []
    strays = sorted(set(expect.payload) - _PAYLOAD_KEYS)
    problems = [
        f"{case.id}: gold payload has no slot {key!r}" for key in strays
    ]
    try:
        validate_payload(expect.payload, registry)
    except PayloadError as exc:
        problems += [f"{case.id}: {problem}" for problem in exc.problems]
    except ValidationError as exc:
        problems.append(f"{case.id}: {_summarise(exc)}")
    return problems
