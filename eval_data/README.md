# The golden set

`golden.jsonl` is the evaluation set: one JSON object per line, each a question
paired with the single answer the service is supposed to give. It is committed
because every number this project reports is only as trustworthy as the cases
behind it, and a set nobody can read is a set nobody can challenge.

It carries **field keys and NYC property-tax vocabulary, and nothing else**. No
real person, company, address or client appears in any question; identifiers
like `C-102` and `<CLIENT_1>` are invented and generic. Keep it that way.

Two paths here are **not** committed and are gitignored: `evals.db` (the run
store) and `reports/` (rendered runs). They are outputs, they churn, and the
only durable input is this file.

## Case format

```json
{"id": "gold-0001",
 "question": "Queens applications for tax year 2024",
 "tags": ["regression"],
 "source": "hand-written",
 "expect": {"kind": "ok", "payload": { ... }}}
```

`expect` is one of three, discriminated on `kind`:

| kind | carries | scored as |
| --- | --- | --- |
| `ok` | `payload` | exact match against the payload the service returned |
| `clarify` | `field` | the field whose vocabulary should come back as candidates |
| `refusal` | `reason` | one of the four closed reasons in `translate/refusals.py` |

Every case also carries **`source`**, which is required and has no default:

| source | means |
| --- | --- |
| `hand-written` | written for the set by a person |
| `paraphrase` | a rewording of another case — see `paraphrase_of` below |
| `correction` | written in response to an observed wrong answer |
| `live-capture` | a question a real caller actually asked |

Provenance is the thing nobody remembers six weeks later, and the analysis
turns on it: the primary comparison excludes paraphrases, which it can only do
if every case says what it is.

Four optional keys:

- **`note`** — one sentence saying why the expectation is what it is, for the
  cases where the correct answer is not the one a reader would first assume. It
  lives on the case rather than in a review document, so anyone "fixing" the
  expectation back to the obvious one has to delete the sentence explaining why
  the obvious one is wrong.
- **`difficulty_override`** — `easy` / `medium` / `hard`, when the derived
  rubric below is wrong about this question. A `note` is then **mandatory**: an
  unexplained override is indistinguishable from a typo, and it silently moves
  a case between the buckets a report compares.
- **`gold_root`** — the root the question should route to, for the clarify and
  refusal cases whose expectation carries no payload to read it off. On an ok
  case the payload already says it, so a `gold_root` there is legal only when
  it agrees. Left out where there is no right answer: "hello" routes nowhere.
- **`paraphrase_of`** — the id of the case this one rephrases. One level only;
  a paraphrase of a paraphrase is rejected, because a chain would make the
  family key depend on which link you walked. The **family** of a case is
  `paraphrase_of` or, for an original, its own id — and every metric that must
  not count one question twice groups on it, including the cluster bootstrap.

`load_golden(path, registry)` in `cert_nlq.evals.golden` reads the file and
checks every case against a registry — an `ok` payload goes through the same
`validate_payload` gate the service itself must pass — reporting *all* problems
at once, each prefixed with the case id.

## Slices and difficulty are derived, never written down

Store only what cannot be derived. A stored copy of a derived fact drifts the
first time the thing it copies is edited; a derived value cannot.

`derive_slices(case, registry)` computes every slice a case belongs to from its
gold payload:

| slice | derived from |
| --- | --- |
| `simple-filter` | exactly one condition, and the payload does nothing else |
| `multi-condition` | two or more conditions |
| `nested-logic` | `where` depth > 1 — a group inside a group |
| `in-op` | some condition uses the `in` operator |
| `join` | the payload reaches a joined table, by its `join` slot or its fields |
| `aggregation` | a non-empty `aggregate` |
| `group-by` | a non-empty `group_by` |
| `having` | a non-empty `having` |
| `coded-vocabulary` | some condition field has a closed vocabulary |
| `clarification` | the expectation is `clarify` |
| `refusal` | the expectation is `refusal` |

**These slices overlap by design.** A question can be a multi-condition join
with a coded value in it, and the point of the axis is to be able to ask "how
does it do on joins" without first deciding the case is *only* a join case. The
consequence: per-slice counts do **not** partition the set, and a report that
sums them is wrong.

A clarify or refusal case picks up no structural slice. There is no gold
payload to read one off, and inferring structure from the question would be the
model's job, not the ruler's.

`derive_difficulty(case, registry)` is the owner's rubric of 2026-09-18:

- **hard** — nested logic, or a mixed-grain aggregate (the aggregated field and
  the `group_by` come from different tables, or the aggregate runs over a
  one-to-many join), or two or more joins, or the question is tagged
  `ambiguous-wording`.
- **easy** — exactly one condition, no join, no aggregate.
- **medium** — everything else.

Read literally, "one condition" is something only an ok case has, so a clarify
or refusal case is `medium` unless it is tagged ambiguous or overridden. That
is deliberate: refusing "hello" is easy and refusing a question about a table
that nearly exists is not, and nothing structural tells them apart, so the
honest default is the middle bucket and the escape hatch is
`difficulty_override` with its mandatory note.

`SLICE_RUBRIC_VERSION` in `evals/golden.py` is bumped whenever either
derivation changes what it says about an unchanged case, so a re-slicing of old
rows is visibly a different ruler rather than a quiet reinterpretation.

## Paraphrases, and the scope-confirmation rule

A paraphrase inherits its original's gold payload **only after a human
confirms it means the same thing**, because a harmless-looking rewording can
change scope: "filed after 2024" and "filed in or after 2024" are different
queries. Generation is automated; inheritance is confirmed. Three layers:

1. **Generation rubric.** A rewording may reorder, resynonymise and change
   register — terse lawyer shorthand, a full sentence, a pile of keywords — but
   must never touch an operative token: comparators ("after", "at least",
   "over"), numbers, years, negations, quantifiers, or coded words.
2. **Mechanical guard, in the loader.** Every number and code-shaped literal in
   the original's question must appear verbatim in the paraphrase's. A
   violation is a load-time failure naming both ids. Cheap, and it catches the
   classic boundary drift.
3. **Review queue.** Every paraphrase is born `review`-tagged and is read by
   the owner as a side-by-side question pair, gold hidden: the only judgement
   is "same query? y/n". The tag comes off on confirmation. The primary
   analysis never depends on paraphrases, so an unconfirmed one cannot
   contaminate the headline — but it is excluded from the secondary analysis
   until it is confirmed.

## Conventions the gold payloads follow

These are decisions, not facts. They are written down so a disagreement is a
conversation about one line here rather than an argument about a hundred
cases.

- **Values are stored codes, not the words a user typed.** A gold payload
  describes the payload *after* value resolution: `"Quee"`, not `"Queens"`;
  `"F"`, not `"filed"`. A coded field's code is a string even when the field's
  type is `number` — `entity` is `"11"`, `relati` is `"1"` — because the
  registry stores codes as strings and `resolve_vocabulary` returns them
  unchanged.
- **Years are integers; money is a float.** `resolve_relative_year` yields an
  `int` and `resolve_money` yields a `float`, so the gold mirrors both.
- **`columns` is `[]` unless the question names its outputs.** Empty means
  "whatever the host defaults to". Only a question that actually asks for
  particular fields ("show me the manager's name and phone") pins them.
- **Dates use `>` / `<` with an inclusive boundary written out**, because the
  registry's date fields offer no `>=` or `<=`: "filed after 2023" is
  `appfiledat > "2023-12-31"`.
- **`in` over an OR group for one field**, matching the translator prompt. A
  nested group appears only where the branches are genuinely different fields
  (`(co-op or condo) and Queens and 2024`).
- **An aggregate alias is `total`** — the one example the translator prompt
  gives — except where that reads as a lie about the number (`average`,
  `highest`). Aliases are free text, so these cases are the most fragile in the
  set and are tagged `review`.

## Tags

`tags` carries **only** these three hand tags, and the loader rejects anything
else — including the retired slice names, so a file half-way through the
migration fails loudly rather than quietly losing its tagging:

| tag | means |
| --- | --- |
| `review` | the author was not confident in this gold answer, or it is new and unconfirmed. Read these first. A `review` tag is a question for the owner, not a known-bad case. |
| `regression` | a real failure this service has already had. Do **not** reword these questions; the leading space in `gold-0002` is deliberate. |
| `ambiguous-wording` | the *question* admits more than one honest reading. |

`ambiguous-wording` is the one hand tag on the slice axis, and the reason it
cannot be derived is that the ambiguity is in the English: the payload records
one reading of it. `derive_difficulty` reads the tag, so tagging a case moves
it to `hard`.

The refusal reason `join_not_available` has **no cases, deliberately**. It
exists in the shared vocabulary because the host can reach a join this service
cannot see, but `translate/refusals.py` says in its own docstring that this
service never constructs it — the router drops an unavailable join rather than
refusing. An outcome this service cannot produce must not be measured as though
it could be, so the questions that probe the unpublished refund / billing /
income-and-expense tables (`gold-0059` .. `gold-0061`) expect
`field_not_in_schema`, which is what the service actually and correctly
returns. Each carries a `note` saying so.
