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
 "tags": ["selection", "regression"],
 "expect": {"kind": "ok", "payload": { ... }}}
```

`expect` is one of three, discriminated on `kind`:

| kind | carries | scored as |
| --- | --- | --- |
| `ok` | `payload` | exact match against the payload the service returned |
| `clarify` | `field` | the field whose vocabulary should come back as candidates |
| `refusal` | `reason` | one of the four closed reasons in `translate/refusals.py` |

A case may also carry an optional `"note"`: one sentence saying why the
expectation is what it is, for the cases where the correct answer is not the
one a reader would first assume. It lives on the case rather than in a review
document, so anyone "fixing" the expectation back to the obvious one has to
delete the sentence explaining why the obvious one is wrong.

`load_golden(path, registry)` in `cert_nlq.evals.golden` reads the file and
checks every case against a registry — an `ok` payload goes through the same
`validate_payload` gate the service itself must pass — reporting *all* problems
at once, each prefixed with the case id.

## Conventions the gold payloads follow

These are decisions, not facts. They are written down so a disagreement is a
conversation about one line here rather than an argument about fifty cases.

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

Every case carries **exactly one slice tag**; the loader rejects a file where
one does not. The slices partition the set, so a per-slice report adds up.

| tag | the case is about |
| --- | --- |
| `selection` | picking rows: borough, block, lot, years, dates, amounts, blanks |
| `values` | resolving a word to a stored code on a coded field |
| `aggregate` | `count` / `sum` / `avg` / `max`, `group_by`, `having`, sorting on an aggregate |
| `join` | a field on `cityproperty` (City Data) or `managers` (Manager) |
| `clarify` | a value that cannot resolve on a field that *has* a vocabulary |
| `refusal:<reason>` | one of the four closed refusal reasons |

`refusal:join_not_available` has **no cases, deliberately**. That reason exists
in the shared vocabulary because the host can reach a join this service cannot
see, but `translate/refusals.py` says in its own docstring that this service
never constructs it -- the router drops an unavailable join rather than
refusing. An outcome this service cannot produce must not have a slice
pretending to measure it, so the three questions that probe the unpublished
refund / billing / income-and-expense tables (`gold-0059` .. `gold-0061`) expect
`field_not_in_schema`, which is what the service actually and correctly
returns. Each carries a `note` saying so.

Borough is treated as `selection` rather than `values` even though it is a
coded field: it appears in most questions as an identity filter, and tagging on
it would make nearly the whole set one slice. `values` is reserved for cases
whose *point* is the code lookup.

Two optional tags may accompany the slice tag:

- `regression` — a real failure this service has already had. Do not reword
  these questions; the leading space in `gold-0002` is deliberate.
- `review` — the author was not confident in this gold answer. Read these
  first. A `review` tag is a question for the owner, not a known-bad case.
