"""Markdown rendering: one run's §11 block, and the paired comparison.

Two entry points. `render_single_run` renders one stored run's full §11
metric block plus its per-slice/difficulty/source tables -- everything
`aggregate()` computed, in §11's own order, `n` beside every number.
`render_compare` renders the frozen three-layer comparison the owner
specified 2026-09-18 (`task-4b-brief.md`, restated in this task's brief):

0. **Error rows (`row.error is not None`) are excluded before layers 1-2
   even see a case** -- the same discipline `aggregate()` applies to layer
   3. An infra failure (a 502, a timeout) is not a model answer; letting it
   into the primary pairing would move the one confirmatory p-value on a
   coin flip. A case enters the comparison only when *both* runs have an
   error-free row for it. Each run's error count is printed in the header.
1. **Primary** -- exact paired McNemar on `e2e_correct`, over the
   error-free CANONICAL cases both runs share. This is the one confirmatory
   test in the whole eval; the canonical filter is applied *here*, not
   inside `stats.py` (`mcnemar_exact`'s own docstring: "the report owns
   that filter").
2. **Secondary** -- the paraphrase-expanded shared set: cluster-bootstrap
   accuracy intervals per model, and `family_accuracy`/`family_consistency`
   side by side per model (never one without the other -- `stats.py`'s own
   rule).
3. **Explanatory cascade** -- the full §11 block per run, in pipeline order,
   plus per-slice/difficulty/source tables from each run's own `aggregate()`
   call. Descriptive only: no p-values here, by the same discipline
   `stats.py`'s module docstring states.
4. The accuracy / p50-latency / cost-per-query trade-off table, each row
   carrying the concurrency that run was measured under.

`render_compare` **refuses**, naming the field(s), when the two runs'
evaluator identity disagrees (`evaluator_version`, `slice_rubric_version`,
`evals_code_hash`, `golden_hash`) -- comparing runs scored by different
rulers would silently misreport a code change as a model difference.
"""
from collections.abc import Sequence
from typing import Any

from ..registry.models import Registry
from .aggregate import Metrics, aggregate
from .golden import GoldenCase, family
from .scoring import Row
from .stats import (
    cluster_bootstrap,
    family_accuracy,
    family_consistency,
    latency_summary,
    mcnemar_exact,
    paired_delta_ci,
    wilson,
)

# --------------------------------------------------------------------------
# cost estimation -- shared by the CLI's pre-run spend gate and the
# post-run trade-off table
# --------------------------------------------------------------------------

#: USD per 1M tokens, {"input": ..., "output": ...}, keyed by model id.
#: **Judgement call** (task-6-report.md): this repo has no committed pricing
#: table anywhere, and the configured model ids (`settings.model`,
#: `settings.claude_model`) are this project's own placeholder names, not
#: ones with a published rate this module could look up. "default" is a
#: deliberately conservative placeholder in the ballpark of contemporary
#: frontier-model list pricing; update it -- and only it -- once a real
#: rate for the deployed model is known. Every dollar figure this module
#: prints is an estimate built on this table, never a substitute for the
#: provider's own billing page.
PRICE_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    "default": {"input": 2.50, "output": 10.00},
}

#: Sec 12a's own measured ballpark ("questions x measured ~3.7K input tokens")
#: -- the pre-run spend estimate's input-token-per-question assumption.
ESTIMATED_INPUT_TOKENS_PER_CASE = 3700
#: §12a: "payloads are a few hundred tokens, and output is the expensive
#: side of the bill" -- deliberately not tiny, since output is priced
#: higher than input in every current frontier-model rate card.
ESTIMATED_OUTPUT_TOKENS_PER_CASE = 300


def _price_for(model_id: str) -> dict[str, float]:
    return PRICE_PER_1M_TOKENS.get(model_id, PRICE_PER_1M_TOKENS["default"])


def has_pricing_for(model_id: str) -> bool:
    """Whether `model_id` has its own entry in `PRICE_PER_1M_TOKENS`, as
    opposed to silently falling back to `"default"`. The CLI's spend gate
    checks this before printing an estimate, so a reader sees "no rate on
    file, using placeholder rates" rather than a confident-looking dollar
    figure that is quietly built on a rate for a different model.
    """
    return model_id in PRICE_PER_1M_TOKENS


def estimate_spend(n_cases: int, model_id: str) -> float:
    """A pre-run spend estimate in USD: `n_cases` questions, each assumed to
    cost `ESTIMATED_INPUT_TOKENS_PER_CASE` input and
    `ESTIMATED_OUTPUT_TOKENS_PER_CASE` output tokens (§12a's own ballpark),
    priced via `PRICE_PER_1M_TOKENS`. This is the number the CLI's
    `--yes-spend` gate prints before any call is made -- see this module's
    docstring for why the price table itself is a placeholder.
    """
    price = _price_for(model_id)
    per_case = (
        ESTIMATED_INPUT_TOKENS_PER_CASE / 1_000_000 * price["input"]
        + ESTIMATED_OUTPUT_TOKENS_PER_CASE / 1_000_000 * price["output"]
    )
    return n_cases * per_case


def estimate_cost_per_query(
    provider: str, model_id: str, tokens_in: float, tokens_out: float
) -> float | None:
    """USD per query from *measured* average token totals (a run's own
    `tokens_per_query`), or `None` only for `"ollama"`, whose cost is
    hardware amortisation, not a per-token bill (spec §15's phase-2 line) --
    there is no dollar figure to compute for it at all.

    **Every other provider gets an estimate here**, including one with no
    entry in `PRICE_PER_1M_TOKENS` (silently priced off `"default"`) --
    2026-09-18 review: this function used to return `None` for "anything
    but openai", which mislabelled a billed non-openai provider (Claude,
    say) as hardware-amortised, exactly as wrong the other way. Whether
    *this* estimate is trustworthy (real rate on file, or the placeholder)
    is `_cost_cell`'s job to say -- via `has_pricing_for` -- not this
    function's; it only ever refuses to price the one provider that
    genuinely has no token bill.
    """
    if provider == "ollama":
        return None
    price = _price_for(model_id)
    return tokens_in / 1_000_000 * price["input"] + tokens_out / 1_000_000 * price["output"]


# --------------------------------------------------------------------------
# generic markdown helpers
# --------------------------------------------------------------------------


def _fmt(metric: dict[str, Any] | None) -> str:
    """One `Metric` (`aggregate.py`'s `{"value": ..., "n": ...} | None`) as
    a table cell -- `n` printed beside every number, `None` rendered as
    `n/a` rather than a blank cell a reader might mistake for zero."""
    if metric is None:
        return "n/a"
    value = metric.get("value")
    if value is None:
        note = metric.get("note")
        return f"n/a ({note})" if note else "n/a"
    n = metric.get("n")
    if isinstance(value, bool):
        return f"{value} (n={n})"
    return f"{value:.3f} (n={n})"


def _fmt_ci(lo: float | None, hi: float | None) -> str:
    if lo is None or hi is None:
        return "n/a"
    return f"[{lo:.3f}, {hi:.3f}]"


def _cost_cell(run: Any, tokens_block: dict[str, Any]) -> str:
    """The trade-off table's "cost per query" cell for one run.

    Three distinct outcomes (2026-09-18 review -- the old version rendered
    every non-openai provider as "hardware-amortised", which is only true
    of `"ollama"`; a billed provider like Claude with no rate on file was
    silently given the *wrong* excuse):

    * `"ollama"` -- genuinely hardware-amortised, no token bill at all.
    * any other provider with no entry in `PRICE_PER_1M_TOKENS` -- "no rate
      on file for `<model>`", the same honest refusal `has_pricing_for`
      backs in the CLI's pre-run spend gate, rather than a dollar figure
      quietly built on a rate for a different model.
    * everything else (openai, or another provider with its own priced
      entry) -- the estimate.
    """
    if run.provider == "ollama":
        return "n/a (hardware-amortised)"
    tokens_in = tokens_block.get("tokens_in")
    tokens_out = tokens_block.get("tokens_out")
    if tokens_in is None:
        return "n/a (unmeasured)"
    if run.provider != "openai" and not has_pricing_for(run.model_id):
        return f"no rate on file for {run.model_id}"
    cost = estimate_cost_per_query(
        run.provider, run.model_id, tokens_in["value"],
        tokens_out["value"] if tokens_out is not None else 0,
    )
    return f"${cost:.4f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


#: The four headline rate-shaped metrics Wilson gets wired onto in the
#: single-run report (2026-09-18 review, closing the "Wilson is a dead
#: seam" finding -- the plan promises Wilson on canonical-set rates and
#: nothing in `src/` called it). Not every rate in the block: Wilson
#: assumes independent Bernoulli trials (`stats.wilson`'s own docstring),
#: which holds for a per-question accuracy count but is already a looser
#: fit for a micro-averaged ratio like field P/R -- extending it further,
#: to join P/R or operator accuracy, would stretch that assumption thinner
#: still for no request behind it. These four are exactly what the brief
#: names: "e2e rate, exact-match rate, field P/R".
_WILSON_KEYS = ("e2e_correct_rate", "exact_match_rate", "field_precision", "field_recall")


def _canonical_wilson_counts(
    rows: Sequence[Row], canonical_ids: set[str]
) -> dict[str, tuple[int, int]]:
    """`(successes, n)` for each of `_WILSON_KEYS`, computed directly from
    already error-excluded **canonical** rows -- exact integer counts, not
    a float round-trip through an aggregated ratio, and restricted to the
    canonical subset because Wilson's independence assumption does not
    survive paraphrase families (`stats.py`'s own module docstring: "on the
    paraphrase-expanded set... use cluster_bootstrap instead").

    Mirrors `aggregate.py`'s own `_rate_over`/`_bool_rate`/`_field_pr`
    denominators exactly, recomputed here rather than imported: those are
    private to that module, and re-deriving four counts from a row list is
    cheaper than widening its API for one caller.
    """
    canonical_rows = [r for r in rows if r.case_id in canonical_ids and r.error is None]

    e2e_n = len(canonical_rows)
    e2e_successes = sum(1 for r in canonical_rows if r.e2e_correct)

    exact_present = [r for r in canonical_rows if r.exact_match is not None]
    exact_n = len(exact_present)
    exact_successes = sum(1 for r in exact_present if r.exact_match)

    measured = [
        r for r in canonical_rows if r.root_correct is True and r.field_tp is not None
    ]
    field_tp = sum(r.field_tp for r in measured)
    field_fp = sum(r.field_fp for r in measured)
    field_fn = sum(r.field_fn for r in measured)

    return {
        "e2e_correct_rate": (e2e_successes, e2e_n),
        "exact_match_rate": (exact_successes, exact_n),
        "field_precision": (field_tp, field_tp + field_fp),
        "field_recall": (field_tp, field_tp + field_fn),
    }


def _wilson_annotation(counts: dict[str, tuple[int, int]], key: str) -> str:
    successes, n = counts[key]
    if n == 0:
        return ""
    lo, hi = wilson(successes, n)
    return f" -- canonical Wilson 95% CI (n={n}): {_fmt_ci(lo, hi)}"


def _block_rows(
    block: Metrics, *, wilson_counts: dict[str, tuple[int, int]] | None = None
) -> list[tuple[str, str]]:
    """The §11 block's headline rows, label -> formatted cell, in §11's own
    order. Shared by the single-run report and the explanatory cascade.

    `wilson_counts` (from `_canonical_wilson_counts`) is `None` for the
    compare report's explanatory cascade -- descriptive only, by that
    section's own discipline -- and supplied only by `render_single_run`,
    which is also the only caller that adds a Wilson annotation to
    `_WILSON_KEYS`' cells. The displayed value/n themselves are always the
    full-set numbers already on `block`; the annotation is an addition, not
    a replacement.
    """

    def cell(key: str, metric) -> str:
        text = _fmt(metric)
        if wilson_counts is not None and key in _WILSON_KEYS:
            text += _wilson_annotation(wilson_counts, key)
        return text

    clar = block["clarification"]
    tokens = block["tokens_per_query"]
    rows = [
        ("Root accuracy", _fmt(block["root_accuracy"])),
        ("Join precision", _fmt(block["join_precision"])),
        ("Join recall", _fmt(block["join_recall"])),
        ("Groups recall (echoed)", _fmt(block["groups_recall_echoed"])),
        ("Groups recall (proxy)", _fmt(block["groups_recall_proxy"])),
        ("Field precision (headline)", cell("field_precision", block["field_precision"])),
        ("Field recall (headline)", cell("field_recall", block["field_recall"])),
        ("Operator accuracy", _fmt(block["operator_accuracy"])),
        ("Exact-match rate", cell("exact_match_rate", block["exact_match_rate"])),
        ("**E2E-correct rate (primary)**", cell("e2e_correct_rate", block["e2e_correct_rate"])),
        ("Clarification rate", _fmt(clar["clarification_rate"])),
        ("Resolution rate", _fmt(clar["resolution_rate"])),
        ("Candidate recall", _fmt(clar["candidate_recall"])),
        ("Latency p50 (ms)", _fmt(block["latency_p50"])),
        ("Latency p95 (ms)", _fmt(block["latency_p95"])),
        ("Tokens in / query", _fmt(tokens["tokens_in"])),
        ("Tokens out / query", _fmt(tokens["tokens_out"])),
        ("Value accuracy (secondary)", _fmt(block["value_accuracy"])),
    ]
    return rows


def _refusal_table(block: Metrics) -> str:
    reasons = block["refusal"]
    if not reasons:
        return "_no refusal reason observed or expected in this bucket_"
    rows = [
        (reason, _fmt(pr["precision"]), _fmt(pr["recall"]))
        for reason, pr in sorted(reasons.items())
    ]
    return _table(("Reason", "Precision", "Recall"), rows)


def _axis_table(buckets: dict[str, Metrics]) -> str:
    if not buckets:
        return "_no buckets_"
    rows = []
    for name, block in sorted(buckets.items()):
        rows.append((
            name,
            _fmt(block["e2e_correct_rate"]),
            _fmt(block["root_accuracy"]),
            _fmt(block["field_precision"]),
            _fmt(block["field_recall"]),
            _fmt(block["latency_p50"]),
        ))
    return _table(
        ("Bucket", "E2E-correct", "Root accuracy", "Field precision", "Field recall",
         "Latency p50 (ms)"),
        rows,
    )


def _errors_line(block: Metrics) -> str:
    errors = block["errors"]
    error_rate = errors["error_rate"]
    rate = f"(rate {error_rate:.3f})" if error_rate is not None else "(rate n/a)"
    return (
        f"- Error rows: {errors['error_count']} of {errors['n']} {rate}; "
        f"unmatched case rows: {errors['unmatched_case_rows']}"
    )


# --------------------------------------------------------------------------
# single-run report
# --------------------------------------------------------------------------


def render_single_run(
    run: Any, rows: Sequence[Row], cases: Sequence[GoldenCase], registry: Registry
) -> str:
    """The §11 aggregate block plus per-slice/difficulty/source tables for
    one stored run. `run` is a `store.RunRecord` (or anything carrying the
    same attributes) -- duck-typed so a caller can pass a plain namespace in
    tests without a live store.
    """
    metrics = aggregate(rows, cases, registry)
    canonical_ids = {c.id for c in cases if c.paraphrase_of is None}
    wilson_counts = _canonical_wilson_counts(rows, canonical_ids)
    lines = [
        f"# Eval run `{run.id}`",
        "",
        f"- Started: {run.started_at}",
        f"- Provider / model: {run.provider} / {run.model_id} "
        f"(router: {run.router_model or run.model_id})",
        f"- Registry version: {run.registry_version}",
        f"- Golden set: {run.golden_path} (`{run.golden_hash}`)",
        f"- Concurrency: {run.concurrency}",
        f"- Canonical-only: {run.canonical_only}"
        + (f", stratified to {run.stratify}" if run.stratify else ""),
        f"- Evaluator identity: evaluator_version={run.evaluator_version}, "
        f"slice_rubric_version={run.slice_rubric_version}, "
        f"evals_code_hash={run.evals_code_hash[:12]}...",
    ]
    if run.notes:
        lines.append(f"- Notes: {run.notes}")
    lines += ["", _errors_line(metrics), ""]

    lines.append("## §11 metric block")
    lines.append(
        "_e2e/exact-match/field P&R rows also carry a Wilson 95% interval over the "
        "canonical subset (independent-Bernoulli only -- see stats.py)._"
    )
    lines.append("")
    lines.append(_table(("Metric", "Value"), _block_rows(metrics, wilson_counts=wilson_counts)))
    lines.append("")
    lines.append("### Refusal precision / recall, by reason")
    lines.append("")
    lines.append(_refusal_table(metrics))

    for title, key in (
        ("by derived slice", "by_slice"),
        ("by difficulty", "by_difficulty"),
        ("by source", "by_source"),
    ):
        lines += ["", f"## {title}", ""]
        if key == "by_slice":
            lines.append(
                "_Slices overlap by design (a case can be `join` and "
                "`coded-vocabulary` at once); counts do not sum to the total._"
            )
            lines.append("")
        lines.append(_axis_table(metrics[key]))

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# compare report
# --------------------------------------------------------------------------


class IdentityMismatch(ValueError):
    """Raised by `render_compare` when two runs' evaluator identity
    disagrees. Comparing runs scored by different rulers -- a bumped
    `EVALUATOR_VERSION`, a re-sliced golden set, an edited scoring module,
    a different golden file -- would silently misreport a ruler change as
    a model difference, which is exactly the failure mode this guards.
    """


_IDENTITY_FIELDS = (
    "evaluator_version",
    "slice_rubric_version",
    "evals_code_hash",
    "golden_hash",
)


def _check_identity(run_a: Any, run_b: Any) -> None:
    mismatched = [f for f in _IDENTITY_FIELDS if getattr(run_a, f) != getattr(run_b, f)]
    if mismatched:
        detail = "; ".join(
            f"{f}: {getattr(run_a, f)!r} vs {getattr(run_b, f)!r}" for f in mismatched
        )
        raise IdentityMismatch(
            f"cannot compare runs {run_a.id!r} and {run_b.id!r}: evaluator identity "
            f"differs on {', '.join(mismatched)} -- {detail}"
        )


def _e2e_rate(rows: Sequence[Row]) -> float | None:
    if not rows:
        return None
    return sum(1 for r in rows if r.e2e_correct) / len(rows)


def render_compare(
    run_a: Any,
    rows_a: Sequence[Row],
    run_b: Any,
    rows_b: Sequence[Row],
    cases: Sequence[GoldenCase],
    registry: Registry,
    *,
    seed: int,
) -> str:
    """The frozen three-layer comparison. Raises `IdentityMismatch` (naming
    the field) rather than rendering anything when `run_a`/`run_b` were
    scored by different rulers -- see that exception's docstring.

    `cases` is the golden set both runs replayed (or a superset of it);
    family membership and the canonical/paraphrase split are read off it,
    never off the rows.

    **Error rows (`row.error is not None`) are excluded before anything
    else here**, the same discipline `aggregate()` already applies (its own
    module docstring): a 502 or a timeout is infra flakiness, not a model
    answer, and letting one into the primary McNemar pairing would move the
    single confirmatory p-value on a coin flip that has nothing to do with
    either model. A case only enters the shared/canonical/expanded sets
    below when **both** runs have an error-free row for it -- a case where
    only one side failed contributes nothing to either side's comparison,
    rather than comparing a real answer against a failure. Section 3's
    `aggregate()` calls receive the *unfiltered* `rows_a`/`rows_b` (they do
    their own, identical filtering internally and additionally report the
    per-run error count/rate), so the two no longer disagree about what an
    "error" run's rate even means.
    """
    _check_identity(run_a, run_b)

    family_map = {c.id: family(c) for c in cases}

    def family_of(row: Row) -> str:
        return family_map.get(row.case_id, row.case_id)

    canonical_ids = {c.id for c in cases if c.paraphrase_of is None}
    errors_a = sum(1 for r in rows_a if r.error is not None)
    errors_b = sum(1 for r in rows_b if r.error is not None)
    by_id_a = {r.case_id: r for r in rows_a if r.error is None}
    by_id_b = {r.case_id: r for r in rows_b if r.error is None}
    shared_ids = sorted(set(by_id_a) & set(by_id_b))
    shared_canonical_ids = [cid for cid in shared_ids if cid in canonical_ids]

    canon_a = [by_id_a[cid] for cid in shared_canonical_ids]
    canon_b = [by_id_b[cid] for cid in shared_canonical_ids]
    expanded_a = [by_id_a[cid] for cid in shared_ids]
    expanded_b = [by_id_b[cid] for cid in shared_ids]

    lines = [
        f"# Compare `{run_a.id}` (A) vs `{run_b.id}` (B)",
        "",
        f"- A: {run_a.provider} / {run_a.model_id}, concurrency {run_a.concurrency}, "
        f"error rows: {errors_a} of {len(rows_a)}, "
        f"canonical_only={run_a.canonical_only}, stratify={run_a.stratify}",
        f"- B: {run_b.provider} / {run_b.model_id}, concurrency {run_b.concurrency}, "
        f"error rows: {errors_b} of {len(rows_b)}, "
        f"canonical_only={run_b.canonical_only}, stratify={run_b.stratify}",
        f"- Shared, error-free cases: {len(shared_ids)} "
        f"(canonical: {len(shared_canonical_ids)})",
        f"- Bootstrap/permutation seed: {seed}",
        "",
    ]

    # -- 1. PRIMARY --------------------------------------------------------
    lines += ["## 1. Primary: paired McNemar on e2e-correctness (canonical cases only)", ""]
    if shared_canonical_ids:
        mcnemar = mcnemar_exact(canon_a, canon_b, outcome=lambda r: r.e2e_correct)
        delta, lo, hi = paired_delta_ci(
            canon_a, canon_b, metric=lambda r: float(r.e2e_correct),
            family_of=family_of, seed=seed,
        )
        if delta is not None:
            delta_line = f"- Paired delta (A - B), 95% CI: {delta:.3f} {_fmt_ci(lo, hi)}"
        else:
            delta_line = "- Paired delta: n/a (no shared canonical families)"
        lines += [
            f"- n = {len(shared_canonical_ids)}, n10 (A right / B wrong) = {mcnemar['n10']}, "
            f"n01 (B right / A wrong) = {mcnemar['n01']}",
            f"- p (exact, two-sided) = {mcnemar['p']:.4f}",
            delta_line,
        ]
    else:
        lines.append("_No shared canonical cases between these two runs -- nothing to test._")

    # -- 2. SECONDARY --------------------------------------------------------
    lines += ["", "## 2. Secondary: paraphrase-expanded set (descriptive)", ""]
    if shared_ids:
        boot_a = cluster_bootstrap(expanded_a, _e2e_rate, family_of, seed=seed)
        boot_b = cluster_bootstrap(expanded_b, _e2e_rate, family_of, seed=seed)

        # A family with ANY errored member, in EITHER run, is excluded
        # whole from family_accuracy/family_consistency (2026-09-18
        # review): those two functions score "every member correct" and
        # "matches the canonical" respectively, and a member missing
        # because it errored -- not because it was never sampled -- would
        # otherwise let the *surviving* members stand in for the whole
        # family (`all(p.e2e_correct for p in [])` is `True`), scoring a
        # family "all correct" or "fully consistent" when one member's
        # real correctness is simply unknown. Read off the *unfiltered*
        # rows_a/rows_b -- an error on a case neither run's sampling even
        # shares is still a reason to distrust that family's completeness.
        errored_case_ids = (
            {r.case_id for r in rows_a if r.error is not None}
            | {r.case_id for r in rows_b if r.error is not None}
        )
        compromised_families = {family_map.get(cid, cid) for cid in errored_case_ids}
        family_ids_present = (
            {family_of(r) for r in expanded_a} | {family_of(r) for r in expanded_b}
        )
        errored_member_families = compromised_families & family_ids_present

        family_metric_a = [r for r in expanded_a if family_of(r) not in compromised_families]
        family_metric_b = [r for r in expanded_b if family_of(r) not in compromised_families]

        fam_acc_a = family_accuracy(family_metric_a, family_of)
        fam_acc_b = family_accuracy(family_metric_b, family_of)
        fam_con_a = family_consistency(family_metric_a, family_of)
        fam_con_b = family_consistency(family_metric_b, family_of)
        rate_a, rate_b = _e2e_rate(expanded_a), _e2e_rate(expanded_b)
        lines.append(_table(
            ("", "A", "B"),
            [
                ("n (shared, expanded)", str(len(expanded_a)), str(len(expanded_b))),
                ("E2E-correct rate", f"{rate_a:.3f}" if rate_a is not None else "n/a",
                 f"{rate_b:.3f}" if rate_b is not None else "n/a"),
                ("Cluster-bootstrap 95% CI", _fmt_ci(*boot_a), _fmt_ci(*boot_b)),
                ("family_accuracy",
                 f"{fam_acc_a['rate']:.3f} (n={fam_acc_a['families']})"
                 if fam_acc_a["rate"] is not None else "n/a",
                 f"{fam_acc_b['rate']:.3f} (n={fam_acc_b['families']})"
                 if fam_acc_b["rate"] is not None else "n/a"),
                ("family_consistency",
                 f"{fam_con_a['rate']:.3f} (n={fam_con_a['paraphrases']})"
                 if fam_con_a["rate"] is not None else "n/a",
                 f"{fam_con_b['rate']:.3f} (n={fam_con_b['paraphrases']})"
                 if fam_con_b["rate"] is not None else "n/a"),
            ],
        ))
        lines.append(
            f"_skipped families -- canonical absent: A={fam_acc_a['skipped']} "
            f"B={fam_acc_b['skipped']}; errored member: {len(errored_member_families)}_"
        )
    else:
        lines.append("_No shared cases between these two runs._")

    # -- 3. EXPLANATORY CASCADE ---------------------------------------------
    lines += ["", "## 3. Explanatory cascade (descriptive, no p-values)", ""]
    metrics_a = aggregate(rows_a, cases, registry)
    metrics_b = aggregate(rows_b, cases, registry)
    cascade_rows = list(zip(_block_rows(metrics_a), _block_rows(metrics_b), strict=True))
    lines.append(_table(
        ("Metric", "A", "B"),
        [(label_a, value_a, value_b) for (label_a, value_a), (_, value_b) in cascade_rows],
    ))
    lines += ["", "### Refusal precision / recall, by reason", "", "**A**", "",
              _refusal_table(metrics_a), "", "**B**", "", _refusal_table(metrics_b)]
    for title, key in (
        ("by derived slice", "by_slice"),
        ("by difficulty", "by_difficulty"),
        ("by source", "by_source"),
    ):
        lines += ["", f"### {title}", "", "**A**", "", _axis_table(metrics_a[key]),
                  "", "**B**", "", _axis_table(metrics_b[key])]

    # -- 4. TRADE-OFF --------------------------------------------------------
    lines += ["", "## 4. Accuracy / latency / cost trade-off", ""]
    lat = latency_summary(
        expanded_a, other=expanded_b, family_of=family_of, seed=seed
    ) if shared_ids else {"p50": None}
    cost_a = _cost_cell(run_a, metrics_a["tokens_per_query"])
    cost_b = _cost_cell(run_b, metrics_b["tokens_per_query"])
    lines.append(_table(
        ("", "A", "B"),
        [
            ("Provider / model", f"{run_a.provider}/{run_a.model_id}",
             f"{run_b.provider}/{run_b.model_id}"),
            ("Concurrency", str(run_a.concurrency), str(run_b.concurrency)),
            ("E2E-correct rate (whole run)", _fmt(metrics_a["e2e_correct_rate"]),
             _fmt(metrics_b["e2e_correct_rate"])),
            ("Latency p50 (ms), at that concurrency", _fmt(metrics_a["latency_p50"]),
             _fmt(metrics_b["latency_p50"])),
            ("Cost per query (est.)", cost_a, cost_b),
        ],
    ))
    median_delta_ci = lat.get("median_delta_ci")
    if median_delta_ci is not None:
        _delta, delta_lo, delta_hi = median_delta_ci
        delta_ci_text = _fmt_ci(delta_lo, delta_hi)
    else:
        delta_ci_text = "n/a"
    lines.append("")
    lines.append(
        f"_Latency delta (A - B), median of paired per-case differences, "
        f"positive = A slower, family-cluster-bootstrap 95% CI: {delta_ci_text}_"
    )

    return "\n".join(lines) + "\n"
