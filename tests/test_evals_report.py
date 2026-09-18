"""`report.py`, exercised against the widget fixture -- `render_single_run`,
`render_compare`'s identity refusal, and the canonical-only primary filter.
"""
import re
import types
from datetime import UTC, datetime

import pytest

from cert_nlq.evals.golden import GoldenCase
from cert_nlq.evals.report import (
    IdentityMismatch,
    estimate_cost_per_query,
    estimate_spend,
    render_compare,
    render_single_run,
)
from cert_nlq.evals.scoring import Row

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _ok_case(case_id, *, source="hand-written", paraphrase_of=None):
    return GoldenCase.model_validate({
        "id": case_id,
        "question": f"question for {case_id}",
        "source": "paraphrase" if paraphrase_of else source,
        "paraphrase_of": paraphrase_of,
        "expect": {
            "kind": "ok",
            "payload": {
                "root": "widget",
                "where": {"field": "widget.serial", "op": ">", "value": 5},
            },
        },
    })


def _row(case_id, e2e_correct, **overrides):
    fields = {
        "case_id": case_id,
        "tags": (),
        "status_expected": "ok",
        "status_actual": "ok",
        "root_correct": e2e_correct,
        "joins_expected": (),
        "joins_actual": (),
        "groups_recall_hit": None,
        "groups_recall_source": None,
        "fields_expected": ("widget.serial",),
        "fields_actual": ("widget.serial",) if e2e_correct else (),
        "field_tp": 1 if e2e_correct else 0,
        "field_fp": 0,
        "field_fn": 0 if e2e_correct else 1,
        "ops_correct": 1,
        "ops_total": 1,
        "values_correct": 1,
        "values_total": 1,
        "exact_match": e2e_correct,
        "refusal_reason_expected": None,
        "refusal_reason_actual": None,
        "clarify_field_expected": None,
        "candidate_hit": None,
        "e2e_correct": e2e_correct,
        "latency_ms": 100.0,
        "tokens_in": 500,
        "tokens_out": 50,
        "error": None,
    }
    fields.update(overrides)
    return Row(**fields)


def _run_identity(**overrides):
    base = {
        "id": "run-a",
        "started_at": datetime(2026, 9, 18, tzinfo=UTC),
        "provider": "openai",
        "model_id": "gpt-5",
        "router_model": "gpt-5-mini",
        "registry_version": "registry-1",
        "golden_path": "eval_data/golden.jsonl",
        "golden_hash": "h" * 64,
        "concurrency": 4,
        "canonical_only": False,
        "stratify": None,
        "notes": None,
        "evaluator_version": "1",
        "slice_rubric_version": "1",
        "evals_code_hash": "c" * 64,
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


# --------------------------------------------------------------------------
# render_single_run
# --------------------------------------------------------------------------


def test_render_single_run_includes_headline_metrics_with_n(registry):
    cases = [_ok_case("c1"), _ok_case("c2")]
    rows = [_row("c1", True), _row("c2", False)]
    run = _run_identity()

    text = render_single_run(run, rows, cases, registry)

    assert "run-a" in text
    assert "E2E-correct rate" in text
    assert "0.500 (n=2)" in text  # 1/2 correct
    assert "## §11 metric block" in text
    assert "## by derived slice" in text
    assert "## by difficulty" in text
    assert "## by source" in text


def test_render_single_run_reports_error_rows_and_unmatched_separately(registry):
    cases = [_ok_case("c1")]
    rows = [_row("c1", True), _row("c-missing", False, error="timeout", status_actual=None,
                                    root_correct=None, exact_match=None, field_tp=None,
                                    field_fp=None, field_fn=None, ops_correct=None,
                                    ops_total=None, values_correct=None, values_total=None,
                                    fields_expected=None, fields_actual=None)]
    run = _run_identity()

    text = render_single_run(run, rows, cases, registry)

    assert "Error rows: 1 of 2" in text


# --------------------------------------------------------------------------
# render_compare: identity refusal
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field", ["evaluator_version", "slice_rubric_version", "evals_code_hash", "golden_hash"]
)
def test_render_compare_refuses_on_each_mismatched_identity_field(registry, field):
    run_a = _run_identity(id="run-a")
    run_b = _run_identity(id="run-b", **{field: "different-value"})

    with pytest.raises(IdentityMismatch) as excinfo:
        render_compare(run_a, [], run_b, [], [], registry, seed=1)

    assert field in str(excinfo.value)


def test_render_compare_succeeds_when_identity_matches(registry):
    cases = [_ok_case("c1")]
    rows_a = [_row("c1", True)]
    rows_b = [_row("c1", False)]
    run_a, run_b = _run_identity(id="run-a"), _run_identity(id="run-b")

    text = render_compare(run_a, rows_a, run_b, rows_b, cases, registry, seed=1)
    assert "run-a" in text
    assert "run-b" in text


# --------------------------------------------------------------------------
# render_compare: the primary McNemar filter is canonical-only
# --------------------------------------------------------------------------


def test_primary_mcnemar_excludes_a_shared_paraphrase_row():
    """A paraphrase row present in both runs, discordant in the same
    direction as a canonical discordant pair, must NOT inflate n10 -- the
    primary test's population is canonical cases only.
    """
    from cert_nlq.registry.models import Registry, RootSpec

    root = RootSpec(root="widget", label="Widgets", key=(), fields=())
    reg = Registry(version="v", roots=(root,))

    canonical_case = _ok_case("c1")
    concordant_case = _ok_case("c2")
    paraphrase_case = _ok_case("c1-p1", paraphrase_of="c1")
    cases = [canonical_case, concordant_case, paraphrase_case]

    # c1: A right, B wrong (discordant, n10). c2: both right (concordant).
    # c1-p1: A right, B wrong too -- would double n10 to 2 if wrongly
    # counted as primary population.
    rows_a = [_row("c1", True), _row("c2", True), _row("c1-p1", True)]
    rows_b = [_row("c1", False), _row("c2", True), _row("c1-p1", False)]
    run_a, run_b = _run_identity(id="run-a"), _run_identity(id="run-b")

    text = render_compare(run_a, rows_a, run_b, rows_b, cases, reg, seed=1)

    match = re.search(r"n10 \(A right / B wrong\) = (\d+)", text)
    assert match is not None
    assert match.group(1) == "1", text


# --------------------------------------------------------------------------
# family_accuracy / family_consistency rendered adjacent
# --------------------------------------------------------------------------


def test_family_metrics_rendered_adjacent(registry):
    canonical_case = _ok_case("c1")
    paraphrase_case = _ok_case("c1-p1", paraphrase_of="c1")
    cases = [canonical_case, paraphrase_case]
    rows_a = [_row("c1", True), _row("c1-p1", True)]
    rows_b = [_row("c1", True), _row("c1-p1", False)]
    run_a, run_b = _run_identity(id="run-a"), _run_identity(id="run-b")

    text = render_compare(run_a, rows_a, run_b, rows_b, cases, registry, seed=1)

    acc_idx = text.index("family_accuracy")
    con_idx = text.index("family_consistency")
    between = text[min(acc_idx, con_idx):max(acc_idx, con_idx)]
    # No section heading between the two -- they sit in the same table.
    assert "##" not in between


# --------------------------------------------------------------------------
# spend estimation
# --------------------------------------------------------------------------


def test_estimate_spend_scales_with_case_count():
    assert estimate_spend(20, "gpt-5") == pytest.approx(2 * estimate_spend(10, "gpt-5"))


def test_estimate_spend_is_positive_for_a_nonzero_case_count():
    assert estimate_spend(50, "gpt-5") > 0


def test_estimate_cost_per_query_none_for_non_openai_provider():
    assert estimate_cost_per_query("ollama", "qwen3:4b", 3700, 300) is None


def test_estimate_cost_per_query_positive_for_openai():
    cost = estimate_cost_per_query("openai", "gpt-5", 3700, 300)
    assert cost is not None
    assert cost > 0
