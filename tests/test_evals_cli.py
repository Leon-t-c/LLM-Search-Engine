"""`python -m cert_nlq.evals`, exercised at the module boundary -- the
spend gate's provider source and the sample-argument validation. Registry
fetch, golden loading, `/healthz`, and the async fan-out are all
monkeypatched onto `cert_nlq.evals.__main__`'s own names; no real network,
no real filesystem golden file, no real store beyond an in-memory one where
a test actually needs to reach it.
"""
import types

import cert_nlq.evals.__main__ as cli_main
from cert_nlq.evals.golden import GoldenCase
from cert_nlq.evals.scoring import Outcome

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _settings(**overrides):
    base = {
        "eval_service_url": "http://fake-eval.test",
        "eval_service_token": "tok",
        "eval_golden_path": "unused.jsonl",
        "eval_db_path": ":memory:",
        "eval_concurrency": 2,
        "registry_url": "http://fake-registry.test",
        "registry_token": "tok",
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _ok_case(case_id):
    return GoldenCase.model_validate({
        "id": case_id,
        "question": f"question for {case_id}",
        "source": "hand-written",
        "expect": {
            "kind": "ok",
            "payload": {
                "root": "widget",
                "where": {"field": "widget.serial", "op": ">", "value": 5},
            },
        },
    })


def _patch_common(monkeypatch, registry, health_body, cases=None):
    monkeypatch.setattr(cli_main, "get_settings", lambda: _settings())
    monkeypatch.setattr(cli_main, "_fetch_registry", lambda settings: registry)
    monkeypatch.setattr(cli_main, "load_golden", lambda path, reg: cases or [_ok_case("c1")])
    monkeypatch.setattr(cli_main, "_golden_hash", lambda path: "h" * 64)
    monkeypatch.setattr(cli_main, "_check_healthz", lambda settings: health_body)


def _forbid_run_benchmark(monkeypatch):
    """Fails the test loudly if the fan-out is ever reached."""
    async def fake_run_benchmark(cases, *, client, concurrency, now_year=None):
        raise AssertionError("run_benchmark must not be called")
    monkeypatch.setattr(cli_main, "run_benchmark", fake_run_benchmark)


# --------------------------------------------------------------------------
# the spend gate is keyed off /healthz's reported provider, not --provider-tag
# --------------------------------------------------------------------------


def test_gate_uses_healthz_provider_not_the_tag(monkeypatch, registry, capsys):
    """healthz reports "openai"; the operator typed --provider-tag ollama.
    The gate must trigger anyway and refuse before any call is placed --
    a wrong tag must not let ~180 billable calls through.
    """
    health_body = {
        "status": "ok", "registry_version": "v1", "provider": "openai",
        "model": "gpt-5", "router_model": "gpt-5-mini",
    }
    _patch_common(monkeypatch, registry, health_body)
    _forbid_run_benchmark(monkeypatch)

    code = cli_main.main(["--provider-tag", "ollama"])

    assert code == 0
    out = capsys.readouterr().out
    assert "Estimated spend" in out
    assert "Refusing to place billable calls without --yes-spend." in out


def test_gate_skips_when_healthz_reports_ollama_even_with_openai_tag(monkeypatch, registry,
                                                                       capsys, tmp_path):
    """The inverse: healthz says ollama, the tag (wrongly) says openai --
    no gate, the free deploy is not blocked by an operator's typo either.
    """
    monkeypatch.chdir(tmp_path)  # this path runs the full happy path, which
    # writes a report -- keep it out of the repo's own eval_data/reports/.
    health_body = {
        "status": "ok", "registry_version": "v1", "provider": "ollama",
        "model": "qwen3:4b-instruct-2507", "router_model": "qwen3:4b-instruct-2507",
    }
    _patch_common(monkeypatch, registry, health_body)

    called = []

    async def fake_run_benchmark(cases, *, client, concurrency, now_year=None):
        called.append(len(cases))
        return [Outcome() for _ in cases]
    monkeypatch.setattr(cli_main, "run_benchmark", fake_run_benchmark)

    # Let record_run/record_rows/render/score all actually run against an
    # in-memory store -- cheapest way to prove the fan-out really happened
    # without also asserting on report contents here.
    code = cli_main.main(["--provider-tag", "openai"])

    assert called == [1]
    out = capsys.readouterr().out
    assert "Estimated spend" not in out
    assert code == 0


def test_gate_triggers_on_an_unknown_or_blank_provider_fail_closed(monkeypatch, registry,
                                                                     capsys):
    health_body = {
        "status": "ok", "registry_version": "v1", "provider": "",
        "model": "", "router_model": "",
    }
    _patch_common(monkeypatch, registry, health_body)
    _forbid_run_benchmark(monkeypatch)

    code = cli_main.main(["--provider-tag", "ollama"])

    assert code == 0
    assert "Refusing to place billable calls without --yes-spend." in capsys.readouterr().out


# --------------------------------------------------------------------------
# visible pricing fallback
# --------------------------------------------------------------------------


def test_pricing_fallback_is_announced_not_silent(monkeypatch, registry, capsys):
    health_body = {
        "status": "ok", "registry_version": "v1", "provider": "openai",
        "model": "some-unpriced-model", "router_model": "some-unpriced-model",
    }
    _patch_common(monkeypatch, registry, health_body)
    _forbid_run_benchmark(monkeypatch)

    cli_main.main(["--provider-tag", "openai"])

    out = capsys.readouterr().out
    assert "no rate on file for some-unpriced-model; using placeholder rates" in out


# --------------------------------------------------------------------------
# --stratify / --limit: an explicit 0 errors loudly rather than being
# silently treated as "flag not given"
# --------------------------------------------------------------------------


def test_stratify_zero_errors_loudly(monkeypatch, capsys):
    monkeypatch.setattr(cli_main, "get_settings", lambda: _settings())
    code = cli_main.main(["--provider-tag", "ollama", "--stratify", "0"])
    assert code == 2
    assert "--stratify" in capsys.readouterr().err


def test_limit_zero_errors_loudly(monkeypatch, capsys):
    monkeypatch.setattr(cli_main, "get_settings", lambda: _settings())
    code = cli_main.main(["--provider-tag", "ollama", "--limit", "0"])
    assert code == 2
    assert "--limit" in capsys.readouterr().err


def test_negative_stratify_errors_loudly(monkeypatch, capsys):
    monkeypatch.setattr(cli_main, "get_settings", lambda: _settings())
    code = cli_main.main(["--provider-tag", "ollama", "--stratify", "-5"])
    assert code == 2
    assert "--stratify" in capsys.readouterr().err


def test_stratify_and_limit_together_are_rejected(monkeypatch, capsys):
    """Combining them would let --limit re-split families --stratify just
    went to the trouble of keeping whole -- mutually exclusive, by design
    (2026-09-18 review).
    """
    monkeypatch.setattr(cli_main, "get_settings", lambda: _settings())
    code = cli_main.main(["--provider-tag", "ollama", "--stratify", "10", "--limit", "5"])
    assert code == 2
    err = capsys.readouterr().err
    assert "--stratify" in err
    assert "--limit" in err
    assert "mutually exclusive" in err


def test_positive_limit_alone_passes_validation_and_truncates_at_a_family_boundary(
    monkeypatch, registry, capsys, tmp_path
):
    """Not an error case -- confirms the `is not None` rewrite didn't
    accidentally block an ordinary positive `--limit`, and that the CLI
    actually calls the family-boundary truncation (a lone-case family here,
    so --limit 2 keeps exactly 2 of the 5 available cases).
    """
    monkeypatch.chdir(tmp_path)  # full happy path -- keep the written report
    # out of the repo's own eval_data/reports/.
    health_body = {
        "status": "ok", "registry_version": "v1", "provider": "ollama",
        "model": "qwen3:4b-instruct-2507", "router_model": "qwen3:4b-instruct-2507",
    }
    cases = [_ok_case(f"c{i}") for i in range(5)]
    _patch_common(monkeypatch, registry, health_body, cases=cases)

    seen = []

    async def fake_run_benchmark(cases, *, client, concurrency, now_year=None):
        seen.append(len(cases))
        return [Outcome() for _ in cases]
    monkeypatch.setattr(cli_main, "run_benchmark", fake_run_benchmark)

    code = cli_main.main(["--provider-tag", "ollama", "--limit", "2"])
    assert code == 0
    assert seen == [2]


def test_positive_stratify_alone_passes_validation(monkeypatch, registry, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    health_body = {
        "status": "ok", "registry_version": "v1", "provider": "ollama",
        "model": "qwen3:4b-instruct-2507", "router_model": "qwen3:4b-instruct-2507",
    }
    cases = [_ok_case(f"c{i}") for i in range(5)]
    _patch_common(monkeypatch, registry, health_body, cases=cases)

    async def fake_run_benchmark(cases, *, client, concurrency, now_year=None):
        return [Outcome() for _ in cases]
    monkeypatch.setattr(cli_main, "run_benchmark", fake_run_benchmark)

    code = cli_main.main(["--provider-tag", "ollama", "--stratify", "3"])
    assert code == 0


# --------------------------------------------------------------------------
# --compare records the bootstrap seed on both runs' notes
# --------------------------------------------------------------------------


def test_compare_mode_appends_bootstrap_seed_to_both_runs_notes(
    monkeypatch, registry, tmp_path
):
    from cert_nlq.evals.store import open_store

    db_path = tmp_path / "evals.db"
    store = open_store(str(db_path))
    identity = {
        "provider": "openai", "model_id": "gpt-5", "router_model": "gpt-5-mini",
        "prompt_hash": "a" * 64, "registry_version": "v1", "service_url": "http://x",
        "concurrency": 2, "golden_path": "unused.jsonl", "golden_hash": "h" * 64,
        "evaluator_version": "1", "slice_rubric_version": "1", "model_config": {},
    }
    from cert_nlq.evals.scoring import Row

    def _row(case_id):
        return Row(
            case_id=case_id, tags=(), status_expected="ok", status_actual="ok",
            root_correct=True, joins_expected=(), joins_actual=(),
            groups_recall_hit=None, groups_recall_source=None,
            fields_expected=(), fields_actual=(), field_tp=1, field_fp=0, field_fn=0,
            ops_correct=1, ops_total=1, values_correct=1, values_total=1,
            exact_match=True, refusal_reason_expected=None, refusal_reason_actual=None,
            clarify_field_expected=None, candidate_hit=None, e2e_correct=True,
            latency_ms=1.0, tokens_in=10, tokens_out=5, error=None,
        )

    run_id_a = store.record_run(**identity)
    run_id_b = store.record_run(**identity, notes="an existing note")
    store.record_rows(run_id_a, [_row("c1")])
    store.record_rows(run_id_b, [_row("c1")])

    monkeypatch.setattr(cli_main, "get_settings", lambda: _settings(eval_db_path=str(db_path)))
    monkeypatch.setattr(cli_main, "_fetch_registry", lambda settings: registry)
    monkeypatch.setattr(cli_main, "_golden_hash", lambda path: "h" * 64)
    monkeypatch.setattr(cli_main, "load_golden", lambda path, reg: [_ok_case("c1")])
    monkeypatch.chdir(tmp_path)

    code = cli_main.main(["--compare", run_id_a, run_id_b, "--seed", "9"])
    assert code == 0

    reloaded_a, _ = store.load_run(run_id_a)
    reloaded_b, _ = store.load_run(run_id_b)
    assert reloaded_a.notes == "bootstrap_seed=9"
    assert reloaded_b.notes == "an existing note bootstrap_seed=9"
