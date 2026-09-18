"""The eval store: run/result persistence, round-tripped through sqlite
`:memory:` so every test runs with no filesystem state left behind.
"""
import builtins
import importlib
import sys
import types
from datetime import UTC, datetime, timedelta, timezone

import pytest

from cert_nlq.evals import store as store_module
from cert_nlq.evals.scoring import Row
from cert_nlq.evals.store import evals_code_hash, open_store

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _identity(**overrides) -> dict:
    """A full, self-consistent run-identity kwargs dict for `record_run`."""
    base = {
        "provider": "openai",
        "model_id": "gpt-5",
        "router_model": "gpt-5-mini",
        "prompt_hash": "a" * 64,
        "registry_version": "registry-1",
        "service_url": "http://localhost:8000",
        "concurrency": 4,
        "golden_path": "eval_data/golden.jsonl",
        "golden_hash": "b" * 64,
        "evaluator_version": "1",
        "slice_rubric_version": "1",
        "model_config": {"router_model": "gpt-5-mini", "translator_model": "gpt-5"},
        "notes": "a smoke run",
        "seed": 42,
        "stratify": 40,
        "canonical_only": True,
    }
    base.update(overrides)
    return base


def _full_row(**overrides) -> Row:
    """A `Row` with every field populated (non-`None` where the type
    allows), regardless of whether the combination is one `score()` would
    itself ever produce -- this is a storage round-trip test, not a
    scoring-semantics one.
    """
    base = {
        "case_id": "case-1",
        "tags": ("regression", "review"),
        "status_expected": "ok",
        "status_actual": "ok",
        "root_correct": True,
        "joins_expected": ("owners",),
        "joins_actual": ("owners", "sales"),
        "groups_recall_hit": True,
        "groups_recall_source": "echo",
        "fields_expected": ("boro", "block"),
        "fields_actual": ("boro",),
        "field_tp": 1,
        "field_fp": 0,
        "field_fn": 1,
        "ops_correct": 2,
        "ops_total": 3,
        "values_correct": 1,
        "values_total": 1,
        "exact_match": False,
        "refusal_reason_expected": "not_a_query",
        "refusal_reason_actual": "not_a_query",
        "clarify_field_expected": "boro",
        "candidate_hit": True,
        "e2e_correct": False,
        "latency_ms": 123.5,
        "tokens_in": 100,
        "tokens_out": 50,
        "error": None,
    }
    base.update(overrides)
    return Row(**base)


# --------------------------------------------------------------------------
# runs: identity round-trip
# --------------------------------------------------------------------------


def test_record_and_load_run_round_trips_every_identity_field():
    store = open_store(":memory:")
    started = datetime(2026, 9, 18, 12, 0, 0, tzinfo=UTC)
    run_id = store.record_run(started_at=started, **_identity())

    run, rows = store.load_run(run_id)

    assert run.id == run_id
    assert run.started_at == started
    assert run.provider == "openai"
    assert run.model_id == "gpt-5"
    assert run.router_model == "gpt-5-mini"
    assert run.prompt_hash == "a" * 64
    assert run.registry_version == "registry-1"
    assert run.service_url == "http://localhost:8000"
    assert run.concurrency == 4
    assert run.golden_path == "eval_data/golden.jsonl"
    assert run.golden_hash == "b" * 64
    assert run.notes == "a smoke run"
    assert run.evaluator_version == "1"
    assert run.slice_rubric_version == "1"
    assert run.evals_code_hash == evals_code_hash()
    assert run.model_config_json == {
        "router_model": "gpt-5-mini",
        "translator_model": "gpt-5",
    }
    assert run.seed == 42
    assert run.stratify == 40
    assert run.canonical_only is True
    assert rows == []


def test_utc_datetime_rejects_a_naive_value_directly():
    """`UTCDateTime.process_bind_param` itself, not through a full
    session flush -- the ORM wraps a `process_bind_param` exception in
    `sqlalchemy.exc.StatementError` by the time it would reach a caller of
    `record_run`, which would make this assertion depend on SQLAlchemy's
    own error-wrapping behaviour rather than on what this type decided.
    Calling the method directly pins the type's own contract instead.
    """
    naive = datetime(2026, 9, 18, 12, 0, 0)  # noqa: DTZ001 - naive on purpose
    with pytest.raises(ValueError, match="tz-aware"):
        store_module.UTCDateTime().process_bind_param(naive, dialect=None)


def test_record_run_rejects_a_naive_started_at_end_to_end():
    """The same rejection, exercised through the real `record_run` path --
    confirms the naive value actually reaches `UTCDateTime` unmodified and
    the flush surfaces *some* error, without pinning SQLAlchemy's wrapper
    exception type.
    """
    store = open_store(":memory:")
    naive = datetime(2026, 9, 18, 12, 0, 0)  # noqa: DTZ001 - naive on purpose
    with pytest.raises(Exception, match="tz-aware"):
        store.record_run(started_at=naive, **_identity())


def test_utc_datetime_converts_a_non_utc_aware_value_to_utc_on_bind():
    # UTC-5: 09:00 there is 14:00 UTC.
    local = datetime(2026, 9, 18, 9, 0, 0, tzinfo=timezone(timedelta(hours=-5)))
    bound = store_module.UTCDateTime().process_bind_param(local, dialect=None)
    assert bound == datetime(2026, 9, 18, 14, 0, 0)  # noqa: DTZ001 - naive storage form
    assert bound.tzinfo is None  # stripped for storage, as UTC


def test_record_run_converts_a_non_utc_aware_started_at_to_utc():
    store = open_store(":memory:")
    # UTC-5: 09:00 there is 14:00 UTC.
    local = datetime(2026, 9, 18, 9, 0, 0, tzinfo=timezone(timedelta(hours=-5)))
    run_id = store.record_run(started_at=local, **_identity())

    run, _ = store.load_run(run_id)

    assert run.started_at == datetime(2026, 9, 18, 14, 0, 0, tzinfo=UTC)
    assert run.started_at.tzinfo == UTC


def test_record_run_generates_a_uuid4_hex_id_by_default():
    store = open_store(":memory:")
    run_id = store.record_run(**_identity())
    assert len(run_id) == 32
    # uuid4 hex: parses back as a valid UUID with version 4.
    import uuid

    assert uuid.UUID(run_id).version == 4


def test_record_run_pinned_evals_code_hash_is_not_recomputed():
    store = open_store(":memory:")
    run_id = store.record_run(evals_code_hash="pinned-hash", **_identity())
    run, _ = store.load_run(run_id)
    assert run.evals_code_hash == "pinned-hash"


def test_run_defaults_seed_stratify_notes_to_none_canonical_only_to_false():
    identity = _identity()
    for key in ("notes", "seed", "stratify", "canonical_only"):
        identity.pop(key)
    store = open_store(":memory:")
    run_id = store.record_run(**identity)
    run, _ = store.load_run(run_id)
    assert run.notes is None
    assert run.seed is None
    assert run.stratify is None
    assert run.canonical_only is False


def test_two_runs_with_different_prompt_hashes_coexist():
    store = open_store(":memory:")
    id_a = store.record_run(**_identity(prompt_hash="a" * 64))
    id_b = store.record_run(**_identity(prompt_hash="c" * 64))

    assert id_a != id_b
    run_a, _ = store.load_run(id_a)
    run_b, _ = store.load_run(id_b)
    assert run_a.prompt_hash == "a" * 64
    assert run_b.prompt_hash == "c" * 64

    all_runs = store.runs()
    assert {r.id for r in all_runs} == {id_a, id_b}


def test_load_run_raises_for_unknown_run_id():
    store = open_store(":memory:")
    with pytest.raises(KeyError):
        store.load_run("does-not-exist")


def test_runs_returns_in_started_at_order():
    store = open_store(":memory:")
    early = datetime(2026, 1, 1, tzinfo=UTC)
    late = datetime(2026, 6, 1, tzinfo=UTC)
    id_late = store.record_run(started_at=late, **_identity())
    id_early = store.record_run(started_at=early, **_identity())
    ordered = [r.id for r in store.runs()]
    assert ordered == [id_early, id_late]


# --------------------------------------------------------------------------
# results: row round-trip, losslessly
# --------------------------------------------------------------------------


def test_rows_attach_to_the_right_run():
    store = open_store(":memory:")
    id_a = store.record_run(**_identity())
    id_b = store.record_run(**_identity())

    store.record_rows(id_a, [_full_row(case_id="a-1")])
    store.record_rows(id_b, [_full_row(case_id="b-1"), _full_row(case_id="b-2")])

    _, rows_a = store.load_run(id_a)
    _, rows_b = store.load_run(id_b)
    assert [r.case_id for r in rows_a] == ["a-1"]
    assert [r.case_id for r in rows_b] == ["b-1", "b-2"]


def test_full_row_round_trips_losslessly():
    store = open_store(":memory:")
    run_id = store.record_run(**_identity())
    original = _full_row()

    store.record_rows(run_id, [original])
    _, (loaded,) = store.load_run(run_id)

    assert loaded == original


def test_row_with_none_fields_round_trips():
    store = open_store(":memory:")
    run_id = store.record_run(**_identity())
    original = _full_row(
        joins_expected=None,
        joins_actual=None,
        fields_expected=None,
        fields_actual=None,
        root_correct=None,
        groups_recall_hit=None,
        groups_recall_source=None,
        field_tp=None,
        field_fp=None,
        field_fn=None,
        ops_correct=None,
        ops_total=None,
        values_correct=None,
        values_total=None,
        exact_match=None,
        refusal_reason_expected=None,
        refusal_reason_actual=None,
        clarify_field_expected=None,
        candidate_hit=None,
        error="timeout",
        status_actual=None,
    )

    store.record_rows(run_id, [original])
    _, (loaded,) = store.load_run(run_id)

    assert loaded == original


def test_empty_tags_round_trip_as_empty_tuple():
    store = open_store(":memory:")
    run_id = store.record_run(**_identity())
    original = _full_row(tags=())

    store.record_rows(run_id, [original])
    _, (loaded,) = store.load_run(run_id)

    assert loaded.tags == ()


def test_record_rows_stores_raw_outcome_alongside_each_row():
    store = open_store(":memory:")
    run_id = store.record_run(**_identity())
    row = _full_row()
    raw = {"status": "ok", "payload": {"root": "sales"}}

    store.record_rows(run_id, [row], raw_outcomes=[raw])

    with store_module.Session(store._engine) as session:
        record = session.query(store_module.ResultRecord).filter_by(run_id=run_id).one()
        assert record.raw_outcome == raw


def test_record_rows_without_raw_outcomes_stores_none():
    store = open_store(":memory:")
    run_id = store.record_run(**_identity())
    store.record_rows(run_id, [_full_row()])

    with store_module.Session(store._engine) as session:
        record = session.query(store_module.ResultRecord).filter_by(run_id=run_id).one()
        assert record.raw_outcome is None


def test_record_rows_mismatched_raw_outcomes_length_raises():
    store = open_store(":memory:")
    run_id = store.record_run(**_identity())
    with pytest.raises(ValueError, match="raw_outcomes"):
        store.record_rows(run_id, [_full_row(), _full_row(case_id="c-2")], raw_outcomes=[{}])


# --------------------------------------------------------------------------
# cascade delete
# --------------------------------------------------------------------------


def test_deleting_a_run_cascades_to_its_results():
    store = open_store(":memory:")
    run_id = store.record_run(**_identity())
    store.record_rows(run_id, [_full_row(), _full_row(case_id="case-2")])

    with store_module.Session(store._engine) as session:
        run = session.get(store_module.RunRecord, run_id)
        session.delete(run)
        session.commit()

    with store_module.Session(store._engine) as session:
        remaining = session.query(store_module.ResultRecord).filter_by(run_id=run_id).all()
        assert remaining == []


def test_bulk_delete_of_a_run_cascades_at_the_database_layer():
    """The FK itself, not the ORM's Python-side cascade.

    `session.delete(run)` (the test above) cascades even without
    `PRAGMA foreign_keys=ON`, because SQLAlchemy's ORM walks the
    `cascade="all, delete-orphan"` relationship in Python regardless of
    what the database enforces. A bulk `DELETE` statement skips that
    relationship walk entirely and asks the database to do the deleting --
    so this only passes if `open_store` actually turned sqlite's
    foreign-key enforcement on.
    """
    from sqlalchemy import delete

    store = open_store(":memory:")
    run_id = store.record_run(**_identity())
    store.record_rows(run_id, [_full_row(), _full_row(case_id="case-2")])

    with store_module.Session(store._engine) as session:
        stmt = delete(store_module.RunRecord).where(store_module.RunRecord.id == run_id)
        session.execute(stmt)
        session.commit()

    with store_module.Session(store._engine) as session:
        remaining = session.query(store_module.ResultRecord).filter_by(run_id=run_id).all()
        assert remaining == []


# --------------------------------------------------------------------------
# evals_code_hash()
# --------------------------------------------------------------------------


def test_evals_code_hash_is_stable_across_two_calls():
    assert evals_code_hash() == evals_code_hash()


def test_evals_code_hash_is_a_sha256_hex_digest():
    digest = evals_code_hash()
    assert len(digest) == 64
    int(digest, 16)  # raises ValueError if not valid hex


def test_evals_code_hash_changes_when_source_text_changes(tmp_path, monkeypatch):
    """Route one of the four hashed modules through a temp copy of its own
    source, then edit the copy -- never the real, committed source file.
    """
    from pathlib import Path

    real_module = importlib.import_module("cert_nlq.evals.scoring")
    real_text = Path(real_module.__file__).read_text(encoding="utf-8")

    fake_path = tmp_path / "scoring_copy.py"
    fake_path.write_text(real_text, encoding="utf-8")

    real_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "cert_nlq.evals.scoring":
            return types.SimpleNamespace(__file__=str(fake_path))
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(store_module.importlib, "import_module", fake_import_module)

    before = evals_code_hash()

    fake_path.write_text(real_text + "\n# a harmless comment\n", encoding="utf-8")
    after = evals_code_hash()

    assert before != after




def test_hashed_modules_includes_runner_excludes_report():
    """Pinned membership (2026-09-18 review): `runner.py`'s wire-status
    translation and usage/route extraction decide what an answer *is*, so
    it must participate; `report.py` only renders already-scored rows and
    must not -- see `_HASHED_MODULES`'s own comment for the full reasoning.
    """
    assert set(store_module._HASHED_MODULES) == {
        "aggregate", "golden", "runner", "scoring", "stats",
    }
    assert "report" not in store_module._HASHED_MODULES


def test_evals_code_hash_changes_when_runner_source_changes(tmp_path, monkeypatch):
    """Same probe as the scoring.py test above, aimed at `runner.py` --
    confirms it is actually read, not just listed.
    """
    from pathlib import Path

    real_module = importlib.import_module("cert_nlq.evals.runner")
    real_text = Path(real_module.__file__).read_text(encoding="utf-8")

    fake_path = tmp_path / "runner_copy.py"
    fake_path.write_text(real_text, encoding="utf-8")

    real_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "cert_nlq.evals.runner":
            return types.SimpleNamespace(__file__=str(fake_path))
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(store_module.importlib, "import_module", fake_import_module)

    before = evals_code_hash()

    fake_path.write_text(real_text + "\n# a harmless comment\n", encoding="utf-8")
    after = evals_code_hash()

    assert before != after


def test_evals_code_hash_never_imports_report():
    """`report.py` changes must never perturb the hash -- checked directly
    against the real, unmocked `evals_code_hash()`: it must not even import
    `cert_nlq.evals.report` while computing the hash.
    """
    real_import_module = importlib.import_module
    imported = []

    def spy_import_module(name, *args, **kwargs):
        imported.append(name)
        return real_import_module(name, *args, **kwargs)

    import unittest.mock as mock
    with mock.patch.object(store_module.importlib, "import_module", spy_import_module):
        evals_code_hash()

    assert "cert_nlq.evals.report" not in imported


# --------------------------------------------------------------------------
# append_note()
# --------------------------------------------------------------------------


def test_append_note_creates_notes_from_blank():
    store = open_store(":memory:")
    run_id = store.record_run(**_identity(notes=None))

    store.append_note(run_id, "bootstrap_seed=7")

    run, _ = store.load_run(run_id)
    assert run.notes == "bootstrap_seed=7"


def test_append_note_appends_space_separated_to_existing_notes():
    store = open_store(":memory:")
    run_id = store.record_run(**_identity(notes="a smoke run"))

    store.append_note(run_id, "bootstrap_seed=7")

    run, _ = store.load_run(run_id)
    assert run.notes == "a smoke run bootstrap_seed=7"


def test_append_note_raises_for_unknown_run_id():
    store = open_store(":memory:")
    with pytest.raises(KeyError):
        store.append_note("does-not-exist", "bootstrap_seed=7")


# --------------------------------------------------------------------------
# missing sqlalchemy
# --------------------------------------------------------------------------


def test_import_fails_with_a_clear_message_when_sqlalchemy_is_absent(monkeypatch):
    """`cert_nlq.evals.store` must fail its own import loudly, naming the
    install command, rather than surfacing a bare `ModuleNotFoundError` from
    inside SQLAlchemy's own import machinery.

    Simulated by intercepting `builtins.__import__` for the `sqlalchemy`
    name only, and forcing a fresh import of the store module (its cached
    entry in `sys.modules` is removed and restored via `monkeypatch`, so
    this leaves every other test's view of the module untouched).
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sqlalchemy" or name.startswith("sqlalchemy."):
            raise ImportError("simulated: sqlalchemy not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "cert_nlq.evals.store", raising=False)

    with pytest.raises(ImportError, match=r"pip install -e \.\[evals\]"):
        importlib.import_module("cert_nlq.evals.store")
