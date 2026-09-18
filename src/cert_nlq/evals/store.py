"""The eval store: SQLAlchemy persistence for runs and per-question results.

A stored run is only comparable to another when its whole identity block
matches -- provider, model, prompt, registry, golden set, *and* the
evaluation-code identity (`evaluator_version`, `slice_rubric_version`,
`evals_code_hash`). The first four are "what was asked"; the last three are
"how the answer was judged" -- a golden file can stay byte-identical while
`derive_slices()` or `end_to_end_correct` changes underneath it, and
`evals_code_hash` is the automatic tripwire for a forgotten version bump
that a human reviewer would otherwise have to notice by eye.

Every `Row` field is a column on `results`, losslessly: tuples serialise to
JSON (and deserialise back to tuples on load, since a `Row` is a frozen
dataclass that declares tuple fields, and a caller comparing a loaded row
against a freshly-scored one should not have to know the storage layer
turned `("a", "b")` into `["a", "b"]` in between). `raw_outcome` is stored
separately from the scored fields, one JSON blob per row, so a later
evaluator-version bump can re-score the same wire responses without a new
harness run.

Importing this module without `sqlalchemy` installed fails loudly, naming
the extra to install, rather than with a bare `ModuleNotFoundError` several
frames deep in SQLAlchemy's own import machinery.
"""
from __future__ import annotations

try:
    from sqlalchemy import (
        JSON,
        DateTime,
        ForeignKey,
        TypeDecorator,
        create_engine,
        event,
        select,
    )
    from sqlalchemy.orm import (
        DeclarativeBase,
        Mapped,
        Session,
        mapped_column,
        relationship,
    )
except ImportError as exc:  # pragma: no cover - exercised via subprocess/import test
    raise ImportError(
        "cert_nlq.evals.store requires sqlalchemy; install it with "
        "`pip install -e .[evals]`"
    ) from exc

import dataclasses
import hashlib
import importlib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .scoring import Row

#: The modules whose *source text* participates in `evals_code_hash()`.
#: Sorted by name so the hash does not depend on import order, and joined
#: with a separator a source file cannot itself contain unescaped (a null
#: byte), so no concatenation of two different splits of the same total
#: text can collide onto the same hash.
_HASHED_MODULES = ("aggregate", "golden", "scoring", "stats")
_HASH_SEPARATOR = "\0"

#: `Row` fields whose stored column is JSON (tuples) rather than the
#: field's own Python type. Every other `Row` field maps straight onto a
#: JSON-capable column via a generic `JSON` type, so this only needs to
#: list the ones that need decoding back into a tuple on load.
_ROW_TUPLE_FIELDS = frozenset({
    "tags", "joins_expected", "joins_actual", "fields_expected", "fields_actual",
})


def evals_code_hash() -> str:
    """sha256 over the source text of `scoring.py + aggregate.py + stats.py
    + golden.py`, in that fixed sorted order.

    This is a hash of *what the evaluator does*, computed at run time, so a
    change to any of the four modules -- including one that should have
    bumped `EVALUATOR_VERSION` or `SLICE_RUBRIC_VERSION` but didn't --
    shows up as a different `evals_code_hash` on the next recorded run. It
    is deliberately not a substitute for those version constants (a
    behaviour-preserving refactor also changes this hash, and that is not
    a new evaluator identity) -- it is the tripwire that catches the case
    the version constants rely on a human to catch: an edit that changed
    scoring and nobody bumped the version beside it.

    Deterministic across machines with identical source text. Line endings
    are part of "identical source text" here -- a checkout with CRLF line
    endings hashes differently from one with LF -- so this is deterministic
    only to the extent the repo's line endings are; this repo normalises
    line endings via `.gitattributes`/git's own `core.autocrlf` handling,
    which is what keeps two clones of the same commit hashing the same.
    """
    digest = hashlib.sha256()
    for name in sorted(_HASHED_MODULES):
        module = importlib.import_module(f"cert_nlq.evals.{name}")
        source = Path(module.__file__).read_text(encoding="utf-8")
        digest.update(source.encode("utf-8"))
        digest.update(_HASH_SEPARATOR.encode("utf-8"))
    return digest.hexdigest()


#: Bound before `Store.record_run` shadows the name with its own
#: `evals_code_hash` parameter -- this is how that method reaches the
#: module-level function to compute a default.
_compute_evals_code_hash = evals_code_hash


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------


class UTCDateTime(TypeDecorator):
    """A `DateTime` that round-trips as UTC, tz-aware, through sqlite.

    SQLite has no native timezone-aware datetime storage -- SQLAlchemy's
    sqlite dialect stores whatever naive/aware datetime it is given as an
    ISO string and hands back a naive one on read, silently dropping
    `tzinfo`. `started_at` is documented as UTC (brief: "started_at
    (UTC)"), so rather than let every caller remember to re-attach
    `tzinfo=UTC` after a load, this type does it once, here: a tz-aware
    value is converted to UTC before the tzinfo is stripped for storage,
    and UTC is reattached to the naive value sqlite hands back on read.

    A **naive** datetime is rejected outright, not guessed at. `started_at`
    feeds a run's identity; silently assuming a naive value already means
    UTC would let a caller's local-time bug (an un-tz-aware
    `datetime.now()`, say) through as a plausible-looking timestamp that is
    wrong by the caller's UTC offset, with no signal anywhere that it
    happened. A caller that has a naive timestamp has a bug upstream, and
    the fix belongs there, not in a coercion here.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(
                "UTCDateTime requires a tz-aware datetime (got a naive one); "
                "attach tzinfo=UTC (or any tzinfo -- it is converted to UTC "
                "here) at the call site rather than relying on this column "
                "to guess"
            )
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return value.replace(tzinfo=UTC)


class Base(DeclarativeBase):
    pass


class RunRecord(Base):
    """One eval run's identity: everything two runs must agree on to be
    comparable, plus the housekeeping fields that just describe it.
    """

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(primary_key=True)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime)
    provider: Mapped[str] = mapped_column()
    model_id: Mapped[str] = mapped_column()
    router_model: Mapped[str] = mapped_column()
    #: sha256 of `ROUTER_SYSTEM + TRANSLATOR_SYSTEM`.
    prompt_hash: Mapped[str] = mapped_column()
    registry_version: Mapped[str] = mapped_column()
    service_url: Mapped[str] = mapped_column()
    concurrency: Mapped[int] = mapped_column()
    golden_path: Mapped[str] = mapped_column()
    #: Caller-supplied, like `prompt_hash` -- this module does not read the
    #: golden file, so it cannot compute this itself. The intended
    #: convention (Task 6's runner is the one that implements it): sha256
    #: of the golden.jsonl file's raw bytes, computed at load time, so two
    #: runs' `golden_hash` values agree iff they replayed byte-identical
    #: golden sets. This comment is the contract until Task 6 lands.
    golden_hash: Mapped[str] = mapped_column()
    notes: Mapped[str | None] = mapped_column(default=None)
    #: `evals.EVALUATOR_VERSION` at run time -- the version of what the
    #: numbers *mean*.
    evaluator_version: Mapped[str] = mapped_column()
    #: `evals.golden.SLICE_RUBRIC_VERSION` at run time.
    slice_rubric_version: Mapped[str] = mapped_column()
    #: `evals_code_hash()` at run time -- see that function's docstring.
    evals_code_hash: Mapped[str] = mapped_column()
    #: The non-secret dict from `/healthz` plus the model ids actually
    #: observed in the usage records -- ground truth over configuration.
    model_config_json: Mapped[dict[str, Any]] = mapped_column("model_config", JSON)
    #: The stats seed (`stats.py`'s bootstrap/permutation seed), when this
    #: run's report drew one. `None` for a run that has not yet been
    #: through the stats layer.
    seed: Mapped[int | None] = mapped_column(default=None)
    #: The `--stratify` sample size, when this run replayed a stratified
    #: subset of the golden set rather than all of it. `None` means "the
    #: whole golden set" -- part of the run's identity, because a run over
    #: a 40-case stratified sample is not comparable to one over the full
    #: 143 without saying so.
    stratify: Mapped[int | None] = mapped_column(default=None)
    #: Whether this run replayed only the canonical (non-paraphrase) cases
    #: -- the preregistered McNemar comparison's own population
    #: (`stats.py`'s module docstring). Also part of run identity.
    canonical_only: Mapped[bool] = mapped_column(default=False)

    results: Mapped[list[ResultRecord]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class ResultRecord(Base):
    """One `Row`, plus the raw wire response it was scored from."""

    __tablename__ = "results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))

    case_id: Mapped[str] = mapped_column()
    tags: Mapped[list[str]] = mapped_column(JSON)
    status_expected: Mapped[str] = mapped_column()
    status_actual: Mapped[str | None] = mapped_column(default=None)
    root_correct: Mapped[bool | None] = mapped_column(default=None)
    joins_expected: Mapped[list[str] | None] = mapped_column(JSON, default=None)
    joins_actual: Mapped[list[str] | None] = mapped_column(JSON, default=None)
    groups_recall_hit: Mapped[bool | None] = mapped_column(default=None)
    groups_recall_source: Mapped[str | None] = mapped_column(default=None)
    fields_expected: Mapped[list[str] | None] = mapped_column(JSON, default=None)
    fields_actual: Mapped[list[str] | None] = mapped_column(JSON, default=None)
    field_tp: Mapped[int | None] = mapped_column(default=None)
    field_fp: Mapped[int | None] = mapped_column(default=None)
    field_fn: Mapped[int | None] = mapped_column(default=None)
    ops_correct: Mapped[int | None] = mapped_column(default=None)
    ops_total: Mapped[int | None] = mapped_column(default=None)
    values_correct: Mapped[int | None] = mapped_column(default=None)
    values_total: Mapped[int | None] = mapped_column(default=None)
    exact_match: Mapped[bool | None] = mapped_column(default=None)
    refusal_reason_expected: Mapped[str | None] = mapped_column(default=None)
    refusal_reason_actual: Mapped[str | None] = mapped_column(default=None)
    clarify_field_expected: Mapped[str | None] = mapped_column(default=None)
    candidate_hit: Mapped[bool | None] = mapped_column(default=None)
    e2e_correct: Mapped[bool] = mapped_column()
    latency_ms: Mapped[float] = mapped_column()
    tokens_in: Mapped[int] = mapped_column()
    tokens_out: Mapped[int] = mapped_column()
    error: Mapped[str | None] = mapped_column(default=None)
    #: The wire response this row was scored from -- kept so a later
    #: evaluator-version bump can re-score without a new harness run.
    raw_outcome: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)

    run: Mapped[RunRecord] = relationship(back_populates="results")


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------


class Store:
    """Thin wrapper around one SQLAlchemy engine. Not thread-safe beyond
    whatever the underlying engine/session already gives you -- callers
    doing concurrent writes should open one `Store` per writer or serialise
    at a higher level.
    """

    def __init__(self, engine) -> None:
        self._engine = engine

    def record_run(
        self,
        *,
        provider: str,
        model_id: str,
        router_model: str,
        prompt_hash: str,
        registry_version: str,
        service_url: str,
        concurrency: int,
        golden_path: str,
        golden_hash: str,
        evaluator_version: str,
        slice_rubric_version: str,
        model_config: dict[str, Any],
        evals_code_hash: str | None = None,
        started_at: datetime | None = None,
        notes: str | None = None,
        seed: int | None = None,
        stratify: int | None = None,
        canonical_only: bool = False,
        run_id: str | None = None,
    ) -> str:
        """Record one run's identity and return its id.

        `evals_code_hash` is computed via the module-level `evals_code_hash()`
        function when not given explicitly -- a caller only needs to pass it
        to pin a value (tests, mostly); a real harness run should let this
        default so the hash always reflects the code that actually scored
        the run.
        """
        record = RunRecord(
            id=run_id or uuid.uuid4().hex,
            started_at=started_at or datetime.now(UTC),
            provider=provider,
            model_id=model_id,
            router_model=router_model,
            prompt_hash=prompt_hash,
            registry_version=registry_version,
            service_url=service_url,
            concurrency=concurrency,
            golden_path=golden_path,
            golden_hash=golden_hash,
            notes=notes,
            evaluator_version=evaluator_version,
            slice_rubric_version=slice_rubric_version,
            evals_code_hash=(
                _compute_evals_code_hash() if evals_code_hash is None else evals_code_hash
            ),
            model_config_json=model_config,
            seed=seed,
            stratify=stratify,
            canonical_only=canonical_only,
        )
        with Session(self._engine) as session:
            session.add(record)
            session.commit()
            return record.id

    def record_rows(
        self,
        run_id: str,
        rows: list[Row],
        raw_outcomes: list[dict[str, Any] | None] | None = None,
    ) -> None:
        """Attach `rows` (and their matching raw wire responses, positionally
        paired) to `run_id`. `raw_outcomes[i]` is stored alongside `rows[i]`;
        omit `raw_outcomes` entirely (or pass `None` at a position) when
        there is nothing to keep.
        """
        if raw_outcomes is not None and len(raw_outcomes) != len(rows):
            raise ValueError(
                f"raw_outcomes has {len(raw_outcomes)} entries but rows has "
                f"{len(rows)}; they must pair up positionally"
            )
        outcomes = raw_outcomes or [None] * len(rows)
        with Session(self._engine) as session:
            for row, raw_outcome in zip(rows, outcomes, strict=True):
                session.add(_row_to_record(run_id, row, raw_outcome))
            session.commit()

    def load_run(self, run_id: str) -> tuple[RunRecord, list[Row]]:
        """The run's identity record, detached from the session, and every
        attached row decoded back into a `Row` -- lossless round-trip of
        every `Row` field, tuples included.
        """
        with Session(self._engine) as session:
            run = session.get(RunRecord, run_id)
            if run is None:
                raise KeyError(f"no run recorded with id {run_id!r}")
            records = (
                session.execute(
                    select(ResultRecord)
                    .where(ResultRecord.run_id == run_id)
                    .order_by(ResultRecord.id)
                )
                .scalars()
                .all()
            )
            rows = [_record_to_row(r) for r in records]
            session.expunge(run)
            return run, rows

    def runs(self) -> list[RunRecord]:
        """Every recorded run, oldest first, detached from the session."""
        with Session(self._engine) as session:
            records = (
                session.execute(select(RunRecord).order_by(RunRecord.started_at)).scalars().all()
            )
            for record in records:
                session.expunge(record)
            return list(records)


def _enable_sqlite_foreign_keys(dbapi_connection, connection_record) -> None:
    """`PRAGMA foreign_keys=ON`, once per new DBAPI connection.

    SQLite ships foreign-key enforcement *off* by default, per connection --
    without this, `ResultRecord.run_id`'s `ForeignKey(..., ondelete="CASCADE")`
    is pure documentation: the ORM's own `cascade="all, delete-orphan"`
    still cleans up a `session.delete(run)`, but a bulk `delete(RunRecord)...`
    statement (or any other write that bypasses the ORM's Python-side
    cascade) leaves orphaned `results` rows behind, silently, because the
    database itself was never asked to enforce the constraint. Registered
    on the engine's `connect` event, not run once at open time, because a
    connection pool can open more than one DBAPI connection over an
    engine's life and each one starts with the pragma off again.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def open_store(path: str) -> Store:
    """Open (creating if needed) a sqlite-backed store at `path`.

    `path` is a plain filesystem path (or `:memory:` for an in-process,
    non-persistent store) -- this builds the `sqlite:///` URL itself, so a
    caller never has to remember SQLAlchemy's URL quoting.
    """
    url = "sqlite:///:memory:" if path == ":memory:" else f"sqlite:///{path}"
    engine = create_engine(url)
    event.listen(engine, "connect", _enable_sqlite_foreign_keys)
    Base.metadata.create_all(engine)
    return Store(engine)


# --------------------------------------------------------------------------
# Row <-> ResultRecord
# --------------------------------------------------------------------------

_ROW_FIELD_NAMES = tuple(f.name for f in dataclasses.fields(Row))


def _row_to_record(run_id: str, row: Row, raw_outcome: dict[str, Any] | None) -> ResultRecord:
    values = {}
    for name in _ROW_FIELD_NAMES:
        value = getattr(row, name)
        values[name] = list(value) if name in _ROW_TUPLE_FIELDS and value is not None else value
    return ResultRecord(run_id=run_id, raw_outcome=raw_outcome, **values)


def _record_to_row(record: ResultRecord) -> Row:
    values = {}
    for name in _ROW_FIELD_NAMES:
        value = getattr(record, name)
        is_tuple_field = name in _ROW_TUPLE_FIELDS and value is not None
        values[name] = tuple(value) if is_tuple_field else value
    return Row(**values)
