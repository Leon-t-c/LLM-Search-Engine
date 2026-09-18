"""`python -m cert_nlq.evals` -- run the golden set against a deployed
`/translate` service, or compare two stored runs.

Two modes, chosen by `--compare`:

**Run mode** (`--provider-tag openai|ollama [--limit N] [--stratify N]
[--canonical-only] [--seed S] [--yes-spend] [--notes "..."]`): load the
golden set, sample it (`--canonical-only` first, `--stratify` second -- see
`runner.stratified_sample`'s own docstring for why that order and not the
reverse: stratifying first would draw proportionally across a set that
still includes paraphrases, then discarding the paraphrases afterwards would
silently unbalance the strata the sampler just built), check `/healthz`
(refusing on anything but `"status": "ok"`), print and gate an OpenAI spend
estimate, fan the sample out through `runner.run_benchmark`, score every
result, record the run and its rows, and render the single-run report to
`eval_data/reports/<run_id>.md`.

**Compare mode** (`--compare RUN_A RUN_B [--seed S]`): load both runs from
the store and render the frozen three-layer comparison
(`report.render_compare`), refusing -- naming the field -- when the two
runs' evaluator identity disagrees.

Only this module drives the event loop (`asyncio.run`, once, around the fan-
out); everything else in the CLI is synchronous, including the `/healthz`
probe and the registry fetch, both of which happen before any async work
starts.
"""
import argparse
import asyncio
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

from ..config import get_settings
from ..registry.client import RegistryClient, RegistryUnavailable
from ..translate.router import ROUTER_SYSTEM
from ..translate.translator import TRANSLATOR_SYSTEM
from . import EVALUATOR_VERSION
from .golden import SLICE_RUBRIC_VERSION, ExpectOk, GoldenCase, load_golden
from .report import IdentityMismatch, estimate_spend, render_compare, render_single_run
from .runner import canonical_only, run_benchmark, stratified_sample
from .scoring import score
from .store import open_store

REPORTS_DIR = Path("eval_data/reports")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cert_nlq.evals",
        description="Replay the golden set against a deployed cert-nlq service, or "
        "compare two stored runs.",
    )
    parser.add_argument(
        "--provider-tag", choices=("openai", "ollama"),
        help="Which deploy this run is against -- required unless --compare is given. "
        "Only 'openai' triggers the spend gate.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap the sampled case count.")
    parser.add_argument(
        "--stratify", type=int, default=None,
        help="Draw a seeded, family-whole sample of about N cases, proportional across "
        "difficulty strata. Applied after --canonical-only.",
    )
    parser.add_argument(
        "--canonical-only", action="store_true",
        help="Replay only cases with no paraphrase_of -- the primary McNemar's own population.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for --stratify's sampling and (in --compare mode) the bootstrap.",
    )
    parser.add_argument(
        "--yes-spend", action="store_true",
        help="Required to actually place calls on --provider-tag openai.",
    )
    parser.add_argument("--notes", default=None, help="Freeform note stored with the run.")
    parser.add_argument(
        "--compare", nargs=2, metavar=("RUN_A", "RUN_B"), default=None,
        help="Render the comparison report for two already-stored run ids instead of "
        "running anything.",
    )
    return parser


def _root_for(case: GoldenCase, registry):
    """The `RootSpec` `score()` needs for this case. An ok case's own gold
    payload names it; a clarify/refusal case may name one via `gold_root`
    (and `score` never reads the parameter for those kinds at all -- see
    its own branches), so any root is a safe fallback when neither is
    available."""
    name = (
        case.expect.payload.get("root")
        if isinstance(case.expect, ExpectOk)
        else case.gold_root
    )
    if name is not None:
        found = registry.root(name)
        if found is not None:
            return found
    return registry.roots[0]


def _golden_hash(path: str) -> str:
    """sha256 of the golden file's raw bytes -- `store.py`'s own documented
    contract for `RunRecord.golden_hash`, computed here at load time."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _prompt_hash() -> str:
    return hashlib.sha256(
        (ROUTER_SYSTEM + TRANSLATOR_SYSTEM).encode("utf-8")
    ).hexdigest()


def _fetch_registry(settings):
    client = RegistryClient(settings.registry_url, settings.registry_token)
    return client.fetch()


def _write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------
# compare mode
# --------------------------------------------------------------------------


def _run_compare(args: argparse.Namespace, settings) -> int:
    store = open_store(settings.eval_db_path)
    run_id_a, run_id_b = args.compare
    try:
        run_a, rows_a = store.load_run(run_id_a)
        run_b, rows_b = store.load_run(run_id_b)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        registry = _fetch_registry(settings)
    except RegistryUnavailable as exc:
        print(f"error: registry unavailable: {exc}", file=sys.stderr)
        return 1

    # Both runs' golden_hash is checked by render_compare (identity); this
    # additionally checks the file *on disk today* still matches run_a's
    # recorded hash -- without it, a golden file edited since the run would
    # silently hand `load_golden` a case set the run never actually saw.
    current_hash = _golden_hash(run_a.golden_path)
    if current_hash != run_a.golden_hash:
        print(
            f"error: {run_a.golden_path} has changed since run {run_a.id} was recorded "
            f"(hash {current_hash} != recorded {run_a.golden_hash}); cannot reconstruct "
            f"the case set it ran over",
            file=sys.stderr,
        )
        return 1

    cases = load_golden(run_a.golden_path, registry)

    try:
        text = render_compare(run_a, rows_a, run_b, rows_b, cases, registry, seed=args.seed)
    except IdentityMismatch as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1

    out_path = REPORTS_DIR / f"compare-{run_a.id}-{run_b.id}.md"
    _write_report(out_path, text)
    print(f"wrote {out_path}")
    return 0


# --------------------------------------------------------------------------
# run mode
# --------------------------------------------------------------------------


def _sample(cases: list[GoldenCase], args: argparse.Namespace, registry) -> list[GoldenCase]:
    if args.canonical_only:
        cases = canonical_only(cases)
    if args.stratify:
        cases = stratified_sample(cases, args.stratify, seed=args.seed, registry=registry)
    if args.limit:
        cases = cases[: args.limit]
    return cases


def _check_healthz(settings) -> dict | None:
    """`None` (with a message already printed) when the service cannot be
    reached, is unreachable, or reports anything but `"status": "ok"` --
    `"degraded"` included, by name, per the brief."""
    try:
        with httpx.Client(base_url=settings.eval_service_url, timeout=10.0) as client:
            response = client.get("/healthz")
    except httpx.HTTPError as exc:
        print(f"error: /healthz unreachable: {exc}", file=sys.stderr)
        return None
    try:
        body = response.json()
    except ValueError as exc:
        print(f"error: /healthz returned a non-JSON body: {exc}", file=sys.stderr)
        return None
    if not isinstance(body, dict) or body.get("status") != "ok":
        print(
            f"error: service is not ready (status {body.get('status')!r} at /healthz); "
            f"refusing to run",
            file=sys.stderr,
        )
        return None
    return body


def _run_benchmark_cli(args: argparse.Namespace, settings) -> int:
    if not args.provider_tag:
        print("error: --provider-tag is required unless --compare is given", file=sys.stderr)
        return 2

    try:
        registry = _fetch_registry(settings)
    except RegistryUnavailable as exc:
        print(f"error: registry unavailable: {exc}", file=sys.stderr)
        return 1

    cases = load_golden(settings.eval_golden_path, registry)
    golden_hash = _golden_hash(settings.eval_golden_path)
    cases = _sample(cases, args, registry)
    if not cases:
        print(
            "error: no cases left to run after --canonical-only/--stratify/--limit",
            file=sys.stderr,
        )
        return 1

    health_body = _check_healthz(settings)
    if health_body is None:
        return 1
    model_config = {
        "provider": health_body.get("provider"),
        "model": health_body.get("model"),
        "router_model": health_body.get("router_model"),
    }
    registry_version = health_body.get("registry_version") or ""

    if args.provider_tag == "openai":
        estimated = estimate_spend(len(cases), model_config.get("model") or "default")
        print(
            f"Estimated spend for {len(cases)} case(s) on model "
            f"{model_config.get('model')!r}: ${estimated:.2f}"
        )
        if not args.yes_spend:
            print("Refusing to place billable calls without --yes-spend.")
            return 0

    now_year = datetime.now(UTC).year

    async def _fan_out():
        headers = {}
        if settings.eval_service_token:
            headers["Authorization"] = f"Bearer {settings.eval_service_token}"
        async with httpx.AsyncClient(
            base_url=settings.eval_service_url, headers=headers
        ) as client:
            return await run_benchmark(
                cases, client=client, concurrency=settings.eval_concurrency, now_year=now_year
            )

    outcomes = asyncio.run(_fan_out())

    rows = [
        score(case, outcome, _root_for(case, registry))
        for case, outcome in zip(cases, outcomes, strict=True)
    ]

    observed_models = sorted({
        u.get("model")
        for outcome in outcomes
        for u in outcome.usage
        if isinstance(u.get("model"), str)
    })
    if observed_models:
        model_config["models_observed"] = observed_models

    store = open_store(settings.eval_db_path)
    run_id = store.record_run(
        provider=model_config.get("provider") or args.provider_tag,
        model_id=model_config.get("model") or "",
        router_model=model_config.get("router_model") or "",
        prompt_hash=_prompt_hash(),
        registry_version=registry_version,
        service_url=settings.eval_service_url,
        concurrency=settings.eval_concurrency,
        golden_path=settings.eval_golden_path,
        golden_hash=golden_hash,
        evaluator_version=EVALUATOR_VERSION,
        slice_rubric_version=SLICE_RUBRIC_VERSION,
        model_config=model_config,
        notes=args.notes,
        seed=args.seed,
        stratify=args.stratify,
        canonical_only=args.canonical_only,
    )
    # The wire body is not retained separately from the parsed `Outcome`
    # (`run_benchmark`'s return type is `list[Outcome]`, per the brief's own
    # interface) -- `raw_outcome` here is the runner's own post-translation
    # `Outcome`, dumped, not the untouched `/translate` response body.
    store.record_rows(
        run_id, rows, raw_outcomes=[o.model_dump(mode="json") for o in outcomes]
    )

    run_record, _ = store.load_run(run_id)
    text = render_single_run(run_record, rows, cases, registry)
    out_path = REPORTS_DIR / f"{run_id}.md"
    _write_report(out_path, text)
    print(f"wrote {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    settings = get_settings()
    if args.compare:
        return _run_compare(args, settings)
    return _run_benchmark_cli(args, settings)


if __name__ == "__main__":
    sys.exit(main())
