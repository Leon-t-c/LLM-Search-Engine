"""Fetch a live registry and report what the translator makes of it.

A development check, not part of the service. It makes **no model call and
costs nothing** — everything here is local work over one HTTP fetch.

    python scripts/check_registry.py

Reads CERT_NLQ_REGISTRY_URL and CERT_NLQ_REGISTRY_TOKEN from the
environment, the same as the service does.

Three questions, in the order they can bite:

1. Does the document parse? A ValidationError here names the field and the
   rule, and is the real answer — the host's own contract test asserts a
   hand-copy of these rules, while this asserts the model itself.
2. What does stage 1 actually see? The routing prompt is printed in full.
   Read it as the model would: a group you cannot place from its
   description and examples alone is a routing miss waiting to happen, and
   the fix is one line of registry text.
3. What does a stage-2 call cost? The generated schema is the input to
   every translation, so its size per slice is the number to watch — and
   it is measurable before spending anything.
"""
import json
import os
import sys

from cert_nlq.ir.schema import json_schema_for
from cert_nlq.registry.client import RegistryClient, RegistryUnavailable
from cert_nlq.translate import router


def main() -> int:
    url = os.environ.get("CERT_NLQ_REGISTRY_URL", "http://127.0.0.1:8000")
    token = os.environ.get("CERT_NLQ_REGISTRY_TOKEN", "")
    if not token:
        print("Set CERT_NLQ_REGISTRY_TOKEN to the host's service token.")
        return 2

    print(f"fetching {url} ...\n")
    try:
        registry = RegistryClient(url, token).fetch()
    except RegistryUnavailable as exc:
        # The message carries the host's own reason: a 401 means the tokens
        # disagree, a validation error names the field that failed.
        print(f"FAILED: {exc}")
        return 1

    print("== 1. it parses ==")
    print(f"  version        {registry.version}")
    for root in registry.roots:
        groups = root.group_names()
        coded = [f for f in root.fields if f.vocabulary]
        print(f"  root           {root.root} — {root.label}")
        print(f"  key            {', '.join(root.key)}")
        print(f"  fields         {len(root.fields)}")
        print(f"  groups         {len(groups)}  identity: {sorted(root.identity_groups())}")
        print(f"  joins          {[(j.table, j.cardinality) for j in root.joins] or 'none'}")
        values = sum(len(f.vocabulary) for f in coded)
        print(f"  coded fields   {len(coded)} carrying {values} values")
        blank = [g.name for g in root.groups if not g.description]
        print(f"  groups with no description: {blank or 'none'}")

    root = registry.roots[0]

    print("\n== 2. the routing prompt, exactly as stage 1 will see it ==\n")
    prompt = (
        f"{router.ROUTER_SYSTEM}\n\nAvailable entities:\n{router._describe(registry)}"
    )
    print(prompt)
    print(f"\n  prompt characters: {len(prompt):,}")

    print("\n== 3. stage-2 schema size, per slice ==")
    slices = [()] + [(name,) for name in root.group_names()[:3]]
    if len(root.group_names()) >= 2:
        slices.append(tuple(root.group_names()[:2]))
    for groups in slices:
        try:
            size = len(json.dumps(json_schema_for(root, groups, ())))
        except Exception as exc:  # a slice with no fields in scope
            print(f"  {str(list(groups)) or 'ALL GROUPS':58s} {exc}")
            continue
        print(f"  {str(list(groups)) or 'ALL GROUPS':58s} {size:>8,} bytes")
    print("\n  This is the input cost of every stage-2 call. A narrowed slice")
    print("  should be a small fraction of the whole.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
