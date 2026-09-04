import subprocess

import cert_nlq


def test_package_imports_and_reports_a_version():
    assert cert_nlq.__version__ == "0.1.0"


def test_no_secret_shaped_strings_are_committed():
    """A credential in the tree is a release blocker, so it fails the suite.

    This repo's history previously contained a committed OpenAI key. The point
    of this test is that the next one fails CI instead of reaching a remote.
    """
    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split()
    offenders = []
    for path in tracked:
        if path.endswith("test_smoke.py"):
            continue  # this file names the markers it looks for
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                body = fh.read()
        except OSError:
            continue
        for marker in ("sk-proj-", "sk-ant-", "AKIA"):
            if marker in body:
                offenders.append(f"{path} contains {marker!r}")
    assert offenders == [], "\n".join(offenders)
