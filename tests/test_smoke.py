import subprocess

import cert_nlq


def test_package_imports_and_reports_a_version():
    assert cert_nlq.__version__ == "0.1.0"


def test_no_secret_shaped_strings_are_committed():
    """A credential in the tree is a release blocker, so it fails the suite.

    Secrets belong in the environment. This is the backstop that stops one
    reaching a remote, where rotation is the only remedy left.
    """
    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split()
    offenders = []
    for path in tracked:
        if path.endswith("test_smoke.py"):
            continue  # this file names the markers it looks for
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                body = fh.read()
        except OSError:
            continue
        for marker in ("sk-proj-", "sk-ant-", "AKIA"):
            if marker in body:
                offenders.append(f"{path} contains {marker!r}")
    assert offenders == [], "\n".join(offenders)


def test_the_readme_describes_what_is_here_now():
    """It claimed "implementation pending" long after that stopped being true.

    A tripwire, not a review: it cannot tell whether the README is *good*,
    only that it has stopped describing an empty repository, and that the
    two things a reader most needs — how to run it and how to test it — are
    named somewhere in it.
    """
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    assert "Implementation pending" not in readme
    assert "/translate" in readme and "/healthz" in readme
    assert "pytest" in readme and "uvicorn" in readme
