"""The git commands the release steps need, run in the repository root."""
from __future__ import annotations

import fnmatch
import subprocess

from .project import ROOT, ReleaseError

# The remote and branch a release is cut from.
REMOTE = "origin"
MAIN = "main"
REMOTE_MAIN = f"{REMOTE}/{MAIN}"

# Paths whose change reaches users of the package: the package itself, and the
# kernel sources and specs it is built from. A pull request that changes one of
# them leaves a release note. fnmatch patterns, matched against paths relative
# to the repository root; `*` also matches `/`.
PACKAGE_PATHS = ("hotkernels/*", "kernels/*/kernel.s", "kernels/*/naive.c", "kernels/*/spec.py")


def run(*args: str) -> str:
    """Run a git command and return its standard output, stripped.

    Args:
        args: The arguments after `git`.

    Returns:
        What the command printed.
    """
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise ReleaseError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def succeeds(*args: str) -> bool:
    """Run a git command that answers yes or no by its exit status.

    Args:
        args: The arguments after `git`.

    Returns:
        Whether it exited with status 0.
    """
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True).returncode == 0


def changed_files(base: str, diff_filter: str) -> list[str]:
    """Return the files a branch changed since it left `base`.

    Args:
        base: The branch or commit the pull request will merge into.
        diff_filter: git's `--diff-filter`, such as `A` for added files only.

    Returns:
        Paths relative to the repository root.
    """
    # `base...HEAD` diffs against the merge base, so changes made on `base`
    # since the branch left it are not counted as the branch's.
    output = run("diff", "--name-only", f"--diff-filter={diff_filter}", f"{base}...HEAD")
    return output.splitlines() if output else []


def touches_package(paths: list[str]) -> list[str]:
    """Return the paths among `paths` that change what users of the package get."""
    return [path for path in paths if any(fnmatch.fnmatch(path, pattern) for pattern in PACKAGE_PATHS)]


def last_tag() -> str | None:
    """Return the most recent release tag reachable from HEAD, or None before the first release."""
    if not succeeds("describe", "--tags", "--abbrev=0", "--match", "v*"):
        return None
    return run("describe", "--tags", "--abbrev=0", "--match", "v*")


def merged_pull_requests(since: str | None) -> list[str]:
    """Return the subject lines of the merges since a tag, newest first.

    Args:
        since: The previous release's tag, or None for all of history.

    Returns:
        One line per merge commit, such as `Merge pull request #23 from ...`.
    """
    span = f"{since}..HEAD" if since else "HEAD"
    output = run("log", "--merges", "--format=%s", span)
    return output.splitlines() if output else []


def tag_exists(tag: str) -> bool:
    """Return whether `tag` exists here or on the remote."""
    if succeeds("rev-parse", "--verify", "--quiet", f"refs/tags/{tag}"):
        return True
    return bool(run("ls-remote", "--tags", REMOTE, f"refs/tags/{tag}"))
