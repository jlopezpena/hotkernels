"""`uv run release`: prepare a release, tag it, and the checks CI runs.

prepare VERSION   bump the version and compile the waiting notes (the release pull request)
tag               tag the release on an up-to-date main and push the tag, which publishes
verify [--tag T]  check the repository holds a finished release; run by the release workflow
check-note        check a pull request left its release note; run on every pull request
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import git
from .project import (
    NOT_NOTES,
    PYPROJECT,
    ROOT,
    UPCOMING_DIR,
    ReleaseError,
    check_release_files,
    notes_file,
    parse_note,
    parse_version,
    read_version,
    tag_for,
    upcoming_notes,
    write_version,
)

TEMPLATES = Path(__file__).resolve().parent / "templates"

# The file GitHub Actions names in this variable collects a step's outputs, one
# `name=value` line each. Set only inside a workflow.
GITHUB_OUTPUT = "GITHUB_OUTPUT"

# Every file a pull request changed (added, copied, modified, renamed, deleted),
# and only the files it added, in git's --diff-filter letters.
ANY_CHANGE = "ACMRD"
ADDED = "A"

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def main() -> None:
    """Configure logging for every command: plain messages on standard error."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def _render(template: str, **values) -> str:
    """Render one of this package's templates.

    Args:
        template: File name in `release/templates/`.
        values: The template's variables.

    Returns:
        The text, without a trailing newline.
    """
    env = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined,
                      trim_blocks=True, lstrip_blocks=True)
    return env.get_template(template).render(**values).rstrip("\n")


def _fail(error: ReleaseError) -> typer.Exit:
    """Log why a command stopped and return the exit that ends it with status 1.

    Args:
        error: What went wrong.

    Returns:
        The exception for the caller to raise.
    """
    log.error("release: %s", error)
    return typer.Exit(code=1)


def _check_new_version(version: str) -> None:
    """Check that `version` can be the next release.

    It must be untagged, at least the version the package carries now, and
    above the last release. The package's current version may itself be the
    one to release, before the first release.

    Args:
        version: The proposed version.
    """
    new = parse_version(version)
    current = parse_version(read_version())
    if new < current:
        raise ReleaseError(f"{version} is below the package's current version {current}")
    previous = git.last_tag()
    if previous is not None and new <= parse_version(previous.removeprefix("v")):
        raise ReleaseError(f"{version} is not above the last release, {previous}")
    if git.tag_exists(tag_for(version)):
        raise ReleaseError(f"{tag_for(version)} exists already")
    if notes_file(version).exists():
        raise ReleaseError(f"{notes_file(version).relative_to(ROOT)} exists already")


def _compile_notes(version: str) -> None:
    """Run towncrier: write the notes file of `version` and delete the notes it gathered.

    Args:
        version: The version being prepared.
    """
    # towncrier deletes the notes with `git rm`, so they leave the index too.
    command = [sys.executable, "-m", "towncrier", "build", "--version", version, "--yes",
               "--config", str(PYPROJECT), "--dir", str(ROOT)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise ReleaseError(f"towncrier failed: {result.stderr.strip() or result.stdout.strip()}")


@app.command()
def prepare(version: str = typer.Argument(..., help="The version to release, such as 0.2.0.")) -> None:
    """Make the release pull request's changes: bump the version and compile the notes.

    Writes `release-notes/VERSION.md` from the notes waiting in
    `release-notes/upcoming/`, deletes them, and sets VERSION in
    `hotkernels/meson.build`. Commits nothing. The highlights at the top of the
    notes are left to be written by hand.

    Args:
        version: The version to release.
    """
    try:
        _check_new_version(version)
        notes = upcoming_notes()
        previous = git.last_tag()
        merges = git.merged_pull_requests(previous)
        # The notes first: if towncrier fails, nothing has changed.
        _compile_notes(version)
        write_version(version)
    except ReleaseError as error:
        raise _fail(error)
    log.info("%s", _render("prepare.txt.j2", version=version, notes=notes, previous=previous, merges=merges,
                           notes_path=notes_file(version).relative_to(ROOT), tag=tag_for(version)))


@app.command()
def tag(push: bool = typer.Option(True, help="Push the tag. --no-push creates it here only.")) -> None:
    """Tag the release on main and push the tag, which starts the release workflow.

    Run on a clean checkout of main after the release pull request is merged.
    The tag is made from the version in `hotkernels/meson.build`.

    Args:
        push: Whether to push the tag to the remote.
    """
    try:
        if git.run("status", "--porcelain"):
            raise ReleaseError("the working tree has changes; tag a clean checkout of main")
        git.run("fetch", git.REMOTE, git.MAIN, "--tags")
        if git.run("rev-parse", "HEAD") != git.run("rev-parse", git.REMOTE_MAIN):
            raise ReleaseError(f"HEAD is not {git.REMOTE_MAIN}; check out main and pull first")
        version = read_version()
        release_tag = tag_for(version)
        if git.tag_exists(release_tag):
            raise ReleaseError(f"{release_tag} exists already; was the version bumped?")
        check_release_files(version)
        git.run("tag", "--annotate", release_tag, "--message", f"hotkernels {version}")
        if push:
            git.run("push", git.REMOTE, f"refs/tags/{release_tag}")
    except ReleaseError as error:
        raise _fail(error)
    log.info("%s", _render("tag.txt.j2", tag=release_tag, pushed=push))


@app.command()
def verify(tag: Optional[str] = typer.Option(None, help="The pushed tag that started the release.")) -> None:
    """Check that the checkout is a release that may be published, and report its version.

    With --tag, the tag must name the version in `hotkernels/meson.build`, the
    commit must be on main, and the notes must be finished. Without it, only
    the version is read. Inside GitHub Actions, writes the step outputs
    `version` and `prerelease` (`true` or `false`).

    Args:
        tag: The tag being released, such as `v0.2.0`, or None.
    """
    try:
        version = read_version()
        parsed = parse_version(version)
        if tag is not None:
            if tag != tag_for(version):
                raise ReleaseError(f"the tag {tag} does not match the package's version {version}; "
                                   f"delete the tag and push {tag_for(version)}")
            check_release_files(version)
            # The workflow fetches main, so origin/main is the branch as it is now.
            if not git.succeeds("merge-base", "--is-ancestor", "HEAD", git.REMOTE_MAIN):
                raise ReleaseError(f"{tag} names a commit that is not on {git.MAIN}")
    except ReleaseError as error:
        raise _fail(error)
    outputs = os.environ.get(GITHUB_OUTPUT)
    if outputs:
        with open(outputs, "a") as handle:
            handle.write(f"version={version}\nprerelease={str(parsed.is_prerelease).lower()}\n")
    log.info("release: version %s%s", version, f", tag {tag} verified" if tag else "")


@app.command(name="check-note")
def check_note(
    base: str = typer.Option(git.REMOTE_MAIN, help="The branch the pull request merges into."),
    pull_request: Optional[int] = typer.Option(None, "--pr", help="The pull request's number."),
) -> None:
    """Check that a pull request that changes the package leaves a release note.

    Every note it adds must be named `<PR>.<type>.md`, with the pull request's
    own number when --pr is given. A release pull request, which adds the
    notes file of the package's version, needs no note. A change no user would
    notice is let through by the `no release note` label, which the workflow
    checks before running this.

    Args:
        base: The branch the pull request merges into.
        pull_request: Its number, or None to skip the check of note names.
    """
    try:
        upcoming_prefix = str(UPCOMING_DIR.relative_to(ROOT)) + "/"
        added = git.changed_files(base, ADDED)
        if str(notes_file(read_version()).relative_to(ROOT)) in added:
            log.info("release: a release pull request; it needs no note")
            return
        notes = [parse_note(path) for path in added
                 if path.startswith(upcoming_prefix) and Path(path).name not in NOT_NOTES]
        upcoming_notes()
        for note in notes:
            if pull_request is not None and note.pull_request != pull_request:
                raise ReleaseError(f"{note.path} is named for pull request {note.pull_request}, "
                                   f"but this is pull request {pull_request}")
        package_changes = git.touches_package(git.changed_files(base, ANY_CHANGE))
    except ReleaseError as error:
        raise _fail(error)
    if package_changes and not notes:
        log.error("%s", _render("missing_note.txt.j2", changes=package_changes, upcoming=upcoming_prefix,
                                pull_request=pull_request))
        raise typer.Exit(code=1)
    log.info("release: %s", f"note {', '.join(n.path for n in notes)}" if notes else "no note needed")
