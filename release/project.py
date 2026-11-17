"""Where the release lives in the repository: the version, the notes and their types.

Everything here reads or writes files in the checkout and runs no git command.
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from packaging.version import InvalidVersion, Version

# release/project.py -> the repository root, one level above this package.
ROOT = Path(__file__).resolve().parent.parent

# The one file that holds the package's version. meson-python reads it from the
# `project()` call, and `hotkernels/pyproject.toml` declares the version dynamic.
MESON_BUILD = ROOT / "hotkernels" / "meson.build"
# The version argument of that `project()` call: `version: '0.1.0'`, with or
# without a language before it. Group 1 is the text before the number, group 2
# the number.
VERSION_PATTERN = re.compile(r"(project\('hotkernels',(?:\s*'c',)?\s*version:\s*')([^']+)'")

# Configuration of towncrier, which also lists the note types.
PYPROJECT = ROOT / "pyproject.toml"
# Where notes wait for the next release, and where each release's notes go.
NOTES_DIR = ROOT / "release-notes"
UPCOMING_DIR = NOTES_DIR / "upcoming"
# A note's file name: the pull request number, its type, and `.md`.
NOTE_NAME = re.compile(r"^(\d+)\.([a-z]+)\.md$")
# Files in UPCOMING_DIR that are not notes: the placeholder that keeps the
# directory in git while it is empty.
NOT_NOTES = frozenset({".gitkeep"})

# Text the notes template leaves where the highlights go. A release cannot be
# tagged while it is still there, so the highlights are always written.
HIGHLIGHTS_MARKER = "HIGHLIGHTS: write"

# A git tag for a version is this prefix and the version.
TAG_PREFIX = "v"


class ReleaseError(Exception):
    """A release step cannot go on; the message says why and what to do."""


@dataclass(frozen=True)
class Note:
    """One note waiting in `release-notes/upcoming/`."""

    # The file, relative to the repository root.
    path: str
    # The pull request that left it.
    pull_request: int
    # Its type, one of `note_types()`.
    note_type: str


def read_version() -> str:
    """Return the package's version, as written in `hotkernels/meson.build`."""
    match = VERSION_PATTERN.search(MESON_BUILD.read_text())
    if match is None:
        raise ReleaseError(f"no version found in {MESON_BUILD.relative_to(ROOT)}")
    return match.group(2)


def write_version(version: str) -> None:
    """Set the package's version in `hotkernels/meson.build`.

    Args:
        version: The new version, already checked with `parse_version`.
    """
    text = MESON_BUILD.read_text()
    MESON_BUILD.write_text(VERSION_PATTERN.sub(lambda m: f"{m.group(1)}{version}'", text, count=1))


def parse_version(text: str) -> Version:
    """Parse a version the way PyPI will.

    Args:
        text: A version such as `0.2.0` or `0.3.0rc1`, without the tag prefix.

    Returns:
        The parsed version.
    """
    try:
        version = Version(text)
    except InvalidVersion:
        raise ReleaseError(f"{text!r} is not a valid version, for example 0.2.0 or 0.3.0rc1") from None
    if str(version) != text:
        raise ReleaseError(f"write the version as {version}, not {text}")
    return version


def tag_for(version: str) -> str:
    """Return the git tag that publishes `version`."""
    return TAG_PREFIX + version


def notes_file(version: str) -> Path:
    """Return the release notes file of `version`."""
    return NOTES_DIR / f"{version}.md"


def note_types() -> list[str]:
    """Return the note types, in the order their sections appear in the notes.

    They are the `[[tool.towncrier.type]]` entries of the root `pyproject.toml`,
    so towncrier and the checks here cannot disagree.
    """
    config = tomllib.loads(PYPROJECT.read_text())
    return [entry["directory"] for entry in config["tool"]["towncrier"]["type"]]


def parse_note(path: str) -> Note:
    """Check a note's file name and read the pull request and type from it.

    Args:
        path: The note, relative to the repository root.

    Returns:
        The note.
    """
    name = Path(path).name
    match = NOTE_NAME.match(name)
    types = note_types()
    if match is None or match.group(2) not in types:
        raise ReleaseError(
            f"{path}: a note is named <pull request number>.<type>.md, with the type one of {', '.join(types)}")
    return Note(path=path, pull_request=int(match.group(1)), note_type=match.group(2))


def upcoming_notes() -> list[Note]:
    """Return every note waiting for the next release, checked and sorted by file name."""
    paths = sorted(p for p in UPCOMING_DIR.iterdir() if p.name not in NOT_NOTES) if UPCOMING_DIR.exists() else []
    notes = [parse_note(str(p.relative_to(ROOT))) for p in paths]
    for note in notes:
        if not (ROOT / note.path).read_text().strip():
            raise ReleaseError(f"{note.path} is empty")
    return notes


def check_release_files(version: str) -> None:
    """Check that the repository holds a finished release of `version`.

    The notes file must exist with its highlights written, and no note may be
    left waiting, since it would belong to this release but not be in it.

    Args:
        version: The version being released.
    """
    notes = notes_file(version)
    relative = notes.relative_to(ROOT)
    if not notes.exists():
        raise ReleaseError(f"{relative} is missing; run `uv run release prepare {version}` first")
    if HIGHLIGHTS_MARKER in notes.read_text():
        raise ReleaseError(f"the highlights in {relative} are not written yet")
    waiting = upcoming_notes()
    if waiting:
        raise ReleaseError(f"{len(waiting)} notes are still in {UPCOMING_DIR.relative_to(ROOT)}; "
                           f"they belong in {relative}")
