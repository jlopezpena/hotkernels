"""Builds the site: pages, templates, static files and data into one directory.

A page is a Markdown file under `pages/`, listed in `site.toml`. It is first
run through Jinja, so it can quote numbers from the repository instead of
repeating them, then rendered to HTML and placed in the page template.

Jinja's usual delimiters collide with the braces in code samples and
diagrams, so pages use square ones: `[[ value ]]`, `[% for ... %]`, `[# note #]`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import tomllib
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import data as D
from .markdown import make_parser, render, render_inline

log = logging.getLogger(__name__)

SITE = Path(__file__).resolve().parent
PAGES = SITE / "pages"
TEMPLATES = SITE / "templates"
STATIC = SITE / "static"
CONFIG = SITE / "site.toml"
# Where the site is built unless the caller says otherwise. `build/` is ignored by git.
DEFAULT_OUT = D.ROOT / "build" / "site"

PAGE_TEMPLATE = "page.html.j2"
# Left in the output directory by a build. A directory without it is not ours to delete.
MARKER = ".docsite"
# Tells GitHub Pages to serve the files as they are, without running Jekyll over them.
NO_JEKYLL = ".nojekyll"
# Appended to the output directory's name for the directory a build is assembled in.
STAGING_SUFFIX = ".staging"
# File name a page is written under, so its URL is the directory.
INDEX = "index.html"
# Directory names inside the output.
STATIC_DIR, DATA_DIR = "static", "data"
# Reading speed used for the "minutes" shown on a page, in words per minute.
WORDS_PER_MINUTE = 230
# Hex digits of the static files' hash appended to their URLs, so a browser
# refetches them when they change and not otherwise.
VERSION_DIGITS = 10
# The layout whose pages form the report: they are numbered and linked in sequence.
REPORT_LAYOUT = "report"

_TAG = re.compile(r"<[^>]+>")


@dataclass
class Page:
    """One page of the site, as `site.toml` lists it."""

    # URL path under the site root, without slashes at either end; "" is the home page.
    path: str
    # Markdown file under `pages/`.
    source: str
    # Title shown at the top of the page and in the browser tab.
    title: str
    # Shorter title for navigation.
    label: str
    # One or two sentences under the title, also used as the page description.
    summary: str
    # Which page template variant to use: "home", "doc" or "report".
    layout: str
    # Position among the report's chapters, counted from 0; None outside the report.
    chapter: int | None = None

    @property
    def url(self) -> str:
        """The page's path from the site root, ending in "/" unless it is the root."""
        return f"{self.path}/" if self.path else ""

    @property
    def root(self) -> str:
        """Relative path from this page back to the site root."""
        return "../" * len(self.path.split("/")) if self.path else ""


def load_config() -> tuple[dict, list[Page]]:
    """Read `site.toml`.

    Returns:
        The site-wide settings, and the pages in the order they are listed,
        with the report's pages numbered in that order.
    """
    config = tomllib.loads(CONFIG.read_text())
    pages, chapter = [], 0
    for entry in config.pop("page"):
        page = Page(label=entry.pop("label", entry["title"]), **entry)
        if page.layout == REPORT_LAYOUT:
            page.chapter, chapter = chapter, chapter + 1
        pages.append(page)
    return config, pages


def _static_version() -> str:
    """Return a short hash of every static file, for cache-busting their URLs."""
    digest = hashlib.sha256()
    for path in sorted(p for p in STATIC.rglob("*") if p.is_file()):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:VERSION_DIGITS]


def _is_ours(directory: Path) -> bool:
    """Say whether a directory may be replaced: it is missing, empty, or an earlier build.

    Args:
        directory: The directory in question.
    """
    return not directory.exists() or not any(directory.iterdir()) or (directory / MARKER).exists()


def _staging(out: Path) -> Path:
    """Create an empty directory next to the output, to build into.

    The site is built there and swapped in only when the build has finished,
    so a build that fails leaves the previous site in place.

    Args:
        out: The directory the finished site will be moved to.

    Returns:
        The staging directory, holding only the build marker and `.nojekyll`.
    """
    if not _is_ours(out):
        raise ValueError(f"{out} exists and was not made by docsite; refusing to replace it")
    staging = out.with_name(out.name + STAGING_SUFFIX)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    (staging / MARKER).write_text("Built by `uv run docsite build`. Safe to delete.\n")
    (staging / NO_JEKYLL).write_text("")
    return staging


def _publish(staging: Path, out: Path) -> None:
    """Replace the output directory with a finished build.

    Args:
        staging: The directory the site was built into.
        out: Where it belongs.
    """
    if out.exists():
        shutil.rmtree(out)
    staging.rename(out)


def _minutes(html: str) -> int:
    """Estimate how long a page takes to read.

    Args:
        html: The page body.

    Returns:
        Whole minutes, at least one.
    """
    words = len(_TAG.sub(" ", html).split())
    return max(1, round(words / WORDS_PER_MINUTE))


def collect_data(md) -> dict:
    """Read everything the pages and figures quote from the repository.

    Args:
        md: The Markdown parser, used to render the text of the attempts logs.

    Returns:
        The data by name. Each entry is also written to `data/<name>.json`
        in the output for the interactive figures.
    """
    python_x86_64 = D.python_timings(D.PYTHON_TIMINGS_X86_64)
    return {
        "attempts": D.attempts(partial(render_inline, md)),
        "findings": D.findings(),
        "headline": D.headline(),
        "python": D.python_timings(),
        # The second architecture, for its chapter of the report.
        "attempts_x86_64": D.attempts(partial(render_inline, md), D.X86_64),
        "findings_x86_64": D.findings_x86_64(),
        "python_x86_64": {"meta": python_x86_64["meta"], "at_size": D.python_at(python_x86_64, D.X86_64_SIZE)},
        # The workloads' recorded runs over time, from the data branch.
        "history": D.history(partial(render_inline, md)),
    }


def build(out: Path = DEFAULT_OUT) -> int:
    """Build the whole site.

    Args:
        out: Directory to build into. It is replaced when the build succeeds.

    Returns:
        The number of pages written.
    """
    config, pages = load_config()
    md = make_parser()
    data = collect_data(md)

    staging = _staging(out)
    shutil.copytree(STATIC, staging / STATIC_DIR)
    (staging / DATA_DIR).mkdir()
    for name, value in data.items():
        (staging / DATA_DIR / f"{name}.json").write_text(json.dumps(value))

    templates = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined, autoescape=True,
                            trim_blocks=True, lstrip_blocks=True)
    content = Environment(loader=FileSystemLoader(PAGES), undefined=StrictUndefined,
                          variable_start_string="[[", variable_end_string="]]",
                          block_start_string="[%", block_end_string="%]",
                          comment_start_string="[#", comment_end_string="#]",
                          trim_blocks=True, lstrip_blocks=True)
    template = templates.get_template(PAGE_TEMPLATE)
    chapters = [p for p in pages if p.layout == REPORT_LAYOUT]
    version = _static_version()

    for page in pages:
        text = content.get_template(page.source).render(data=data, site=config, page=page, root=page.root,
                                                        chapters=chapters)
        rendered = render(md, text, page.root, config["repo"])
        neighbours = {}
        if page.chapter is not None:
            neighbours["previous"] = chapters[page.chapter - 1] if page.chapter > 0 else None
            neighbours["next"] = chapters[page.chapter + 1] if page.chapter + 1 < len(chapters) else None
        html = template.render(site=config, page=page, pages=pages, chapters=chapters, root=page.root,
                               content=rendered.html, toc=rendered.toc, has_diagram=rendered.has_diagram,
                               has_math=rendered.has_math, minutes=_minutes(rendered.html), version=version,
                               **neighbours)
        target = staging / page.path / INDEX
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html)
    _publish(staging, out)
    log.info("Built %d pages into %s", len(pages), out)
    return len(pages)
