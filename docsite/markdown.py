"""Markdown to HTML for the site's pages.

Pages are CommonMark with a few additions: tables, footnotes, `$maths$`,
`::: note` callouts, fenced `mermaid` diagrams, and two link shorthands.
A link to `/report/judge/` is rewritten to be relative to the page it is on,
so the site works under any path prefix, and a link to `repo:judge/bench.py`
points at that file on GitHub.
"""
from __future__ import annotations

from dataclasses import dataclass
from html import escape

from markdown_it import MarkdownIt
from mdit_py_plugins.anchors import anchors_plugin
from mdit_py_plugins.container import container_plugin
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.footnote import footnote_plugin
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

# The fence language that marks a diagram instead of code.
MERMAID = "mermaid"
# Link prefix for a file in the repository, and where such links lead.
REPO_SCHEME = "repo:"
REPO_BLOB_PATH = "/blob/main/"
# The kinds of callout a page can use: `::: note Optional title`.
CALLOUTS = ("note", "aside", "warning", "takeaway")
# Headings that get an id, and the level listed in a page's table of contents.
ANCHOR_MIN_LEVEL, ANCHOR_MAX_LEVEL = 2, 3
TOC_TAG = "h2"
# Lexer used when a fence names no language or one Pygments does not know.
PLAIN_LEXER = "text"


@dataclass
class Rendered:
    """A page's Markdown, rendered."""

    # The page body as HTML.
    html: str
    # The page's second-level headings, as (id, text) pairs in document order.
    toc: list[tuple[str, str]]
    # Whether the page holds a diagram, so the template loads the renderer.
    has_diagram: bool
    # Whether the page holds maths, so the template loads the typesetter.
    has_math: bool


def _fence(self, tokens, idx, options, env) -> str:
    """Render a fenced block: a diagram for `mermaid`, highlighted code otherwise.

    Args:
        self: The markdown-it renderer.
        tokens: The token stream.
        idx: Index of the fence token.
        options: The parser's options. Unused.
        env: The render environment. Unused.

    Returns:
        The block's HTML. A diagram is left as text for the browser to draw;
        anything after `mermaid` on the opening line becomes its caption.
    """
    token = tokens[idx]
    language, _, rest = token.info.strip().partition(" ")
    if language == MERMAID:
        caption = f"<figcaption>{escape(rest)}</figcaption>" if rest else ""
        return f'<figure class="diagram"><pre class="mermaid">{escape(token.content)}</pre>{caption}</figure>\n'
    try:
        lexer = get_lexer_by_name(language or PLAIN_LEXER)
    except ClassNotFound:
        lexer = get_lexer_by_name(PLAIN_LEXER)
    code = highlight(token.content, lexer, HtmlFormatter(nowrap=True))
    return f'<pre class="code" data-lang="{escape(language)}"><code>{code}</code></pre>\n'


def _link_open(self, tokens, idx, options, env) -> str:
    """Render the opening tag of a link, expanding the two shorthands.

    Args:
        self: The markdown-it renderer.
        tokens: The token stream.
        idx: Index of the link token.
        options: The parser's options.
        env: The render environment: `root` is the relative path from the page
            to the site root, and `repo` the repository's URL.

    Returns:
        The `<a>` tag.
    """
    token = tokens[idx]
    href = token.attrGet("href") or ""
    if href.startswith(REPO_SCHEME):
        token.attrSet("href", env["repo"] + REPO_BLOB_PATH + href[len(REPO_SCHEME):])
    elif href.startswith("/"):
        token.attrSet("href", env["root"] + href[1:])
    return self.renderToken(tokens, idx, options, env)


def _table_open(self, tokens, idx, options, env) -> str:
    """Open a table inside a wrapper that scrolls sideways on a narrow screen."""
    return '<div class="table-wrap">' + self.renderToken(tokens, idx, options, env)


def _table_close(self, tokens, idx, options, env) -> str:
    """Close a table and its scrolling wrapper."""
    return self.renderToken(tokens, idx, options, env) + "</div>"


def _callout(self, tokens, idx, options, env) -> str:
    """Render the opening or closing of a `::: kind Title` callout.

    Args:
        self: The markdown-it renderer.
        tokens: The token stream.
        idx: Index of the container token.
        options: The parser's options. Unused.
        env: The render environment. Unused.

    Returns:
        An `<aside>` tag carrying the callout's kind, with its title if it has one.
    """
    token = tokens[idx]
    if token.nesting != 1:
        return "</aside>\n"
    kind, _, title = token.info.strip().partition(" ")
    heading = f'<p class="callout-title">{escape(title)}</p>' if title else ""
    return f'<aside class="callout {kind}">{heading}\n'


def make_parser() -> MarkdownIt:
    """Build the Markdown parser every page is rendered with.

    Returns:
        A parser with the site's extensions and render rules installed.
    """
    md = MarkdownIt("commonmark", {"html": True, "typographer": True}).enable(["table", "replacements", "smartquotes"])
    md.use(anchors_plugin, min_level=ANCHOR_MIN_LEVEL, max_level=ANCHOR_MAX_LEVEL)
    md.use(footnote_plugin)
    md.use(dollarmath_plugin, allow_labels=False, double_inline=False)
    for kind in CALLOUTS:
        md.use(container_plugin, name=kind, render=_callout)
    md.add_render_rule("fence", _fence)
    md.add_render_rule("link_open", _link_open)
    md.add_render_rule("table_open", _table_open)
    md.add_render_rule("table_close", _table_close)
    return md


def _toc(tokens: list) -> list[tuple[str, str]]:
    """Collect a page's table of contents from its tokens.

    Args:
        tokens: The page's block tokens, after the anchors plugin has run.

    Returns:
        The (id, text) of every second-level heading, in order.
    """
    entries = []
    for i, token in enumerate(tokens):
        if token.type == "heading_open" and token.tag == TOC_TAG:
            text = "".join(child.content for child in tokens[i + 1].children or [] if child.type in ("text", "code_inline"))
            entries.append((token.attrGet("id") or "", text))
    return entries


def render(md: MarkdownIt, text: str, root: str, repo: str) -> Rendered:
    """Render one page.

    Args:
        md: The parser from `make_parser`.
        text: The page's Markdown.
        root: Relative path from the page to the site root, ending in "/" or empty.
        repo: URL of the repository on GitHub, without a trailing slash.

    Returns:
        The page's HTML and what the template needs to know about it.
    """
    env = {"root": root, "repo": repo}
    tokens = md.parse(text, env)
    html = md.renderer.render(tokens, md.options, env)
    return Rendered(
        html=html,
        toc=_toc(tokens),
        has_diagram='class="mermaid"' in html,
        has_math='class="math' in html,
    )


def render_inline(md: MarkdownIt, text: str) -> str:
    """Render a single line of Markdown, without a surrounding paragraph.

    Args:
        md: The parser from `make_parser`.
        text: One line of Markdown.

    Returns:
        The line as HTML.
    """
    return md.renderInline(text, {"root": "", "repo": ""})
