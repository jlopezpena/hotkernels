"""Static checks on kernel assembly source, run before anything is executed.

A kernel must be a pure leaf function: no calls, no system calls, no writable
data, and no use of registers the platform reserves.
"""
from __future__ import annotations

import re

# A label at the start of a line, such as `Lloop:`.
_LABEL = re.compile(r"^\s*[\w.$]+:")
# A C-style block comment.
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
# Directives that create writable data.
_DATA_DIRECTIVES = re.compile(
    r"^\s*\.(data|bss|comm|lcomm|zerofill|tbss|tdata)\b|^\s*\.section\s+__DATA", re.IGNORECASE
)

# Per-architecture rules.
#   calls:         mnemonics that call another function
#   syscalls:      mnemonics that enter the operating system
#   reserved:      pattern matching a register the platform reserves
#   reserved_name: that register's name, for the message
#   line_comment:  extra characters that start a comment, besides `//`
_RULES = {
    "arm64": {
        "calls": {"bl", "blr", "blraa", "blrab", "blraaz", "blrabz"},
        "syscalls": {"svc", "hvc", "smc"},
        "reserved": re.compile(r"\b[xw]18\b", re.IGNORECASE),
        "reserved_name": "x18",
        "line_comment": [";"],
    },
}


def _strip_comments(text: str, line_comment: list[str]) -> list[str]:
    """Return the source lines with comments removed, keeping line numbers intact."""
    # Replace each block comment with the newlines it contained.
    text = _BLOCK_COMMENT.sub(lambda m: "\n" * m.group().count("\n"), text)
    lines = []
    for line in text.splitlines():
        for marker in ["//", *line_comment]:
            line = line.split(marker, 1)[0]
        lines.append(line)
    return lines


def _strip_labels(line: str) -> str:
    """Remove any leading labels, leaving the instruction or directive."""
    while _LABEL.match(line):
        line = _LABEL.sub("", line, count=1)
    return line


def lint_source(text: str, arch: str) -> list[str]:
    """Return a list of rule violations in a kernel's assembly source.

    An empty list means the source passes. Architectures without rules pass.
    """
    rules = _RULES.get(arch)
    if rules is None:
        return []
    issues = []
    for number, raw in enumerate(_strip_comments(text, rules["line_comment"]), start=1):
        if _DATA_DIRECTIVES.search(raw):
            issues.append(f"line {number}: declares writable data")
            continue
        line = _strip_labels(raw)
        tokens = line.split()
        if not tokens:
            continue
        mnemonic = tokens[0].lower()
        if mnemonic in rules["calls"]:
            issues.append(f"line {number}: calls a function ({mnemonic})")
        if mnemonic in rules["syscalls"]:
            issues.append(f"line {number}: makes a system call ({mnemonic})")
        is_directive = mnemonic.startswith(".")
        if not is_directive and rules["reserved"].search(line):
            issues.append(f"line {number}: uses the reserved register {rules['reserved_name']}")
    return issues
