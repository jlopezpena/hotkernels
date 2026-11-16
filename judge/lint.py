"""Static checks on kernel assembly source, run before anything is executed."""
from __future__ import annotations

import re

_LABEL = re.compile(r"^\s*[\w.$]+:")

_RULES = {
    "arm64": {
        "calls": {"bl", "blr", "blraa", "blrab", "blraaz", "blrabz"},
        "syscalls": {"svc", "hvc", "smc"},
        "reserved": re.compile(r"\b[xw]18\b", re.IGNORECASE),
        "reserved_name": "x18",
    },
}

_DATA_DIRECTIVES = re.compile(
    r"^\s*\.(data|bss|comm|lcomm|zerofill|tbss|tdata)\b|^\s*\.section\s+__DATA", re.IGNORECASE
)


def _strip_comments(text: str, arch: str) -> list[str]:
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group().count("\n"), text, flags=re.DOTALL)
    lines = []
    for line in text.splitlines():
        line = line.split("//", 1)[0]
        if arch == "arm64":
            line = line.split(";", 1)[0]
        lines.append(line)
    return lines


def lint_source(text: str, arch: str) -> list[str]:
    rules = _RULES.get(arch)
    if rules is None:
        return []
    issues = []
    for number, raw in enumerate(_strip_comments(text, arch), start=1):
        if _DATA_DIRECTIVES.search(raw):
            issues.append(f"line {number}: declares writable data")
            continue
        line = raw
        while _LABEL.match(line):
            line = _LABEL.sub("", line, count=1)
        tokens = line.split()
        if not tokens:
            continue
        mnemonic = tokens[0].lower()
        if mnemonic in rules["calls"]:
            issues.append(f"line {number}: calls a function ({mnemonic})")
        if mnemonic in rules["syscalls"]:
            issues.append(f"line {number}: makes a system call ({mnemonic})")
        if not mnemonic.startswith(".") and rules["reserved"].search(line):
            issues.append(f"line {number}: uses the reserved register {rules['reserved_name']}")
    return issues
