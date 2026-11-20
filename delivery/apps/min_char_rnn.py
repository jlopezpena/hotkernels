"""Runs Andrej Karpathy's `min-char-rnn.py` for a fixed number of iterations.

The program is copied unchanged in `vendor/min_char_rnn/`. It is Python 2,
reads `input.txt` and loops for ever, so it cannot be imported. This wrapper
reads its source, makes the textual changes listed in REWRITES, none of which
touches a NumPy call, and executes it with a text supplied in place of the
file.

The program's arrays are as large as its vocabulary: it has a weight matrix of
100 hidden units by one column per distinct character, and its gradient. It
is sized through its input alone. The small size trains on this repository's
licence, which has a few dozen distinct characters; the large one on a text of
a thousand.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

TITLE = "min-char-rnn.py, a character-level recurrent network"
SOURCE = "https://gist.github.com/karpathy/d4dee566867f8291f086 (Andrej Karpathy), copied unchanged"
LICENCE = "BSD"
REQUIRES = ()

# The program as published.
PROGRAM = Path(__file__).resolve().parent / "vendor" / "min_char_rnn" / "min-char-rnn.py"
# The text the small size is trained on: this repository's licence, which does not change.
TEXT = Path(__file__).resolve().parents[2] / "LICENSE"
# Training iterations of one run. Each is a forward and backward pass over 25 characters.
ITERATIONS = 300
# Distinct characters in the text of the large size.
LARGE_VOCABULARY = 1_000
# Times each of them occurs in that text, in a fresh random order each time:
# enough text for every iteration to read new characters.
LARGE_REPEATS = 10
# The first character of the block the large text's characters are taken from:
# CJK Unified Ideographs, a script with thousands of characters.
LARGE_FIRST_CHARACTER = 0x4E00
# Seed of NumPy's global generator, which the program draws its weights and samples from.
SEED = 0
# Changes made to the source before it runs: (pattern, replacement, what for).
REWRITES = (
    (r"\bprint (.*?)(\s+#.*)?$", r"print(\1)", "Python 2 print statements"),
    (r"\bxrange\b", "range", "Python 2 xrange"),
    (r"^while True:", f"while n < {ITERATIONS}:", "the program loops for ever"),
)
# Names of the program's globals returned as its results.
RESULTS = ("Wxh", "Whh", "Why", "bh", "by", "hprev", "smooth_loss")
# The text's distinct characters at the small size, counted when the module is loaded.
SMALL_VOCABULARY = len(set(TEXT.read_text()))
# One line per size saying how large its data is.
SIZES = {
    size: f"{ITERATIONS} iterations, 100 hidden units, sequences of 25 characters, a vocabulary of {vocabulary:,}"
    for size, vocabulary in (("small", SMALL_VOCABULARY), ("large", LARGE_VOCABULARY))
}


def large_text() -> str:
    """Build the text of the large size.

    Returns:
        LARGE_REPEATS shuffled copies of LARGE_VOCABULARY distinct
        characters, so the program's vocabulary is exactly that many.
    """
    rng = np.random.default_rng(SEED)
    codes = np.concatenate([rng.permutation(LARGE_VOCABULARY) for _ in range(LARGE_REPEATS)])
    return "".join(chr(LARGE_FIRST_CHARACTER + int(code)) for code in codes)


def make_data(size: str) -> tuple:
    """Prepare the program and its input.

    Args:
        size: One of SIZES.

    Returns:
        (compiled program, training text).
    """
    source = PROGRAM.read_text()
    for pattern, replacement, _ in REWRITES:
        source = re.sub(pattern, replacement, source, flags=re.MULTILINE)
    text = TEXT.read_text() if size == "small" else large_text()
    return compile(source, str(PROGRAM), "exec"), text


def run(program, text: str) -> tuple:
    """Run the program from the start.

    Args:
        program: The compiled program.
        text: The text it trains on.

    Returns:
        Its weights, its last hidden state and its smoothed loss.
    """

    class _Input:
        """Stands in for the file the program opens."""

        def read(self) -> str:
            """Return the training text."""
            return text

    def _silent(*_args) -> None:
        """Take the program's progress lines and discard them."""

    np.random.seed(SEED)
    scope = {"open": lambda *_args: _Input(), "print": _silent, "__name__": "min_char_rnn"}
    exec(program, scope)
    return tuple(scope[name] for name in RESULTS)
