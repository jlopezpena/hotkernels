"""The four measurements of the gradient-step study.

1. `in_loop`: the update of `mlp_train`'s 784 x 256 weights inside the real
   training loop, written six ways, with the time of the statement, the page
   faults it takes and the time of the whole training.
2. `after_reader`: `w -= b` alone, after each of the things the training loop
   does to `w` before it, to find which one makes the statement slow.
3. `allocator`: both applications on NumPy's allocator and on a caching one.
4. `accumulations`: the two accumulations of `min_char_rnn`, timed alone at
   the program's sizes against what fusing each would leave.

Every timing alternates the ways it compares, round by round, so that a change
in the machine's load falls on all of them.
"""
from __future__ import annotations

import os
import resource
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

import hotkernels

from ..apps import mlp_train, min_char_rnn
from . import native

# Bytes ahead of the loads at which the study's loops prefetch `w` for a store.
# Chosen from 0 to 8,192 bytes on the Apple M1 Pro after a threaded matrix
# product had read `w`: 4,096 and 8,192 were the fastest, and it costs nothing
# when `w` is in the cache already.
PREFETCH_AHEAD = 4096
# Training runs per way of writing the update, alternating.
ROUNDS = 5
# Calls timed per point of `after_reader`, of which the median is kept.
REPEATS = 40
# Calls timed per point of `accumulations`.
ACCUMULATION_REPEATS = 2000
# The size of each application the study runs: the one it was first recorded
# at (docs/applications-findings.md), where the gradient step was found.
STUDIED_SIZE = {"mlp_train": "large", "min_char_rnn": "small"}
# Inputs and hidden units of mlp_train at that size: 784 and 256.
INPUTS, HIDDEN = mlp_train.NETWORKS[STUDIED_SIZE["mlp_train"]]
# Shapes `after_reader` times: the weights of mlp_train, and a square matrix
# small enough that Accelerate multiplies it on one thread.
READER_SHAPES = ((INPUTS, HIDDEN), (256, 256))
# Rows of the batch the matrix product of `after_reader` multiplies `w` by.
READER_BATCH = mlp_train.BATCH
# The parameter of mlp_train whose update is measured.
MEASURED = "w1"
# Environment variable that limits Accelerate's threads. The study reports it,
# since the effect it finds depends on it.
THREADS_VARIABLE = "VECLIB_MAXIMUM_THREADS"
# Scale of the values `after_reader` subtracts, so that `w` stays the same
# size however often the statement runs.
SMALL = 1e-12
# Nanoseconds in a microsecond and in a millisecond.
NS_PER_US, NS_PER_MS = 1e3, 1e6
# Seed of the arrays the study makes up.
SEED = 0


def _faults() -> int:
    """Return the page faults this process has taken that needed no disk read."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_minflt


def _ns(call: Callable[[], object]) -> int:
    """Time one call in nanoseconds."""
    start = time.perf_counter_ns()
    call()
    return time.perf_counter_ns() - start


# ------------------------------------------------------------------ in_loop


@dataclass(frozen=True)
class Way:
    """One way of writing the gradient step, as `in_loop` runs it."""

    # Short name used in the report.
    name: str
    # Updates `w` from `g`; given the weights, the gradient and a buffer of
    # their shape that the way may use as it likes.
    update: Callable[[np.ndarray, np.ndarray, np.ndarray], None]
    # True to run the whole training with hotkernels installed.
    installed: bool = False
    # True when the update leaves a different result from NumPy's on purpose.
    # None does.
    inexact: bool = False


def _as_written(w: np.ndarray, g: np.ndarray, buffer: np.ndarray) -> None:
    """The statement as mlp_train writes it: `w -= LEARNING_RATE * g`."""
    w -= mlp_train.LEARNING_RATE * g


def _reused(w: np.ndarray, g: np.ndarray, buffer: np.ndarray) -> None:
    """The same two passes, with the product written into a buffer that lives across steps."""
    np.multiply(mlp_train.LEARNING_RATE, g, out=buffer)
    w -= buffer


def _fused(ahead: int, w: np.ndarray, g: np.ndarray, buffer: np.ndarray) -> None:
    """One pass in C: `w[i] = w[i] - (lr * g[i])`, the product rounded first.

    Args:
        ahead: Prefetch distance in bytes, 0 for none.
        w: The weights, updated in place.
        g: The gradient.
        buffer: Not used.
    """
    native.loops().study_fused(w.ctypes.data, g.ctypes.data, mlp_train.LEARNING_RATE, w.size, ahead)


def _two_loops(ahead: int, w: np.ndarray, g: np.ndarray, buffer: np.ndarray) -> None:
    """NumPy's multiply into a new temporary, then the study's subtract loop.

    Args:
        ahead: Prefetch distance in bytes, 0 for none.
        w: The weights, updated in place.
        g: The gradient.
        buffer: Not used.
    """
    product = mlp_train.LEARNING_RATE * g
    native.loops().study_subtract(w.ctypes.data, product.ctypes.data, w.size, ahead)


WAYS = (
    Way("as written", _as_written),
    Way("product into a reused buffer", _reused),
    Way("fused, one pass", lambda w, g, b: _fused(0, w, g, b)),
    Way("NumPy multiply, subtract with prefetch", lambda w, g, b: _two_loops(PREFETCH_AHEAD, w, g, b)),
    Way("fused, one pass, with prefetch", lambda w, g, b: _fused(PREFETCH_AHEAD, w, g, b)),
    Way("as written, hotkernels installed", _as_written, installed=True),
)


@dataclass(frozen=True)
class Split:
    """The statement as written, in four timed parts, medians in nanoseconds."""

    # Taking the temporary from the allocator: np.empty_like(g).
    allocate: float
    # The multiply into it, its first touch.
    multiply: float
    # The subtraction from w.
    subtract: float
    # Giving the temporary back.
    free: float
    # Page faults per step in each of the four parts.
    faults: tuple[float, float, float, float]


@dataclass(frozen=True)
class WayResult:
    """What `in_loop` measured for one way of writing the update."""

    # The way's name.
    name: str
    # Median time of the update of MEASURED, per step, in nanoseconds.
    step: float
    # Page faults per update, on average.
    faults: float
    # Fastest and median time of the whole training, in nanoseconds.
    best_run: float
    median_run: float
    # Whether the trained parameters are bit for bit those of "as written".
    identical: bool
    # Whether hotkernels was installed.
    installed: bool


def _initial_parameters(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Return mlp_train's initial parameters, drawn as mlp_train.run draws them."""
    return {
        "w1": rng.standard_normal((INPUTS, HIDDEN)) * np.sqrt(2.0 / INPUTS),
        "b1": np.zeros(HIDDEN),
        "w2": rng.standard_normal((HIDDEN, mlp_train.CLASSES)) * np.sqrt(2.0 / HIDDEN),
        "b2": np.zeros(mlp_train.CLASSES),
    }


def _train(update: Callable[[np.ndarray, np.ndarray, np.ndarray], None] | None,
           data: tuple) -> tuple[dict, list[int], list[int], list[tuple], int]:
    """Run mlp_train's training loop with the update of MEASURED replaced.

    The loop is mlp_train.run's, statement for statement, calling the
    application's own `loss_and_gradients`; only the update of MEASURED is
    the way under test, and it alone is timed. The other parameters are
    updated as the application writes it.

    Args:
        update: The way to update MEASURED, or None to time the statement as
            written in its four parts (see Split).
        data: mlp_train.make_data()'s result at STUDIED_SIZE.

    Returns:
        (the trained parameters, the time of each update in nanoseconds, the
        page faults of each, the four parts of each when `update` is None,
        the time of the whole training in nanoseconds).
    """
    # The held-out data and the hidden units, which INPUTS and HIDDEN already give, are not used.
    x, y, *_ = data
    rng = np.random.default_rng(mlp_train.SEED + 1)
    params = _initial_parameters(rng)
    buffer = np.empty_like(params[MEASURED])
    times, faults, parts = [], [], []
    start = time.perf_counter_ns()
    for _ in range(mlp_train.EPOCHS):
        order = rng.permutation(x.shape[0])
        for first in range(0, x.shape[0], mlp_train.BATCH):
            batch = order[first:first + mlp_train.BATCH]
            _, gradients = mlp_train.loss_and_gradients(params, x[batch], y[batch])
            for name in params:
                if name != MEASURED:
                    params[name] -= mlp_train.LEARNING_RATE * gradients[name]
                elif update is None:
                    parts.append(_split_step(params[name], gradients[name]))
                else:
                    before = _faults()
                    times.append(_ns(lambda: update(params[name], gradients[name], buffer)))
                    faults.append(_faults() - before)
    return params, times, faults, parts, time.perf_counter_ns() - start


def _split_step(w: np.ndarray, g: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Run `w -= LEARNING_RATE * g` as its four parts, timing each.

    Args:
        w: The weights, updated in place.
        g: The gradient.

    Returns:
        (nanoseconds of each part, page faults of each part).
    """
    marks = [(time.perf_counter_ns(), _faults())]
    product = np.empty_like(g)
    marks.append((time.perf_counter_ns(), _faults()))
    np.multiply(mlp_train.LEARNING_RATE, g, out=product)
    marks.append((time.perf_counter_ns(), _faults()))
    w -= product
    marks.append((time.perf_counter_ns(), _faults()))
    del product
    marks.append((time.perf_counter_ns(), _faults()))
    pairs = list(zip(marks, marks[1:]))
    return tuple(b[0] - a[0] for a, b in pairs), tuple(b[1] - a[1] for a, b in pairs)


def in_loop(rounds: int = ROUNDS) -> tuple[list[WayResult], Split]:
    """Measure every way of writing the update inside mlp_train's training.

    Args:
        rounds: Trainings per way, alternating between the ways.

    Returns:
        (one result per way, in WAYS order; the statement as written in parts).
    """
    data = mlp_train.make_data(STUDIED_SIZE["mlp_train"])
    native.loops()
    runs: dict[str, list] = {way.name: [] for way in WAYS}
    split_parts = []
    for _ in range(rounds):
        for way in WAYS:
            if way.installed:
                hotkernels.install()
            try:
                runs[way.name].append(_train(way.update, data))
            finally:
                if way.installed:
                    hotkernels.uninstall()
        split_parts.extend(_train(None, data)[3])
    reference = runs[WAYS[0].name][0][0]
    results = []
    for way in WAYS:
        trained = runs[way.name]
        totals = [run[4] for run in trained]
        results.append(WayResult(
            name=way.name,
            step=float(np.median([t for run in trained for t in run[1]])),
            faults=float(np.mean([f for run in trained for f in run[2]])),
            best_run=float(min(totals)),
            median_run=float(np.median(totals)),
            identical=all(np.array_equal(run[0][k], reference[k]) for run in trained for k in reference),
            installed=way.installed,
        ))
    times = np.array([p[0] for p in split_parts], dtype=float)
    faults = np.array([p[1] for p in split_parts], dtype=float)
    medians = np.median(times, axis=0)
    split = Split(*(float(m) for m in medians), faults=tuple(float(f) for f in faults.mean(axis=0)))
    return results, split


# ------------------------------------------------------------- after_reader


@dataclass(frozen=True)
class ReaderResult:
    """`w -= b` timed after one thing was done to `w`, three ways."""

    # Shape of w.
    shape: tuple[int, int]
    # What was done to w before each timed call.
    reader: str
    # Median nanoseconds: NumPy's loop, the study's loop without a prefetch,
    # and with a prefetch for a store PREFETCH_AHEAD bytes ahead.
    numpy: float
    plain: float
    prefetch: float


def _readers(w: np.ndarray, x: np.ndarray, v: np.ndarray, copy: np.ndarray) -> list[tuple[str, Callable]]:
    """Return the things `after_reader` does to `w` before timing `w -= b`.

    Args:
        w: The matrix.
        x: A batch of rows to multiply by w.
        v: A vector to multiply w by.
        copy: An array of w's shape and values, to write into w.

    Returns:
        (description, call) pairs.
    """
    return [
        ("nothing", lambda: None),
        ("x @ w, a matrix product", lambda: x @ w),
        ("w @ v, a matrix-vector product", lambda: w @ v),
        ("x @ w, then w.sum()", lambda: (x @ w, w.sum())),
        ("x @ w, then np.copyto(w, copy)", lambda: (x @ w, np.copyto(w, copy))),
    ]


def after_reader(repeats: int = REPEATS) -> list[ReaderResult]:
    """Time `w -= b` after each reader, on each of READER_SHAPES.

    Args:
        repeats: Timed calls per point, of which the median is kept.

    Returns:
        One result per shape and reader.
    """
    lib = native.loops()
    rng = np.random.default_rng(SEED)
    results = []
    for shape in READER_SHAPES:
        w = rng.standard_normal(shape)
        b = rng.standard_normal(shape) * SMALL
        x = rng.standard_normal((READER_BATCH, shape[0]))
        v = rng.standard_normal(shape[1])
        copy = w.copy()
        loops = {
            "numpy": lambda: w.__isub__(b),
            "plain": lambda: lib.study_subtract(w.ctypes.data, b.ctypes.data, w.size, 0),
            "prefetch": lambda: lib.study_subtract(w.ctypes.data, b.ctypes.data, w.size, PREFETCH_AHEAD),
        }
        for reader, before in _readers(w, x, v, copy):
            times: dict[str, list[int]] = {name: [] for name in loops}
            for _ in range(repeats):
                for name, loop in loops.items():
                    before()
                    times[name].append(_ns(loop))
            results.append(ReaderResult(shape, reader, *(float(np.median(times[name])) for name in loops)))
    return results


# ---------------------------------------------------------------- allocator


@dataclass(frozen=True)
class AllocatorResult:
    """One application on NumPy's allocator and on the study's caching one."""

    # Application name.
    name: str
    # Fastest run, in nanoseconds, on each allocator.
    plain: float
    cached: float
    # Page faults per run, the median, on each.
    plain_faults: float
    cached_faults: float
    # Whether the results were identical.
    identical: bool


def _timed_run(app, data: tuple) -> tuple[int, int, tuple]:
    """Run an application once.

    Args:
        app: The application's module.
        data: Its make_data() result.

    Returns:
        (nanoseconds, page faults, its results).
    """
    before = _faults()
    start = time.perf_counter_ns()
    out = app.run(*data)
    elapsed = time.perf_counter_ns() - start
    return elapsed, _faults() - before, out


def _same(a: tuple, b: tuple) -> bool:
    """Say whether two applications' results are equal, element for element and bit for bit."""
    return all(np.array_equal(np.asarray(p), np.asarray(q), equal_nan=True) for p, q in zip(a, b, strict=True))


def allocator(rounds: int = ROUNDS) -> tuple[list[AllocatorResult], tuple[int, int, int]]:
    """Run mlp_train and min_char_rnn on NumPy's allocator and on the caching one.

    Args:
        rounds: Runs per application and allocator, alternating.

    Returns:
        (one result per application; the cache's hits, misses and bytes held
        at the end).
    """
    cache = native.allocator()
    results = []
    for app in (mlp_train, min_char_rnn):
        data = app.make_data(STUDIED_SIZE[app.__name__.rsplit(".", 1)[-1]])
        plain, cached = [], []
        for _ in range(rounds):
            plain.append(_timed_run(app, data))
            cache.install()
            try:
                cached.append(_timed_run(app, data))
            finally:
                cache.uninstall()
        results.append(AllocatorResult(
            name=app.__name__.rsplit(".", 1)[-1],
            plain=float(min(r[0] for r in plain)),
            cached=float(min(r[0] for r in cached)),
            plain_faults=float(np.median([r[1] for r in plain])),
            cached_faults=float(np.median([r[1] for r in cached])),
            identical=all(_same(plain[0][2], r[2]) for r in cached),
        ))
    return results, cache.counts()


# ------------------------------------------------------------ accumulations


@dataclass(frozen=True)
class AccumulationResult:
    """One accumulation of min_char_rnn, timed alone at the program's sizes."""

    # The statement.
    statement: str
    # Shape of the array accumulated into.
    shape: tuple[int, ...]
    # How often one run of the program executes it.
    per_run: int
    # Median nanoseconds of the statement as written.
    written: float
    # Median nanoseconds of what a fused version would still do: the product
    # alone for the matrix product, one pass in C for the square.
    fused: float


def _program_sizes() -> tuple[int, int, int]:
    """Return min_char_rnn's hidden size, steps per iteration and iterations per run."""
    source = min_char_rnn.PROGRAM.read_text()
    hidden = int(_assignment(source, "hidden_size"))
    steps = int(_assignment(source, "seq_length"))
    return hidden, steps, min_char_rnn.ITERATIONS


def _assignment(source: str, name: str) -> str:
    """Return the value assigned to `name` at the top of a program's source.

    Args:
        source: The program.
        name: A variable assigned once as `name = value # comment`.

    Returns:
        The value's text.

    Raises:
        ValueError: The program does not assign it.
    """
    for line in source.splitlines():
        if line.startswith(f"{name} ="):
            return line.split("=", 1)[1].split("#", 1)[0].strip()
    raise ValueError(f"{name} is not assigned in the program")


def _median_ns(call: Callable[[], object], repeats: int) -> float:
    """Return the median time of `repeats` calls, in nanoseconds."""
    return float(np.median([_ns(call) for _ in range(repeats)]))


def accumulations(repeats: int = ACCUMULATION_REPEATS) -> list[AccumulationResult]:
    """Time min_char_rnn's two accumulations alone, as written and as a fused version would leave them.

    `dWhh += np.dot(dhraw, hs[t-1].T)` is an outer product of two columns of
    `hidden` added into a square matrix: fusing it would mean the product
    writing into dWhh, and what is left is the product alone. `mWhh += dWhh *
    dWhh`, Adagrad's memory, is a multiply and an add: fused, one pass.

    Args:
        repeats: Timed calls per statement.

    Returns:
        One result per statement.
    """
    hidden, steps, iterations = _program_sizes()
    rng = np.random.default_rng(SEED)
    column, row = rng.standard_normal((hidden, 1)), rng.standard_normal((hidden, 1))
    accumulated = np.zeros((hidden, hidden))
    gradient = rng.standard_normal((hidden, hidden)) * SMALL
    memory = np.zeros((hidden, hidden))
    lib = native.loops()
    outer = AccumulationResult(
        statement="dWhh += np.dot(dhraw, hs[t-1].T)",
        shape=(hidden, hidden),
        # Three such statements per step, for dWxh, dWhh and dWhy.
        per_run=3 * steps * iterations,
        written=_median_ns(lambda: accumulated.__iadd__(np.dot(column, row.T)), repeats),
        fused=_median_ns(lambda: np.dot(column, row.T), repeats),
    )
    square = AccumulationResult(
        statement="mWhh += dWhh * dWhh",
        shape=(hidden, hidden),
        # One per parameter per iteration; Wxh and Why are hidden x vocabulary.
        per_run=iterations,
        written=_median_ns(lambda: memory.__iadd__(gradient * gradient), repeats),
        fused=_median_ns(lambda: lib.study_fused_square_add(memory.ctypes.data, gradient.ctypes.data,
                                                            memory.size), repeats),
    )
    return [outer, square]


def threads_setting() -> str:
    """Return the value of THREADS_VARIABLE, or "not set"."""
    return os.environ.get(THREADS_VARIABLE, "not set")
