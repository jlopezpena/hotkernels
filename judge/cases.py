"""Generates the inputs a kernel is checked against.

Every case is fully determined by its description, so any failure can be rerun.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .memory import ARRANGEMENTS
from .spec import Kernel, np_dtype

# Value classes per kind of data. "Positional" classes place one interesting
# element at a chosen index.
_PLAIN = {
    "f": ["random", "equal", "asc", "desc", "ties", "zeros", "denormal", "inf_mix",
          "nan_all", "nan_some", "extremes"],
    "b": ["random", "all_false", "all_true", "sparse", "dense", "alternating"],
    "i": ["random", "small", "equal", "asc", "desc", "extremes"],
}
_POSITIONAL = {
    "f": ["nan_at", "max_at", "min_at", "posinf_at", "neginf_at", "two_max_at"],
    "b": ["one_true_at", "one_false_at"],
    "i": ["max_at", "min_at"],
}


def _kind(dtype: np.dtype) -> str:
    return "i" if dtype.kind in "iu" else dtype.kind


def classes(dtype_name: str) -> tuple[list[str], list[str]]:
    kind = _kind(np_dtype(dtype_name))
    return _PLAIN[kind], _POSITIONAL[kind]


def _crc(text: str) -> int:
    return zlib.crc32(text.encode())


def base(dtype_name: str, n: int, seed: int, arg: str) -> np.ndarray:
    """The default random array for an argument."""
    dt = np_dtype(dtype_name)
    rng = np.random.default_rng([seed, n, _crc(arg)])
    kind = _kind(dt)
    if kind == "f":
        return rng.standard_normal(n).astype(dt)
    if kind == "b":
        return rng.random(n) < 0.5
    info = np.iinfo(dt)
    return rng.integers(info.min, info.max, n, dtype=dt, endpoint=True)


def make(dtype_name: str, cls: str, p: int, n: int, seed: int, arg: str, base_arr: np.ndarray) -> np.ndarray:
    dt = np_dtype(dtype_name)
    kind = _kind(dt)
    rng = np.random.default_rng([seed, n, _crc(arg), _crc(cls)])
    if cls == "random":
        return base_arr

    if kind == "f":
        info = np.finfo(dt)
        if cls == "equal":
            return np.full(n, 1.5, dt)
        if cls == "asc":
            return np.sort(base_arr)
        if cls == "desc":
            return np.sort(base_arr)[::-1].copy()
        if cls == "ties":
            return rng.integers(0, 4, n).astype(dt)
        if cls == "zeros":
            return np.where(rng.random(n) < 0.5, 0.0, -0.0).astype(dt)
        if cls == "denormal":
            return (rng.integers(-50, 50, n) * info.smallest_subnormal).astype(dt)
        if cls == "nan_all":
            return np.full(n, np.nan, dt)
        if cls == "extremes":
            return rng.choice(np.array([info.max, -info.max, info.tiny, -info.tiny, 0.0], dt), n)
        a = base_arr.copy()
        if cls == "inf_mix":
            m = rng.random(n)
            a[m < 0.1] = np.inf
            a[m > 0.9] = -np.inf
            return a
        if cls == "nan_some":
            a[rng.random(n) < 0.2] = np.nan
            return a
        if cls == "nan_at":
            a[p] = np.nan
            return a
        if cls == "posinf_at":
            a[p] = np.inf
            return a
        if cls == "neginf_at":
            a[p] = -np.inf
            return a
        np.clip(a, -1, 1, out=a)
        if cls == "max_at":
            a[p] = 10
        elif cls == "min_at":
            a[p] = -10
        elif cls == "two_max_at":
            a[p] = 10
            a[n - 1 - p] = 10
        else:
            raise ValueError(cls)
        return a

    if kind == "b":
        if cls == "all_false":
            return np.zeros(n, bool)
        if cls == "all_true":
            return np.ones(n, bool)
        if cls == "sparse":
            return rng.random(n) < 0.05
        if cls == "dense":
            return rng.random(n) < 0.95
        if cls == "alternating":
            return (np.arange(n) % 2).astype(bool)
        if cls == "one_true_at":
            a = np.zeros(n, bool)
            a[p] = True
            return a
        if cls == "one_false_at":
            a = np.ones(n, bool)
            a[p] = False
            return a
        raise ValueError(cls)

    info = np.iinfo(dt)
    if cls == "small":
        return rng.integers(0, 4, n).astype(dt)
    if cls == "equal":
        return np.full(n, 7, dt)
    if cls == "asc":
        return np.sort(base_arr)
    if cls == "desc":
        return np.sort(base_arr)[::-1].copy()
    if cls == "extremes":
        return rng.choice(np.array([info.min, info.max, 0], dt), n)
    a = (base_arr // 2).astype(dt)
    if cls == "max_at":
        a[p] = info.max
    elif cls == "min_at":
        a[p] = info.min
    else:
        raise ValueError(cls)
    return a


@dataclass(frozen=True)
class Case:
    n: int
    arr: str  # buffer arrangement, see memory.ARRANGEMENTS
    seed: int
    picks: tuple[tuple[str, str, int], ...]  # (argument, class, index)

    def __str__(self) -> str:
        parts = [f"n={self.n}", f"arr={self.arr}", f"seed={self.seed}"]
        for arg, cls, p in self.picks:
            parts.append(f"{arg}={cls}@{p}" if cls.endswith("_at") else f"{arg}={cls}")
        return " ".join(parts)

    @staticmethod
    def parse(text: str) -> "Case":
        fields = dict(part.split("=", 1) for part in text.split())
        n, arr, seed = int(fields.pop("n")), fields.pop("arr"), int(fields.pop("seed"))
        picks = []
        for arg, value in fields.items():
            cls, _, p = value.partition("@")
            picks.append((arg, cls, int(p or 0)))
        return Case(n, arr, seed, tuple(picks))


def build_inputs(k: Kernel, case: Case, cache: dict | None = None) -> dict[str, np.ndarray]:
    picks = {arg: (cls, p) for arg, cls, p in case.picks}
    inputs = {}
    for a in k.ins:
        key = (a.name, case.n, case.seed)
        if cache is not None and key in cache:
            base_arr = cache[key]
        else:
            base_arr = base(a.dtype, case.n, case.seed, a.name)
            if cache is not None:
                cache[key] = base_arr
        cls, p = picks.get(a.name, ("random", 0))
        inputs[a.name] = np.ascontiguousarray(
            make(a.dtype, cls, p, case.n, case.seed, a.name, base_arr), np_dtype(a.dtype)
        )
    return inputs


def sizes(min_n: int, max_pow: int = 22) -> list[int]:
    """Every size up to 65, then each power of two with its neighbours."""
    out = set(range(min_n, 66))
    for e in range(7, max_pow + 1):
        out |= {2**e - 1, 2**e, 2**e + 1}
    return sorted(s for s in out if s >= min_n)


def fixed_cases(k: Kernel, seed: int = 1) -> Iterator[Case]:
    ins = k.ins

    def case(n, arr, chosen):
        return Case(n, arr, seed, tuple((a.name, *chosen.get(a.name, ("random", 0))) for a in ins))

    for n in sizes(k.min_n):
        small, big = n <= 65, n > 65536
        for a in ins:
            plain, positional = classes(a.dtype)
            if big:  # large arrays are slow to generate; keep one class of each sort
                plain, positional = plain[:1], positional[:1]
            for cls in plain:
                for arr in ARRANGEMENTS if small else ARRANGEMENTS[:2]:
                    yield case(n, arr, {a.name: (cls, 0)})
            if n == 0:
                continue
            if small:
                indices = range(n)
            elif big:
                indices = [0, n - 1]
            else:
                indices = [0, n // 2, n - 1]
            for cls in positional:
                for i, p in enumerate(indices):
                    arr = ARRANGEMENTS[(p + n) % 5] if small else ARRANGEMENTS[i % 2]
                    yield case(n, arr, {a.name: (cls, p)})
        if len(ins) == 2 and n <= 4096:
            first, second = ins
            for c1 in classes(first.dtype)[0]:
                for c2 in classes(second.dtype)[0]:
                    yield case(n, ARRANGEMENTS[n % 2], {first.name: (c1, 0), second.name: (c2, 0)})


def fresh_cases(k: Kernel, seed: int, count: int = 400) -> Iterator[Case]:
    """Random cases from a seed that changes on every run."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        top = 5000 if rng.random() < 0.1 else 300
        n = int(rng.integers(k.min_n, top))
        picks = []
        for a in k.ins:
            plain, positional = classes(a.dtype)
            pool = plain + (positional if n > 0 else [])
            cls = str(rng.choice(pool))
            p = int(rng.integers(0, n)) if cls.endswith("_at") else 0
            picks.append((a.name, cls, p))
        yield Case(n, str(rng.choice(ARRANGEMENTS)), seed, tuple(picks))
