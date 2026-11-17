"""The kernel spec format.

Each kernel directory has a `spec.py` that defines `KERNEL = Kernel(...)`. The
spec tells the judge how to call the kernel and what the right answer is.

The format is provisional: it is the smallest thing that covers the kernels we
have. Extend it when a kernel needs something it cannot express.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

# Element types a buffer can have: spec name -> (NumPy dtype, C type).
DTYPES: dict[str, tuple[type, str]] = {
    "f32": (np.float32, "float"),
    "f64": (np.float64, "double"),
    "i8": (np.int8, "int8_t"),
    "i16": (np.int16, "int16_t"),
    "i32": (np.int32, "int32_t"),
    "i64": (np.int64, "int64_t"),
    "u8": (np.uint8, "uint8_t"),
    "u16": (np.uint16, "uint16_t"),
    "u32": (np.uint32, "uint32_t"),
    "u64": (np.uint64, "uint64_t"),
    "bool": (np.bool_, "bool"),
}

# Types a kernel can return: spec name -> C type. Every element type, plus
# `usize` for counts and `void` for kernels that only write to a buffer.
RETURNS: dict[str, str] = {name: c for name, (_, c) in DTYPES.items()} | {
    "usize": "size_t",
    "void": "void",
}

EXACT = "exact"
# Keys of a tolerance mode: the allowed error in units in the last place, and
# an optional function giving the magnitude that unit is taken at.
TOLERANCE_ULP = "tolerance_ulp"
TOLERANCE_SCALE = "scale"
_MODE_KEYS = {TOLERANCE_ULP, TOLERANCE_SCALE}
USED_RETURN = "return"
USED_CAPACITY = "capacity"


def np_dtype(name: str) -> np.dtype:
    """Return the NumPy dtype for a spec type name such as `f32`."""
    return np.dtype(DTYPES[name][0])


# Scalar argument types: spec name -> C type. Integers travel in the integer
# argument registers and floats in the floating-point ones.
SCALARS: dict[str, str] = {"usize": "size_t", "i64": "int64_t", "f32": "float", "f64": "double"}
FLOAT_SCALARS = ("f32", "f64")

# The value of `len` and `capacity` that means "as many elements as the Size argument says".
SIZE_N = "n"

# Name of the scalar set in which every Scalar has its own `value`. It always
# exists, is always checked and is always timed.
DEFAULT_SCALARS = "default"
# Words a case description uses for its own fields (`n=4 arr=end seed=1
# scalars=wide x=random`), which an argument or a scalar set must not be named.
RESERVED_NAMES = ("n", "arr", "seed", "scalars")
# Start of the name of the value class an argument with `draws_from` gets:
# `from_a` for one drawn from the argument `a`.
DRAWN_PREFIX = "from_"


def drawn_class(arg) -> str | None:
    """Return the name of the value class an argument draws from another, if any.

    Args:
        arg: A buffer argument.

    Returns:
        DRAWN_PREFIX followed by the source's name for an In with
        `draws_from`, otherwise None.
    """
    source = getattr(arg, "draws_from", None)
    return f"{DRAWN_PREFIX}{source}" if source else None


@dataclass(frozen=True)
class In:
    """A read-only input buffer."""

    # Argument name, as used in the C prototype and passed to the oracle.
    name: str
    # Element type: a key of DTYPES.
    dtype: str
    # Element count: SIZE_N for the kernel's Size argument, a fixed number for
    # a buffer whose length does not depend on it (a set of test values, for
    # example), or the name of an integer Scalar argument whose value is the
    # length (the needles of a search, whose number has nothing to do with the
    # length of the array searched). A length given by a scalar is checked for
    # the values the kernel's scalar sets give it, and for no others.
    len: str | int = SIZE_N
    # Applied to every generated array before the call, for kernels that are
    # only defined on some inputs: `np.sort` for one that needs sorted data,
    # for example. Takes the array and returns one of the same length. Timing
    # inputs pass through it too, unless a distribution supplies them.
    prepare: Callable | None = None
    # Value classes of this argument alone, tried next to the ones every
    # argument of its type gets: name -> function. For inputs a caller sends
    # that the general classes do not produce, such as a set of test values
    # padded with repeats. Each function takes the argument's default random
    # array and returns an array of the same length and type, which it must
    # not build by modifying the one it was given. A name appears in case
    # descriptions (`test=padded_3`), so it must not be the name of a
    # general value class.
    classes: dict[str, Callable] | None = None
    # Name of another input argument whose values this one may be drawn from:
    # the needles of a search, drawn from the array searched, so that some of
    # them equal its elements. It gives this argument one more value class,
    # DRAWN_PREFIX + the source's name (`v=from_a`), whose elements are
    # elements of the source as the kernel receives it. The source has the
    # same element type and does not itself draw from another argument.
    draws_from: str | None = None


@dataclass(frozen=True)
class Out:
    """An output buffer the kernel writes to."""

    # Argument name, as used in the C prototype.
    name: str
    # Element type: a key of DTYPES.
    dtype: str
    # How many elements the buffer can hold: SIZE_N for the kernel's Size
    # argument, a fixed number (a table of counts, for example), or the name
    # of an integer Scalar argument whose value is the capacity (a histogram's
    # bin count, for example).
    capacity: str | int = SIZE_N
    # How much of the buffer the kernel fills: USED_CAPACITY means all of it,
    # USED_RETURN means the kernel returns the number of elements it wrote.
    used: str = USED_CAPACITY
    # True lets the kernel write anywhere inside the capacity, not only the
    # elements it reports. Only the reported elements are compared with the
    # oracle; the rest of the buffer may hold anything afterwards. Writing past
    # the capacity is still a failure. Needs used=USED_RETURN. A binding must
    # never show a caller the unreported part of such a buffer.
    scratch: bool = False


@dataclass(frozen=True)
class InOut:
    """A buffer the kernel reads and modifies in place.

    It starts as an input array and is compared, in full, with the oracle's
    result afterwards. The oracle receives the array like any other input and
    must return the modified copy without changing its argument.
    """

    # Argument name, as used in the C prototype and passed to the oracle.
    name: str
    # Element type: a key of DTYPES.
    dtype: str
    # An in-place buffer always has as many elements as the Size argument says,
    # is filled completely, and is never scratch. These mirror the fields of In
    # and Out so the judge can treat it as either.
    len = SIZE_N
    capacity = SIZE_N
    used = USED_CAPACITY
    scratch = False
    prepare = None
    classes = None
    draws_from = None


@dataclass(frozen=True)
class Size:
    """The element count, passed to the kernel as a `size_t`."""

    # Argument name, as used in the C prototype.
    name: str


@dataclass(frozen=True)
class Scalar:
    """A scalar argument: a bound, a bin count, a fill value.

    A scalar has no value classes. It takes its `value` in the default scalar
    set, and whatever the kernel's `scalar_sets` give it in the others. The
    kernel is tested for those combinations and no others.
    """

    # Argument name, as used in the C prototype and passed to the oracle.
    name: str
    # Scalar type: a key of SCALARS.
    dtype: str
    # The value in the default scalar set.
    value: int | float


@dataclass(frozen=True)
class ScalarSet:
    """One more combination of values for a kernel's Scalar arguments.

    Scalars are given together because they are rarely independent: a lower
    bound must not exceed the upper one, and a bin count belongs to a range.
    """

    # Name, as it appears in a case description (`scalars=<name>`) and in the
    # heading of a timing table.
    name: str
    # Scalar argument name -> value. A Scalar left out keeps its default value.
    values: dict[str, int | float]
    # True to time the kernel with these values as well as check it. Each
    # timed set adds a full timing sweep, so keep them few.
    timed: bool = False


def element_count(arg: In | Out | InOut, n: int, capacity: bool = False,
                  scalars: dict[str, int | float] | None = None) -> int:
    """Return how many elements a buffer argument has.

    Args:
        arg: A buffer argument.
        n: The value of the kernel's Size argument in this call.
        capacity: Read the buffer's `capacity` instead of its `len`. An In has
            only a length and an Out only a capacity; an InOut has both.
        scalars: The scalar values of this call, by argument name. Needed only
            for a buffer whose capacity is given by a Scalar.

    Returns:
        `n` when the spec says SIZE_N, the scalar's value when it names a
        Scalar, otherwise the fixed element count it gives.

    Raises:
        ValueError: The size is a scalar and `scalars` was not given.
    """
    declared = arg.capacity if capacity or isinstance(arg, Out) else arg.len
    if declared == SIZE_N:
        return n
    if isinstance(declared, str):
        if scalars is None:
            raise ValueError(f"{arg.name}: its size is the scalar {declared!r}, "
                             "so the scalar values of the call are needed")
        return int(scalars[declared])
    return declared


@dataclass
class Kernel:
    """Everything the judge needs to know about one kernel."""

    # Kernel name. Also the C function name and the directory name.
    name: str
    # Return type: a key of RETURNS.
    returns: str
    # Arguments in C order: a mix of In, Out, InOut, Size and Scalar.
    args: list
    # Python function that returns the correct answer. It receives one NumPy
    # array per In or InOut, in argument order, followed by one value per
    # Scalar, in argument order: `oracle(x, lo, hi)` for a kernel declared as
    # `f(x, out, n, lo, hi)`.
    oracle: Callable
    # How output is compared with the oracle: EXACT for byte equality, or
    # {"tolerance_ulp": k} to allow k units in the last place on floats. The
    # unit is taken at the size of the value itself, unless the dict also has
    # "scale": a function over the same arguments as the oracle that returns, per
    # output element, the magnitude the unit is taken at. A running sum needs
    # that: its rounding error is bounded by the sizes of the terms, not by
    # the size of a result that may have cancelled to almost nothing.
    mode: str | dict = EXACT
    # Smallest element count the kernel is ever called with.
    min_n: int = 0
    # Largest element count used for correctness, or None for the judge's full
    # range. For kernels whose NumPy oracle is too slow to run on millions of
    # elements once per case. Timing always uses the full sweep.
    check_max_n: int | None = None
    # Named input distributions used for timing: name -> fn(rng, n) -> {arg: array}.
    # When absent, timing uses uniformly random inputs.
    distributions: dict[str, Callable] | None = None
    # What is timed as NumPy, when that is not the oracle. Takes the same
    # arguments. Needed for in-place operations, where the oracle has to copy
    # its input and the statement a user would write does not.
    numpy: Callable | None = None
    # Further combinations of scalar values to check the kernel with, beyond
    # the default one made of every Scalar's own `value`. Without them a kernel
    # with scalars is verified for a single combination only.
    scalar_sets: list[ScalarSet] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Reject specs the judge cannot handle, with a message naming the kernel."""
        if self.returns not in RETURNS:
            raise ValueError(f"{self.name}: unknown return type {self.returns!r}")
        for a in self.args:
            if isinstance(a, (In, Out, InOut)) and a.dtype not in DTYPES:
                raise ValueError(f"{self.name}: unknown dtype {a.dtype!r} for {a.name}")
            if isinstance(a, Scalar) and a.dtype not in SCALARS:
                raise ValueError(f"{self.name}: unknown scalar type {a.dtype!r} for {a.name}")
            if not isinstance(a, Size) and a.name in RESERVED_NAMES:
                raise ValueError(f"{self.name}: an argument cannot be named {a.name!r}; "
                                 "case descriptions use that word")
        self._check_scalar_sets()
        self._check_scalar_sizes()
        self._check_draws()
        if len(self.outs) > 1:
            raise ValueError(f"{self.name}: at most one Out or InOut argument is supported")
        if self.out and self.out.used == USED_RETURN and self.returns != "usize":
            raise ValueError(f"{self.name}: used='return' needs returns='usize'")
        if self.out and self.out.scratch and self.out.used != USED_RETURN:
            raise ValueError(f"{self.name}: scratch=True needs used='return'; a buffer "
                             "that is filled completely has no unreported part")
        if self.mode != EXACT and self.tolerance_ulp is None:
            raise ValueError(f"{self.name}: mode must be 'exact' or {{'tolerance_ulp': k}}, "
                             "optionally with 'scale': fn")
        if isinstance(self.mode, dict) and set(self.mode) - _MODE_KEYS:
            raise ValueError(f"{self.name}: unknown keys in mode: {sorted(set(self.mode) - _MODE_KEYS)}")

    def _check_scalar_sizes(self) -> None:
        """Reject a buffer whose size names something other than an integer Scalar.

        An Out's capacity and an In's length may be given by a scalar. The
        scalar is then an element count, so no scalar set may make it negative.
        """
        integer_scalars = {a.name for a in self.scalars if a.dtype not in FLOAT_SCALARS}
        for a in self.buffers:
            declared = a.capacity if isinstance(a, Out) else a.len
            if not isinstance(declared, str) or declared == SIZE_N:
                continue
            what = "capacity" if isinstance(a, Out) else "length"
            if declared not in integer_scalars:
                raise ValueError(f"{self.name}: {what} of {a.name} is {declared!r}, "
                                 "which is not an integer Scalar argument")
            for set_name in self.scalar_set_names:
                if self.scalar_values(set_name)[declared] < 0:
                    raise ValueError(f"{self.name}: scalar set {set_name!r} makes {declared} negative, "
                                     f"and it is the {what} of {a.name}")

    def _check_draws(self) -> None:
        """Reject an argument that draws from something it cannot draw from.

        The source must be another input of the same element type that does
        not itself draw from an argument: the judge builds every source before
        the arguments drawn from it, in one pass.
        """
        ins = {a.name: a for a in self.ins}
        for a in self.ins:
            if not a.draws_from:
                continue
            source = ins.get(a.draws_from)
            if source is None or source is a:
                raise ValueError(f"{self.name}: {a.name} draws from {a.draws_from!r}, "
                                 "which is not another input argument")
            if source.dtype != a.dtype:
                raise ValueError(f"{self.name}: {a.name} is {a.dtype} and draws from "
                                 f"{source.name}, which is {source.dtype}")
            if source.draws_from:
                raise ValueError(f"{self.name}: {a.name} draws from {source.name}, "
                                 "which itself draws from another argument")

    def _check_scalar_sets(self) -> None:
        """Reject scalar sets with a bad name or a value for an unknown scalar."""
        known = {a.name for a in self.scalars}
        seen = {DEFAULT_SCALARS}
        for scalar_set in self.scalar_sets:
            name = scalar_set.name
            if name in seen or name in RESERVED_NAMES or not name.isidentifier():
                raise ValueError(f"{self.name}: scalar set name {name!r} is a duplicate, "
                                 "reserved, or not a plain identifier")
            seen.add(name)
            unknown = sorted(set(scalar_set.values) - known)
            if unknown:
                raise ValueError(f"{self.name}: scalar set {name!r} gives values for "
                                 f"{unknown}, which are not Scalar arguments")

    @property
    def scalars(self) -> list[Scalar]:
        """The Scalar arguments, in argument order."""
        return [a for a in self.args if isinstance(a, Scalar)]

    @property
    def scalar_set_names(self) -> list[str]:
        """The names of every scalar set that is checked: the default one first."""
        return [DEFAULT_SCALARS] + [scalar_set.name for scalar_set in self.scalar_sets]

    @property
    def timed_scalar_sets(self) -> list[str]:
        """The names of the scalar sets that are timed: the default one first."""
        return [DEFAULT_SCALARS] + [s.name for s in self.scalar_sets if s.timed]

    def scalar_values(self, set_name: str = DEFAULT_SCALARS) -> dict[str, int | float]:
        """Return the value of every Scalar argument in one scalar set.

        Args:
            set_name: DEFAULT_SCALARS, or the name of one of `scalar_sets`.

        Returns:
            Scalar argument name -> value, in argument order. Empty for a
            kernel without scalars.
        """
        values = {a.name: a.value for a in self.scalars}
        if set_name == DEFAULT_SCALARS:
            return values
        for scalar_set in self.scalar_sets:
            if scalar_set.name == set_name:
                return values | scalar_set.values
        raise ValueError(f"{self.name}: no scalar set named {set_name!r}; "
                         f"it has {', '.join(self.scalar_set_names)}")

    def oracle_scalars(self, set_name: str = DEFAULT_SCALARS) -> list:
        """Return the scalar values of one set as the oracle receives them.

        Args:
            set_name: DEFAULT_SCALARS, or the name of one of `scalar_sets`.

        Returns:
            One value per Scalar, in argument order: a Python int for an
            integer, a Python float for an f64, and a `np.float32` for an f32,
            so that the oracle computes with the value the kernel is passed
            and not with a double that rounds differently.
        """
        values = self.scalar_values(set_name)
        converted = []
        for a in self.scalars:
            if a.dtype == "f32":
                converted.append(np.float32(values[a.name]))
            elif a.dtype == "f64":
                converted.append(float(values[a.name]))
            else:
                converted.append(int(values[a.name]))
        return converted

    @property
    def ins(self) -> list[In | InOut]:
        """The buffers the oracle receives, in argument order: inputs and in-place buffers."""
        return [a for a in self.args if isinstance(a, (In, InOut))]

    @property
    def outs(self) -> list[Out | InOut]:
        """The buffers the kernel writes, in argument order: outputs and in-place buffers."""
        return [a for a in self.args if isinstance(a, (Out, InOut))]

    @property
    def buffers(self) -> list[In | Out | InOut]:
        """Every pointer argument, in argument order."""
        return [a for a in self.args if isinstance(a, (In, Out, InOut))]

    @property
    def out(self) -> Out | InOut | None:
        """The single buffer the kernel writes, or None for kernels that only return a value."""
        outs = self.outs
        return outs[0] if outs else None

    def largest_buffer(self, n: int, set_name: str = DEFAULT_SCALARS) -> int:
        """Return the element count of the largest buffer in one call.

        Args:
            n: The value of the kernel's Size argument.
            set_name: The scalar set the call is made with.

        Returns:
            The largest element count among the kernel's buffers, or `n` for a
            kernel without buffers. For a kernel whose buffers all have `n`
            elements or fewer this is `n`; it is larger when a scalar gives a
            buffer more elements than that.
        """
        scalars = self.scalar_values(set_name)
        counts = [element_count(a, n, scalars=scalars) for a in self.buffers]
        return max([n, *counts])

    @property
    def tolerance_ulp(self) -> int | None:
        """Allowed error in units in the last place, or None in exact mode."""
        if isinstance(self.mode, dict):
            return self.mode.get(TOLERANCE_ULP)
        return None

    @property
    def tolerance_scale(self) -> Callable | None:
        """The function giving the magnitude each output element's error is measured at.

        Returns:
            The spec's scale function, which takes the oracle's arguments and
            returns one magnitude per output element, or None when the unit in
            the last place is taken at the size of the value itself.
        """
        if isinstance(self.mode, dict):
            return self.mode.get(TOLERANCE_SCALE)
        return None


def load_spec(kernel_dir: Path) -> Kernel:
    """Import `spec.py` from a kernel directory and return its `KERNEL`."""
    path = Path(kernel_dir) / "spec.py"
    if not path.exists():
        raise FileNotFoundError(f"no spec.py in {kernel_dir}")
    module_spec = importlib.util.spec_from_file_location(f"_hk_spec_{path.parent.name}", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    kernel = getattr(module, "KERNEL", None)
    if not isinstance(kernel, Kernel):
        raise ValueError(f"{path} must define KERNEL = Kernel(...)")
    return kernel
