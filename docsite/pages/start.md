There is very little to do. Install the package, call `install()` once, and NumPy uses the kernels for the rest of the process.

## Requirements

| | |
|---|---|
| Machine | A Mac with Apple Silicon (ARM64), or Linux on x86-64 with a processor of the x86-64-v3 level (AVX2, FMA, BMI2: Intel since 2013, AMD since 2015) |
| Python | 3.11 or later, the standard build with the interpreter lock |
| NumPy | 2.3 or a later NumPy 2 |

Wheels are published for CPython 3.11 to 3.14. With another Python the package is built from source when it is installed, which takes a few seconds and needs the Xcode command line tools (`xcode-select --install`) on a Mac, or a C compiler and GNU binutils on Linux.

The figures on this site were measured on an Apple M1 Pro. On x86-64 the same calls are served by a second set of kernels, written separately, and NumPy is faster to begin with for some of them, so the gains are not the same.

On any other machine the package still installs, with no compiler, and `install()` does nothing, so code that calls it stays portable.

## Install

```bash
pip install hotkernels
```

or `uv pip install hotkernels`.

To work on the kernels themselves, clone the repository and run `uv sync`; the judge and the other development tools come with it.

## Without changing your code

`python -m hotkernels` runs a script with hotkernels installed, as `python` would run it:

```bash
python -m hotkernels my_script.py --its --own --arguments
python -m hotkernels -m pytest            # a module, as python -m runs it
python -m hotkernels --report my_script.py  # print report() to standard error at the end
```

`--inexact` installs the kernels that are not bit-identical to NumPy as well.

## Turn it on

```python
import numpy as np
import hotkernels

hotkernels.install()            # NumPy now uses the kernels, in the whole process

x = np.random.standard_normal(1_000_000)
x[x > 0]                        # served by a kernel
np.clip(x, 0, 1)                # served by a kernel
x[::2].sum()                    # not something a kernel handles: NumPy does it, as before

hotkernels.uninstall()          # NumPy is exactly as it was
```

That is the whole interface:

| Call | What it does |
|---|---|
| `hotkernels.install()` | Points NumPy at every kernel that returns NumPy's result bit for bit |
| `hotkernels.install(inexact=True)` | Also installs the two that do not: `np.cumsum` on float32 and float64 |
| `hotkernels.uninstall()` | Puts every pointer back |
| `with hotkernels.enabled():` | Installs for the duration of a block |
| `hotkernels.report()` | What is installed, and how often each kernel ran |

Importing the package changes nothing; only `install()` does. Calling it twice is harmless, and so is `uninstall()` without `install()`.

::: note It covers the whole process
`install()` changes NumPy itself, so every thread and every library that calls NumPy gets the kernels, not only your own code. That is the point: pandas, scikit-learn or your own helpers speed up without knowing. It also means `with hotkernels.enabled():` is a way to limit hotkernels in time, not to one thread.
:::

## See what it did

`print(hotkernels.report())` after your code has run:

```text
hotkernels 0.2.0: installed, 26 of 28 kernels (NumPy 2.5.5)

NumPy call         Elements Kernel    Result            Kernel calls  Left to NumPy
np.argmax          float32  assembly  exact                        0              0
np.argmax          float64  assembly  exact                        1
np.clip            float64  assembly  exact                        1
x[mask]            float64  assembly  exact                        1
x[mask] = v        float64  assembly  exact                        1              0
np.cumsum          int64    assembly  exact                        1              0
np.cumsum          float64  assembly  within tolerance not installed: needs install(inexact=True)
x.astype(float32)  uint8    C         exact                        1              3
np.sum             uint8    assembly  exact                        1              0
...
```

**Left to NumPy** is the column to read when nothing got faster. It counts calls that reached a hook and did not meet a kernel's conditions: another element type, a strided array, an argument the kernel does not implement. A large number there tells you what your arrays look like.

## What gets faster

A kernel takes a call only when the array is a plain, contiguous `ndarray` of the kernel's element type and every other argument is one the kernel implements. Any other call goes to NumPy as it was written.

| You write | Element types | A kernel takes it when |
|---|---|---|
| `np.argmax(x)`, `np.argmin(x)`, `x.argmax()`, `x.argmin()` | float32, float64 | For these types, including strided input and `axis=` (one kernel call per row or column). A strided run or a row shorter than 128 elements stays with NumPy, which is as fast there |
| `np.clip(x, lo, hi)`, `x.clip(lo, hi)` | float32, float64 | Scalar bounds with `lo <= hi`; the output is not the input. Small arrays gain most with bounds that are Python numbers and no `out` |
| `x[mask]` | float32, float64, int64 | The mask has the shape of `x`, of any number of dimensions. A mask over the rows of a two-dimensional `x` stays with NumPy |
| `np.compress(mask, x)`, `np.extract(mask, x)` | float32, float64, int64 | `x` and `mask` the same size, `x` of any shape. For `np.compress` the mask is one-dimensional and there is no `axis` or `out` |
| `x[mask] = v` | float64 | As for `x[mask]`, and `v` is a Python float or int |
| `np.putmask(x, mask, v)`, `np.copyto(x, v, where=mask)` | float64 | `v` is a Python float or int, and `x` and `mask` are the same size (`np.putmask`) or the same shape (`np.copyto`, with no `casting`), of any shape |
| `np.nonzero(mask)`, `mask.nonzero()`, `np.where(mask)` | bool | One-dimensional; `np.where` with nothing but the mask |
| `np.where(mask, a, b)` | float64, int64 | `mask` is a boolean array and `a` and `b` are arrays of its exact shape and one element type, of any number of dimensions; or, for float64, one of `a` and `b` is such an array and the other a single number: a Python float, a Python int up to 2^53, or an `np.float64`. Shapes that would broadcast, and mixed element types, stay with NumPy |
| `np.cumsum(x)`, `x.cumsum()` | int64, bool; float32 and float64 with `inexact=True` | No `axis`, `dtype` or `out` |
| `x.astype(np.float32)` | uint8 | Other arguments at their defaults |
| `np.maximum.accumulate(x)` | int64 | `x` is one-dimensional, with at least one element; no `dtype` or `out`, and `axis` left out, 0 or -1 |
| `np.fmax.accumulate(x)` | int64, float64 | As for `np.maximum.accumulate`, and a float64 `x` holds no signalling NaN, no negative zero, and NaNs of one bit pattern only |
| `np.fmax.reduceat(x, starts)` | float64 | `x` is one-dimensional, with at least one element; `starts` is a one-dimensional int64 array, not a list, with at least one start and every start inside `x`; no `dtype` or `out`, and `axis` left out, 0 or -1; `x` holds no signalling NaN. Fastest on short runs: see below |
| `np.sum(x)`, `x.sum()` | uint8, bool | No other arguments |
| `np.count_nonzero(x)` | int64, float64 | No `axis` |
| `np.histogram(x, bins, range)` | float64 | An integer bin count; no weights or density |
| `np.bincount(x)` | uint8 | No weights |
| `np.isin(x, values)` | int64 | At most eight test values; flags at their defaults |
| `np.searchsorted(a, v)`, `a.searchsorted(v)`, `np.digitize(x, bins)` | float64 | `a` is one-dimensional; `v` is an array of any shape or a Python number; either `side`; no `sorter`. For `np.digitize`, with or without `right=True`, when the bins increase |
| `np.interp(x, xp, fp)` | float64 | `xp` and `fp` are one-dimensional and the same length; no `left`, `right` or `period`; `xp` holds no NaN; at most 64 sample points per query |

"Contiguous" means any shape for the operations that work element by element or reduce the whole array: a contiguous 40 by 25 array reaches the `clip` kernel as one run of 1,000 elements.

How much faster each one is, from Python and at three sizes, is in [the results]([[ root ]]report/results/).

## Things worth knowing

- **Float `cumsum` is opt-in for a reason.** The fast kernels add in a different order from NumPy. The result is as accurate, but not bit-identical, and the output of a non-negative input is not guaranteed to be non-decreasing. Code such as `np.searchsorted(np.cumsum(p), r)` relies on that. Use `install(inexact=True)` only when it does not matter to you. [Why it cannot be both exact and fast]([[ root ]]report/arithmetic/#cumsum-the-chain-you-cannot-break).
- **`x[mask] = v` rewrites elements it does not change**, and so do `np.putmask` and `np.copyto(x, v, where=mask)`. On a dense mask the kernel stores unselected elements back with their own bits. The result is identical; the array must be writable throughout, which a normal array is.
- **A mask holding a byte other than 0 or 1 goes to NumPy.** NumPy treats any non-zero byte as True, and a boolean view of other data can hold a 2. Every mask is read once before a kernel sees it, in the indexing syntax, in the function forms, in `mask.sum()`, in `np.cumsum(mask)` and in `np.where(mask, a, b)` alike.
- **The mask functions are counted under the indexing syntax.** In `report()`, `np.compress` and `np.extract` count on the `x[mask]` rows, `np.putmask` and `np.copyto` on `x[mask] = v`, and `np.where(mask)` on `np.nonzero`. `np.copyto` without `where` is another operation and is not counted at all. `np.where(mask, a, b)` has a row of its own, `np.where(c, a, b)`, whose element type reads `float64`: it counts int64 calls too, for the same reason as `x[mask]` below. `mask.sum()` counts on the `np.sum` row, whose element type reads `uint8`: one kernel adds the bytes of both. In the same way `x[mask]` on int64 counts on the float64 row of `x[mask]`: that kernel moves 64-bit elements without computing on them, so it serves both types.
- **`np.maximum.accumulate` on floats stays with NumPy.** It carries a NaN forward from where it appears, which no kernel does. `np.fmax.accumulate`, the running maximum that skips NaN, is served on float64 and gives NumPy's result bit for bit, the payload of a leading NaN included.
- **A float64 array holding a signalling NaN goes to NumPy in `np.fmax.accumulate`.** NumPy turns one into a quiet NaN and forgets the maximum so far; the kernel would skip it. Arithmetic never produces a signalling NaN, so you only meet one in data built bit by bit, but every float64 array is read for them before the kernel sees it.
- **A float64 array holding a negative zero, or NaNs of different bit patterns, also goes to NumPy in `np.fmax.accumulate`.** There NumPy's own answer depends on how it was built: its x86-64 and ARM64 builds give a running maximum of `0.0` and `-0.0` different signs. You get whatever your NumPy gives. The same read finds them.
- **`np.fmax.reduceat` gains most on short runs.** On runs of 16 it is several times faster than NumPy. On runs of several hundred elements it is about as fast as NumPy, and on long runs of an array too large for the processor's second-level cache it can be slower (0.8x at runs of 1,024 in a million elements): every element is read once more, for signalling NaNs. Give `starts` as an int64 array; a list or int32 starts go to NumPy.
- **`np.fmax.reduceat` leaves two kinds of answer to NumPy.** When `x` holds a negative zero and the maximum of some run is a zero, or `x` holds NaNs of different bit patterns and some run is all NaN, the whole call goes to NumPy. Which zero or which NaN NumPy returns there depends on the vector instructions it picked when it started, which differ between processors, and hotkernels does not guess. Values from ordinary arithmetic have one NaN pattern, and a zero maximum among negative values only matters if a negative zero is present.
- **`np.clip(x, 0, 1)` is faster by less than `np.clip(x, -1, 1)`.** A lower bound of exactly zero needs extra care for negative zeros. It is still several times faster than NumPy.
- **A small `np.clip` is fast when its bounds are plain numbers.** `np.clip(x, lo, hi)` and `x.clip(lo, hi)` skip NumPy's own call path, most of a call on a small array, when the bounds are Python floats or ints, or NumPy scalars of the array's own type, and there is no `out`, `dtype`, `casting` or `where`. The result has the element type NumPy gives it: a float32 array clipped to Python floats stays float32. Other calls, such as `np.float64` bounds on a float32 array (NumPy returns float64), `out=`, or a bound of `None`, go through NumPy's call path first; the kernel still serves them where it served them before, and they gain only on larger arrays.
- **`np.isin` is fastest when the test values are close together**: all between 0 and 65,534, or all within 32,767 of the first one. Otherwise it runs at the speed of compiled C, which is still ahead of NumPy.
- **`np.searchsorted` and `np.interp` trust you that the array is sorted, as NumPy does.** Neither checks. On an array that is not sorted both return a meaningless result, and the kernel's is not the same meaningless result as NumPy's. Sorted data, including NaNs at the end of `a`, gives NumPy's answer bit for bit.
- **`np.digitize` is served when its bins increase.** It calls `np.searchsorted`, on either side. With decreasing bins it searches them reversed, as a view no kernel takes, and runs at NumPy's speed.
- **`np.interp` with a few queries against a long table stays with NumPy.** A NaN among the sample points is something the kernel cannot handle, so `xp` is read once before every call. With more than 64 sample points per query that pass would cost more than the kernel saves.
- **Free-threaded Python is not supported.** `install()` does nothing there, and `report()` says so.
- **NumPy 2.0 to 2.2 are not supported.** They answer a few calls differently from later releases (`np.clip` drops the sign of a zero on 2.0, and `np.count_nonzero(x)` gives a Python `int` before 2.3), and the kernels give the later answer. Installing hotkernels brings NumPy 2.3 or later; with an older NumPy in place `install()` does nothing, and `report()` says so.
- **Subclasses mostly stay with NumPy.** `x[mask]`, the methods and the hooked functions take a kernel only for a plain `ndarray`; masked arrays, memory maps and your own subclasses go to NumPy there. `clip` and `argmax` are also hooked beneath NumPy's own handling of array types, so for a subclass they serve whatever plain buffer NumPy hands them, and NumPy wraps the result as it always did.

## Checking it yourself

From a clone of the repository:

```bash
uv run delivery demo          # run examples/workload.py with and without hotkernels, and compare
uv run delivery check         # 96,734 NumPy calls, both ways; results, errors and warnings must agree
uv run delivery time          # time each statement from Python on your machine
uv run delivery numpy-tests   # NumPy's own test suite, both ways; about ten minutes
```

`delivery demo` is the quickest way to see the effect: an ordinary NumPy program on two million elements that runs in 45 ms on plain NumPy and 8.4 ms with the one line added, with the same results bit for bit.
