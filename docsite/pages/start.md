There is very little to do. Install the package, call `install()` once, and NumPy uses the kernels for the rest of the process.

## Requirements

| | |
|---|---|
| Machine | A Mac with Apple Silicon (ARM64) |
| Python | 3.11 or later, the standard build with the interpreter lock |
| NumPy | 2.0 or later |

Wheels are published for CPython 3.11 to 3.14. With another Python the package is built from source when it is installed, which takes a few seconds and needs the Xcode command line tools (`xcode-select --install`).

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
hotkernels 0.1.0: installed, 20 of 22 kernels (NumPy 2.5.5)

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
| `np.argmax(x)`, `np.argmin(x)`, `x.argmax()` | float32, float64 | For these types, including strided input and `axis=` (one kernel call per row or column). Runs shorter than 128 elements reached through the method stay with NumPy, which is as fast there |
| `np.clip(x, lo, hi)`, `x.clip(lo, hi)` | float32, float64 | Scalar bounds with `lo <= hi`; the output is not the input |
| `x[mask]` | float32, float64 | Both one-dimensional and the same length |
| `x[mask] = v` | float64 | As above, and `v` is a Python float or int |
| `np.nonzero(mask)`, `mask.nonzero()` | bool | One-dimensional |
| `np.cumsum(x)`, `x.cumsum()` | int64; float32 and float64 with `inexact=True` | No `axis`, `dtype` or `out` |
| `x.astype(np.float32)` | uint8 | Other arguments at their defaults |
| `np.sum(x)`, `x.sum()` | uint8 | No other arguments |
| `np.count_nonzero(x)` | int64, float64 | No `axis` |
| `np.histogram(x, bins, range)` | float64 | An integer bin count; no weights or density |
| `np.bincount(x)` | uint8 | No weights |
| `np.isin(x, values)` | int64 | At most eight test values; flags at their defaults |
| `np.searchsorted(a, v)`, `a.searchsorted(v)` | float64 | `a` is one-dimensional; `v` is an array of any shape or a Python number; `side='left'`, which is the default, and no `sorter` |
| `np.interp(x, xp, fp)` | float64 | `xp` and `fp` are one-dimensional and the same length; no `left`, `right` or `period`; `xp` holds no NaN; at most 64 sample points per query |

"Contiguous" means any shape for the operations that work element by element or reduce the whole array: a contiguous 40 by 25 array reaches the `clip` kernel as one run of 1,000 elements.

How much faster each one is, from Python and at three sizes, is in [the results]([[ root ]]report/results/).

## Things worth knowing

- **Float `cumsum` is opt-in for a reason.** The fast kernels add in a different order from NumPy. The result is as accurate, but not bit-identical, and the output of a non-negative input is not guaranteed to be non-decreasing. Code such as `np.searchsorted(np.cumsum(p), r)` relies on that. Use `install(inexact=True)` only when it does not matter to you. [Why it cannot be both exact and fast]([[ root ]]report/arithmetic/#cumsum-the-chain-you-cannot-break).
- **`x[mask] = v` rewrites elements it does not change.** On a dense mask the kernel stores unselected elements back with their own bits. The result is identical; the array must be writable throughout, which a normal array is.
- **`np.clip(x, 0, 1)` is faster by less than `np.clip(x, -1, 1)`.** A lower bound of exactly zero needs extra care for negative zeros. It is still several times faster than NumPy.
- **`np.isin` is fastest when the test values are close together**: all between 0 and 65,534, or all within 32,767 of the first one. Otherwise it runs at the speed of compiled C, which is still ahead of NumPy.
- **`np.searchsorted` and `np.interp` trust you that the array is sorted, as NumPy does.** Neither checks. On an array that is not sorted both return a meaningless result, and the kernel's is not the same meaningless result as NumPy's. Sorted data, including NaNs at the end of `a`, gives NumPy's answer bit for bit.
- **`np.digitize` is served only with `right=True`.** It calls `np.searchsorted`, and with the default `right=False` it asks for `side='right'`, which has no kernel.
- **`np.interp` with a few queries against a long table stays with NumPy.** A NaN among the sample points is something the kernel cannot handle, so `xp` is read once before every call. With more than 64 sample points per query that pass would cost more than the kernel saves.
- **Free-threaded Python is not supported.** `install()` does nothing there, and `report()` says so.
- **Subclasses mostly stay with NumPy.** `x[mask]`, the methods and the hooked functions take a kernel only for a plain `ndarray`; masked arrays, memory maps and your own subclasses go to NumPy there. `clip` and `argmax` are hooked beneath NumPy's own handling of array types, so they serve whatever plain buffer NumPy hands them, and NumPy wraps the result as it always did.

## Checking it yourself

From a clone of the repository:

```bash
uv run delivery demo          # run examples/workload.py with and without hotkernels, and compare
uv run delivery check         # 75,307 NumPy calls, both ways; results, errors and warnings must agree
uv run delivery time          # time each statement from Python on your machine
uv run delivery numpy-tests   # NumPy's own test suite, both ways; about ten minutes
```

`delivery demo` is the quickest way to see the effect: an ordinary NumPy program on two million elements that runs in 45 ms on plain NumPy and 8.4 ms with the one line added, with the same results bit for bit.
