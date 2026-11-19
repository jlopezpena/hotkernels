# Bottleneck: NaN-aware reductions that a library sends past NumPy

This began as a proposal, written before the code, for [issue 192](https://github.com/sofeikov/hotkernels/issues/192), and is now implemented in the same branch. It answers the issue's three questions with measurements and says what was built. Where the building departed from the proposal, [What changed in the building](#what-changed-in-the-building) says so, and [Status](#status) holds the result against the exit criterion. The rest of the text stands as proposed.

Every figure here is ARM64's, measured on an Apple M1 Pro with CPython 3.12.7, NumPy 2.5.5, bottleneck 1.6.1, empyrical-reloaded 0.7.3 and pandas 3.1.1, at a load average of 2.1 to 2.7 with other work running. No x86-64 figure has been taken.

## Summary

- **Where the time is.** The `empyrical_stats` application makes 300 calls of bottleneck's `nanmean`, `nanstd`, `nanmin` and `nanmax` on matrices of 252 × 4,789, each reduced along axis 0, and 840 more on one-dimensional series. The 300 take 582 ms of a 1,334 ms run; the 840, 9 ms.
- **They are worth a kernel.** A plain C loop that keeps bottleneck's order within each column and works on 32 columns at a time gives bottleneck's answers bit for bit and is 4.3x to 7.1x faster at the application's shape, 1.7x to 6x at others. Done as kernels, the application would take about 1.5x less time. See [Question 1](#question-1-are-nan-aware-reductions-worth-a-kernel).
- **Exactness is what decides, and it can be kept.** `nanmin` and `nanmax` do not depend on order except for the sign of a zero, which bottleneck takes from the last zero. `nanmean` and `nanstd` do, and bottleneck's order (a sequential sum from +0.0, NaNs skipped) is reproduced exactly as long as each column is summed in order. One more detail is the build's: bottleneck's macOS ARM64 wheel computes `asum += d * d` as one fused multiply-add. The kernel does the same on ARM64 and the two roundings on x86-64, and `install()` checks this against the bottleneck in use.
- **They can be reached for every caller.** bottleneck's functions are C function objects, and CPython calls such an object through a pointer inside it, as it calls NumPy's functions. Changing that pointer serves `bn.nanmean`, empyrical's wrapper that captured it at import, and pandas' `nanops`, which captured it too, without replacing any name. It is the function hook hotkernels already uses for NumPy, on an object of another library. See [Question 2](#question-2-where-would-they-be-reached).
- **Proposed:** four kernels, `nanmean_cols_f64`, `nanstd_cols_f64`, `nanmin_cols_f64` and `nanmax_cols_f64`, shipped as C on both architectures, and four hooks on bottleneck's functions. `install()` applies them by default, since they are bit-identical; `install(bottleneck=False)` leaves them out. It never imports bottleneck: if bottleneck is not imported yet, the hooks are applied when it is. `report()` shows them as rows of their own. See [Design](#design).
- **Not proposed now:** kernels for NumPy's own `np.nanmean` and `np.nanstd`, which would need NumPy's order and not bottleneck's; one-dimensional and last-axis forms, which this application does not need; assembly. Each is measured below and left for an issue of its own.

## Status

Against the [exit criterion](#exit-criterion), on an Apple M1 Pro with CPython 3.12.7, NumPy 2.5.5 and bottleneck 1.6.1:

| # | Criterion | Result |
|---|---|---|
| 1 | `judge check` passes for the four kernels | Met on ARM64, with `--unlocked` since the specs are not on `main`: 158,874 cases each for `nanmean_cols_f64`, `nanmin_cols_f64` and `nanmax_cols_f64`, 170,918 for `nanstd_cols_f64`. With the square and add of `nanstd` unfused on ARM64 the reference fails the judge at its first case. x86-64 is not checked here; CI runs `judge check nanstd_cols_f64 --unlocked` on Linux |
| 2 | `delivery check` passes, with the new cases | Met: 221,347 calls over the 23 NumPy operations and the four bottleneck functions, 1,529 of them bottleneck's, none failing; and the 24 checks of `install()` itself, among them that it never imports bottleneck, that a later import is served, and that a bottleneck of another layout leaves the hooks off with `install()` still working |
| 3 | `empyrical_stats` identical and faster | Met. `uv run --group apps delivery apps --only empyrical_stats`, before the accumulation along axis 0 of [accumulate-rows-design.md](accumulate-rows-design.md) was below it in the stack: 1,290.8 ms on plain NumPy, 826.0 ms after `install()`, 1.56x, results identical, at a load average of 3.7 (the study's run before this change gave 1.02x). With that change below it: 1,287.8 ms and 360.5 ms, 3.57x, results identical, at a load average of 9; in alternating runs at the same load, `install(bottleneck=False)` took 860.1 ms and `install()` 364.1 ms, so the bottleneck hooks take the application 2.36x further. The ratios come from alternating runs |
| 4 | `report()` shows the four rows, counting | Met: `bn.nanmean`, `bn.nanstd`, `bn.nanmin` and `bn.nanmax`, "waiting" until bottleneck is imported. In one run of the application they took 480 calls and left 660, the one-dimensional ones, to bottleneck |

From Python, best of 15 calls, at 252 × 4,789, bottleneck's own call against the hooked one (load average 14):

| Call | Rolling windows | C order |
|---|---|---|
| `bn.nanmean(a, axis=0)` | 1.578 ms to 0.302 ms, 5.23x | 1.632 ms to 0.401 ms, 4.07x |
| `bn.nanstd(a, axis=0, ddof=1)` | 3.900 ms to 0.538 ms, 7.25x | 3.911 ms to 0.712 ms, 5.49x |
| `bn.nanmin(a, axis=0)` | 1.341 ms to 0.302 ms, 4.44x | 1.378 ms to 0.403 ms, 3.42x |
| `bn.nanmax(a, axis=0)` | 1.341 ms to 0.302 ms, 4.45x | 1.366 ms to 0.403 ms, 3.39x |

**Which bottleneck and CPython.** The function objects were checked to be what the hook expects (exactly `builtin_function_or_method`, `METH_VARARGS | METH_KEYWORDS`, a NULL call pointer), every probe passed, and 200 random matrices with NaN and negative zeros gave the same bits with and without the hooks in all four functions, for bottleneck 1.4.2, 1.5.0, 1.6.0 and 1.6.1 on CPython 3.12, and for 1.6.1 on CPython 3.11, 3.13 and 3.14, all on NumPy 2.5.5. bottleneck 1.3.8, built against NumPy 1, does not import on NumPy 2, so the development environment asks for 1.4 or later. On a bottleneck whose function differs in any of those three ways, the hook's `apply` refuses before changing anything, `install()` leaves that hook off and lists it in `report()`, and the call is bottleneck's as before; `delivery check` proves this with a stand-in module of Python functions. A bottleneck that rounds `nanstd` differently from the platform's reference fails that hook's probe, with the same result.

## What changed in the building

| Proposed | Built | Why |
|---|---|---|
| The hooks are applied by `install()` or by the finder | The same; the finder's probes are taken off the counters again, by a new `forget()` in the binding | `install()` resets the counters after its probes. A hook applied later would otherwise report its probes' four calls as the program's |
| The hook takes a matrix whose kept axis has a stride of 8 bytes | The same, which also takes a 5,040 × 1 column (one result), as empyrical passes for its alpha | The rule did not need an exception for it; `delivery check` has shapes of one row and of one column |
| The attempts logs say no assembly was tried | Each holds one attempt, the C itself | The documentation site requires a numbered attempt in every log |
| No version test (open decision 2) | No version test; the development environment asks for bottleneck 1.4 or later | 1.3.8 does not import on NumPy 2. The checks of the object's layout and the probes are the guard, as for NumPy |

Not built, and proposed as issues: assembly for the four kernels, the last-axis and one-dimensional forms, and kernels for NumPy's own `np.nanmean` and `np.nanstd`.

## What the application calls

Every bottleneck call one run of `empyrical_stats` makes, recorded by wrapping empyrical's functions (the script is in the pull request's description):

| Function | Shape | Strides (bytes) | Axis | Calls | Time | Per call |
|---|---|---|---|---|---|---|
| `nanstd`, `ddof=1` | 252 × 4,789 | 8, 8 | 0 | 60 | 231.8 ms | 3.86 ms |
| `nanmean` | 252 × 4,789 | 8, 8 | 0 | 60 | 96.0 ms | 1.60 ms |
| `nanmin` | 252 × 4,789 | 38,312, 8 | 0 | 60 | 92.9 ms | 1.55 ms |
| `nanmax` | 252 × 4,789 | 8, 8 | 0 | 60 | 80.9 ms | 1.35 ms |
| `nanmin` | 252 × 4,789 | 8, 8 | 0 | 60 | 80.6 ms | 1.34 ms |
| `nanmean`, `nanstd`, `nanmin`, `nanmax` | 5,040 or 5,040 × 1 | 8 | 0 | 840 | 8.8 ms | 10 µs |

The run took 1,334 ms. The five large rows are 582 ms of it, 44%; the profile in [applications-findings.md](applications-findings.md#empyrical_stats-102x) gave 46.5% on a busier machine.

The matrices with strides of 8 and 8 are empyrical's rolling windows: `rolling_window(returns, 252).T`, a view in which column j is the 252 returns from day j on. The columns overlap, the whole view is 5,040 doubles, and each column is contiguous in memory. The one with a row stride of 38,312 bytes is a C-contiguous 252 × 4,789 array, the drawdown of every window, reduced down its columns. In both, the elements of one row are adjacent: element (i, j) is at `a + 8 × (i × step + j)`, with `step` 1 for the windows and 4,789 for the drawdown.

bottleneck reduces each column on its own, in order, one scalar operation at a time ([`reduce_template.c`](https://github.com/pydata/bottleneck/blob/v1.6.1/bottleneck/src/reduce_template.c)). A sum of one column waits on the latency of an add for each element. That is why it is slow, and also why it can be made fast without changing a bit: the columns are independent.

## Question 1: are NaN-aware reductions worth a kernel?

### What bottleneck computes

From bottleneck 1.6.1's source, for float64 along one axis of an array of two or more dimensions (`*_one_float64`); the loops have not changed since 1.3.2, to which bottleneck reverted in 2022:

| Function | Per column |
|---|---|
| `nanmean` | `asum = 0; count = 0;` then for each element `ai`, in order: `if (ai == ai) { asum += ai; count++; }`. The result is `asum / count`, or NaN when count is 0 |
| `nanstd` | The same sum and count. If `count > ddof`: `amean = asum / count`, then `asum = 0` and for each element that is not NaN, in order, `ai -= amean; asum += ai * ai`. The result is `sqrt(asum / (count - ddof))`, else NaN |
| `nanmin`, `nanmax` | `extreme = +inf` (or -inf); for each element, `if (ai <= extreme) { extreme = ai; allnan = 0; }` (or `>=`). The result is `extreme`, or NaN when no comparison held |

What a kernel must keep:

- **The order of the additions within a column.** That alone fixes `nanmean`, and `nanstd`'s two passes. Masking a NaN to +0.0 and adding it is the same as skipping it: the sum starts at +0.0 and, in the default rounding, never becomes -0.0, and every number plus +0.0 is itself.
- **`count` converted to a double, then a division.** Not a multiplication by an inverse, which bottleneck uses only for integer input.
- **Whether `asum += ai * ai` is fused.** In C the compiler may contract it into one fused multiply-add, rounded once, and clang does so on ARM64: the disassembly of `nanstd_one_float64` in the macOS ARM64 wheel has `fsub d3, d2, d1` then `fmadd d3, d3, d3, d0`. An x86-64 wheel built for the baseline instruction set cannot fuse. So the answer bottleneck gives depends on its build, as `np.interp`'s does on NumPy's ([kernels/interp_f64/naive.c](../kernels/interp_f64/naive.c)). The kernel follows the platform, as `interp_f64` does, and `install()` checks it against the bottleneck in use ([Self-test](#self-test)).
- **For `nanmin` and `nanmax`, only the sign of a zero.** Every other value has one bit pattern among the elements equal to it. With `<=`, a later equal element replaces an earlier one, so a zero result has the sign of the last zero in the column. An all-NaN column gives NaN, with no warning; NumPy's `np.nanmin` warns "All-NaN slice encountered". A column of +inf gives +inf.
- **The NaN.** bottleneck writes `NPY_NAN`, the positive quiet NaN, for an empty or all-NaN column.

bottleneck and NumPy do not agree with each other. On the application's windows, `np.nanstd` differs from `bn.nanstd` in 715 of 4,789 results (by up to 3.8 × 10⁻¹⁶ relative): NumPy squares and adds in two roundings. Along the last axis NumPy sums pairwise and differs in most results of both `nanmean` and `nanstd`. `np.nanmin([-0.0, 0.0])` is -0.0 where bottleneck's is +0.0. A kernel for bottleneck's calls therefore has to be judged against bottleneck, not against NumPy.

### A prototype

Plain C, compiled with `clang -O3 -mcpu=apple-m1`, called through `ctypes`. For the layout of the application, element (i, j) at `a[i × step + j]`, it takes 32 columns at a time and, for each row in turn, updates 32 accumulators, one per column. Each column's elements are still added in order, so the result is bottleneck's bit for bit; the 32 additions of a row are independent, and the compiler turns them into vector instructions. `nanstd` uses `__builtin_fma` for the square and add. `nanmin` keeps a flag per column for "some comparison held". The probe also has a form for the last axis of a C-contiguous matrix, which runs eight rows' sums side by side, and one-dimensional forms. The source is in the pull request's description.

Checked first: 300 random shapes from 1 × 1 to 59 × 79, as windows and as C-contiguous matrices, with NaN, zeros of both signs, infinities, all-NaN and constant columns, against bottleneck on the same arrays: every result identical, for all four functions, `ddof` 0, 1 and 2, all three layouts.

Best of 15 calls, per call:

| Layout | Shape | Function | bottleneck | NumPy's `np.nan*` | Prototype | Faster than bottleneck |
|---|---|---|---|---|---|---|
| Windows, step 1 | 252 × 4,789 | `nanmean` | 1.596 ms | 1.254 ms | 0.314 ms | 5.08x |
| | | `nanstd` | 3.869 ms | 2.968 ms | 0.546 ms | 7.09x |
| | | `nanmin` | 1.321 ms | 0.204 ms | 0.308 ms | 4.30x |
| | | `nanmax` | 1.317 ms | 0.204 ms | 0.308 ms | 4.28x |
| C order, axis 0 | 252 × 4,789 | `nanmean` | 1.637 ms | 1.333 ms | 0.400 ms | 4.10x |
| | | `nanstd` | 3.903 ms | 3.119 ms | 0.658 ms | 5.93x |
| | | `nanmin` | 1.343 ms | 0.213 ms | 0.403 ms | 3.33x |
| | 1,000 × 1,000 | `nanstd` | 3.351 ms | 2.516 ms | 0.609 ms | 5.50x |
| | 65,536 × 16 | `nanmean` | 1.637 ms | 1.869 ms | 0.516 ms | 3.17x |
| | 4 × 262,144 | `nanmean` | 0.726 ms | 1.547 ms | 0.420 ms | 1.73x |
| | | `nanmin` | 0.725 ms | 0.290 ms | 0.420 ms | 1.73x |
| C order, axis 1 | 4,789 × 252 | `nanmean` | 1.599 ms | 1.043 ms | 0.333 ms | 4.80x |
| | | `nanstd` | 3.832 ms | 2.890 ms | 0.731 ms | 5.25x |
| | 65,536 × 16 | `nanmean` | 0.607 ms | 1.520 ms | 0.298 ms | 2.04x |
| | | `nanmin` | 0.542 ms | 0.838 ms | 0.289 ms | 1.87x |
| One dimension | 5,040 | `nanmin` | 6 µs | 2 µs | 1 µs | 4.5x |
| | 1,048,576 | `nanmin` | 1,343 µs | 119 µs | 114 µs | 11.8x |
| | 5,040 | `nanmean` | 8.2 µs | 10.9 µs | 5.9 µs | 1.39x |
| | 1,048,576 | `nanmean` | 1,628 µs | 850 µs | 1,043 µs | 1.56x |
| | 1,048,576 | `nanstd` | 3,803 µs | 2,262 µs | 2,402 µs | 1.58x |

What the table says:

- **Along axis 0, every function is 3.3x to 7.1x faster than bottleneck at the application's shape,** and still 1.7x on four rows, where there is little to do per column. `nanstd` gains most because it makes two passes, each limited by the latency of a chain of adds.
- **Along the last axis the same holds,** 1.9x to 6.2x, from running eight rows' chains side by side.
- **In one dimension `nanmean` and `nanstd` can gain at most about 1.6x.** One chain of additions has no independent work beside it. bottleneck's loop puts the NaN test on the chain (an add and a select per element, about 1.55 ns); taking the test off it, by adding +0.0 for a NaN, leaves one add per element, about 0.99 ns, which is the latency of an add. Nothing faster is bit-identical. `nanmin` and `nanmax` in one dimension do not depend on order and gain 4.5x to 12x.
- **NumPy's `np.nanmin` is faster than the C prototype along axis 0,** and already vector code (it is `np.fmin.reduce` and a test for NaN in the result). A NEON kernel for bottleneck's `nanmin` could close that gap: per two columns it needs a compare, a select and an OR. The C does not reach it.

**What it is worth to the application.** At the prototype's speeds the 300 large calls would take about 113 ms in place of 582 ms, and the run about 865 ms in place of 1,334 ms: about 1.5x. The 840 one-dimensional calls are 9 ms in all, too little to matter here.

**Which calls can be served bit for bit, then:** all four functions along axis 0 of a two-dimensional float64 array whose rows' elements are adjacent (any row stride), at the speeds above; the same along the last axis of a C-contiguous one; `nanmin` and `nanmax` in one dimension. `nanmean` and `nanstd` in one dimension can be served exactly, but at most 1.6x faster.

## Question 2: where would they be reached?

### How the callers bind bottleneck

- **empyrical** (`empyrical/utils.py`) imports bottleneck at its own import, wraps each function once, `nanmean = _wrap_function(bn.nanmean)`, and `empyrical/stats.py` imports the wrapped names with `from .utils import nanmean, ...`. The wrapper is a closure that holds the bottleneck function object and calls it as `f(*args, **kwargs)`. Replacing `bottleneck.nanmean` after empyrical is imported changes nothing it calls.
- **pandas** (`pandas/core/nanops.py`) imports bottleneck when `nanops` is imported, if it is installed, and its decorator `bottleneck_switch` looks the function up once, `bn_func = getattr(bn, bn_name)`, when the module is defined. At call time it uses bottleneck when the option `compute.use_bottleneck` is on (the default), `skipna` is true, the array is not empty and of a type bottleneck takes, and no mask is given; and then not for `nanmean` and `nansum`, which pandas keeps on NumPy for precision. So pandas sends `nanstd`, `nanvar`, `nanmin` and `nanmax` to bottleneck, with `axis=None` for a Series and the last axis of a block's values for a DataFrame. pandas also calls the function again with NumPy if bottleneck's result holds an infinity.
- **bottleneck** (`bottleneck/__init__.py`) does `from .reduce import nanmean, ...`. Each function is one C function object of the type `builtin_function_or_method`, defined by a `PyMethodDef` with `METH_VARARGS | METH_KEYWORDS`. It parses `(a, axis=None, ddof=0)`, turns a non-array into an array, sends a byte-swapped array to its pure-Python fallback, and then picks a loop by element type and by whether it reduces everything (`axis=None`, or a one-dimensional array) or one axis. There is no table of loops and no other point of dispatch to replace: the loops are static functions inside one C function.

### The options

| Option | Reaches `bn.nanmean(a)` | Reaches empyrical and pandas, imported before | Undone by `uninstall()` | Cost |
|---|---|---|---|---|
| Replace the module attributes `bottleneck.nanmean` and the rest | Yes | No: both looked the function up at their own import | Yes | None |
| Also replace the names in modules already imported (`empyrical.utils`, `pandas.core.nanops`, ...) | Yes | Only the names found: empyrical's closures hold the object itself, not a name | Only if every replaced name is tracked | A walk of `sys.modules`, and wrong for any module not foreseen |
| A hook inside bottleneck's dispatch | | | | There is none |
| **Change the call pointer inside each function object** | **Yes** | **Yes: it is the same object** | **Yes: one pointer back** | Nothing on a call that no kernel takes, beyond the hook's own tests |

The last is the function hook of [delivery-design.md](delivery-design.md#the-five-hooks), applied to an object of another library. CPython calls a `builtin_function_or_method` through a `vectorcall` pointer at an offset its type publishes, `tp_vectorcall_offset`, and that is a CPython guarantee, not bottleneck's. For a function defined with `METH_VARARGS | METH_KEYWORDS` the pointer is NULL and CPython builds a tuple and a dictionary and calls the type's `tp_call`. Setting the pointer sends every call of that object to the binding, however the object was found: `bn.nanmean`, empyrical's closure, pandas' `bn_func`, `functools.partial`. A call that takes no kernel goes on to the type's `tp_call` with the arguments as they came, which is the path CPython takes for it today. `uninstall()` puts NULL back.

It changes nothing in bottleneck that its authors control, and nothing CPython caches. It does not reach bottleneck's pure-Python fallbacks in `bottleneck.slow`, which are other functions.

**Not importing bottleneck.** `install()` looks for `bottleneck.reduce` in `sys.modules` and never imports it. When it is not there, `install()` puts a finder at the front of `sys.meta_path` that does nothing but notice when `bottleneck.reduce` is executed, through whichever finder actually loads it, and then applies the hooks. This matters for `python -m hotkernels script.py`, which installs before the script imports anything. `uninstall()` removes the finder with the hooks.

**Testing it.** `delivery check` runs each case once without hotkernels and once with it and compares everything a caller can observe. A case that calls `bn.nanmean(a, axis=0)` fits as it is: the run without hotkernels is bottleneck's own answer. bottleneck becomes a dependency of the development environment, so that the check and the judge have it.

### NumPy's own `np.nanmin` and the rest

Code that calls NumPy directly, best of seven:

| Call | 16 elements | 4,096 | 1,048,576 | 252 × 4,789, axis 0 |
|---|---|---|---|---|
| `np.nanmin(a)` | 1.59 µs | 1.87 µs | 107.5 µs | 0.216 ms |
| `np.fmin.reduce(a)`, which it calls | 0.29 µs | 0.56 µs | 104.0 µs | 0.213 ms |
| `np.nanmean(a)` | 5.35 µs | 9.18 µs | 796 µs | 1.327 ms |
| `np.nanstd(a)` | 12.5 µs | 21.4 µs | 2,123 µs | 2.978 ms |
| `np.sum(a)`, for scale | | | | 0.210 ms |

- **`np.nanmin` and `np.nanmax` are not slow on large arrays.** They are `np.fmin.reduce` plus a test for NaN in the result, and `np.fmin.reduce` is vector code: at a million elements it is 12x faster than bottleneck. What is left is NumPy's Python wrapper on short arrays, 1.3 µs of the 1.59 µs at 16 elements. A function hook on `np.nanmin` (an `_ArrayFunctionDispatcher`, the type of `np.median`) could remove that, as it did for `np.argmax`; that is the subject of [issue 99](https://github.com/sofeikov/hotkernels/issues/99), not of a new kernel.
- **`np.nanmean` and `np.nanstd` are slow:** Python code that copies the array with NaNs replaced by zero, counts, sums and divides, and for `nanstd` subtracts, squares and sums again. At the application's shape `np.nanmean` takes 6.3 times as long as `np.sum`. A kernel for them is worth having, but it must reproduce NumPy's order: pairwise along a contiguous axis, sequential across rows, the sum starting from the first element and not from +0.0, two roundings for the square. That is another spec, with NumPy as its oracle, and another hook (the function hook on `np.nanmean` and `np.nanstd`). It is proposed as an issue of its own and not built here.

## Question 3: what should the study and `report()` show?

- **`report()` shows the bottleneck kernels as rows,** under the call a user writes, `bn.nanmean` and the rest, with the calls each took. The column "Left to NumPy" holds, on these rows, the calls that reached the hook and were left to bottleneck. When bottleneck has not been imported, the rows say so and that the hooks will be applied when it is; when `install(bottleneck=False)` left them out, the rows say that. The first line's count of kernels counts them like any other.
- **The study needs no change of its own.** `delivery apps` builds its "Kernel calls" and "Left to NumPy" columns from `report()`, so the bottleneck calls appear there once they have a hook, and the comparison of results with and without hotkernels covers them. The profile already names bottleneck's functions under "Other compiled code"; with the hooks, that time moves to the hotkernels class. `applications-findings.md` is updated with a new run of `empyrical_stats`.
- **What `report()` cannot show** is a call to a library it has no hook in. Nothing cheap fixes that: it would take a hook in every compiled library. The study's profile remains the way to find the next one.

## Design

### Kernels

Four kernels, one per function, with one layout: a two-dimensional float64 array reduced along axis 0, whose elements in a row are adjacent and whose rows are `step` elements apart.

```c
size_t nanmean_cols_f64(const double *a, double *out, size_t n, size_t cols, size_t step);
size_t nanstd_cols_f64(const double *a, double *out, size_t n, size_t cols, size_t step, int64_t ddof);
size_t nanmin_cols_f64(const double *a, double *out, size_t n, size_t cols, size_t step);
size_t nanmax_cols_f64(const double *a, double *out, size_t n, size_t cols, size_t step);
```

- `a` holds `n` elements. Element (i, j) is `a[i × step + j]`, for `j < cols` and `i < rows`, where `rows` is `(n − cols) / step + 1` when `n >= cols`, else 0: as many rows as fit. `step >= 1`, `cols >= 1`. A step below `cols` makes the rows overlap, as the application's windows do.
- The kernel writes one result per column to `out` and returns `cols`, or returns 0 and writes nothing when there are no rows.
- The result of each column is bottleneck's, bit for bit, for its build on that architecture: the oracle in `spec.py` calls bottleneck on the same view. `ddof` is any int64; `count − ddof` is computed in int64 and converted.
- **C on both architectures.** `naive.c` is the blocked loop of the prototype: 32 columns at a time, one accumulator each, rows in order. It is simple, it is what the prototype measured, and on ARM64 the compiler makes vector code of it. Assembly is for a later run, once the specs are on `main`; the room is in `nanmin` and `nanmax`, where NumPy's vector loop is 1.5x faster than the C.
- `nanstd_cols_f64`'s `naive.c` fuses the square and add on ARM64 and does not on x86-64, by an `#if` like `interp_f64`'s. Its oracle is bottleneck on the machine it runs on, so each architecture's `naive.c` is checked against that architecture's bottleneck, and CI checks it on both, as it does `interp_f64`.

The last-axis and one-dimensional forms are not built: this application does not use them, pandas does not send `nanmean` to bottleneck, and `nanmean` and `nanstd` in one dimension gain at most 1.6x. They are left for an issue.

### Hooks

One hook per function, on `bottleneck.reduce.nanmean`, `.nanstd`, `.nanmin` and `.nanmax`, a new kind shown as "bottleneck". The wrapper takes a kernel when the call is:

- `f(a, axis)`, `f(a, axis=...)`, and for `nanstd` also `ddof` by position or keyword; no other keyword.
- `a` exactly an `ndarray` of float64 in native byte order with two dimensions, each at least one long.
- `axis` a Python int naming one of the two axes (0, 1, -1 or -2). The other axis, the one kept, has a stride of 8 bytes; the reduced axis has a positive stride that is a multiple of 8. That is axis 0 of a C-contiguous matrix, axis 1 of a Fortran-ordered one, and any view whose kept axis is contiguous, such as empyrical's windows.
- `ddof` absent or a Python int that bottleneck's own conversion to a C `int` accepts.

The result is a new one-dimensional float64 array of the kept axis's length, as bottleneck's is. Anything else, a one-dimensional array or `axis=None` included, goes to the type's `tp_call` with the caller's arguments, and is counted as left. The interpreter lock is released for the kernel from 8,192 elements, as in the other hooks and as bottleneck itself does. The floating-point status flags are left as the kernel leaves them, as bottleneck's own loops leave theirs; `delivery check` compares the warnings NumPy raises afterwards.

### Self-test

Each hook's probe reduces a probe matrix with NaN, zeros of both signs, infinities and random values, along axis 0, once with the hook removed and once with it applied, and requires the two results to be the same bits and the kernel's counter to have moved. The random values make a fused and an unfused square-and-add differ in several columns, so a bottleneck build that rounds differently from the kernel fails the probe of `nanstd` and leaves that hook out, with a line in `report()`. The other hooks do not depend on the build.

### `install()`

- `install(inexact=False, bottleneck=True)`. With `bottleneck=False` the four hooks are left out and the finder is not added.
- The four hooks are applied after NumPy's, when `bottleneck.reduce` is in `sys.modules`, or later by the finder. A hook applied by the finder is proved by its probe then, like the others.
- `uninstall()` removes them and the finder. `delivery check` gains cases for the four functions, and its checks of `install()` itself gain: bottleneck's function objects are the same objects after `install()` and `uninstall()`; with `bottleneck=False` no bottleneck hook is applied; and, in a fresh interpreter, `install()` does not import bottleneck and an import of bottleneck afterwards is served.

### Exit criterion

1. `judge check` passes for the four kernels, on both architectures for `nanstd_cols_f64`.
2. `delivery check` passes, with the new cases.
3. `uv run --group apps delivery apps --only empyrical_stats` reports identical results and a run measurably faster than without hotkernels.
4. `report()` shows the four rows, counting.

## Considered and not proposed

| Option | Why not |
|---|---|
| Replace `bottleneck.nanmean` and the other module attributes | Misses empyrical and pandas, which hold the function objects from their own import. Shown by reading their code, above |
| A tolerance instead of bottleneck's order | Not needed: the order costs nothing, since the speed comes from working on columns side by side. A tolerance would also make the kernels opt-in |
| Serving bottleneck's calls with NumPy's results | They differ from bottleneck's in the last bit for `nanstd` and along the last axis; a user's numbers would change by installing hotkernels |
| `np.nanmean` and `np.nanstd` kernels in the same change | NumPy's order is not bottleneck's, so they are other kernels with another oracle, and no application of the study calls them. Left for an issue |
| Importing bottleneck at `install()` to hook it | Imports a module the user did not ask for, and costs its import time in every process that installs |

## Decisions still open

1. **Whether binding a second library is wanted at all.** The issue leaves this to the maintainer. Proposed: yes, for bottleneck, because the binding is the same hook on the same kind of object, bit-identical and cheap to test; and on by default, with `install(bottleneck=False)` to leave it out.
2. **Which bottleneck versions.** Resolved: no version test, as proposed, and `delivery check` on the bottleneck in `uv.lock`. Checked on 1.4.2 to 1.6.1; see [Status](#status).
3. **Assembly, the other layouts, and NumPy's own `nanmean` and `nanstd`:** issues of their own.
