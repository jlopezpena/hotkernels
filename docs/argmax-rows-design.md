# `argmax` and `argmin` along the rows of a matrix, and how a call is counted

This answers [issue 190](https://github.com/sofeikov/hotkernels/issues/190). It is written before the code, in the same pull request, as the maintainer asked for one pull request per issue.

Every figure here is ARM64's, measured on an Apple M1 Pro with NumPy 2.5.5, from Python, while other agents were timing on the same machine (load average 1.6 to 3.0). The ratios are what to read.

## Summary

- **The call is worth serving, not only the counting.** `np.argmax(p, axis=1)` on 300,000 rows of ten float64 takes NumPy 8.3 ns a row: a function call through the type's table per row, around a loop of ten elements. The existing `argmax_f64` kernel, called once per row from C, takes 2.3 ns a row. For float32 rows of ten NumPy takes 22 ns a row and the kernel 1.6.
- **Rows too short for the kernels' vector paths go through a loop of the binding's own** that puts one row in each lane of a vector and walks the columns: 0.4 to 0.9 ns a row for rows of two to four, where NumPy takes 7 to 13.
- **A call-level hook.** The function and method hooks of `argmax` and `argmin`, which today take only calls without an axis, also take `axis=1` and `axis=-1` on a C-ordered two-dimensional float32 or float64 array. `m.any(axis=1)` set the precedent.
- **One Python call counts once.** The type slot is reached once per row, and counted each time, in both columns of `report()`. A call made from Python now counts once, under the kernel that served it or under "Left to NumPy". The report's own footnote already says it counts "per NumPy call".

## What NumPy does

`np.argmax(a, axis=1)` goes through NumPy's Python wrapper to `a.argmax(axis=1, out=None)`, which is `PyArray_ArgMax` (`numpy/_core/src/multiarray/calculation.c`). For an axis that is not the last it transposes and copies; for the last axis of a C-ordered array it uses the data as it is. It then calls the type's `argmax` function once per row, with the interpreter lock released:

```c
for (i = 0; i < n; i++) {
    arg_func(ip, m, rptr, ap);
    ip += elsize * m;
    rptr += 1;
}
```

That function is the type slot hotkernels already hooks. Below 128 elements the hook passes the row on to NumPy's own function, because it would have to take the interpreter lock back for the kernel ([delivery-design.md](delivery-design.md#threads)), and counts one call left to NumPy. 300,000 rows count 300,000.

## Measurements

From Python, `np.argmax(a, axis=1)` on random data, nanoseconds per row. "NumPy" is plain NumPy; "With hotkernels" is the code before this change, where every row of fewer than 128 elements reaches the slot and is passed on. The two prototype columns are loops over the rows called through `ctypes` on the same arrays, with their results compared with NumPy's: the existing assembly kernels called once per row, and a loop with one row per vector lane (four float64 rows, or eight float32 rows, per 32 bytes of vector), written with the compiler's vector types. A blank was not measured.

| Rows × columns | Type | NumPy | With hotkernels | Kernel per row | Row per lane |
|---|---|---|---|---|---|
| 1,000,000 × 2 | float64 | 7.4 | 7.8 | 5.5 | 0.41 |
| 666,666 × 3 | float64 | 11.0 | 11.5 | 9.2 | 0.65 |
| 500,000 × 4 | float64 | 13.5 | | 1.57 | 0.90 |
| 250,000 × 8 | float64 | 6.3 | | 1.57 | 1.89 |
| 300,000 × 10 | float64 | 8.3 | 8.5 | 2.29 | 2.38 |
| 125,000 × 16 | float64 | 10.1 | 10.4 | 2.39 | 4.10 |
| 31,250 × 64 | float64 | 34.2 | | 8.21 | 18.3 |
| 15,625 × 128 | float64 | 70.8 | | 16.3 | 38.0 |
| 1,000 × 1,000 | float64 | 651 | 135 | 102 | 282 |
| 1,000,000 × 2 | float32 | 6.8 | 7.1 | 5.5 | 0.42 |
| 666,666 × 3 | float32 | 10.3 | 10.8 | 9.0 | 0.60 |
| 500,000 × 4 | float32 | 13.1 | | 11.6 | 0.73 |
| 250,000 × 8 | float32 | 19.5 | | 1.57 | 1.42 |
| 300,000 × 10 | float32 | 22.1 | 22.1 | 1.58 | 1.77 |
| 125,000 × 16 | float32 | 10.1 | 10.7 | 1.56 | 3.12 |
| 31,250 × 64 | float32 | 31.7 | | 4.01 | 14.4 |
| 1,000 × 1,000 | float32 | 422 | 82 | 49 | 245 |

`np.argmin` took the same times as `np.argmax` within a few per cent wherever both were measured.

What the table says:

- **NumPy pays 6 to 8 ns a row whatever the row.** That is the call through the table and the function's own entry; the row's elements cost little until rows are long.
- **The kernels already beat it per row from four float64 or eight float32 elements.** Below one vector's worth they run a scalar path, which takes 5 to 12 ns.
- **One row per lane wins on narrow rows** and loses on wide ones, where it walks a column at a time while the kernel reads a row a vector at a time. The crossover is near eight columns for both types. The thresholds are set again on the built code; see [Thresholds](#thresholds).
- **A loop that branches on each comparison loses to NumPy on random data.** The loop of `naive.c`, one row after another, took 11.5 ns a row on rows of ten float64 (against NumPy's 8.3), from mispredicted branches. A row-at-a-time loop that selects instead of branching took 6.0 ns, a 1.4x gain only.

So the answer to the issue's first question is yes: the call can be made faster than NumPy, not only freed of the hook's overhead. About 3.5x for float64 rows of ten, 14x for float32 rows of ten, and 10x to 18x for rows of two to four.

## Design

### Which hook

The function hook (`np.argmax(m, axis=1)`) and the method hook (`m.argmax(axis=1)`) of each operation take a call when:

- the array is exactly an `ndarray`, of float32 or float64 in native byte order, two-dimensional and C-ordered, with at least one column;
- `axis` is the Python int 1 or -1, by position or as `axis=`;
- `out` is absent or None (NumPy's wrapper passes `out=None` to the method), and nothing else is given: `keepdims` is NumPy's.

Zero rows are taken: the result is an empty `np.intp` array. Zero columns are NumPy's, which raises. The result is a new one-dimensional `np.intp` array of one element per row, as NumPy's is.

The function hook is not needed for the call to be served, since NumPy's wrapper reaches the method. It takes the call itself because the wrapper costs a few hundred nanoseconds, which is most of a call on a few hundred rows, as it does for `np.argmax(x)` without an axis.

The type slot stays as it is. It still serves the other routes to the same function: `axis=0`, arrays of three dimensions, strided and Fortran-ordered arrays, which NumPy copies or walks into rows before the slot sees them.

### Two paths

- **From a threshold width up, the operation's existing kernel once per row.** Nothing new is judged: the kernel is `argmax_f64` (or one of the other three) on a contiguous run of `cols` elements, which is exactly what the judge checks it on. The interpreter lock stays held, as it does where the function and method hooks call these kernels today, because the kernels read a row twice ([issue 9](https://github.com/sofeikov/hotkernels/issues/9)).
- **Below it, a loop built into the binding, one row per lane.** For each lane it keeps the best value and its column, and takes column `j` when the best value is a number and `!(x[j] <= best)`: when `x[j]` is larger, or is a NaN. A NaN therefore wins from the first one on and is never replaced, a later equal value never replaces an earlier one, and `-0.0` and `0.0` compare equal, so the first of them is kept: NumPy's rules, and `naive.c`'s. `argmin` turns the comparison round, `!(x[j] >= best)`. Rows left over after the last whole group of lanes take the same rule one at a time. The loop reads each element once, so it runs with the interpreter lock released above `HK_NOGIL_MIN_N` elements: a writer at the same time can make a row's answer wrong, not take the loop outside the matrix.

The loop is written once, with the compiler's vector types (`__attribute__((vector_size(32)))`), which clang and gcc compile to NEON on ARM64 and, inside a function marked for the kernels' level, to AVX2 on x86-64.

### Why a loop in the binding and not a kernel in `kernels/`

A kernel in `kernels/` is judged, and ships as assembly or as its naive C on both architectures. A new judged kernel for narrow rows would need:

- assembly for x86-64, which cannot be written and timed here: there is no x86-64 machine at hand, and a figure from emulation does not count;
- or its naive C on x86-64, which [release-design.md](release-design.md) rules out until the Linux wheel is built with clang, and which, as a loop that branches on each comparison, is slower than NumPy on random rows (above).

The binding already has loops of its own for calls on short arrays, where NumPy's call path is nearly all of the cost: `clip_i64`, `round_f64` and `median_f64`. A row of two to seven elements is that case, repeated. Such a loop is held to NumPy by `delivery check`, which runs on Linux x86-64 in continuous integration as well as on ARM64, and is shown in `report()` as C and not counted among the kernels.

### Thresholds

The narrowest row given to the kernel is a constant per element type, set from the built code by timing widths either side of the crossover from Python. The prototype puts it near eight columns for both types.

**On x86-64 the thresholds are ARM64's, and the speed is not measured.** The x86-64 kernels have their own short-array paths, and NumPy's per-row cost there is not known here. Correctness on x86-64 is covered by `delivery check` in continuous integration. Timing it, and setting the thresholds for that processor, is owed; see [Left for later](#left-for-later).

### Counting

Today the type slot counts every row: one Python call on 300,000 rows of ten shows as 300,000 calls left to NumPy, and one on 1,000 rows of 1,000 as 1,000 kernel calls. Both columns of `report()` then count something the user did not write, and the footnote under the table, "It is counted per NumPy call", is wrong for them.

Three ways were weighed:

| Way | For | Against |
|---|---|---|
| Count Python-level calls, in both columns | The unit is the line the user wrote, as for every other operation. A large number again means many calls | None found. A call cannot be split between served and passed rows: every row of one call has the same length |
| Report rows passed on separately, with their element count | Shows the work done in NumPy, not only the calls | A further column, for one operation. The user's question, "were my calls served?", is still answered in rows |
| Leave the counts per row, and explain in the footnote | No code | The number stays the largest in the report of both applications and still reads as a loss |

**Chosen: one count per Python call.** The method hook, which every Python-level `argmax` and `argmin` on an ndarray reaches (NumPy's function sends the call there), marks the thread as inside a call before it hands a float32 or float64 array on to NumPy. While the mark is set, the type slot records in it whether it took a kernel or passed a row on, instead of counting. When NumPy returns, the method hook counts the call once: under the kernel that took its rows, or else as left to NumPy. The mark is per thread, because NumPy runs the slot with the interpreter lock released and other threads may be in calls of their own. A slot reached by C code without the method, which no NumPy function does, still counts as it does today.

The applications and workloads reports read these counters through `report()`, so their tables change with it.

## What changes

| Where | Change |
|---|---|
| `hotkernels/src/hotkernels/_binding/` | The rows path of the function and method hooks; the loop with one row per lane; the per-thread mark and its use in the slot |
| `hotkernels/src/hotkernels/hooks.py` | Four loops built into the binding, `argmax_rows_f32` and so on, with a probe each; shown in `report()` as `  axis=1` under their operation |
| `delivery/cases.py` | Widths either side of the threshold and of the kernels' own vector sizes, every value class (NaN first, last and in between, ties, signed zeros, infinities), the spellings that take the call and those that must not, and the layouts |
| `delivery/check.py` | That `report()` counts one call for one Python call that NumPy splits into rows |
| Documents | The hook table in `delivery-design.md`, `docsite/pages/start.md`, `hotkernels/README.md`, `kernel-findings.md`, the applications findings and recorded results |

## Exit criterion

- `delivery check` passes, with the new cases.
- From Python, `np.argmax(p, axis=1)` on 300,000 × 10 float64 is faster than NumPy by about 3x, and rows of 2, 3 and 10 of both types are faster than NumPy.
- `report()` shows the `sklearn_metrics` and `mlp_train` applications with no `np.argmax` left to NumPy.
- `docsite build` succeeds.

## Left for later

- Timing the rows path on x86-64 and setting its thresholds there.
- Arrays of more than two dimensions with `axis=-1`, which still go to the type slot and NumPy row by row, now counted once.
