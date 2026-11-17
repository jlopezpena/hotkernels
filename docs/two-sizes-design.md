# Two sizes: `np.searchsorted` and `np.interp` served by their kernels

This began as a proposal and is now implemented. Where the building departed from it, [What changed in the building](#what-changed-in-the-building) says so, and [Status](#status) holds the result against the exit criterion. The rest of the text stands as proposed.

It closes [issue 16](https://github.com/sofeikov/hotkernels/issues/16): `searchsorted_f64` and `interp_f64` exist, pass the judge and are faster than NumPy, and no call ever reaches them. It follows from "What the kernels need first" in [delivery-design.md](delivery-design.md#what-the-kernels-need-first).

The kernels and every figure here are ARM64's, measured on an Apple M1 Pro.

## Summary

- **The problem is one argument.** Both kernels take a single length and use it for the sorted array and for the values searched for. A real call has two lengths, so the binding has nothing it can hand them.
- **The judge needs one small extension, not a new concept.** A spec may already size an output buffer by an integer scalar argument. This proposal lets it size an input the same way. The second length becomes a scalar, and scalar sets, which exist, vary it independently of the first, check each value and time the ones marked.
- **The assembly changes are small.** Both kernels already choose their code path from the length of the sorted array and count their loop in needles; the two numbers just arrive in one register today. About fifteen instructions change in each.
- **Two hooks.** `ndarray.searchsorted` as a method hook, which `np.searchsorted` and `np.digitize` reach, and `np.interp` as a function hook.
- **One thing the binding must screen for.** A NaN among `np.interp`'s sample points makes the kernel read before its buffer ([issue 9](https://github.com/sofeikov/hotkernels/issues/9)). The binding reads `xp` once before the kernel sees it.

## Status

Against the [exit criterion](#exit-criterion), on an Apple M1 Pro with NumPy 2.5.5:

| Criterion | Result |
|---|---|
| `judge selftest` passes, including the new fixture | Met. `rank_u8` passes, and both of its broken variants are caught for the reason they were built to trigger |
| `judge check` passes for both kernels with the new specs | Met, with `--unlocked`: `searchsorted_f64` 141,872 cases, `interp_f64` 155,063. All 22 kernels pass under the changed judge |
| `delivery check` passes with the new cases | Met. 2,132 cases for `searchsorted` and 3,144 for `interp`; 75,307 in all, none failing |
| NumPy's test suite gives the same outcome | Met. 47,234 tests, none differs, with both hooks installed |
| `report()` shows both kernels installed and counting | Met: 20 of 22 kernels, 22 with `inexact=True` |
| From Python, at a million needles against a million elements, both are faster than NumPy by a factor in the judge's range | Met: `np.searchsorted` 7.76x, `np.interp` 30.5x. The judge's earlier figures in that shape were 8.84x and 22.7x |

From Python, on a machine in use (load average 3.9), so rough:

| Statement | Sorted array | Needles | NumPy | hotkernels | Faster by |
|---|---|---|---|---|---|
| `np.searchsorted(a, v)` | 16 | 16 | 569 ns | 418 ns | 1.36x |
| `np.searchsorted(a, v)` | 4,096 | 4,096 | 255 µs | 13.4 µs | 19.1x |
| `np.searchsorted(a, v)` | 1,048,576 | 1,048,576 | 148 ms | 19.1 ms | 7.76x |
| `np.searchsorted(a, v)` | 1,000 | 4,096 | 207 µs | 8.79 µs | 23.5x |
| `np.searchsorted(a, v)` | 1,000 | 1,048,576 | 60.7 ms | 2.24 ms | 27.2x |
| `np.interp(x, xp, fp)` | 16 | 16 | 592 ns | 109 ns | 5.44x |
| `np.interp(x, xp, fp)` | 4,096 | 4,096 | 256 µs | 13.3 µs | 19.2x |
| `np.interp(x, xp, fp)` | 1,048,576 | 1,048,576 | 571 ms | 18.7 ms | 30.5x |
| `np.interp(x, xp, fp)` | 1,000 | 4,096 | 216 µs | 11.2 µs | 19.3x |
| `np.interp(x, xp, fp)` | 1,000 | 1,048,576 | 55.6 ms | 2.91 ms | 19.2x |

Not done, and owed: every judge run here was made with `--unlocked`, because the work changes locked files, and on a machine in use. The published figures for these two kernels have not been replaced. A locked run on a quiet machine, a recorded `delivery time`, and the README's chart follow once this is in `main` ([issue 38](https://github.com/sofeikov/hotkernels/issues/38)).

## What changed in the building

| Proposed | Built | Why |
|---|---|---|
| The fixture's broken variants read the input using `n`, and write `n` outputs | One reads a byte past the scalar-sized input; one handles the default count whatever the scalar says | A variant that confuses the two lengths fails on the first case, where `n` is 0, as a wrong result whichever way the judge sized the buffer. These two are caught only if the buffer is placed by the scalar's value and the other sets are run |
| `np.interp` takes the kernel for fewer than 2^32 sample points | No upper limit | The kernel already sends a longer table down its one-at-a-time path |
| The needle of `searchsorted` may be a Python float | Or a Python int that a double holds exactly | `np.searchsorted(a, 3)` is as common, and NumPy converts the int the same way |
| Nothing about what a call costs when it is passed on | Keyword names are compared in place instead of through a library call | `side='right'` and `left=` reach the hooks' argument matching before being passed on, which cost about 35 ns per call. It is now 10 to 25 ns, on calls of 300 ns and up. A call rejected by its element type still costs under 2 ns |
| Check time not discussed | `judge check searchsorted_f64` takes 22 seconds, down from about two minutes, with three times the cases | Reusing prepared inputs, which the proposal introduced to keep the scalar sets affordable, removed most of what the check cost before |

| Left for later | Issue |
|---|---|
| A locked re-run of both kernels on a quiet machine, a recorded `delivery time`, and the README's chart | [38](https://github.com/sofeikov/hotkernels/issues/38) |
| A path for fewer than 64 needles against an array of 64 or more, which go one at a time | [39](https://github.com/sofeikov/hotkernels/issues/39) |
| A judge check that the search kernels stay inside their buffers on unsorted input | [40](https://github.com/sofeikov/hotkernels/issues/40) |
| `side='right'` for `np.searchsorted`, which `np.digitize` asks for by default | [41](https://github.com/sofeikov/hotkernels/issues/41), done since: [searchsorted-right-design.md](searchsorted-right-design.md) |

## What is wrong today

```c
void searchsorted_f64(const double *a, const double *v, int64_t *out, size_t n);
void interp_f64(const double *x, const double *xp, const double *fp, double *out, size_t n);
```

(This section and the ones after it describe the state before the work, as proposed.) `n` is the length of `a` and of `v`; of `xp`, `fp` and of `x`. `np.searchsorted(a, v)` with a table of 1,000 values and a million needles cannot be expressed. `hotkernels.report()` lists both kernels as "not bound".

The cause is in the judge. A spec has one `Size`, and `judge/spec.py` rejects an input whose length is anything else ("the length of input ... cannot be a scalar"). The kernels were written to the only shape the judge could check.

## The judge

### An input may take its length from a scalar

An output's capacity can already be the name of an integer `Scalar` argument; `histogram_f64` sizes its counts that way. The same is allowed for an input:

```python
In("v", "f64", len="needles"),
Out("out", "i64", capacity="needles"),
Size("n"),                             # the length of a, as before
Scalar("needles", "usize", 64),
```

The check that forbids it exists because inputs are generated before the scalar values are looked up. That is an ordering inside `build_inputs`, not a property of the design: a case already names its scalar set, so the values are known when the case is. `build_inputs` and the timing inputs take the scalar values and pass them to `element_count`, which already resolves a scalar name.

Rules for a scalar used as a length: it is an integer scalar, and every value it takes in any scalar set is zero or more. An `InOut` buffer keeps its fixed length; nothing needs it otherwise.

### Why not a second `Size`

Considered: a second size argument with its own sweep, a new field in the case description, and a strategy for pairing the two sizes.

| | A scalar as a length | A second `Size` |
|---|---|---|
| New concepts in the spec | None | A second size, how the two are paired in checking, which one timing sweeps |
| Case descriptions | Unchanged: `n=1024 arr=end seed=1 scalars=many v=nan_at@3` | A new field, and a parser that tells it from a value class |
| Timing harness | Unchanged: scalars already reach the kernel | A second count through the generated C loop |
| Values the second length takes | The ones the spec lists | Any |

The cost of the first column is the last row: the second length is checked at the values the spec names, not at every value. For these kernels that is the right trade. The second length only decides how many groups the loop runs and how the leftover is handled, and the groups are 16, 64 and 128 wide; the interesting values are the ones around those, and a spec can list them. The first length, which selects the code path and the search depth, keeps the full sweep. The README already says of scalars that they "are verified for the values each spec lists"; that sentence now covers these two lengths as well, and the specs' lists are written to make it a small statement.

`judge/spec.py` describes itself as "the smallest thing that covers the kernels we have". This is that.

### Which length is the `Size`

The length of the sorted array: `a` for `searchsorted`, `xp` and `fp` for `interp`.

- It selects the code path (four in `searchsorted`, three in `interp`) and sets the depth of the search. Every path boundary and every leftover count of the search has to be checked, and the `Size` is what is tried at every value from `min_n` to 65 and at every power of two with its neighbours.
- It is what makes a timing sweep informative: the same number of needles against a table that fits the first-level cache, then the second, then none.

The number of needles (queries, for `interp`) is the scalar. The default set gets the full sweep of value classes, memory arrangements and positions. Each further set gets the thinner sweep every scalar set gets.

### Three changes that follow

1. **Positions come from the buffer's own length.** A positional value class (`nan_at@p`) is tried at every index when the case has at most 65 elements. Today the indices are taken from `n`. For a buffer sized by a scalar they are taken from that buffer's length, so a NaN needle is tried at every index of a 64-needle group whatever the table's length.
2. **The pool of timing inputs is sized by the largest buffer.** At small sizes the timing loop cycles through several inputs, so the processor cannot learn one input's branches. The number of inputs is chosen from `n` so that all of them fit the first-level cache. A table of 16 elements with 1,000 needles would get 64 inputs; the count is taken from the largest buffer of the call instead.
3. **Prepared inputs are kept while the size does not change.** `prepare=np.sort` runs for every case, and most cases of one size share the same sorted array: only the other argument's value class changes. The check keeps the last prepared array per argument and reuses it when the recipe is the same. To make that effective, the fixed cases are ordered by size first and scalar set second; the set of cases is the same. Without this, every scalar set would sort several four-million-element arrays again.

### What the judge's self-test gains

A fixture kernel with a scalar-sized input, and deliberately broken versions of it that the judge must reject: one that reads the input using `n` instead of the scalar (caught by the guard page when the scalar is smaller), one that writes `n` outputs.

## The specs

Both are locked files and change in the pull request that changes the judge.

### `searchsorted_f64`

```c
void searchsorted_f64(const double *a, const double *v, int64_t *out, size_t n, size_t needles);
```

`n` is the length of `a`, from 0. `needles` is the length of `v` and of `out`.

| Scalar set | `needles` | Why |
|---|---|---|
| default | 64 | One full group of the widest path that works in cache. Full sweep; timed |
| `none`, `one` | 0, 1 | The empty call, and a single needle |
| `k15`, `k16`, `k17` | 15, 16, 17 | Either side of the first group width, and one leftover |
| `k63`, `k65` | 63, 65 | Either side of the second |
| `k127`, `k128`, `k129` | 127, 128, 129 | Either side of the third |
| `many` | 1,000 | Several groups and a leftover in every path (1,000 leaves 8, 40 and 104). Timed |

### `interp_f64`

```c
void interp_f64(const double *x, const double *xp, const double *fp, double *out, size_t n, size_t queries);
```

`n` is the length of `xp` and `fp`, from 1 (NumPy raises on empty sample points). `queries` is the length of `x` and of `out`. The correctness cap on `n` stays at 65,537.

| Scalar set | `queries` | Why |
|---|---|---|
| default | 64 | One full group. Full sweep; timed |
| `none`, `one` | 0, 1 | |
| `k15`, `k16`, `k17`, `k63`, `k65` | | Either side of both group widths |
| `many` | 1,000 | Timed |

### What is timed

Each kernel is timed twice across the sweep of `n`: 64 needles and 1,000 needles per call. `compare` therefore measures both, and a change that speeds one up at the other's expense shows as such.

The figures published so far (8.84x and 22.7x over NumPy "at 1,048,576 elements") were taken with both lengths equal to a million. That shape is no longer timed. The findings and the README are re-measured in the new shape and say which two numbers a figure was taken at.

## The assembly

### `searchsorted_f64`

The path is chosen by the table's length and needs a minimum number of needles, because leftovers are handled by redoing the last full group:

| Path | Table length | Needles at once | Used when |
|---|---|---|---|
| `Lscalar` | any | 1 | Fewer needles than the path below needs |
| Sixteen | 16 to 63 | 16 | `needles >= 16` |
| `Lv64` | 64 to 2^32 - 1 | 64 | `needles >= 64`, and the table is under 1,048,576 or `needles < 128` |
| `Lbig` | from 1,048,576 | 128 | `needles >= 128` |

`Lv64` keeps its indices in 32-bit lanes and steps through the search with a loop, so it serves any table up to 2^32 elements; today it is only entered below 1,048,576 because the large path is better there. With fewer than 128 needles on a large table it is the best path available.

What changes: the three places that compute "the last full group" from `n` use `needles`; `Lsmall` loops over `needles`; the entry tests both numbers. Nothing in a search loop changes.

**Left for a tuning run:** fewer than 64 needles against a table of 64 or more go one at a time through `Lscalar`. That is correct and no slower than NumPy's own search, and a call that small is mostly call overhead. A path for it is a kernel run's job, under the one rule, once the spec is in `main`.

### `interp_f64`

The same shape: the search stage is chosen by the table (16 queries at once under 64 sample points, 64 from there), and each needs that many queries. Fewer go to `Lslow`, which is `naive.c` line by line. One sample point writes `fp[0]` for every query.

## The binding

### `searchsorted`

A method hook on `ndarray.searchsorted`, which uses the fast calling convention the method hook already supports. `np.searchsorted(a, v)` calls the method, and so does `np.digitize`.

The kernel takes `a.searchsorted(v, side, sorter)` when:

- `a` is a plain one-dimensional contiguous float64 array;
- `v` is a plain contiguous float64 array of any shape, or a Python float or int. The result has `v`'s shape; for a Python float it is the `np.intp` scalar NumPy returns;
- `side` is absent or `'left'`, and `sorter` is absent or `None`.

`np.digitize(x, bins)` asks for `side='right'` by default and stays with NumPy. `np.digitize(x, bins, right=True)` on increasing bins asks for `'left'` and is served.

### `interp`

A function hook on `np.interp`. The kernel takes `np.interp(x, xp, fp)` when:

- `x` is a plain contiguous float64 array of any shape; `xp` and `fp` are plain one-dimensional contiguous float64 arrays of the same length, at least 1;
- `left`, `right` and `period` are absent or `None`;
- `xp` holds no NaN (below).

### Inputs outside the kernels' contracts

| Input | What the kernel does | What the binding does |
|---|---|---|
| A NaN in `xp` ([issue 9](https://github.com/sofeikov/hotkernels/issues/9), item 1) | At `xp[0]` it reads `xp[-1]` and `fp[-1]`. Elsewhere it returns something other than NumPy, whose answer then depends on which elements its own search happens to probe | Reads `xp` once. Any NaN sends the call to NumPy |
| `xp` not increasing, or `a` not sorted | Stays inside the buffers: every probe index is bounded by the length, whatever the data. The result is as meaningless as NumPy's and is not the same meaningless result | Nothing. NumPy documents both functions for sorted input only and does not check; a check here would cost a pass over the table on every call |

**The cost of reading `xp`.** About 0.13 ns per sample point, measured: 130 µs for a million. NumPy's `interp` takes 150 to 500 ns per query on a table that size. So the pass is noise when there are many queries and the whole cost when there are few. The kernel is taken only when `len(xp) <= 64 * len(x)`; with fewer queries than that the call goes to NumPy, which is not slow there in absolute terms. Confirmed: with 1,000 sample points and 16 queries, right at the limit, the call is 3.8x faster than NumPy from Python.

`searchsorted` needs no pass: a NaN in `a` is ordered last by NumPy and by the kernel, and the judge checks it.

**The last row is not verified by the judge.** The specs sort their input, so the judge never runs these kernels on unsorted data. The bound is by construction (the search adds to an index only amounts that keep it inside `[0, n]`). A value class that skips `prepare` and checks memory safety only would close this; it is filed as [issue 40](https://github.com/sofeikov/hotkernels/issues/40) and is not part of this work. Outside the judge, the built kernels were run 2,952 times on unsorted arrays (random, descending, with NaNs, with infinities; lengths from 0 to 1,048,577; 0 to 1,000 needles) with every buffer against a guard page at its end and at its start: no fault, and every index `searchsorted_f64` returned was inside `[0, n]`.

### Threads

Both hooks release the interpreter lock above the usual threshold, counted in needles. The kernels read their inputs once and write only the output the binding allocated.

### Self-test, report and tests

- Each hook gets a probe in `hotkernels/hooks.py`: one small call with a known answer, as for every other hook.
- `report()` lists both kernels as installed, with their counters. The count it prints becomes 20 of 22, and 22 with `inexact=True`.
- `delivery check` gets cases for both: equal and unequal lengths either way round, the boundary counts above, NaN needles and NaN queries, infinities, repeated sample points, a NaN in `xp` (expected to go to NumPy), `side='right'`, a `sorter`, `left=`/`right=`/`period=`, strided and float32 inputs (all expected to go to NumPy), a Python float needle, `np.digitize` both ways.
- `delivery time` gets a statement for each.

## Order of work

1. **Judge and specs** (one pull request, locked files): the spec extension, the three changes that follow, the self-test fixture, the two specs and their `naive.c`, and the two assembly kernels made to match, since a spec and a kernel that disagree cannot both be in `main`. `judge selftest` and `judge check` for both kernels pass. This pull request changes locked files, so its runs are made with `--unlocked` and its timings are indicative only.
2. **Binding** (stacked on 1): the two hooks, probes, `delivery check` cases, timing, the user-facing pages and the release note.
3. **After both are merged:** a locked `judge run` of both kernels on a quiet machine for the published figures, and a tuning run for the short-call path.

## Exit criterion

- `uv run judge selftest` passes, including the new fixture.
- `uv run judge check searchsorted_f64` and `interp_f64` pass with the new specs.
- `uv run delivery check` passes with the new cases, and `uv run delivery numpy-tests` gives the same outcome for every NumPy test as before.
- `hotkernels.report()` shows both kernels installed and counting.
- From Python, at a million needles against a table of a million, both statements are faster than NumPy by a factor in the range the judge reports for the kernel alone.

## Out of scope

- `side='right'` for `searchsorted`, which needs a second comparison in the kernel, and with it `np.digitize`'s default.
- Element types other than float64.
- `left`, `right` and `period` for `interp`.
- A faster path for a handful of needles against a large table.
- x86-64. [x86-64-design.md](x86-64-design.md) writes these kernels afresh; they are written to the new specs.

## Decisions still open

| | Question | Proposed |
|---|---|---|
| 1 | Is "checked for the needle counts the spec lists" enough, or must the second length be swept like the first? | Enough. The lists cover both sides of every group width, and fresh random cases draw from them on every run |
| 2 | Should `np.interp` with few queries on a large table take the kernel after a pass over `xp`? | No: only when `len(xp) <= 64 * len(x)` |
| 3 | Should unsorted input be detected? | No. NumPy does not, and the kernels stay inside their buffers |
