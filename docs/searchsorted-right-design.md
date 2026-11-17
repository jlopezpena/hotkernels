# The right side: `np.searchsorted(a, v, side='right')` and `np.digitize`'s default

This began as a proposal and is now implemented. Where the building departed from it, [What changed in the building](#what-changed-in-the-building) says so, and [Status](#status) holds the result against the exit criterion. The rest of the text stands as proposed.

It closes [issue 41](https://github.com/sofeikov/kernel-arena/issues/41), which [two-sizes-design.md](two-sizes-design.md#out-of-scope) left out of scope.

The kernel and every figure here are ARM64's, measured on an Apple M1 Pro.

## Summary

- **The gap is the common spelling.** `np.digitize(x, bins)` with its default arguments asks for `side='right'`. So does `np.searchsorted(a, v, side='right')`. Both are passed to NumPy today, while the left side is 8x to 27x faster than NumPy from Python at a million needles.
- **A second kernel, `searchsorted_right_f64`.** It has the signature of `searchsorted_f64`, its own spec and naive C, and is checked by the judge like any other. The existing kernel and its locked files do not change.
- **The assembly is the left kernel's with one comparison turned round**, `a[i] <= key` in place of `a[i] < key`, and a simpler answer for a NaN needle: the length of the array.
- **One more condition in the binding.** The method hook for `ndarray.searchsorted` sends `side='right'` to the new kernel. `np.searchsorted` and `np.digitize` reach it through the method, as they do today.
- **No change to the judge.** One gap in what it checks is recorded below and covered from Python.

## Status

Against the [exit criterion](#exit-criterion), on an Apple M1 Pro with NumPy 2.5.5:

| Criterion | Result |
|---|---|
| `judge check searchsorted_right_f64` passes | Met, with `--unlocked`: 141,872 cases |
| `judge run` reports it faster than naive C and than NumPy at every size | Met. With 1,000 needles, 13.3x to 24.7x over naive C and 4.26x to 20.7x over NumPy; with 64 needles, 3.92x to 23.7x and 2.09x to 27.7x. Load average 3.8, so indicative |
| Each mutation of the comparison is caught | Met. Ten by the judge, and the two in the long-array path by `delivery check`, once a case was added for the first step. The table is in `kernels/searchsorted_right_f64/attempts.md` |
| `delivery check` passes with the new cases | Met. 4,306 cases for `searchsorted`, up from 2,132; 77,481 in all, none failing |
| NumPy's test suite gives the same outcome | Met. 47,234 tests, none differs |
| `report()` shows the kernel installed and counting | Met: 21 of 23 kernels, 23 with `inexact=True` |
| From Python, both statements are faster than NumPy by a factor in the range of the left side's | Met, in the table below |
| `docsite build` succeeds | Met |

From Python, in one run on a machine in use (load average 5.1), so rough:

| Statement | Sorted array | Needles | NumPy | hotkernels | Faster by |
|---|---|---|---|---|---|
| `np.searchsorted(a, v)` | 1,048,576 | 1,048,576 | 145 ms | 21.5 ms | 6.73x |
| `np.searchsorted(a, v, side='right')` | 16 | 16 | 636 ns | 439 ns | 1.45x |
| `np.searchsorted(a, v, side='right')` | 4,096 | 4,096 | 253 µs | 13.4 µs | 18.9x |
| `np.searchsorted(a, v, side='right')` | 1,048,576 | 1,048,576 | 145 ms | 19.0 ms | 7.66x |
| `np.searchsorted(a, v)` | 1,000 | 1,048,576 | 61.0 ms | 2.26 ms | 27.0x |
| `np.searchsorted(a, v, side='right')` | 1,000 | 4,096 | 208 µs | 8.81 µs | 23.6x |
| `np.searchsorted(a, v, side='right')` | 1,000 | 1,048,576 | 61.3 ms | 2.27 ms | 27.1x |
| `np.digitize(x, bins)` | 1,000 | 16 | 1.64 µs | 1.49 µs | 1.10x |
| `np.digitize(x, bins)` | 1,000 | 4,096 | 210 µs | 9.75 µs | 21.5x |
| `np.digitize(x, bins)` | 1,000 | 1,048,576 | 61.4 ms | 2.31 ms | 26.5x |

Not done, and owed: the judge runs were made with `--unlocked`, because the kernel is not on `main`, and on a machine in use. A locked run on a quiet machine, a recorded `delivery time` and a row in the README's table and chart follow with the left side's ([issue 38](https://github.com/sofeikov/kernel-arena/issues/38)).

## What changed in the building

| Proposed | Built | Why |
|---|---|---|
| `delivery check` already has needles drawn from the array against the two long arrays; add them for the right side | It had them only up to 4,097 elements; the long arrays were searched for needles with NaNs. Needles drawn from the array were added there for both sides | The proposal misread `delivery/cases.py`. Without them the long-array path was unchecked on ties from Python as well, for the left side too |
| Needles drawn from the array catch a wrong comparison anywhere in the long-array path | They catch every step but the first. An array of one repeated value was added for that | The first step picks between two starting points that both reach the answer unless elements equal to the needle lie on both sides of `a[p - 1]`. The mutation run showed it |
| `np.digitize` with decreasing bins always goes to NumPy | It does when the bins are a contiguous array. Bins that are themselves a reversed view, `np.digitize(x, a[::-1])`, are contiguous again once `np.digitize` reverses them, and take the kernel | Found by the first run of the new cases. The result is NumPy's either way |
| The kernel's row in `report()` | Shown as `side='right'`, indented under `np.searchsorted` | The report's first column is 19 characters wide |
| The timing row for a call that gains nothing was `side='right'` | It is now float32 needles | The old row is served |
| Nothing about the naive C | The two sides' references are not comparable baselines | clang compiles the right side's with a branch per probe and the left side's without one. The kernel's own times are the same; its factor over naive C is not |

| Left for later | Issue |
|---|---|
| A value class in the judge that draws needles from the array. Done: [Values drawn from another argument](judge-design.md#values-drawn-from-another-argument), and the judge now catches both long-array mutations | [48](https://github.com/sofeikov/kernel-arena/issues/48) |
| A locked re-run on a quiet machine, a recorded `delivery time`, and the README's table and chart | [38](https://github.com/sofeikov/kernel-arena/issues/38) |
| A path for fewer than 64 needles against a long array, on both sides | [39](https://github.com/sofeikov/kernel-arena/issues/39) |

## What is wrong today

(This section and the ones after it describe the state before the work, as proposed.)

| Call | Asks the method for | Today |
|---|---|---|
| `np.searchsorted(a, v)`, `a.searchsorted(v)` | `side='left'` | Kernel |
| `np.digitize(x, bins, right=True)`, increasing bins | `side='left'` | Kernel |
| `np.searchsorted(a, v, side='right')` | `side='right'` | NumPy |
| `np.digitize(x, bins)`, increasing bins | `side='right'` | NumPy |
| `np.digitize` with decreasing bins | Either side, on `bins[::-1]`, which is not contiguous | NumPy |

`try_searchsorted` in `hotkernels/src/hotkernels/_binding/hook_method.c` returns to NumPy for any `side` other than `'left'`. `docsite/pages/start.md` tells users that `np.digitize` is served only with `right=True`.

## What the right side is

For each needle `key`, the result is the number of elements that are not greater than it, in NumPy's ordering for floats, where NaN is greater than everything and equal to itself:

```c
static inline bool less(double a, double b) { return a < b || (b != b && a == a); }
// side='left':   first i with !less(a[i], key)
// side='right':  first i with  less(key, a[i])
```

| Needle | `side='left'` | `side='right'` |
|---|---|---|
| Not NaN, not in `a` | The same index on both sides | |
| Not NaN, equal to elements `a[i..j]` | `i` | `j + 1` |
| `-0.0` or `+0.0` | Before every zero of either sign | After every zero of either sign |
| NaN | The index of the first NaN in `a`, or `n` | `n` |

For a needle that is not NaN the right side is the count of `a[i] <= key`. The comparison is false for a NaN element, which is right: NaNs are last and no ordinary needle goes after one. A NaN needle goes after everything, so its answer needs no search.

## Where the right side goes

Three ways to serve it were considered.

| | A second kernel | A flag on `searchsorted_f64` | Left kernel on changed needles |
|---|---|---|---|
| How | `searchsorted_right_f64`, a copy with the comparison turned | A `side` scalar; the kernel picks one of two code paths | The binding replaces each needle by the next double above it, searches left, and sets NaN needles to `n` afterwards |
| Locked files changed | None | `spec.py`, `naive.c` and the signature of an installed kernel | None |
| Checked by the judge | Yes | Yes | No: the trick lives in the binding |
| Cost per call | None | A branch per call | A pass over the needles, a buffer as long as them, and a pass over the result |
| New assembly | About 1,500 lines, of which about 215 differ | The same, inside one file | None |

**Proposed: a second kernel.**

- The flag buys nothing. The comparison is an instruction, not data, so the kernel would hold both versions of every path anyway, and the left side, which is installed and has published figures, would have to be checked and timed again.
- Changing the needles is correct (`a[i] <= key` is `a[i] < next(key)`, with `+inf` and NaN treated apart), and it is the cheapest to build. But it puts arithmetic that decides the answer where the judge cannot see it, and against a short array its two extra passes are of the size of the search itself. It stays available as a fallback if the kernel disappoints.

The cost of the choice is duplication: two files of about 1,500 lines that differ in one instruction per probe. They are kept as two files, because the judge builds and locks one `kernel.s` per kernel and a shared source would need a rule changed. An improvement to one has to be tried on the other; both `attempts.md` files say so.

## The spec and the naive C

`kernels/searchsorted_right_f64/spec.py` is the left kernel's with the oracle changed:

```python
def oracle(a, v, needles):
    """For each needle, the last index at which it could be inserted to keep `a` sorted."""
    return np.searchsorted(a, v, side="right")
```

```c
void searchsorted_right_f64(const double *a, const double *v, int64_t *out, size_t n, size_t needles);
```

`n` is the length of `a` and the swept `Size`; `needles` is a scalar that sizes `v` and `out`. The scalar sets are the same twelve: the code paths and group widths are the same, so the counts that matter are the same. The default (64 needles) and `many` (1,000) are timed.

`naive.c` is the same binary search with `less(key, a[mid])` deciding the step.

### What the judge can and cannot tell apart

The two sides differ only for a needle that equals an element (and for a NaN needle). Random needles against a random array never produce one. The judge produces them in three ways:

| Where | Needles equal to elements |
|---|---|
| Pairings of value classes (`ties` with `ties`, `equal` with `equal`, `zeros` with `zeros`), default scalar set | Up to 4,096 elements |
| Fresh random cases, any scalar set | Up to 5,000 elements |
| Everything else: larger arrays, and the fixed cases of the other eleven scalar sets | None |

So three of the four code paths are checked on ties. The fourth, for arrays of 1,048,576 elements or more, is not: a kernel that compared the wrong way there would pass. The same is true of `searchsorted_f64` today, in the other direction.

This proposal does not change the judge. It covers the gap twice:

- `delivery check` already has needles drawn from the array itself (`_hits` in `delivery/cases.py`) against arrays of 1,048,576 and 1,048,579 elements. Those cases are added for the right side, and they run in CI.
- Before the kernel is proposed it is mutated: in each path in turn, one comparison is put back to the left side's, and the judge or `delivery check` must fail. The pull request reports which caught each.

Closing it in the judge (a value class that draws one argument from another) is [issue 48](https://github.com/sofeikov/kernel-arena/issues/48). That is now done: the judge draws needles from the array at every size and in every scalar set ([judge-design.md](judge-design.md#values-drawn-from-another-argument)), and both long-array mutations fail `judge check`.

## The assembly

`kernels/searchsorted_right_f64/arm64/kernel.s` starts as `searchsorted_f64`'s. The structure, the four paths, the group widths and the register use are unchanged. What changes:

| Place | Left | Right |
|---|---|---|
| Every probe, in all four paths (`fcmgt key, a[...]`, scalar and vector) | `a[i] < key` | `fcmge key, a[...]`: `a[i] <= key` |
| The one-at-a-time search (`fcmp` then `b.mi`) | step right if `a[mid] < key` | `b.ls`: step right if `a[mid] <= key`; false when `a[mid]` is NaN |
| A NaN needle in the one-at-a-time path (`Lnan_search`) | Binary search for the first NaN | Store `n` |
| A NaN needle in a group (`Lnanfix`) | Binary search for the first NaN | Store `n` |

The invariant holds unchanged: the answer is in `[base, base + len]` and `base + len <= n`, because each step adds to `base` only what the left kernel adds, on a comparison that is true for a subset of the same sorted prefix. Every load is at an index the left kernel could also load, so the proof that it stays inside the buffer carries over, including on unsorted input ([issue 40](https://github.com/sofeikov/kernel-arena/issues/40)).

This is a first version, not a tuned one: it inherits the left kernel's choices, which were made on this processor for the same loads and the same dependency chain. It is logged as attempt 1 in `kernels/searchsorted_right_f64/attempts.md`. It also inherits the one-at-a-time path for fewer than 64 needles against a long array ([issue 39](https://github.com/sofeikov/kernel-arena/issues/39)).

## The binding

`try_searchsorted` accepts `side` absent, `'left'` or `'right'`, compared as today (an exact `str`, in place). It calls the kernel for that side, for an array of needles and for a single Python number alike. Everything else about the call is as it is: `a` one-dimensional contiguous float64, needles a contiguous float64 array of any shape or a Python number, no `sorter`.

| Call | After |
|---|---|
| `np.searchsorted(a, v, side='right')`, `a.searchsorted(v, 'right')` | `searchsorted_right_f64` |
| `np.digitize(x, bins)`, increasing bins | `searchsorted_right_f64` |
| `np.digitize(x, bins, right=True)`, increasing bins | `searchsorted_f64`, as today |
| `np.digitize` with decreasing bins, either side | NumPy, as today: it searches a reversed view |
| Any other `side`, or a `sorter` | NumPy, which raises or sorts as it does |

- **Counters.** The kernel gets its own counter, `searchsorted_right_f64`. The operation stays `searchsorted`, so calls passed on are counted where they are today.
- **Self-test.** The method hook gets a second probe: needles that are elements of the array, with `side='right'`. Its answer differs from the left side's at every needle, so a hook that reached the wrong kernel fails it.
- **`report()`.** One more row. The count becomes 21 of 23 kernels, 23 with `inexact=True`.
- **Threads and unsorted input.** As for the left side: the kernel runs without the interpreter lock for large calls, and an unsorted array gives a meaningless result inside the buffers.

## Tests

`delivery/cases.py`:

- The cases that expect no kernel for `np.searchsorted(a, v, side='right')` and `np.digitize(v, a)` expect `searchsorted_right_f64`.
- Every case the left side has is repeated for the right: each table size against each needle count, the value classes of both arguments including needles drawn from the array, the two long arrays, the spellings (`'right'` positional and by keyword, a Python number, a NaN) and the layouts.
- `np.digitize(v, a)` with decreasing bins stays a NumPy case.

`delivery/timing.py`: the row for `side='right'`, which is there to show a call that gains nothing, becomes two timed rows matching the left side's, and `np.digitize(x, bins)` is added.

## Documents that change

| Document | What |
|---|---|
| `docsite/pages/start.md` | "What gets faster": `side` no longer restricted. "Things worth knowing": the `np.digitize` caveat becomes decreasing bins only. The sample `report()` output |
| `hotkernels/README.md` | The same conditions |
| `docs/delivery-design.md` | The hook table and the conditions table |
| `docs/kernel-findings.md` | A row and a paragraph for the kernel; the count |
| `README.md`, `docsite/pages/index.md`, `docsite/pages/report/` | 22 kernels become 23, 20 installed become 21, and the counts derived from them |
| `docs/two-sizes-design.md` | The "left for later" row points here |
| `docs/x86-64-design.md` | The kernel joins the list that needs a verdict there |
| `kernels/searchsorted_f64/attempts.md` | A note that the right side is a copy |
| `release-notes/upcoming/` | One `kernel` note |

## Order of work

1. `spec.py` and `naive.c`; `uv run judge check searchsorted_right_f64 --unlocked` passes on the naive C.
2. `kernel.s`; `judge check` passes; the mutations are caught.
3. The binding, the probe, the delivery cases; `uv run delivery check` passes.
4. `uv run judge run` and `uv run delivery time`; the documents; the release note.

The kernel does not exist on `main`, so its judge runs are made with `--unlocked` and their figures are indicative until it is merged and run locked.

## Exit criterion

- `uv run judge check searchsorted_right_f64` passes, and `uv run judge run` reports it faster than naive C and than NumPy at every size.
- Each mutation of the comparison is caught by the judge or by `delivery check`.
- `uv run delivery check` passes with the new cases, and `uv run delivery numpy-tests` gives the same outcome for every NumPy test as before.
- `hotkernels.report()` shows the kernel installed and counting.
- From Python, `np.searchsorted(a, v, side='right')` and `np.digitize(x, bins)` are faster than NumPy by a factor in the range of the left side's.
- `uv run docsite build` succeeds.

## Out of scope

- `np.digitize` with decreasing bins.
- A `sorter`, and element types other than float64.
- A path for fewer than 64 needles against a long array ([issue 39](https://github.com/sofeikov/kernel-arena/issues/39)); it applies to both sides.
- A value class in the judge that draws needles from the array.
- x86-64. [x86-64-design.md](x86-64-design.md) writes the search kernels afresh.
- Locked figures on a quiet machine, with the left side's ([issue 38](https://github.com/sofeikov/kernel-arena/issues/38)).

## Decisions still open

| | Question | Proposed |
|---|---|---|
| 1 | A second kernel, a flag, or changed needles? | A second kernel |
| 2 | Its name | `searchsorted_right_f64`. The left kernel keeps its name; renaming it would touch a locked spec and every published figure |
| 3 | Two assembly files, or one source built twice? | Two files. One source needs the judge to build a kernel from a file outside its directory |
| 4 | Is `delivery check` enough for ties on the long-array path, or must the judge check them first? | Enough for now, with the mutation run recorded; the judge's gap is an issue of its own |
