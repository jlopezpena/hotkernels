# Sorting: a stable `argsort` kernel

This is a proposal, built in the same pull request: the maintainer asked for one pull request per issue, so the document is its first commit and the code follows it. Where the building departed from the proposal, [What changed in the building](#what-changed-in-the-building) says so, and [Status](#status) holds the result against the exit criterion.

It answers [issue 188](https://github.com/sofeikov/hotkernels/issues/188): two of the five applications of the [applications study](applications-findings.md) spend more of their time sorting than doing anything else, and no kernel sorts. The issue asks for a survey entry first and leaves to the maintainer whether a sort kernel belongs in the project. This document makes the case that it does, for the stable sort, and says what is left out and why.

Every figure here is ARM64's, measured on an Apple M1 Pro with NumPy 2.5.5, on a machine in use (load average 3 to 5), so the ratios are what to read.

## Summary

- **A stable `argsort` of 64-bit keys by radix sort is exact and about ten to twenty times faster than NumPy's.** NumPy sorts float64, int64 and uint64 stably with timsort: 120 ms for a million random float64. A least-significant-digit radix sort that carries each element's index with its key takes 7 ms, and returns the same array in every case, ties, NaNs and zeros of both signs included.
- **Stable is the kind that can be exact.** A stable sort has one right answer. The default kind is not stable, and which of two equal elements NumPy's quicksort puts first depends on its partitioning; no other algorithm reproduces that. So the kernels serve `kind='stable'`, `kind='mergesort'` and `stable=True`, and the default kind stays NumPy's. See [The default kind](#the-default-kind).
- **Three kernels, `argsort_f64`, `argsort_i64` and `argsort_u64`, as C.** The gain is the algorithm, and the loops that carry it are bound by memory, not by instructions. They ship as their C on both architectures, as `cast_u8_f32` does on ARM64.
- **One method hook, on `ndarray.argsort`.** NumPy 2.5 no longer sorts through the per-type function table the `argmax` kernels are hooked into, so the hook is on the method, which `np.argsort` calls.
- **One addition to the judge's spec format**: an output buffer whose capacity is a multiple of `n`, so that a kernel can be lent working memory in proportion to its input.
- **What it is worth.** The four `argsort` calls of `sklearn_metrics` are half of that application.

## What was measured

### The applications' calls

A trace of every sort NumPy ran, taken by wrapping NumPy's sort loops for the purpose (not part of the package):

| Application | Calls | Served by this design |
|---|---|---|
| `sklearn_metrics` | 4 × `argsort`, float64, 1,000,000 elements, stable (`np.argsort(y_score, kind="mergesort")` in `_binary_clf_curve`) | Yes |
| `sklearn_metrics` | 300,000 × `argsort`, float64, 10 elements, stable (one call, `axis=1`, in `top_k_accuracy_score`) | No: rows of ten |
| `trimesh_mesh` | 2 × `argsort`, uint64, 245,760 elements, stable; 1 × 1,532 elements | Yes |
| `trimesh_mesh` | 2 × `argsort`, uint64, 245,760 elements, and 2 × int64, 40,962 elements, default kind | No: not stable |
| `trimesh_mesh` | 617,463 × `sort`, int64, 2 elements, default kind (one call, `np.sort(edges, axis=1)`) | No: rows of two, a different kernel |

### NumPy against a radix sort

Stable `argsort`, NumPy against the prototype of the kernel, both called on the same array:

| Elements | Data | NumPy | Radix sort | Faster by |
|---|---|---|---|---|
| 1,000,000 | float64, uniform in [0, 1) | 124.7 ms | 6.97 ms | 17.9x |
| 1,000,000 | float64, the scores of `sklearn_metrics` | 125.0 ms | 6.99 ms | 17.9x |
| 1,000,000 | float64, normal | 122.2 ms | 9.96 ms | 12.3x |
| 1,000,000 | int64, the whole range | 128.0 ms | 7.75 ms | 16.5x |
| 1,000,000 | int64 in [0, 1000) | 102.0 ms | 1.39 ms | 73x |
| 100,000 | float64, uniform | 9.33 ms | 0.65 ms | 14.3x |
| 16,384 | float64, uniform | 1.22 ms | 0.116 ms | 10.6x |
| 4,096 | float64, uniform | 244 µs | 29.4 µs | 8.3x |
| 1,000 | float64, uniform | 37.6 µs | 9.5 µs | 4.0x |

NumPy's default `argsort` (introsort) takes 83 ms for the million float64 and 67 ms for the int64, so the radix sort is also ten times faster than the sort it does not replace.

### Where NumPy's timsort is fast

Timsort is quick on input that is already in order, and only there:

| 1,000,000 float64 | NumPy stable `argsort` |
|---|---|
| Random | 120 ms |
| Sorted | 0.98 ms |
| Reversed | 1.23 ms |
| All equal | 1.01 ms |
| Sorted, then 1,000 random elements appended | 3.07 ms |
| Sorted, then 100,000 random elements appended | 13.6 ms |
| Sorted, with 10 random pairs swapped | 6.6 ms |
| Two sorted halves | 7.4 ms |
| 1,024 sorted runs | 70.3 ms |

Merging is what is slow: two sorted halves already take as long as the radix sort takes on random data. So NumPy wins only when nearly the whole array is one run, and the binding can tell that without reading the array (below).

## The kernel

`size_t argsort_f64(const double *x, int64_t *out, size_t n)`, and the same for `int64_t` and `uint64_t` keys. It writes to `out[0..n)` the indices that sort `x` stably, and returns `n`. `out` has room for `3n` elements; the kernel uses the rest as working memory.

### Keys

Each element becomes an unsigned 64-bit key whose order is NumPy's order:

- **uint64**: the element.
- **int64**: the element with its sign bit inverted.
- **float64**: a double is a sign and a magnitude, and magnitudes order like the values they stand for. A positive number's key is 2^63 plus its magnitude and a negative number's 2^63 minus it. NumPy's comparison calls `-0.0` and `+0.0` equal, and any two NaNs, and a stable sort must keep equal elements in their original order, so they must get equal keys: both zeros have magnitude 0 and so the same key, and every NaN, whatever its sign and payload, gets the largest key there is, which also puts NaN last, where NumPy puts it.

Two elements compare equal for NumPy exactly when their keys are equal, and less exactly when the key is less. A stable sort is determined by that order, so any stable sort by key returns what NumPy's timsort returns.

### The sort

A least-significant-digit radix sort: one counting pass per digit, each stable, from the lowest digit up.

- **Only the bits that differ are sorted.** One pass over the keys ORs and ANDs them; the bits between the lowest and the highest that differ are the window. Integers in a small range have a narrow window and need one or two passes.
- **One 64-bit word per element.** A pass moves the key and the index. Once a digit has been sorted on, its bits are not needed again, so a word holds the remaining key bits above the index: for a million elements the index takes 20 bits and leaves 44 for the key. When the window is wider than that, the first pass sorts on as many low bits as must go (up to 13), reading the keys from `x` and writing packed words. When even that is not enough (full-range integers from about 8,000 elements up), the first pass writes pairs of words and the second packs them.
- **Digits of 11 bits**, 8 below 2,048 elements. All the counts are taken in one pass before the first scatter, and a pass in which every element has the same digit is skipped.
- **The last pass writes indices alone** into `out[0..n)`. The buffers are chosen by the parity of the number of passes so that it lands there without a copy.
- **Sixteen elements or fewer are sorted by insertion.**

The counts live on the stack: about 96 KB. Nothing is written outside `out`.

### What it needs of its caller

- `out` has `3n` elements and belongs to the caller of the kernel, not to the user: it is a scratch output in the judge's sense.
- `n` is below 2^26 (67 million). Above it the pairs' first two passes would need digits wider than 13 bits.

## The judge

The spec format gains one thing: `Out("out", "i64", capacity=Times(3), used=USED_RETURN, scratch=True)`, an output whose capacity is a multiple of the `Size` argument. A kernel reports `n` elements; they are compared with the oracle, `np.argsort(x, kind="stable")`; the rest of the buffer is working memory the kernel may leave in any state, which is what a scratch output already means. Nothing else in the judge changes. Value classes for float keys already include NaN, infinities, both zeros and ties.

The alternative was a second writable buffer argument. The judge supports one output per kernel, and lifting that touches the worker, the glue and the timing harness for something no other kernel needs.

## The binding

### The hook

NumPy 2.3 reached every sort through the element type's function table (`PyArray_ArrFuncs.argsort`), where a type-slot hook like `argmax`'s would have served every caller. NumPy 2.5 does not: `PyArray_ArgSort` asks the type for a sort method object held in a private structure (`NPY_DT_SLOTS(dtype)->argsort_meth`) and the table is read only by `np.lexsort`. A probe that replaced the table's entries saw `np.lexsort` and nothing else.

So the hook is a method hook on `ndarray.argsort`, the kind `cumsum` and `searchsorted` have. `np.argsort(x, kind=...)` calls the method, as do `np.unique(x, return_index=True)` and scikit-learn.

It takes the kernel when:

- the array is exactly an `ndarray`, one-dimensional, contiguous, of float64, int64 or uint64 in native byte order;
- `kind` is the string `'stable'` or `'mergesort'` with `stable` absent or None, or `kind` is absent or None and `stable` is `True`;
- `axis` is absent, -1 or 0; `order` is absent or None; `descending`, where NumPy has it, is absent, None or False;
- it has at least `ARGSORT_MIN_N` and fewer than 2^26 elements;
- it is not nearly one run already (next section).

Everything else goes to NumPy unchanged. The result is a new `intp` array. The binding allocates `3n` words, runs the kernel with the interpreter lock released, and copies the first `n` into the result.

### Input that is already in order

A run long enough for NumPy to win covers nearly the whole array, so it contains the middle element. The binding starts at the middle and walks outwards while the elements are in order (non-decreasing, or strictly decreasing). On random data that stops after a few elements. If the run it finds leaves out fewer than one element in 32, the call is NumPy's; otherwise the kernel's. The walk costs time only on input NumPy is about to sort in a millisecond.

### The default kind

`np.argsort(x)` with no `kind` is the common spelling, and it is not served. On an array with no two equal elements every correct sort returns the same permutation, so the kernel's answer would be NumPy's; with ties it is a correct sort and not NumPy's permutation. Two ways to serve it exist and neither is in this proposal:

1. **Sort, then verify there were no ties**, and hand the call to NumPy if there were. Exact, and a loss of the radix sort's time (about 8% of NumPy's) on every array with ties that a cheap test beforehand does not catch.
2. **Under `install(inexact=True)`**, without verification. The kernel is then exact for one kind and inexact for another, which `report()` has no way to say.

Both are in [issue 207](https://github.com/sofeikov/hotkernels/issues/207). `np.sort(x, kind='stable')` and `np.lexsort`, which still reads the type's table and passes an index array that is not in order, are in [issue 208](https://github.com/sofeikov/hotkernels/issues/208), and the rows of two that `trimesh` sorts in [issue 209](https://github.com/sofeikov/hotkernels/issues/209).

## Exit criterion

1. `uv run judge selftest` passes with the spec format's addition.
2. `uv run judge check` passes for the three kernels, and `uv run judge run` reports each faster than NumPy from 1,024 elements up.
3. `uv run delivery check` passes with cases for `argsort`: every kind and spelling, ties, NaNs, both zeros, sorted and nearly sorted input, sizes either side of each limit.
4. `uv run delivery time` shows no `argsort` statement slower than NumPy at any size it measures.
5. `sklearn_metrics` runs measurably faster with hotkernels than without.
6. `report()` shows the three kernels and counts their calls.

## Status

To be filled in when built.

## What changed in the building

| Proposed | Built | Why |
|---|---|---|
| The key of a negative float is its bits inverted | It is 2^63 minus its magnitude | Inverting turns the zero low bits of a whole number into ones, so whole numbers of both signs had keys that differ in all 64 bits: 15.2 ms for a million of them, against 4.7 ms with the subtraction. It also gives both zeros one key without a test |
| The binding walks out from the middle element to find one long run | It counts falls from the start, at most eight, and then measures the run that begins at the last fall in the first thirty-second of the array | The walk from the middle missed the commonest ordered input: a sorted array with one pair of elements swapped has no long run through its middle, takes NumPy 1.2 ms and the kernel 7. And a comparison per element cost 0.5 ms per million on sorted input, half of NumPy's whole time; the scan now passes over 64 elements at a time and costs 0.16 ms |
| Input already in order is NumPy's | A sorted array is answered `0, 1, 2, ...` by the hook, and a strictly descending one `n-1, ..., 0` | Once the scan has reached the end without a fall the answer is known. From Python a million sorted float64 take 0.5 ms, against NumPy's 1.0 ms |
| The kernel reads the caller's array | It reads a copy the binding makes: four words of memory per element, not three | The kernel reads its input three times and sizes the places it scatters to from the first reading. An array that changes in between can send a write outside its buffers. `argmax` has the same exposure and keeps the interpreter lock; a sort of a million elements is too long to hold it, and a copy made under the lock is safe against every writer for about a fiftieth of the sort |
| `ARGSORT_MIN_N`, to be measured | 256 | From Python the kernel was level with NumPy at 128 elements (0.95x to 1.30x by element type), 1.2x to 1.6x faster at 192 and 1.6x to 1.95x at 256. At 64 and 96 it took up to 1.6 times as long: NumPy sorts so few by insertion |
| `descending` absent, None or False | Absent | NumPy 2.3, the oldest the package installs on, has no such parameter and raises for it. A call one supported NumPy rejects is not the hook's to take |
| The counts take about 96 KB of stack | 112 KB | Two tables for digits of up to 13 bits and six for 11 |
| Each attempt on a kernel is judged by `uv run judge compare` | The attempts were timed in a harness outside the repository, and the result by `uv run judge run` | `compare` builds a `kernel.s` at two revisions; a kernel that is its C has none. The logs say which figures are which |
| Nothing about the Linux wheels | The argsort kernels ship as C on x86-64 and need the wheel build to use clang | [release-design.md](release-design.md#more-than-one-platform) allows gcc there only while no kernel ships as C on x86-64. The change is made by the pull request for [issue 189](https://github.com/sofeikov/hotkernels/issues/189), whose `unique_i64` is the first such kernel and which this one is stacked on |
| Three follow-ups in one issue | Three issues | The default kind ([207](https://github.com/sofeikov/hotkernels/issues/207)), `np.sort(x, kind='stable')` and `np.lexsort` ([208](https://github.com/sofeikov/hotkernels/issues/208)), and sorting short rows, which the trace found in both applications ([209](https://github.com/sofeikov/hotkernels/issues/209)) |

Tried and not kept, in `kernels/argsort_f64/arm64/attempts.md`: the scatter loop of a pass in assembly, which tied the compiler's (1.00x to 1.03x), and ranking the sign and exponent bits of a float to narrow the window, which made data of one sign 5% to 19% slower for a gain of 5% to 11% on some sizes of data of both.

Not built: a sort for 17 to 255 elements. The kernel sorts 16 or fewer by insertion and is 16 times faster than a NumPy call there, but the binding does not take arrays under 256, and between the two sizes the kernel has nothing NumPy's insertion sort lacks.

## Documents that change

`README.md` (the count of kernels, the table of speeds), `hotkernels/README.md`, `docsite/pages/start.md`, the hook table in `docs/delivery-design.md`, `docs/kernel-findings.md`, `docs/judge-design.md` and `kernels/README.md` (the spec format), `docs/survey-design.md` where it lists the candidates, and the applications findings where they say no kernel sorts.
