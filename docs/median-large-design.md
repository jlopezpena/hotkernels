# `np.median` of a large `float64` array

This is a proposal, written before the code. It answers the `np.median` part of [issue 220](https://github.com/sofeikov/hotkernels/issues/220), found by the second applications study ([applications-second-findings.md](applications-second-findings.md)): in `astropy_timeseries`, NumPy's `introselect` under `np.median` is 17.6% of the program on the large size, and bottleneck's `nanmedian` another 15.9%.

Every figure here is ARM64's, measured on an Apple M1 Pro with NumPy 2.5.5, from Python. The machine also hosts the project's CI runners, so the load average is given with each table; the ratios are what to read.

## Summary

- **hotkernels answers `np.median(x)` only up to 128 elements** today, with a loop of the binding's own that sorts a copy on the stack (`median_f64`, [delivery-design.md](delivery-design.md#calling-the-kernel-above-numpys-wrapper)). Above that NumPy partitions a copy with `introselect`: 10.1 ms on 1,048,576 normally distributed elements, where `np.mean` takes 0.19 ms.
- **Extend the same loop to arrays of any size** with an exact selection on the bits of the elements: map each double to an order-preserving 64-bit integer key, count the keys by the eleven bits below the bits they all share, keep only the elements of the digit that holds the middle, and go on among those. That is three reads of the array and work on a small buffer.
- **A throwaway probe of the selection, in C called from Python**, took 1.14 ms on the same million elements: 8.8x. On an astronomical image like Astropy's it was 7.0x, on ten distinct whole numbers 4.8x. It gave NumPy's answer bit for bit on every array it was tried on.
- **On data already in order NumPy is fast too,** about 1 ms at a million elements, because `introselect` picks good pivots there; the probe is level with it (0.8x to 1.3x). This is the one place the proposal does not gain.
- **The same call and guards, one more spelling.** `np.median(x)` and now `np.median(x, axis=None)`, which Astropy writes, on an exactly-`ndarray` contiguous float64 array of any shape and at least one element, with no NaN. A mean of the two middle elements that NumPy would warn about, and a zero median beside a negative zero, go to NumPy as today.
- **Memory: 64 KB of counters and a buffer of the candidates,** at most eight bytes an element and usually under a few per cent of that, where NumPy's own call copies the whole array. A call that cannot get it goes to NumPy. The interpreter lock is released from 8,192 elements; the selection is written so that a thread writing the array meanwhile cannot make it write outside its buffers.
- **Left for later:** bottleneck's `nanmedian` and `np.quantile`/`np.percentile`, the other two parts of #220, and `np.median` along an axis.

## What is slow today

From Python, NumPy 2.5.5, minimum of seven batches, normally distributed values unless the row says otherwise; load average 2.9:

| Elements | `np.median(x)` | `np.mean(x)` | `np.sort(x)` |
|---|---|---|---|
| 1,000 | 7.5 µs | 1.6 µs | |
| 100,000 | 916 µs | 20.4 µs | |
| 1,048,576 | 10.1 ms | 0.19 ms | 37.7 ms |
| 1,048,576, `normal(100, 5)` with 2% bright stars | 9.4 ms | 0.19 ms | |
| 1,048,576, ascending | 1.12 ms | 0.19 ms | |

## How NumPy answers it

From `numpy/lib/_function_base_impl.py`, the same in 2.3, 2.4 and 2.5:

1. `np.median(a, axis=None, out=None, overwrite_input=False, keepdims=False)` is a function object made by `array_function_dispatch`, which the function hook already changes. It calls `_ureduce`, which calls `_median`.
2. `_median` sets the partition indexes: `[n//2 - 1, n//2]` for an even count, `[(n-1)//2]` for an odd one, and for a float array also `-1`, so that a NaN, which sorts last, is found at the end.
3. Without `overwrite_input` it calls `np.partition(a, kth)` with `axis=None`, which flattens the array into a new copy and partitions the copy with `introselect`.
4. It takes `np.mean` of the middle one or two elements, and `_median_nancheck` replaces the result with NaN, and warns, if the last element of the partition is a NaN.

So the result is a function of the values alone: the middle element of the sorted values, or NumPy's mean of the two middle ones, `(a + b) / 2` computed by `np.add.reduce` and a true division. Any algorithm that finds the same two values gives the same bits, provided the arithmetic after it is NumPy's. The short-array loop already relies on this, and its guards (below) are the cases where that arithmetic raises a flag NumPy would report, or where NumPy's own partition decides which of two zeros, equal as numbers, it returns.

`np.median(x)` and `np.median(x, axis=None)` are the same call: `axis` defaults to None.

## What Astropy calls

In `astropy_timeseries` the large medians come from `astropy.stats` 8.0.3:

| Caller | Call | Array |
|---|---|---|
| `biweight_location`, `biweight_scale` | `median_func(data, axis=axis)`, where `median_func` is `np.median` for an ndarray without `ignore_nan`, and `axis=None` is passed by keyword | each 1,024 × 1,024 float64 image, contiguous |
| `median_absolute_deviation`, which both call | `func(data, axis=axis)` and `func(np.abs(data - data_median), axis=axis)`, with `func` `np.median` and `axis=None` by keyword | the image, and its absolute deviations |
| `mad_std` of each light curve | the same, through `median_absolute_deviation` | 5,000 float64 |

So every one is spelt with `axis=None`, which the hook does not take today even on 128 elements. With bottleneck installed, as here, `median_absolute_deviation` does not pass `overwrite_input=True`; without bottleneck it does, and that spelling stays NumPy's (below). `sigma_clipped_stats` calls bottleneck's `nanmedian`, which is not `np.median` and is out of scope.

## Where the call is caught

The existing function hook on `np.median`, `call_median` in `hook_function.c`. It takes a call when all of these hold, and passes every other call to NumPy unchanged:

- The arguments are the array alone, by position, or the array by position and `axis` as `None`, by position or by keyword. `a=` by keyword, any other `axis`, `out`, `overwrite_input` and `keepdims`, even at their defaults, are NumPy's. `overwrite_input=True` in particular lets NumPy reorder the caller's array, which a call answered here would not do, and the caller can see the difference.
- The array is exactly an `ndarray` of float64 in native byte order, C-contiguous, of any number of dimensions (NumPy flattens it), with at least one element and fewer than 2^32 (the selection counts in 32 bits).
- It holds no NaN.
- The mean of the two middle elements raises no flag NumPy reports (`median_raises`, unchanged): a sum that overflows, infinities of both signs, or a half that underflows. NumPy warns or raises for those by `np.errstate`.
- The median is not a zero in an array holding a negative zero (`median_has_negative_zero`, unchanged): which zero NumPy's partition leaves in the middle depends on where it moves equal elements.

Up to 128 elements the middle elements are found as today, by sorting a copy on the stack; the limit is to be timed again against the selection. Above it, by the selection below. Either way the result is the middle element, or the two added and divided by two in NumPy's order with the precision flag cleared afterwards, as an `np.float64` from `hk_scalar`, and the call is counted under `median_f64`. Every call not taken, for whatever reason, is counted by `hk_pass(HK_OP_MEDIAN)` as today, so `report()` keeps its one row, `np.median` on float64, `C`, `exact`.

## The selection

**Keys.** For the bits `b` of a double, the key is `b` with every bit flipped when the sign bit is set, and with only the sign bit set otherwise. As unsigned integers the keys are in the order of the numbers: `-inf` lowest, `+inf` highest, `-0.0` just below `+0.0`, and the NaNs beyond the infinities. That is the order NumPy's partition uses for everything but the two zeros, which compare equal there; a zero median beside a negative zero is NumPy's anyway, and for any other median the order of the zeros between themselves does not change which values are in the middle. The argsort kernels order floats by the same key ([sort-design.md](sort-design.md)).

**Passes.**

1. **Range.** One read of the array for the smallest and the largest key, in vectors (two lanes of 64 bits on ARM64, four on x86-64), with no branch on the data. A key below that of `-inf` or above that of `+inf` is a NaN, and the call is NumPy's: found with integer operations, so a signalling NaN raises no flag.
2. **Count.** The bits above the highest bit where the smallest and largest keys differ are the same in every key. The elements are counted by the eleven bits just below them: 2,048 counters of 32 bits. The keys and digits of each block of eight elements are computed in vectors, and the block is counted one element at a time into eight sets of counters, element `i` of the block into set `i`, which are added together afterwards. Real data puts most elements in a few digits, and data in order puts them in one digit at a time; with one set, each increment would wait for the previous one to the same counter. Walking the counters gives the digit that holds each of the two middle ranks.
3. **Gather.** When both ranks fall in one digit, the elements of that digit are copied, as keys, to a buffer as long as its count. A sparse digit is gathered by testing a block's digits in vectors and looking at its elements only when one of them is in the digit; a dense one (more than one element in 64) by a compaction without a branch, which would otherwise be mispredicted on every other element. The selection goes on among the candidates, whose keys share eleven more bits: their range, a count, a gather in place. Each round removes eleven varying bits, so there are at most six.
4. **Finish.** A set whose keys are all equal is its own answer, and one of at most 64 keys is sorted by insertion. When the two ranks of an even count fall in two digits, the lower is the largest key in its digit and the upper the smallest in its, which one more read of the set finds.

The keys are turned back into doubles, and the arithmetic after them is the short-array loop's.

On normally distributed values the digit that holds the middle has a few hundred of a million elements, so the array is read three times (range, count, gather) and the rest is noise. Data in one binade is the hard case for top bits, and the range is what handles it: on `normal(100, 5)`, Astropy's background, the smallest and largest keys share the sign and most of the exponent, so the eleven bits counted are the low bits of the exponent and the top of the fraction. A few whole numbers are the hard case for the range: on 0 to 9 the digit is the exponent, and the middle digit holds four of the ten values, gathered densely and counted once more. An array whose elements are all equal stops after the range. Nothing depends on the order of the elements except the cost of the counting, which the eight sets keep level.

**Exactness.** The two keys found are those of the elements of rank `(n-1)//2` and `n//2` in ascending order, by construction: every step after the range is integer counting and copying, and every element is counted once whatever its value. The result is then made by the same operations as NumPy's mean of them.

**Memory.** The counters, 8 × 2,048 × 4 bytes = 64 KB, and the buffer of candidates, eight bytes for each element of the middle digit after the first count, plus a block: a few hundred elements on normal data, 4% on the image, 40% on ten whole numbers, and at most every element. Both come from `malloc`; NumPy's own call allocates a copy of the whole array, so a call that fits NumPy fits this. A call whose memory cannot be had is NumPy's. Below 4,096 elements one set of counters is used, as clearing 64 KB would cost more than the sets save.

**The interpreter lock.** The argsort hook copies its array with the lock held and sorts the copy without it, because its kernel sizes its buckets from one read and fills them on another: an array changed in between would make it write outside its buffers. The selection is written so that this cannot happen: every count is masked to its 2,048 counters; the walk over the counters ends inside them, because the counts add up to the number of elements whatever their values; and the gather stops at the capacity its count gave, plus at most a block of slack in the buffer, and reports a difference, which sends the call to NumPy. The buffer of candidates is the binding's own, which nobody else writes. So the array is not copied, and the lock is released from 8,192 elements (`HK_NOGIL_MIN_N`), as the other loops over large arrays do. A thread that writes the array while it is read gets a meaningless median, as it would from NumPy's partition of a copy taken while it wrote.

## The probe

A stand-in of the selection above, in C (`clang -O2`), called from Python through `ctypes` on the array's buffer, against `np.median(x)` on the same array; minimum of seven batches each; load average 2 to 4. It differs from the design in one detail: the range is taken by comparing the doubles as numbers, with a zero at either end widened to both zeros.

| Data | Elements | `np.median(x)` | Probe | Faster by |
|---|---|---|---|---|
| normal | 1,000 | 8 µs | 3 µs | 2.3x |
| normal | 100,000 | 922 µs | 112 µs | 8.2x |
| normal | 1,048,576 | 10.1 ms | 1.14 ms | 8.8x |
| `normal(100, 5)` | 1,048,576 | 8.4 ms | 1.14 ms | 7.3x |
| sky: `normal(100, 5)`, 2% stars `+ exponential(500)` | 1,048,576 | 9.2 ms | 1.32 ms | 7.0x |
| `1e9 + uniform(0, 1)` | 1,048,576 | 10.2 ms | 1.11 ms | 9.2x |
| ten whole numbers | 1,048,576 | 10.1 ms | 2.12 ms | 4.8x |
| ascending | 1,048,576 | 1.13 ms | 1.15 ms | 1.0x |
| descending | 1,048,576 | 1.29 ms | 1.15 ms | 1.1x |
| ascending, 1% of the elements swapped | 1,048,576 | 1.40 ms | 1.14 ms | 1.2x |
| ascending plus `normal(0, 0.01)` | 1,048,576 | 0.89 ms | 1.15 ms | 0.8x |
| `sin(0.001 i)` | 1,048,576 | 1.47 ms | 1.17 ms | 1.3x |
| ascending whole numbers 0 to 999 | 1,048,576 | 1.46 ms | 1.73 ms | 0.8x |
| ascending | 100,000 | 111 µs | 120 µs | 0.9x |
| ascending whole numbers 0 to 999 | 100,000 | 112 µs | 175 µs | 0.6x |

At a million normal elements the three reads take 0.26 ms (range), 0.49 ms (count) and 0.39 ms (gather).

It agreed with `np.median` bit for bit on every array of the table; the first stand-in, below, also on each of the first six kinds at 129, 130, 1,001 and 1,048,577 elements.

**How the parameters were chosen.** An earlier stand-in, with the counting and the gather one element at a time and the range taken on keys, compared digit widths and sets of counters at a million elements (same load):

| Digit bits / sets | normal | `normal(100, 5)` | sky | ascending | ten whole numbers |
|---|---|---|---|---|---|
| 11 / 1 | 2.30 ms | 1.50 ms | 1.92 ms | 3.10 ms | 7.03 ms |
| 11 / 4 | 1.39 ms | 1.36 ms | 1.74 ms | 1.54 ms | 6.01 ms |
| 12 / 4 | 1.37 ms | 1.35 ms | 1.51 ms | 1.55 ms | 4.03 ms |
| 16 / 2 | 1.44 ms | 1.59 ms | 1.47 ms | 2.06 ms | 2.27 ms |
| 16 / 1, the top sixteen bits, no range | 1.50 ms | 4.95 ms | 5.07 ms | 3.10 ms | 2.67 ms |

Without the range, data in one binade puts a third of its elements in the middle digit (the last row). Sixteen-bit digits are best on few distinct values but need 256 KB of counters a set, and at 1,000 elements took 12 to 30 µs, more than NumPy's 7 to 8, for clearing them. Computing the digits in vectors and testing blocks in the gather took the normal case from 1.71 ms to 1.08 ms; eight sets instead of four helped data in order (ascending 1.24 to 1.13 ms); and the dense gather took ten whole numbers from 6.4 ms to 2.2 ms.

**Data in order.** NumPy's `introselect` takes the median of three as its pivot, which on data in order or nearly so is close to the middle at once: it then costs about a read of the array, 0.9 to 1.5 ms at a million elements. The selection costs the same three reads whatever the order. A stand-in that read the array once, choosing its digits from a sample and gathering a window of digits while it counted, was no faster: 0.9 to 1.5 ms at a million, 1.0 to 1.1 ms on data in order, for its single pass did more work per element. So on such data the proposal is about level with NumPy, slower by up to 0.8x in the worst case measured at a million and 0.6x at 100,000.

## Alternatives

| Way | For | Against |
|---|---|---|
| Sort a copy and take the middle | Simple; exact | `np.sort` of a million float64 takes 3.7 times as long as NumPy's median |
| Choose two pivots from a sample and keep the elements between them (Floyd–Rivest) | One read of the array that keeps a small fraction | Its cost and its memory depend on the data: a sample can miss, ties at a pivot need care, and a miss means another read or NumPy's call |
| Radix select on the top sixteen bits | No range pass | Data in one binade puts a third of the elements in one digit; 256 KB of counters a set |
| One read, digits from a sample, a window gathered while counting | One read | Measured no faster, above, and more code |
| **Radix select below the shared bits (proposed)** | A fixed number of reads whatever the data; exact by construction; 64 KB of counters | Level with NumPy on data in order |

## What changes

| Where | Change |
|---|---|
| `hotkernels/src/hotkernels/_binding/hook_function.c` | The selection, and `try_median` and `call_median` taking `axis=None` and arrays above the sort's limit |
| `hotkernels/src/hotkernels/hooks.py` | The `median_f64` probe also on PROBE_N and PROBE_N - 1 elements, the even one as `np.median(x, axis=None)` |
| `delivery/cases.py` | Every value class at every size of the general sweep; large odd and even counts; duplicates; all equal; infinities of one sign and of both; negative zeros at and away from a zero median; subnormals; middle elements whose mean underflows; data in one binade; ascending and descending; two dimensions; `axis=None` by keyword and by position, and the spellings beside it that stay NumPy's; signalling NaNs in a large array |
| `delivery/timing.py` | The `np.median(x)` row has no upper limit |
| Documents | The hook table in `delivery-design.md`, `docsite/pages/start.md`, `hotkernels/README.md`, `README.md`, and the applications findings |

Not a kernel in `kernels/`: like the short-array loop it extends, the selection is control flow over a few small buffers, the same C on both architectures, and is held to NumPy by `delivery check`, which runs on Linux x86-64 in CI too.

## Exit criterion

- `delivery check` passes, with the new cases.
- From Python, `np.median(x)` on 1,048,576 normally distributed float64 is several times faster than NumPy, and faster than NumPy on 1,000.
- `delivery apps --only astropy_timeseries` gives identical results and is faster on the large size.
- `docsite build` succeeds.

## x86-64

There is no x86-64 machine here. The selection is portable C: the vector loops use the compiler's vector types (four 64-bit lanes on x86-64, compiled for the kernels' AVX2 level, two on ARM64), and the rest is scalar. Correctness on x86-64 is checked by `delivery check` in CI. No x86-64 speed figure is claimed.

## Left for later

- **bottleneck's `nanmedian`** (15.9% of `astropy_timeseries`): the same selection after leaving out the NaNs, reached through the bottleneck hooks. Its result and its handling of an all-NaN array are bottleneck's, not NumPy's.
- **`np.quantile` and `np.percentile`** (6.5% of `pandas_sales`): several ranks at once, and NumPy's interpolation between neighbours (`_lerp`) after them.
- **`np.median` along an axis**, and with `overwrite_input=True`.
- **Data in order**, where NumPy is as fast.
