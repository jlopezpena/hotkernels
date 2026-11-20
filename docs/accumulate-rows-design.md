# Accumulation along the first axis, a row at a time

This is a design for [issue 187](https://github.com/sofeikov/hotkernels/issues/187), written before the code and built in the same pull request. Where the building departed from it, [What changed in the building](#what-changed-in-the-building) says so; [Results](#results) holds the measurements.

`np.fmax.accumulate(a, axis=0)` and `a.cumprod(axis=0)` on a C-ordered two-dimensional float64 array are 38% of the `empyrical_stats` application ([applications-findings.md](applications-findings.md#empyrical_stats-102x)). hotkernels takes `np.fmax.accumulate` only on one-dimensional arrays and has nothing for a cumulative product, so both go to NumPy, which walks each column in turn with a stride of one row. The same work a row at a time was six times faster even from Python. This design takes those calls in the binding, with a loop of its own, for `np.add`, `np.multiply`, `np.maximum` and `np.fmax`.

## What arrives where

A throwaway build of the binding printed what the `ufunc.accumulate` method hook receives (NumPy 2.5.5, Apple M1 Pro):

| Call | Arrives at `ufunc.accumulate` as |
|---|---|
| `a.cumprod(axis=0)`, `np.cumprod(a, axis=0)` | `np.multiply.accumulate(a, 0)` |
| `a.cumprod(axis=0, out=a)` | `np.multiply.accumulate(a, 0, out=a)`, `out` the same object as `a` |
| `a.cumsum(axis=0)`, `np.cumsum(a, 0)`, `np.nancumsum(a, axis=0)` | `np.add.accumulate(a, 0)`, after the `method.cumsum` hook passed it on |
| `a.cumsum(axis=0, dtype=np.float64)` | `np.add.accumulate(a, 0, dtype=float64)` |
| `np.cumulative_sum(a, axis=0)`, `np.cumulative_prod(a, axis=0)` | `np.add.accumulate(a, axis=0, dtype=None, out=None)`, and the same for `np.multiply` |
| `np.fmax.accumulate(a, axis=0)` | itself |

NumPy's `PyArray_CumSum` and `PyArray_CumProd` look the `accumulate` attribute up on the ufunc and call it, so they reach the entry of the ufunc type's method table that the hook already replaces ([issue 97](https://github.com/sofeikov/hotkernels/issues/97)). `ndarray.cumprod` is not hooked and needs no hook. `ndarray.cumsum` is, and passes every call with an axis on.

## Why it is exact

NumPy's accumulation along axis 0 hands its inner loop one column at a time: the previous output element, the input element and the output element, each a row apart, and computes `out[i] = op(out[i-1], x[i])` for i from 1, with `out[0] = x[0]`. The loop proposed computes the same `op(out[i-1], x[i])` for every element, a row at a time instead of a column at a time. Each output element is the same operation on the same two operands, so the order of traversal can change the result only where the operation is not a function of its operands' values:

- **`np.add` and `np.multiply`.** IEEE addition and multiplication are correctly rounded in the current rounding mode and commutative in value, the sign of a zero included. Each column is computed in NumPy's own order: there is no reassociation, unlike the float `cumsum` kernels. Two things are not fixed by the values: which NaN comes out when both operands are NaNs with different bits (it depends on the processor and on the order the compiler puts the operands in), and the floating-point flags, which NumPy turns into warnings or errors. So the loop stops at the first block of rows that produces a NaN or raises a flag NumPy reports (invalid, divide by zero, overflow, underflow), and NumPy computes the rest; see below.
- **`np.maximum` and `np.fmax`.** Every correct maximum of two numbers is the same bits, except that the maximum of `-0.0` and `+0.0` may be either zero, and that two NaNs with different bits may give either. NumPy's answer for those depends on its build and its loop ([fmax-references-design.md](fmax-references-design.md)). The array is read first by the guard `np.fmax.accumulate` already has on one dimension (`fmax_accumulate_left_to_numpy`), and one that holds a signalling NaN, a negative zero or NaNs of more than one bit pattern is NumPy's. Without those, every NaN in the array is one quiet NaN, `np.maximum` returns it, `np.fmax` returns the other operand, and the two zeros cannot meet: the result is the same bits however it is computed.

## The calls taken

`ufunc.accumulate(a, axis=0)` with `ufunc` one of `np.add`, `np.multiply`, `np.maximum`, `np.fmax`, and `a.cumsum(axis=0)` at the method hook, when:

- `a` is a plain `ndarray` of float64 in native byte order, C-contiguous, with two or more dimensions, at least one row and at least one element per row. An array of more dimensions is taken as rows of everything after the first axis, which is what C order makes it.
- `axis` is the Python int 0 or `-a.ndim`, or absent for `accumulate`, whose default is 0. `cumsum` without an axis flattens and is not this.
- `dtype` is absent or None.
- `out` is absent or None, or a plain float64 `ndarray` of `a`'s shape, C-contiguous and writable, that is either `a`'s memory exactly (in place, as empyrical's `out.cumprod(axis=0, out=out)`) or shares no byte with it. A partial overlap is NumPy's.

Everything else goes on to NumPy as before: other element types, other axes, Fortran order and strided arrays (where NumPy's columns are already contiguous), tuples for `out`, and one-dimensional arrays, which keep the kernels they have.

## The loop

Plain C in `hook_method.c`, compiled for the kernels' level (`HK_TARGET_KERNEL_LEVEL`), with one function per operation over one row: `out[j] = op(prev[j], x[j])` for the row's columns, where `prev` is the output's previous row. The pointers do not overlap, so the compiler makes vector code of it. Row 0 is a copy.

In place, the input row and the output row are the same memory. Rows are taken a block at a time (at least one row, and as many as fit in 16 KB): the block's input is copied to a scratch buffer, and the operation reads from the copy. The copy is in the first-level cache when it is read, and it is also what lets the call be handed to NumPy part way, below. (Built the other way round: see [What changed in the building](#what-changed-in-the-building).)

**Handing the rest to NumPy.** For `np.add` and `np.multiply` the flags are cleared before the first block and read after each one, and each row reports whether it produced a NaN. At the first block that does either, the block's rows are put back as they were (from the copy in place, or by copying the input's remaining rows into the output otherwise), and NumPy's own `accumulate` is called on the output from the row before the block, in place: `ufunc.accumulate(r[b-1:], 0, out=r[b-1:])`. Row `b-1` already holds the accumulation up to it, so NumPy's result for the remaining rows is the one it would have computed for the whole call, with its own warnings or errors for the operations that raised them; none was raised before block `b`. The call goes to NumPy's original function, never back through the hook. Such a call is counted as left to NumPy.

For `np.maximum` and `np.fmax` the guard has already sent every array whose answer could differ to NumPy, so the loop runs to the end; the comparisons may set the invalid flag for a quiet NaN, which NumPy's loop does not leave behind, so the flags are cleared afterwards, as after `cumfmax_f64`.

The interpreter lock is released from 8,192 elements, as elsewhere in the binding, and taken back before NumPy is called.

## Counting

Four loops built into the binding, each with a counter, a self-test probe and a row in `report()`, implementation C and result exact: `cumsum_rows_f64`, `cumprod_rows_f64`, `cummax_rows_f64` and `cumfmax_rows_f64`. They are installed by default: they are bit-identical to NumPy.

- `a.cumsum(axis=0)` is taken at the `method.cumsum` hook itself, by the same function, so that a call is not counted as passed on there and then taken at `ufunc.accumulate`.
- A call to `np.maximum` or `np.fmax` that is not taken is counted under `accumulate`, as now. A call to `np.add` or `np.multiply` that is not taken is not counted: nearly all of them are one-dimensional accumulations no kernel serves, and `np.cumsum` counts its own at `method.cumsum`. A call handed to NumPy part way is counted under the operation of the hook it arrived at.

These loops differ from the ones built into the binding so far (`clip_i64`, `round_f64`, `median_f64`), which serve short arrays where NumPy's call path is the cost: here the gain is the order in which memory is walked, at every size. [delivery-design.md](delivery-design.md) settled that loops whose gain is not in their arithmetic belong in the binding as plain C, not in `kernels/`: the judge times a kernel on flat buffers and would see nothing of the traversal.

## Why not a kernel, or NumPy's own loop

- **A kernel** would take the matrix and its shape and do the same rows. The work per element is one operation on two loaded values and a store: the loop is bound by memory, and the C the compiler makes of it is the vector code a kernel would hold. To be confirmed by measuring the loop against a copy of the array.
- **NumPy's inner loop for the type**, called once per row through the ufunc's loop table, would keep NumPy's arithmetic, but not its order of operands nor its NaN choices on strided columns, so the guards would be the same; and it needs the loop looked up per ufunc and type through NumPy's internal tables. The plain C loop is measured against it.

## What is measured

From Python with `uv run delivery time`, which times each statement with and without hotkernels in alternating batches, on matrices of 4 × 4, 64 × 64 and 256 × 4,096 elements; with a script for 253 × 4,789 (empyrical's shape) and a larger one; and the `empyrical_stats` application before and after with `uv run --group apps delivery apps --only empyrical_stats`.

## Tested

`uv run delivery check`, in `delivery/cases.py`: every spelling above and the ones that must stay with NumPy, on C-ordered arrays of two and three dimensions with one to thousands of columns, in place and with a separate `out`, over the judge's value classes; Fortran order, strided rows, a read-only array, a partially overlapping `out`, and other element types; and the cases that hand the call to NumPy part way: a NaN, an overflow and an underflow in the middle of a column, under `np.errstate` settings that warn and raise, with the output compared after an error too.

## What changed in the building

| Proposed | Built | Why |
|---|---|---|
| In place, copy the block's input to scratch, then compute into the array from the copy | Compute the block into scratch, then copy it over the array | The first was as slow as NumPy: 4.5 ms for `o.cumprod(axis=0, out=o)` on 253 × 4,789, against 0.43 ms out of place. A loop in C outside the binding showed the same on the Apple M1 Pro: 2.4 to 9 ms computing into the rows just read, 0.5 to 0.8 ms through scratch. A stop part way now needs nothing put back in place: the block's rows are written only after it passed |
| The row loops measured against NumPy's inner loop called per row | Measured against NumPy's contiguous loop over the same 252 rows in one call | Calling NumPy's loop per row through its internal tables was not built. NumPy's contiguous `np.multiply` over the same element count into a preallocated array took 363 µs; the binding's whole call, allocating its result, 420 µs; the issue's Python loop per row 595 µs. The arithmetic is not where the time is |
| `np.maximum` and `np.fmax` as plain C | The same; ARM64 `fmax` and `fmaxnm` intrinsics were tried | Same time (549 µs against 558 µs): the guard's read of the array is the difference from the product (about 0.13 ms of 0.55) |

Nothing was left open: every point the design listed was built as described otherwise.

## Results

From Python, alternating batches with and without hotkernels (a script in the manner of `delivery time`), Apple M1 Pro, NumPy 2.5.5, at a load average of 3.8 to 4.6 while other work ran, so the figures are rough:

| Statement, float64 | 4 × 4 | 64 × 64 | 253 × 4,789 | 2,520 × 4,789 | 100,000 × 2 |
|---|---|---|---|---|---|
| `np.fmax.accumulate(a, axis=0)` | 2.6x | 9.3x | 7.6x (4.24 ms to 0.56 ms) | 14.3x | 1.1x |
| `np.maximum.accumulate(a, axis=0)` | 2.6x | 9.3x | 7.6x | 15.2x | 1.1x |
| `a.cumprod(axis=0)` | 1.7x | 11.4x | 9.9x (4.24 ms to 0.43 ms) | 21.9x | 1.4x |
| `a.cumsum(axis=0)` | 3.2x | 13.0x | 10.1x | 22.7x | 1.4x |
| `o.cumprod(axis=0, out=o)` after a copy | 1.3x | 6.0x | 7.2x | 6.2x | 1.3x |

A copy of the 253 × 4,789 array takes 0.18 ms, so the product runs at a little over twice the time of a copy. `uv run delivery time --only accumulate` and `--only cumsum` time the four statements at 4 × 4, 64 × 64 and 256 × 4,096.

`uv run --group apps delivery apps --only empyrical_stats`, on the same machine at a load average of 4 to 6: 1.01x before the change (1,328.9 ms to 1,310.0 ms), 1.57x after (1,324.0 ms to 842.3 ms), results identical. Each of the 60 two-dimensional `cumprod` and `np.fmax.accumulate` calls of a run is taken.

The x86-64 build is the same C compiled for AVX2; it has not been timed, and is checked by CI's `delivery check` on Linux.
