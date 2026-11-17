# The first kernels: results and findings

This records the first pass over the shortlist in [survey-design.md](survey-design.md): 22 kernels for 15 NumPy operations, each with a spec, a naive C reference and an attempt at beating that C in ARM64 assembly. Twenty-one ship as assembly and one as C. It covers the kernel half of phase 3 in [core-design.md](core-design.md). The other half, the binding and `install()`, is in [delivery-design.md](delivery-design.md).

The numbers are the judge's. What is open to review is the set of decisions at the end.

The results so far are ARM64's. The kernels are being carried over to x86-64 ([x86-64-design.md](x86-64-design.md)), and this document is organised to hold both.

## How this document is organised

- **What is true of an operation is said once.** Each family under [Results by family](#results-by-family) opens with "The operation": what NumPy requires of the result, why the simple loop is slow, the idea a fast kernel is built on, and any proof it rests on. None of that depends on the processor.
- **What is true of one architecture sits under its name.** "On ARM64" and "On x86-64" hold the instructions chosen, the block sizes, the measured speeds, what was tried and did not help, and what limits the kernel on that processor. A sentence such as "this is the hardware's limit" is only ever about the hardware of its heading.
- **A kernel's verdict is per architecture.** The same kernel can ship as assembly on one, as C on another, and not be installed on a third.
- **The x86-64 sections are empty until its kernels exist.** What each one will try first is listed in [x86-64-design.md](x86-64-design.md#candidates-per-family).

## Summary

### ARM64

- **Twenty-one of the 22 kernels have assembly that the judge reports as significantly faster than naive C.** The exception, `cast_u8_f32`, ties the compiler and ships as its C. Every kernel is faster than NumPy at every size measured. Twenty are bit-identical to NumPy. The two float `cumsum` kernels are within a declared tolerance, which is what makes them faster at all.
- **The survey's prediction for the "C kernels" was wrong.** It expected assembly to tie the compiler on `clip`, `astype`, `np.sum` on `uint8` and `count_nonzero`, because clang vectorises those loops. Tried, assembly beat the C on five of the six: by 5.9x on `sum_u8`, by 1.3x to 1.4x on the counts, and by 1.35x to 1.38x on `clip` for bounds that allow its fast path. A vectorised loop is not necessarily a good one.
- **Three results are narrower than their headline number.** `isin_i64` is fast only for test values that lie close together, and the two `clip` kernels only for some bounds and, with a lower bound of zero, only for data without negative zeros. Each is set out below.
- **The judge has blind spots that mattered.** It tested one value per scalar argument and, from 4,096 elements up, repeats one input per size. Both hid things that were found by other means. The first is now closed: a spec lists several combinations of scalar values, and the `clip`, `histogram` and masked-assignment kernels are checked with them.

| Kernel | Input | vs naive C | at 16 | at 4,096 | at 1,048,576 | at 16,777,216 | vs NumPy | at 1,048,576 |
|---|---|---|---|---|---|---|---|---|
| `argmax_f32` | random | 23.4x [23.39 – 23.41] | 6.05x | 45.9x | 26.6x | 18.7x | 23.2x | 8.82x |
| `argmin_f32` | random | 23.4x [23.39 – 23.41] | 6.04x | 45.8x | 26.7x | 18.8x | 23.2x | 8.86x |
| `argmax_f64` | random | 12.5x [12.53 – 12.54] | 4.11x | 22.9x | 12.7x | 9.37x | 17.2x | 6.86x |
| `argmin_f64` | random | 12.6x [12.60 – 12.65] | 4.11x | 22.7x | 13.2x | 9.38x | 17.3x | 7.10x |
| `compress_f32` | half selected | 2.76x [2.73 – 2.76] | 1.97x | 3.54x | 2.70x | 2.62x | 42.8x | 41.7x |
| `compress_f32` | few selected | 5.14x [5.13 – 5.14] | 1.97x | 7.61x | 5.70x | 3.70x | 21.8x | 17.3x |
| `compress_f32` | all | 3.77x [3.74 – 3.77] | | | | | 30.5x | |
| `compress_f64` | half selected | 1.85x [1.85 – 1.86] | 1.69x | 2.61x | 1.67x | 1.47x | 27.1x | 22.5x |
| `compress_f64` | few selected | 4.15x [4.14 – 4.15] | 1.69x | 7.63x | 4.10x | 2.06x | 15.3x | 11.0x |
| `compress_f64` | all | 2.77x [2.77 – 2.78] | | | | | 20.4x | |
| `nonzero_bool` | half selected | 2.68x [2.68 – 2.68] | 2.20x | 2.98x | 2.47x | 2.49x | 10.7x | 5.15x |
| `nonzero_bool` | few selected | 6.89x [6.89 – 6.90] | 2.20x | 8.09x | 8.08x | 7.60x | 22.4x | 19.2x |
| `nonzero_bool` | all | 4.30x [4.30 – 4.30] | | | | | 15.5x | |
| `maskfill_f64` | half selected | 7.19x [7.04 – 7.26] | 2.88x | 3.54x | 19.0x | 18.7x | 24.3x | 21.4x |
| `maskfill_f64` | few selected | 2.87x [2.87 – 2.88] | 1.60x | 2.75x | 4.64x | 3.14x | 11.9x | 7.48x |
| `maskfill_f64` | all | 4.54x [4.50 – 4.57] | | | | | 17.0x | |
| `isin_i64` | mostly misses | 4.54x [4.53 – 4.54] | 2.12x | 5.64x | 5.34x | 4.39x | 108.2x | 75.6x |
| `isin_i64` | wide values | 0.98x [0.98 – 0.98] | 0.87x | 1.00x | 1.00x n.s. | 1.00x n.s. | 9.47x | 1.83x |
| `isin_i64` | all | 2.11x [2.11 – 2.11] | | | | | 32.0x | |
| `searchsorted_f64` | random | 8.82x [8.80 – 8.86] | 4.84x | 9.70x | 8.42x | 12.6x | 13.2x | 8.84x |
| `interp_f64` | random | 4.95x [4.92 – 4.96] | 1.81x | 5.03x | 6.02x | 8.69x | 21.8x | 22.7x |
| `histogram_f64` | random | 3.48x [3.47 – 3.49] | 2.27x | 3.82x | 3.59x | 3.57x | 31.6x | 9.70x |
| `bincount_u8` | random | 1.24x [1.24 – 1.25] | 1.07x | 1.29x | 1.31x | 1.30x | 8.14x | 6.74x |
| `cumsum_i64` | random | 1.90x [1.89 – 1.93] | 1.36x | 2.43x | 1.95x | 1.69x | 24.3x | 12.5x |
| `cumsum_f64` | random | 4.46x [4.43 – 4.47] | 1.74x | 5.76x | 5.48x | 4.78x | 31.7x | 20.1x |
| `cumsum_f32` | random | 6.10x [6.02 – 6.11] | 1.74x | 7.76x | 7.83x | 7.22x | 43.4x | 29.0x |
| `sum_u8` | random | 5.85x [5.84 – 5.85] | 3.33x | 8.87x | 5.07x | 4.40x | 73.4x | 15.7x |
| `count_nonzero_i64` | random | 1.28x [1.28 – 1.28] | 1.73x | 1.50x | 1.03x | 1.07x | 8.26x | 3.25x |
| `count_nonzero_f64` | random | 1.41x [1.41 – 1.41] | 1.39x | 1.71x | 1.28x | 1.08x | 7.80x | 3.79x |
| `clip_f32` | bounds (-0.5, 0.5), random | 1.33x [1.32 – 1.35] | 1.16x | 1.58x | 1.13x | 1.22x | 26.6x | 8.67x |
| `clip_f32` | bounds (0, 1), random | 1.21x [1.20 – 1.22] | 1.17x | 1.32x | 1.28x | 1.04x n.s. | 24.7x | 9.78x |
| `clip_f32` | bounds (0, 1), sparse -0.0 | 1.02x [1.01 – 1.03] | 1.17x | 0.99x n.s. | 1.00x n.s. | 1.01x n.s. | 20.2x | 7.58x |
| `clip_f32` | bounds (0, 1), rounded | 1.03x [1.01 – 1.04] | 1.15x | 1.00x n.s. | 1.00x n.s. | 1.18x n.s. | 20.2x | 7.48x |
| `clip_f32` | all six tables | 1.19x [1.18 – 1.19] | | | | | 23.7x | |
| `clip_f64` | bounds (-0.5, 0.5), random | 1.23x [1.21 – 1.25] | 1.09x | 1.61x | 1.09x n.s. | 1.12x | 12.5x | 3.65x |
| `clip_f64` | bounds (0, 1), random | 1.15x [1.14 – 1.16] | 1.09x | 1.33x | 1.22x | 1.06x | 12.1x | 4.25x |
| `clip_f64` | bounds (0, 1), sparse -0.0 | 1.01x [1.00 – 1.03] | 1.10x | 1.02x | 1.00x n.s. | 0.96x n.s. | 10.8x | 3.50x |
| `clip_f64` | bounds (0, 1), rounded | 1.00x n.s. [0.99 – 1.01] | 1.09x | 1.01x n.s. | 1.00x n.s. | 0.96x n.s. | 10.5x | 3.58x |
| `clip_f64` | all six tables | 1.14x [1.14 – 1.15] | | | | | 12.0x | |
| `cast_u8_f32` | random | C kernel | | | | | 9.05x | 2.68x |

Every figure is how many times faster the kernel is. "vs naive C" and "vs NumPy" are geometric means over the eleven sizes from 16 to 16,777,216 elements, with the 95% interval for the first; the four columns after it are the speedup over naive C at single sizes. "n.s." marks a difference that is not statistically significant. Kernels timed on two kinds of input, or with two sets of scalar values, have a row for each and one for both. The NumPy mean is dominated by NumPy's fixed cost per call at small sizes (for example 1.3 µs for `np.sum` on 16 bytes), which is a property of the call path and not of the loop. The column at 1,048,576 elements shows the loops alone.

### x86-64

No results yet. NumPy is a different opponent there: it already has vector loops for `argmax`, `clip` and `astype`, so the ARM64 figures above are not a forecast. What a first measurement suggests is in [x86-64-design.md](x86-64-design.md#what-was-measured-before-writing-this).

## How the numbers were taken

### ARM64

- **Machine.** Apple M1 Pro, macOS (Darwin 27.2.0), NumPy 2.5.5, Apple clang 21. Native, not emulated.
- **Command.** `uv run judge run <kernel> --unlocked` for each kernel, one after another, with nothing else running. 30 interleaved samples per size.
- **Three passes.** The `histogram_f64` row is from a second pass of the same command, taken after scalar sets were added to the judge and the faster `histogram` path was adopted. The `clip` rows are from a third, taken after the path for a zero lower bound was written, while the machine was in use (load average about 6). In that pass the sizes that fit the caches are steady and the sizes from 262,144 elements up are not: the row for bounds (-0.5, 0.5), whose code did not change, read 1.44x and 11.1x at 1,048,576 elements for float32 on the idle machine, and 1.30x and 4.88x for float64. The `clip` kernels are also timed on bounds (-0.5, 0.5) with the two kinds of data that hold negative zeros; those two tables match the random one and are left out of the table above, though the "all six" row includes them. Every other row is from the first pass. `maskfill_f64` was timed again in the second pass, but the machine was no longer idle and its sizes from 1,048,576 elements up varied by a factor of two between runs; its kernel had not changed, so its first-pass row stands.
- **Unlocked.** The judge refuses to time a kernel whose `spec.py`, `naive.c` or judge differs from `main`, unless told otherwise. The specs and the judge changes are new in the same pull request as the assembly, so every run here needed `--unlocked` and the judge marks the results as untrusted. Nothing is recorded under `results/` for that reason. Once this is merged, the same commands without the flag give results that can be recorded.
- **How the kernels were written.** One agent per family of kernels, each in its own worktree, following the rule in [AGENTS.md](../AGENTS.md): keep an edit only if `check` passes and `compare --against HEAD` reports it significantly faster. Each kernel's `attempts.md` lists every attempt, kept or not. Attempts labelled "local" or "private" there were timed outside the judge and are not judge results.

### x86-64

Nothing has been measured by the judge on x86-64 yet. The judge has to be ported first.

## What the judge had to learn

The two fixtures needed nothing beyond arrays and a size. The real kernels needed scalar arguments, several combinations of values for them, fixed-size buffers and buffers sized by a scalar, buffers modified in place, inputs that must be sorted, kernels without assembly, and a tolerance measured at a scale other than the result. These are described in [judge-design.md](judge-design.md#what-a-spec-can-express). All of them change the judge; its selftest passes with them.

## Results by family

### `argmax` and `argmin` on floats

#### The operation

- **What NumPy requires.** The index of the first element equal to the maximum (or minimum). A NaN wins over every number, and among several NaNs the first one wins.
- **The idea.** The position does not have to be tracked element by element. Reduce each block of elements to its maximum, remember which block held the best value so far, and search only that block for the position at the end. A NaN has to be noticed block by block, because it decides the result wherever it sits.

#### On ARM64

- **Technique.** One pass that reduces each block of elements (64 for float32, 32 for float64) to its maximum with `fmax`, and remembers which block held the best value so far with one scalar compare and a `csel`. No index vector is carried, and no branch depends on the data unless it holds a NaN. At the end only the winning block is scanned for the first position that equals the maximum. `fmax` returns NaN when either operand is NaN, so a block that holds one shows up in the same compare as unordered, and a scalar loop then finds the first NaN. Short arrays take straight-line code with no loop instead: 8 to 64 and 129 to 256 elements for float32, 4 to 64 for float64. Below those sizes a scalar loop runs.
- **Why it is this far ahead.** NumPy's loop is scalar for floats. The float32 kernel runs at about 0.03 ns per element in the first-level cache and 0.05 to 0.07 beyond it. On input without a NaN its speed does not depend on the data. A NaN ends the block loop early, but sends the straight-line sizes to a scalar scan of the whole array: 256 float32 elements take 12 ns without a NaN and 112 ns with one in the last position.
- **What did not help:** software prefetch (slower), aligning the loop, a vector search for the first NaN, cheaper reductions of a block to a scalar.

#### On x86-64

Not written yet. NumPy's loop for floats is not scalar there, so "why it is this far ahead" will not read the same. The candidates for this family are in [x86-64-design.md](x86-64-design.md#candidates-per-family).

### Mask selection: `compress` and `nonzero`

#### The operation

- **What NumPy requires.** The selected elements (for `nonzero`, their positions) in their original order, and how many there are.
- **Why the simple loop is slow.** It asks "is this element selected?" before each store. On a mask with no pattern the processor guesses that answer wrong about half the time.
- **The idea.** Do not ask. Store at the current output position and move the position forward by the number selected, a block of elements per store. A mask that selects few elements is better served by finding the selected positions directly, so a kernel may have a path for each.
- **It relies on scratch output.** The stores write past the elements that are kept, inside the buffer's capacity. That is what `scratch=True` permits; the trade-offs are in [judge-design.md](judge-design.md#scratch-output-writing-inside-the-capacity).

#### On ARM64

- **Technique.** Dense masks are handled eight mask bytes at a time, and each kernel compacts differently. `compress_f32` packs the eight bytes into an index into a 256-entry table of shuffle patterns, and `tbl` moves the selected elements to the front of a block that is stored whole. `nonzero_bool` uses the same index into a table that lists the positions of the true bytes. `compress_f64` has no table and no `tbl`: it stores the elements in pairs, and the mask decides the address each pair is loaded from. In all three the write pointer advances by the number selected. Sparse masks: 64 mask bytes are reduced to a 64-bit mask and the set bits are extracted with `clz`, a fixed number at a time, so there is no data-dependent branch per element. The kernel switches between the two by measured density.
- **A weak spot the judge does not time.** Around 10% density the switch between the two paths mispredicts. Measured outside the judge, `nonzero_bool` costs 0.127 ns per element at 10% against 0.045 at 5% and 0.110 at 20%. The judge times 50% and 5%.
- **Limit.** With half the elements selected, `compress_f64` writes as fast as memory allows from 16,384 elements up.

#### On x86-64

Not written yet. The candidates for this family are in [x86-64-design.md](x86-64-design.md#candidates-per-family).

### Masked assignment

#### The operation

- **What NumPy requires.** Selected elements become the value. NumPy's `x[mask] = v` does not touch the others.
- **Why the simple loop is slow.** It is a store under a condition, element by element. Compiled as a branch, it is guessed wrong about half the time on a mask with no pattern.
- **The idea.** Work on whole blocks. Either write the block back with the value blended in, which also writes the unselected elements with their own bits, or use a store that leaves them alone where the architecture has one. Which is available, and what it asks of the caller, is per architecture.

#### On ARM64

- **Technique.** Dense blocks: load 16 elements, blend with the value through `bsl` under a mask widened by `tbl`, store all 16. Sparse blocks: no load; one store per selected element at an address chosen without branching.
- **It rewrites elements it does not change.** A dense block stores unselected elements back with their own bits. The array must be writable throughout, and a write by another thread to an unselected element between the load and the store would be undone. NumPy's `x[mask] = v` does not touch unselected elements. This is why the compiler cannot do it and why it has to be a decision: see "Decisions".
- **The jump at 65,536 elements is naive C slowing down**, not the kernel speeding up. Below that size the branch predictor memorises the judge's repeated input and naive C looks better than it is.

#### On x86-64

Not written yet. A store that writes only the selected elements exists there, so the rewrite above may not be needed. The candidates for this family are in [x86-64-design.md](x86-64-design.md#candidates-per-family).

### `isin` with eight test values

#### The operation

- **What NumPy requires.** One boolean per element: whether it equals any of the eight test values.
- **The work.** Done literally, eight compares per element. A kernel gets ahead of a compiler only where the test values allow something cheaper than comparing against each one.
- **The idea.** When the test values lie close together, an element can be narrowed to a small key and looked up in a table, with every value outside the window made unable to match.

#### On ARM64

- **Technique.** When the test values allow it, each element is narrowed to a 16-bit key (with saturation, so values outside the window can never match) and membership is four `tbl` lookups on the nibbles of the key, or three when every test value is below 4,096. They allow it when all eight lie between 0 and 65,534, or all lie within 32,767 of the first test value. Otherwise the kernel compares at 64 bits, eight compares per vector, which is what clang already does.
- **The speedup depends on the test values, and on their order.** For small or closely spaced test values the kernel is several times faster than naive C. Because the second window is centred on the first test value, the same eight values can be fast in one order and slow in another: eight values spanning 60,000 ran at 0.17 ns per element with a middle value first and 0.62 with the lowest first. For unrelated 64-bit values it ties naive C, and at 16 elements it is slower, because it has tried and rejected the narrow set-ups first. The judge's original distribution only ever took the fastest path, so a second one (`wide values`) was added to show the other.
- **Why the general case cannot be beaten:** eight 64-bit compares per vector of two elements is the work, the compiler emits exactly that, and there is no wider compare on this hardware.

#### On x86-64

Not written yet. A compare of four 64-bit elements at once exists there, so the general case is not settled by the ARM64 result. The candidates for this family are in [x86-64-design.md](x86-64-design.md#candidates-per-family).

### `searchsorted` and `interp`

#### The operation

- **What NumPy requires.** For each needle, the first position in the sorted array whose element is not less than it, with NaN ordered after every number. `interp` runs the same search over `xp` and interpolates between the two neighbours. Its result is bit-identical only if the division and the multiply-add are done the way NumPy's build for that architecture does them.
- **Why the simple loop is slow.** It runs one search at a time. Each probe waits for the compare before it, and on a large array each probe is a cache miss the next one queues behind.
- **The idea.** A branch-free binary search run for many needles at once. The step size depends only on the array length, so all searches stay in step and their cache misses overlap instead of queueing.

#### On ARM64

- **Two lengths.** `searchsorted_f64(a, v, out, n, needles)` and `interp_f64(x, xp, fp, out, n, queries)`: `n` is the length of the sorted array (of `xp` and `fp`), and the number of needles or queries is separate. The first selects the code path and the depth of the search; the second is how many times the loop runs.
- **Technique.** Every step loads one probe per needle, compares, and moves each needle's base by the compare mask. Sixteen needles at a time when the array has fewer than 64 elements, 64 at a time up to 1,048,576, and blocks of 128 beyond.
- **A path needs one full group of needles.** Leftover needles are covered by redoing the last full group, so a call with fewer needles than its path's group takes a narrower one: 64 to 127 needles against an array of 1,048,576 or more use the 64-needle path, and anything smaller is searched one needle at a time. That last case is correct and untuned: a handful of needles against a large array is mostly the cost of the call.
- **`interp`** runs the same search over `xp`, then the interpolation for two queries per vector with a real `fdiv` and a fused multiply-add, which is what NumPy's build uses. Any group whose result contains a NaN is redone by a line-by-line copy of the reference, which keeps the output bit-identical.
- **What limits it.** In cache, which elements are probed, not the instruction count: making every search probe the same elements ran 1.25x to 1.33x faster with identical instructions. Beyond the cache, the number of cache and TLB misses, which the kernel overlaps but cannot reduce.
- **The number of needles is verified for the values the specs list:** 0, 1, and the counts either side of 16, 64 and 128 (15, 16, 17, 63, 64, 65, 127, 128, 129), and 1,000. Fresh random cases draw from the same list.
- **The figures in the table above were taken with as many needles as elements**, before the kernels had a second length; the search loops have not changed since. The judge now times each kernel with 64 and with 1,000 needles per call across the sweep of array lengths. From a run with the judge unlocked on a machine in use, so indicative only:

  | Kernel | Needles | vs naive C | at 16 | at 4,096 | at 1,048,576 | at 16,777,216 | vs NumPy | at 1,048,576 |
  |---|---|---|---|---|---|---|---|---|
  | `searchsorted_f64` | 64 | 8.77x [8.74 – 8.80] | 5.27x | 9.60x | 10.1x | 10.2x | 5.30x | 2.38x |
  | `searchsorted_f64` | 1,000 | 9.05x [9.02 – 9.08] | 5.43x | 9.47x | 8.94x | 15.8x | 9.44x | 6.53x |
  | `interp_f64` | 64 | 5.17x [5.16 – 5.19] | 1.82x | 5.00x | 7.46x | 9.32x | 16.9x | 15.7x |
  | `interp_f64` | 1,000 | 4.97x [4.96 – 4.99] | 1.80x | 4.91x | 6.25x | 10.6x | 16.3x | 19.6x |

  The sizes are lengths of the sorted array. With 64 needles a call is short enough that NumPy's fixed cost per call is part of its figure.
- **Not verified by the judge:** `interp_f64` is checked up to 65,537 sample points (the NumPy oracle is too slow beyond), so its path from 1,048,576 was compared with the reference outside the judge, without guard pages. Neither kernel is run on unsorted input, where the result is as meaningless as NumPy's and the kernel is required only to stay inside its buffers; that holds by construction and is not tested.

#### On x86-64

Not written yet. Whether NumPy's x86-64 build fuses the multiply and the add in `interp` has to be established before the interpolation is written; it need not match ARM64. The candidates for this family are in [x86-64-design.md](x86-64-design.md#candidates-per-family).

### `histogram` and `bincount`

#### The operation

- **What NumPy does for `histogram`.** It computes a bin estimate by divide, multiply and truncate, then corrects it against the two neighbouring bin edges. A bit-identical kernel either performs those operations in the same order, each rounded on its own, or proves for the element in hand that neither correction can fire.
- **The proof.** An element whose position inside its bin is far enough from both edges needs no correction, provided the bin count and range pass a check made once per call. That check has to reject a product that overflowed. The bound was derived for the operations as one kernel performs them, so it is rechecked for every architecture; the ARM64 one is below and in its `kernel.s` header.
- **`bincount`.** One counter is loaded and stored per element, and that is the limit whatever surrounds it. Equal neighbouring values wait on each other unless they are counted into separate tables.

#### On ARM64

- **`histogram_f64`, the literal path.** NumPy computes a bin estimate by divide, multiply and truncate, then corrects it against the two neighbouring bin edges. The kernel can do those operations two elements per vector, in the same order and individually rounded, falling back to scalar code for a lane that is out of range, NaN or in the last bin. That takes 14 vector operations per two elements and about 0.88 ns per element, and nothing cheaper computes the same thing.
- **The path the kernel takes instead.** It computes the estimate exactly as NumPy does and skips the two edge corrections for an element whose position inside its bin is at least 2^-12 of a bin away from either edge: about 2,047 elements in 2,048. An error bound in the `kernel.s` header shows that neither correction can fire for such an element, provided the bin count and range pass a check made once per call; when they do not, the literal path is used. The check compares two products and has to reject one that overflowed: as first written it took two infinities for equal, so narrow ranges near the largest float64 values got the fast path where the bound is false, and elements near a bin edge could land in the neighbouring bin. Every other element goes to a scalar copy of the reference. Four vector operations per two elements, 0.69 ns per element, 1.30x [1.30 – 1.30] over the literal path.
- **Why it was held back, and what changed.** It replaces operations with a proof, and the proof depends on the bin count and range, of which the judge could test one combination. The spec now lists sixteen, and the kernel passes all of them. The judge's data seldom lands near a bin edge, so the evidence about edges is differential tests outside the judge: the author's against the reference (410 million elements), and one against `np.histogram` itself with values on and within three ulps of NumPy's edges and around the 2^-12 threshold (430,000 rounds, 126 million elements, no mismatch; a histogram without the corrections fails 2,086 of 3,000 rounds of the same inputs). Neither reached the top of the float64 range, which is where the overflow above was found in review, by a third test of the same kind. With the check corrected, 6,000 ranges of magnitude 2^960 and up, 636 of them with both products infinite, give no mismatch over 1.6 billion elements; before, 28 of 3,000 did.
- **`bincount_u8`** is limited by one counter load and store per element. Reading eight input bytes per load and, for large arrays, counting into four tables on the stack so that equal neighbouring bytes do not wait on each other, is worth 1.24x. A 65,536-entry pair table and `dc zva` for zeroing were slower.

### `cumsum`

#### The operation

- **Integers are exact in any order.** So the additions can be regrouped: prefix sums inside each block are built from the inputs alone, and the running total is added to a whole block at a time, so the chain from one block to the next is one add per block instead of one per element.
- **Floats cannot be both exact and fast.** Bit-identical output requires NumPy's order of additions, each depending on the one before, so n elements take n times the latency of one floating-point add. No exact kernel can beat that, on any processor.
- **So float `cumsum` is in tolerance mode**, as [survey-design.md](survey-design.md) decided.
- **Infinities and NaN must still land where NumPy has them.** A regrouped sum can overflow where the sequential one does not, and can return to a finite value where the sequential one stays infinite. A kernel has to prove, for each stretch of output, that neither happened, and redo the stretch in NumPy's order when it cannot.
- **The tolerance needed a scale.** The judge's tolerance was measured at the size of the result. A regrouped running sum over data that cancels differs from NumPy's by millions of such units while being exactly as accurate, because a small result carries the rounding error of its large terms. The allowed error of element i is now 4 units in the last place of `(i + 1) * (|x[0]| + ... + |x[i]|)`, the standard bound for adding in any order.

#### On ARM64

- **Integers.** In cache the scheme runs at about 0.13 ns per element; from 4,194,304 elements it runs at the speed of a plain copy.
- **The exact float chain, measured.** The latency of one floating-point add is 0.93 ns here. Naive C and an exact assembly kernel both measured 0.95 to 0.97 ns per element from 4,096 elements up. The first version of these specs was exact by mistake and produced that tie; the specs now declare a tolerance.
- **The float kernels** use the integer scheme, with two running sums per block for float64 and four for float32, and are limited by the number of vector operations per block, not by the latency of an add.
- **Infinities and NaN still land where NumPy has them.** A regrouped sum can overflow where the sequential one does not, and can return to a finite value where the sequential one stays infinite. The kernel works in chunks of 1,024 elements and accumulates the square of every output; a finite sum of squares proves every output of the chunk was finite and well below the overflow threshold, so the sequential sum cannot have overflowed either. A chunk that fails the test is redone one element at a time in NumPy's order. The argument is in each `kernel.s` header.
- **Accuracy, measured outside the judge** against an exactly computed sum over 80,303 inputs per kernel: the kernels are at least as accurate as NumPy's sequential sum in most classes tried. On all-equal values, the worst case for a sequential sum, NumPy's error was 7 to 16 times the kernel's. They are less accurate in two. Where cancellation makes every one of NumPy's partial sums exact, NumPy's error is zero and the kernel's is not: on the repeating float32 pattern `-2^24 + 2, 1, 2^24, 1` the kernel loses 1 per repetition, 250,000 at a million elements where the true sum is 1,000,000, and float64 does the same with 2^53. And on magnitudes that grow quickly the kernel's error was up to about 3 times NumPy's in review, at most about 2.5 units in the last place. Both are inside the declared tolerance; small integers stored as floats, where any order of additions is exact, come out exact.
- **Costs of the scheme.** Running sums above about 1e153 (float64) or 4e18 (float32) fail the squares test and run at the speed of naive C, with correct results. Fewer than 32 (float64) or 20 (float32) elements use the exact sequential chain. Float32 arrays of 2^29 elements or more use the sequential loop throughout, because the proof does not hold there.

#### On x86-64

Not written yet. A dependent chain is cheaper on a faster clock, so the scalar loop the vector scheme has to beat is not the ARM64 one. The candidates for this family are in [x86-64-design.md](x86-64-design.md#candidates-per-family).

### `sum`, `count_nonzero`, `clip`, `astype`

#### The operation

- **`sum` on `uint8`.** The result is a 64-bit sum, so the bytes have to be widened somewhere. Where, and how often, is the whole design.
- **`count_nonzero` on floats.** A NaN counts as non-zero and both zeros count as zero.
- **`clip`.** NumPy's result is what compare-and-select gives, `v < lo ? lo : (v > hi ? hi : v)`: an element equal to a bound keeps its own sign of zero, and a NaN passes through. A shorter sequence built from a minimum and a maximum is usable only where it gives the same bits, and that depends on how the architecture's minimum and maximum treat zeros and NaN. The reference is defined for ordered bounds only; see [below](#where-the-result-is-narrower-than-the-number).

#### On ARM64

These were expected to ship as C. Five ship as assembly.

- **`sum_u8`.** clang widens every byte to 64 bits before adding: 24 vector instructions per 32 bytes. The kernel adds bytes pairwise into 16-bit accumulators with `uadalp`, sixteen accumulators per step, and widens every 128 steps. In cache it runs at three 16-byte loads per cycle, which is the core's limit.
- **`count_nonzero_i64`.** A 64-bit element is non-zero exactly when the larger of its two halves is, so three levels of `umaxp` reduce eight vectors to sixteen bytes before one test and one accumulate: nine vector instructions per eight vectors against clang's sixteen. From 65,536 to 1,048,576 elements both run at the rate the second-level cache delivers and tie.
- **`count_nonzero_f64`.** One bit test against a mask that ignores the sign handles NaN and both zeros, then the same narrowing.
- **`clip`.** Compare-and-select needs four vector operations per vector and clang is already there. `fmin(fmax(x, lo), hi)` needs two, but differs from NumPy on signed zeros at the bounds and on signalling NaNs, so the kernel takes it only when the bounds rule the first out and detects the second afterwards. For a lower bound of +0.0, the bound `np.clip(x, 0, 1)` has, the only element the two operations get wrong is -0.0, so a third operation per vector watches for one and a chunk that held one is done again by compare and select. See the caveats below.
- **`cast_u8_f32` stays C.** clang's loop is already at the limit of one 16-byte store per cycle. The assembly gained at 16 and 64 elements (no entry checks, no scalar remainder) and at 16,777,216 (non-temporal loads), and tied at the eight sizes between: 1.05x [0.99 – 1.08] overall in one run and 1.06x [1.00 – 1.07] in the next. That is a tie, and the design ships C on a tie. The assembly is in the git history and its attempts log is kept.

#### On x86-64

Not written yet. The minimum and maximum instructions there treat zeros and NaN differently from ARM64's, so which `clip` path is exact is a separate question. The candidates for this family are in [x86-64-design.md](x86-64-design.md#candidates-per-family).

## Where the result is narrower than the number

The second item is about the reference and holds on every architecture. The others describe the ARM64 kernels. Whether the `clip` ones arise on x86-64 at all is part of [open issue 1](x86-64-design.md#open-issue-1-techniques-particular-to-x86-64) in the x86-64 design.

- **`clip`: how fast depends on the bounds and, for a zero lower bound, on the data.** The two-operation path is taken when neither `lo` is `+0.0` nor `hi` is `-0.0`, because `fmax(-0.0, +0.0)` is `+0.0` where NumPy keeps `-0.0`. A lower bound of `+0.0`, as in `np.clip(x, 0, 1)`, has a three-operation path of its own: 1.31x to 1.33x over naive C at 4,096 elements on data without negative zeros, against 1.6x for the two-operation path. That path cannot be exact on its own, and no exact form in fewer than the four operations of compare-and-select was found, so it checks each chunk for a -0.0 and redoes the chunk when it finds one. A redo costs about 75 ns, so the path is quick to give up, and it is never tried again in the same call: one -0.0 among the first 7,264 float32 elements (3,632 for float64) sends the rest of the array to compare-and-select, however long the array is. Later than that, a single one costs one redo and the path carries on. At 65,536 float32 elements a call takes 6.1 microseconds with no negative zero, 7.5 with one at index 7,000, which is the speed of naive C, and 6.2 with one at index 8,000. On data with a negative zero that early the kernel runs at the speed of naive C from about 1,000 elements up, and up to about 14% slower at the smallest sizes that take the path, 256 elements for float64 and 512 for float32. Rounded data is such data: a value between -0.5 and 0 rounds to -0.0. The specs now time both kinds, which is where the figures come from. The judge also gained a value class with a single negative zero among ordinary values: without it nothing reached the redo of a chunk followed by more fast chunks, and a kernel whose redo stored nothing passed the check. An upper bound of `-0.0` still takes compare-and-select.
- **`clip`: defined for ordered bounds only.** For `lo > hi` NumPy returns `hi`, and for a NaN bound it returns NaN. The reference does neither. Compared outside the judge over 100 pairs of bounds per type, the reference matches NumPy bit for bit on every ordered pair and on none of the others. The specs now say so, the judge checks twelve ordered pairs (zeros of both signs, equal bounds, infinite, denormal and near-largest bounds), and the binding must send any other call to NumPy.
- **`clip`: reads and writes the floating-point status register.** To stay exact when the data holds a signalling NaN, the kernel reads FPSR after the fast loop and redoes the array if the invalid-operation flag was raised. If the caller left that flag set, it is cleared first and set again afterwards. No rule forbids this, flags are only ever added, and it is how a two-operation path can be exact at all; it is still more than "a pure function over its buffers" suggests, so it is listed under "Decisions".
- **`isin_i64`: fast only for test values between 0 and 65,534, or within 32,767 of the first one**, as above.

## What the judge does not see

Items 1, 2, 4 and 5 are about the judge and hold on every architecture. The measurements quoted in them, and items 3 and 6, come from the ARM64 kernels and from the references as compiled for ARM64.

1. **Scalar values nobody listed.** The judge used to pass one value per scalar. That hid the clip reference being wrong for reversed and NaN bounds and the clip kernel's dependence on the bounds, and it was the reason the faster `histogram` path could not be judged. A spec now lists further combinations ([judge-design.md](judge-design.md#scalar-sets)): twelve pairs of bounds for `clip`, sixteen bin counts and ranges for `histogram`, six values for masked assignment. All four kernels and their references pass them. What remains unseen is any combination that is not listed, and data that depends on the scalars: nothing places elements on a histogram's bin edges. Both let the `histogram_f64` overflow through. No listed range was near the largest float64 values, and the two sets added there since would not have caught it either, because the only values the judge generates inside such a range sit on its outer edge.
2. **One input per size from 4,096 elements up.** Small sizes cycle through a pool of inputs; large sizes repeat one. A branch predictor memorises a repeated input, so code with data-dependent branches looks faster than it is up to about 16,384 elements. Among the references only naive `maskfill_f64` branches on every element, so the speedup reported for masked assignment at those sizes is understated. The naive `compress` and `nonzero` loops are branchless and are not affected. It also hides the mask kernels' own mispredictions when they switch between their dense and sparse paths.
3. **Densities and distributions between the ones declared.** The mask kernels are timed at 50% and 5%, and are weakest near 10%. `bincount_u8` is timed on uniformly random bytes; all-equal bytes run at 2.1 ns per byte below 8,192 elements.
4. **Paths behind a size cap.** `interp_f64` above 65,537 elements.
5. **A wide tolerance at large sizes for float32.** The `cumsum` bound grows with the element count. Past some tens of thousands of float32 elements it no longer notices a single skipped element. Small sizes are not affected.
6. **Prefetch addresses.** The two count kernels prefetch up to 256 KB past the end of the array when it is larger than 16 MB. A prefetch never faults and reads nothing, so the judge passes it.

## What a caller must know

These are constraints on the binding, collected from the kernels. The last column says where a constraint comes from. One that comes from the spec or the reference holds on every architecture. One that comes from the ARM64 kernel is neither assumed nor ruled out for x86-64; the x86-64 rows are added with its kernels.

| Kernel | Constraint | Holds on |
|---|---|---|
| `compress_*`, `nonzero_bool` | Scratch output: allocate the full capacity, return only the reported elements, never run on a buffer the caller owns | Every architecture: the spec declares it |
| `maskfill_f64` | Rewrites unselected elements with their own bits in dense blocks | ARM64 |
| `clip_*` | `lo <= hi`, neither NaN, otherwise fall back | Every architecture: the reference is defined for ordered bounds only |
| `clip_*` | `x` and `out` must not overlap; touches FPSR; with `lo = +0.0`, data with a negative zero in its first few thousand elements runs at the speed of the C | ARM64 |
| `interp_f64` | From 2^32 elements it takes the scalar path | ARM64 |
| `cumsum_f32`, `cumsum_f64` | Not bit-identical to NumPy; `report()` must mark them | Every architecture: the spec declares a tolerance |
| `cumsum_f32`, `cumsum_f64` | Adjacent outputs come from separate running sums, so non-negative input does not give a non-decreasing output, and the steps back are not confined to the last bit: on uniform [0, 1) float32 data at a million elements about one adjacent pair in six decreases, by up to about 5, and after `2^24, 0.9, 0.2` a run of zeros alternates between 2^24 and 2^24 + 2 | ARM64. Any kernel that regroups the additions can do the same |
| `histogram_f64` | Written for uniform bins with a finite range, `lo < hi` and at least one bin; the judge tests sixteen bin counts and ranges | Every architecture |
| Every kernel with a `Scalar` | Verified by the judge for the combinations its spec lists, and no others | Every architecture |

## Decisions

Each decision says which architecture it applies to. One taken for an ARM64 kernel is not taken for x86-64 by default.

Decided since the first version of this document:

- **`histogram_f64` takes the faster path.** 1.30x more, relying on an error-bound proof instead of performing NumPy's corrections on every element. It was adopted once the judge could test other bin counts and ranges. *ARM64. The proof is rechecked before an x86-64 kernel relies on it.*
- **The judge has scalar sets.** Several combinations of scalar values per spec, checked on kernel and reference alike and optionally timed. *Every architecture.*
- **`clip` has a path for a zero lower bound**, timed by the judge on data with and without negative zeros. *ARM64.*

Still open:

1. **`maskfill_f64`: is rewriting unselected elements acceptable?** It is the whole of the gain on dense masks. Recommended: yes for arrays the caller owns exclusively, which is NumPy's normal case, with the caveat documented. *ARM64. On x86-64 a store that leaves unselected elements alone may make the question unnecessary.*
2. **`clip`: is using FPSR acceptable?** Without it the kernel is the compare-and-select loop, which ties the C in cache. Recommended: yes, since the register is per thread and flags are only added, but it should be a stated exception to the kernel rules. *ARM64. On x86-64 the question may not arise.*
3. **Float `cumsum`: one overflow test per block instead of four?** About 12% faster from 256 to 16,384 elements. The proof that infinities land where NumPy has them would no longer hold for a running sum within rounding distance of the largest finite value. Recommended: no. *ARM64. It comes up again for any kernel that keeps the scheme.*

## What is next

- Re-run every kernel locked after this is merged, record the results and generate the leaderboard.
- The binding and `install()` are done: [delivery-design.md](delivery-design.md). What it showed the kernels still need is listed there.
- The x86-64 kernels: [x86-64-design.md](x86-64-design.md). Their results fill the x86-64 sections of this document.
