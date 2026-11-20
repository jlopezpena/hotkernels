# The kernels and Polars: what a probe found

This is a proposal for review. It asks whether the kernels can serve [Polars](https://pola.rs/) as well as NumPy, reports a probe that timed Polars against them, and lists the ways forward. Nothing is built yet.

Every figure here is ARM64's, measured on an Apple M4 Max. No x86-64 figure has been taken.

## Summary

- **Calling the kernels from another language is easy.** Each kernel is a C-ABI leaf function, such as `int64_t argmax_f64(const double *x, size_t n)`, with no state and no calls. A Rust crate could assemble the kernel set that `hotkernels/meson.build` already writes and declare the kernels `extern "C"`.
- **Polars cannot take them the way NumPy does.** Polars has no dispatch table to replace at run time, so there is no `install()` for it. It also stores booleans as bits, has a null mask beside each column, splits a column into chunks, and may define NaN and ties differently from NumPy. See [What does not carry over](#what-does-not-carry-over).
- **The probe found three groups.** Three Polars operations (`arg_true`, `cum_sum`, `clip`) are 20x to 33x slower than the kernel, and slower than NumPy: most of that gap is a slow path in Polars, which a plain loop would close. Most of the others are 1.8x to 6.8x slower than the kernel in Polars, which is already fast there: that is what assembly adds. Polars wins one: `filter` on a sparse mask, because its mask is one bit per element. See [Results](#results).
- **Done first:** Polars' slow paths are reported as [pola-rs/polars#30903](https://github.com/pola-rs/polars/issues/30903) (`clip`), [pola-rs/polars#30904](https://github.com/pola-rs/polars/issues/30904) (`arg_true`) and [pola-rs/polars#30905](https://github.com/pola-rs/polars/issues/30905) (`cum_sum`), each with a reproducible example and figures from ARM64 and x86-64. No other step is taken until Polars has answered. That costs this project nothing and helps every Polars user more than a binding would.
- **Proposed next, if wanted:** a Rust crate of the kernels, and a Polars expression plugin for the five operations of the second group. A user would opt in per call; nothing would be transparent. See [Ways forward](#ways-forward).

## How the probe was run

A throwaway script, not in the repository, timed each operation three ways on the same data. The script and the commands that build the kernels for it are in [a comment on pull request 178](https://github.com/sofeikov/hotkernels/pull/178#issuecomment-7009343245). The three ways:

- **Polars:** the Series method a Polars user writes, such as `s.arg_max()`.
- **The kernel:** the ARM64 `kernel.s` from `main`, assembled into a shared library and called through `ctypes`. Output buffers were allocated inside the timed call, as Polars allocates its result.
- **References:** NumPy, and the kernel's `naive.c` compiled with `clang -O3 -mcpu=native`.

1,048,576 elements, no nulls, one chunk, best of seven repeats. `searchsorted` with 1,000 needles in a sorted array, `histogram` with 50 bins over (-3, 3), `is_in` against 8 values, masks with 50% or 5% of elements true. The script checked that Polars and the kernel agree where their results are directly comparable: `arg_max`, `arg_min`, `search_sorted`, `sum`, the nonzero count and the length of `filter`.

- **Machine.** Apple M4 Max, macOS, CPython 3.14.8, NumPy 2.5.5, Polars 2.1.0 (runtime `rt32`, index type `UInt32`), Apple clang 21. Native. Load average 1.5 to 2.4.
- **Threads.** Polars on one thread (`POLARS_MAX_THREADS=1`), as the kernels and NumPy run. With Polars' default of 16 threads every figure was within 5%, except two named below.

## Results

Times in microseconds per call.

| Operation | Polars call | Polars | NumPy | naive C | Kernel | Polars / kernel | naive C / kernel |
|---|---|---|---|---|---|---|---|
| Indices of true elements, 50% true | `m.arg_true()` | 3,235 | 359 | 276 | 99 | 32.7x | 2.78x |
| Running sum, int64 | `s.cum_sum()` | 2,813 | 1,510 | 260 | 130 | 21.7x | 2.00x |
| Clip to (-0.5, 0.5), float64 | `s.clip(lo, hi)` | 2,269 | 264 | 141 | 104 | 21.9x | 1.36x |
| Running sum, float64 | `s.cum_sum()` | 2,808 | 2,358 | 510 | 142 | 19.8x | 3.59x |
| Membership in 8 values, int64 | `s.is_in(test)` | 645 | 5,414 | 489 | 95 | 6.83x | 5.18x |
| Sum, uint8 | `s.sum()` | 45 | 133 | 46 | 11 | 4.03x | 4.14x |
| Index of the maximum, float64 | `s.arg_max()` | 163 | 528 | 936 | 72 | 2.26x | 13.0x |
| Index of the minimum, float64 | `s.arg_min()` | 166 | 528 | 936 | 73 | 2.26x | 12.8x |
| Count of non-zeros, float64 | `(s != 0).sum()` | 180 | 257 | 97 | 78 | 2.30x | 1.24x |
| Sorted search, 1,000 needles | `s.search_sorted(v)` | 28 | 18 | 94 | 16 | 1.82x | 6.03x |
| Histogram, 50 equal bins | `s.hist(bin_count=50)` | 1,066 | 4,443 | 1,591 | 533 | 2.00x | 2.99x |
| Select by mask, 50% true | `s.filter(m)` | 197 | 3,152 | 273 | 139 | 1.41x | 1.96x |
| Select by mask, 5% true | `s.filter(m)` | 45 | 618 | 270 | 63 | **0.71x** | 4.27x |

Notes on the rows:

- **Running sum, float64.** The kernel regroups the additions, which is where most of its speed comes from, and its result is not bit-identical to a sequential sum. hotkernels installs it only on request (`install(inexact=True)`). Polars would have to accept the same.
- **Histogram.** `s.hist(bins=edges)` with the 51 edges written out takes 15,181 µs, 14 times as long as `bin_count=50`: it searches the edges for each element. NumPy does the same with an array of edges (22,548 µs). The row above uses `bin_count`, which takes its range from the data rather than from (-3, 3); the work is the same. With 16 threads, `bins=edges` takes 4,671 µs.
- **Sorted search.** With 16 threads Polars took 55 µs, twice its time on one.
- **Count of non-zeros.** Polars builds a mask and then sums it, two passes; the kernel makes one.
- **Masks.** The kernels take NumPy's masks, one byte per element. Polars' `filter` and `arg_true` take Polars' own, one bit per element. Each side was timed with its own mask, so the rows compare the operation, not the layout.

## What the results say

**First group: slow paths in Polars.** `arg_true`, `cum_sum` and `clip` take 2.3 to 3.2 ms in Polars, about 2.2 to 3.1 ns per element. A plain C loop takes 0.14 to 0.51 ms, and NumPy is faster than Polars in all four rows. The gap is Polars' own: a plain Rust loop in `polars-compute` would close most of it, and the kernel adds only its lead over C (1.4x to 3.6x) on top. Assembly is not the answer here; a report to Polars is.

**Second group: what assembly adds.** Where Polars is already fast, the kernels still lead: 2.3x on `arg_max` and `arg_min`, where Polars is already 5.7x faster than the naive C; 6.8x on `is_in` against a short list; 4.0x on `sum` of uint8; 1.8x on `search_sorted`; 2.0x on `hist`. These are the operations a binding would be for.

**Polars wins one.** On a mask with 5% true, Polars' `filter` takes 45 µs and the kernel 63 µs. The kernel reads one byte of mask per element and Polars one bit, so Polars reads 8 times less mask. The kernel packs NumPy's bytes into bits before it works on them, but it has to read the bytes first.

## What does not carry over

| Difference | What it means for a binding |
|---|---|
| **No hook.** Polars is compiled Rust, with no dispatch table to replace at run time | No transparent `install()`. A user calls the kernel by name, through a plugin |
| **Booleans are bits.** Arrow packs a boolean column eight to a byte | The mask kernels (`compress`, `maskfill`, `nonzero`, `where`, `cumsum_bool`, `any_rows_bool`) do not take Polars' masks. They would need variants that read bits, and Polars' `filter` already wins on sparse masks |
| **Nulls.** Each column has a validity mask beside its values | A kernel can run only on a chunk with no nulls, or where the values under a null do not change the result |
| **Chunks.** A column can be several arrays | The binding calls the kernel per chunk and combines: a running sum carries its total into the next chunk, `arg_max` compares the chunks' winners |
| **Semantics.** The kernels match NumPy: the first NaN wins `argmax`, the first maximum wins a tie, signed zeros are kept | Each operation is checked against Polars, not NumPy, before it is offered. The judge's oracle is NumPy, so a Polars oracle is new work |
| **Index type.** Polars returns `UInt32` indices by default | A conversion, or a kernel variant that writes 32-bit indices |

## Ways forward

In order. Each step is worth taking only if the one before it was.

1. **Report the slow paths to Polars.** Done: [pola-rs/polars#30903](https://github.com/pola-rs/polars/issues/30903) (`clip`), [pola-rs/polars#30904](https://github.com/pola-rs/polars/issues/30904) (`arg_true`) and [pola-rs/polars#30905](https://github.com/pola-rs/polars/issues/30905) (`cum_sum`). Each has a reproducible example in Polars and NumPy only, figures from the M4 Max and an Intel Core i9-14900K, and the likely cause in Polars' source. `hist` is left out: its fast path exists, and the slow one is NumPy's behaviour too. The fixes are left to Polars' maintainers. Polars accepts AI-written code only from a contributor who has verified it by hand, so this project sends no patch.
2. **A Rust crate of the kernels**, `hotkernels-sys`: the kernel set `hotkernels/meson.build` already writes, assembled with the `cc` crate in a `build.rs`, the kernels declared `extern "C"`, and the naive C on any other target. Useful on its own to any Rust user, and the base for step 3. Its own design document.
3. **A Polars expression plugin** ([`pyo3-polars`](https://github.com/pola-rs/pyo3-polars)) for the second group: `arg_max`, `arg_min`, `is_in` against a short list, `sum` of uint8, `search_sorted`. Called as `hk.arg_max(pl.col("x"))`. Each operation needs the Polars semantics checked and the chunk and null handling of [What does not carry over](#what-does-not-carry-over). Its own design document, after step 2.
4. **The techniques upstream, in Rust.** The kernels' ideas, written with `std::arch` intrinsics, offered to `polars-compute`. The proofs and dead ends in each `attempts.md` make that easier. Whether Polars wants per-architecture code is Polars' decision.

## Considered and not proposed

| Option | Why not |
|---|---|
| A transparent `install()` for Polars | Polars has nothing to hook. A patched build of Polars would be a fork |
| Mask kernels that read bits | They would serve Arrow's layout, but Polars already wins on sparse masks, and NumPy has no bit layout to serve. Not before step 3 shows a need |
| Offering the assembly files to Polars as they are | Two architectures of hand-written assembly, checked by a judge Polars does not run, is a lot to ask a project to take. Step 4 is the version Polars could review |

## Decisions still open

1. **Whether to go past step 1.** Proposed: wait for Polars' answer to the issue. If it fixes the first group, the case for a binding is the second group only, at 1.8x to 6.8x.
2. **x86-64.** The second group may differ there, both Polars' figures and the kernels' lead. Proposed: run the same probe on the Intel Core i5-14600KF before any design of step 2.

No release note: `docs/` only. No existing statement changes.
