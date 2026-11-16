# compress_f64 attempts (arm64)

All judge commands in this run were passed `--unlocked`, because the spec is not on `main` yet. Other agents were timing kernels on the same machine, so intervals are wider than usual.

## 1. Scalar baseline

- **Idea:** the naive loop in assembly: store every element at `out[k]`, advance `k` by the mask byte.
- **check:** pass
- **compare:** not run (first version, nothing to compare against)
- **Kept:** yes
- **Notes:** starting point only; expected to match naive C.

## 2. Eight elements per step, compaction by load address

- **Idea:** a pair needs one 16-byte store: `{x[p], x[p+1]}` if `mask[p]`, else `{x[p+1], x[p+2]}`. So load each pair from `x + 8*(p + 1 - mask[p])` and store it whole; no shuffle and no table. Mask bytes loaded as a u64 `m`; `C - (m << 3)` gives the four load offsets in 16-bit fields, `m * 0x0808…08` gives the store offsets (running counts times 8) in bytes. Write pointer advances once per 8 elements. Main loop needs 9 elements left (the pair load reads one past the pair); exactly 8 left go through a block whose last pair uses scalar stores; scalar tail below that.
- **check:** pass
- **compare:** significantly faster, 1.94x [1.93 – 1.96]
- **Kept:** yes
- **Notes:** 2.4x in L1 (about 0.139 ns per element), 2.0x for few_selected beyond L1, 1.2 to 1.8x for half_selected beyond L1. n=16,384 is an outlier (1.5 to 1.6x): x and out are each 128 KB there, the size of L1.

## 3. Sparse path: 64-bit bitmask and unconditional `clz` extractions, tiers of 6 and 12

- **Idea:** same scheme as nonzero_bool attempts 3 to 5. Per 64 elements, build a bitmask in NEON (`ushl` by 0..7, three `addp`, `fmov`, `rbit`) and count it. If at most 6 are selected, find 6 positions with `clz` / `lsr` / `bic` and copy `x[pos]` with scalar `ldr d` / `stp d`, no branches; 7 to 12 selected use a second copy with 12 extractions; more than 12 falls back to the dense code (attempt 2) for those 64 elements. After a dense stretch, go back to sparse if at most 1 in 8 was selected, else run a longer dense stretch (up to 512 elements). An exhausted `clz` gives 64 and copies `x[64]` as rubbish, so the sparse loop needs 65 elements left; exactly 64 left use a copy that masks the position with `and #63`.
- **check:** pass
- **compare:** significantly faster, 1.43x [1.43 – 1.44]
- **Kept:** yes
- **Notes:** few_selected 2.07x: 2.2x to 4.4x up to 16,384, 2.0 to 2.1x up to 1M, only 1.05x at 4M and 16M. half_selected 0.99x (n=64 0.94x and n=256 0.97x from the failed sparse attempt on the first block). At 4M and 16M the sparse path reads x at scattered positions, the hardware prefetcher does not follow, and the time is set by how many cache misses fit in the out-of-order window (about 31 cycles per 64 elements).
