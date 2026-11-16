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
