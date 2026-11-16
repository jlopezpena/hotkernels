# nonzero_bool attempts (arm64)

All judge commands in this run were passed `--unlocked`, because the spec is not on `main` yet. Other agents were timing kernels on the same machine, so intervals are wider than usual.

## 1. Scalar baseline

- **Idea:** the naive loop in assembly: store every index at `out[k]`, advance `k` by the mask byte.
- **check:** pass
- **compare:** not run (first version, nothing to compare against)
- **Kept:** yes
- **Notes:** starting point only; expected to match naive C.

## 2. Eight mask bytes per step with a 256-entry position table

- **Idea:** load 8 mask bytes as a u64; one multiply packs them into an 8-bit mask, another gives 8x the true count. The mask indexes a 2 KB table whose 8-byte entry lists the positions of the true bytes in order. Widen those to 64 bits (`ushll` x3), add the block's base index (`uaddw`/`uaddw2` x4) and store all 64 bytes at the write pointer (`stp q` x2). Write pointer advances once per 8 elements. Scalar tail.
- **check:** pass
- **compare:** significantly faster, 2.55x [2.55 – 2.56]
- **Kept:** yes
- **Notes:** half_selected 2.37x, few_selected 2.75x. About 0.119 ns per element in L1 (roughly 3 cycles per 8). half_selected falls to 1.5 to 2.2x above 1M elements, where the output stream is memory-bound.

## 3. Sparse path: 64 mask bytes to a bitmask, 8 unconditional extractions

- **Idea:** per 64 mask bytes, build a 64-bit bitmask in NEON (`ushl` by 0..7, three rounds of `addp`, `fmov`) and count it (`cnt`, `addv`). If at most 8 bytes are true, extract 8 positions with `rbit`/`clz`/`x & (x-1)`, no branches, and store all 8 (extras are rubbish in the scratch buffer). Otherwise run the dense table code (attempt 2) for those 64 bytes. After a dense stretch, go back to sparse if at most 1 in 8 bytes was true, else run a longer dense stretch (up to 512 bytes) before looking again. About 70 instructions per 64 bytes in the sparse path against about 170 dense.
- **check:** pass
- **compare:** significantly faster, 1.24x [1.24 – 1.24]
- **Kept:** yes
- **Notes:** few_selected 1.65x (1.9x from n=1,024 up). half_selected 0.93x: unchanged from n=1,024 to 1M, but n=16 is 0.67x (3.0 to 4.5 ns), n=64 0.85x, n=256 0.94x, from the extra setup and the failed sparse attempt on the first block. 16M half_selected read 0.88x in this run; other agents were timing at the same time and that size is memory-bound, so it needs a recheck. Next: cut the small-n overhead.
