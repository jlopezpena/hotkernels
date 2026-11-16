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
