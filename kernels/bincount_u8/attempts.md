# bincount_u8 attempts

All runs on an Apple M1 Pro with `--unlocked` (the spec was not on `main` yet), while other
agents were timing other kernels on the same machine, so timings are noisier than usual.

## 1. Baseline: zero with `stp q`, one byte per iteration

- **Idea:** simplest correct version. Zero the 2,048 bytes with a 64-byte `stp q0, q0` loop, then load byte, load count, add, store.
- **check:** pass
- **compare:** n/a (first version). `run`: 0.88x [0.87 – 0.88] vs naive C overall; 0.85x above 256 elements (naive C unrolls by two and runs at 0.29 ns/byte), parity at 16 and 64 where zeroing dominates.
- **Kept:** yes
- **Notes:** starting point only.
