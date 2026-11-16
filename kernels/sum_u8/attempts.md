# sum_u8 attempts

All runs on an Apple M1 Pro with `--unlocked` (the spec was not on `main` yet), while other
agents were timing other kernels on the same machine, so absolute timings are noisier than usual.

What the compiler does with `naive.c` (`clang -O3 -mcpu=native`, Apple clang 21): it widens every
byte to 64 bits before adding. Each 32-byte step is eight `tbl` (to spread the bytes into 32-bit
lanes) and sixteen `uaddw`/`uaddw2` into sixteen 64-bit accumulators: 24 vector instructions on
four vector ports, six cycles per 32 bytes. That is 0.059 ns per byte (about 17 GB/s) at every
size, well below what the caches or memory can deliver, so there is room at every size.

## 1. Baseline: `uaddlv` per 16 bytes, scalar tail

- **Idea:** simplest vector version. One 16-byte load per step, `uaddlv` adds its sixteen bytes to one 16-bit value, moved to a general register and added to the total. Leftover bytes one at a time.
- **check:** pass
- **compare:** n/a (first version). `run`: 2.43x [2.43 – 2.43] vs naive C overall; 2.75x at 16, 2.33x at 64, 2.2–2.9x above.
- **Kept:** yes
- **Notes:** 0.026 ns per byte at large sizes, about 2.4 cycles per 16 bytes: one `uaddlv` and one vector-to-general move per step.
