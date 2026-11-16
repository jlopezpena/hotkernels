# Survey results: Apple M1 Pro

- NumPy 2.5.5, Python 3.12.7, macOS-27.2.1-arm64-arm-64bit
- Naive C: Apple clang version 21.0.0 (clang-2100.4.1.1), `-O3 -mcpu=native -std=c11 -Wall -Wextra`
- Reference loop drift over the run: 1.2%
- Git revision: `d2dea79`

Each ratio is NumPy time divided by naive C time. Above 1 means NumPy is slower than the naive loop.
`C SIMD` gives the vector width clang's loop vectoriser used for the naive kernel, or `no`. `alloc` marks statements where NumPy allocates
its result; the C side then mallocs and frees its output on every call as well.

| Candidate | NumPy statement | C SIMD | alloc | large-n gap | n=16 | n=4,096 | n=65,536 | n=1,048,576 | n=16,777,216 | C ns/elem at 1M | NumPy ns/elem at 1M | NumPy at n=1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| compress_f64 p=0.5 (branchless C) | `x[mask]` | no | yes | **13.8x** | 9.21x | 7.85x | 14.1x | 14.0x | 13.2x | 0.34 | 4.80 | 201 ns |
| compress_f64 p=0.05 (branchless C) | `x[mask]` | no | yes | **2.70x** | 8.54x | 1.49x | 2.58x | 2.72x | 2.82x | 0.34 | 0.93 | 85 ns |
| nonzero_bool p=0.05 (branchless C) | `np.nonzero(mask)[0]` | no | yes | **2.24x** | 12.4x | 1.08x | 1.89x | 2.41x | 2.45x | 0.36 | 0.88 | 310 ns |
| nonzero_bool p=0.5 (branchless C) | `np.nonzero(mask)[0]` | no | yes | **1.95x** | 13.1x | 2.43x | 1.96x | 1.93x | 1.96x | 0.38 | 0.72 | 307 ns |
