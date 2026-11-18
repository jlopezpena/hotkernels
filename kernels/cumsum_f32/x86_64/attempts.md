# cumsum_f32 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, locked, pinned to one performance core by the judge. Read with the `cumsum_i64` and `cumsum_f64` logs: this kernel was written after both and took its structure from `cumsum_f64` as that stood after its attempt 3.

What the compiler does with `naive.c` (`clang -O3 -march=native`): the scalar chain, one `vaddss` per element at two cycles each: 0.38 ns per element from 1,024 to 1,048,576. NumPy takes 1.6 ns per element at 1,048,576.

## 1. Baseline: window sums of eight from overlapping loads, a magnitude test per chunk

- **Idea:** eight elements per vector. `P` = input plus the load one element back; `P'` = the same two elements further back, two more loads; `D` = `P` + `P'`, the four elements ending at each lane; `W` = `D` plus `D` four elements back, one `vperm2f128`; output = previous output plus `W`. Four loads, five adds and one permute per eight elements, and a chain of one `vaddps`. The shift by two elements is done with loads and not in registers because across the two halves of a register it takes two shuffles, and shuffles share a port with the adds. From `cumsum_f64`: loads ahead of the stores, `vmovntps` from 24 MB of output, a short path below 64 elements, and the magnitude test per chunk. **The proof and its limit are the ARM64 kernel's:** a chunk is accepted when every output is below 2^64, which is what a finite sum of squares guarantees there, and arrays of 2^29 elements or more take the plain loop. Fewer than 16 elements take the plain chain.
- **check:** pass. Tested outside the judge as `cumsum_f64` was: 20,990 runs, large sizes up to 6,400,010 elements so that the cache-bypassing stores are reached. No failure.
- **compare:** not run (first version). `run`: 2.97x over naive C overall, 26.7x over NumPy.
- **Kept:** yes
- **Notes:** 4.4x to 4.9x over the naive C from 1,024 to 65,536. No stall in the close-placement table.

## 2. The adds with a memory operand as fused multiply-adds by one

- **Idea:** the five adds and the permute of a vector all go to two ports, three cycles per vector. `vfmadd231ps` with a register of ones computes the same sum with the same single rounding and runs on a third port. Used for the two adds per `D` that have a load as their operand.
- **check:** pass
- **compare:** significantly slower, 0.979x [0.976 – 0.982]
- **Kept:** no
- **Notes:** 0.94x to 0.95x at 64, 1,024 and 4,096; within 2% elsewhere. The multiply-add takes four cycles where the add takes two, and the ports it frees were not what held the loop in the judge's timing.

## Where it stands

Final `judge run`:

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 2.3 ns | 3.1 ns | 1.34x [1.34 – 1.34] | 732.9 ns | 317x [317 – 318] |
| 64 | 6.5 ns | 13.9 ns | 2.13x [2.12 – 2.13] | 787.9 ns | 121x [120 – 121] |
| 256 | 20.8 ns | 84.5 ns | 4.06x [4.05 – 4.06] | 1.1 us | 53.4x [53.3 – 53.4] |
| 1,024 | 81.2 ns | 380.1 ns | 4.68x [4.66 – 4.71] | 2.4 us | 29.7x [29.6 – 29.9] |
| 4,096 | 322.2 ns | 1.6 us | 4.86x [4.86 – 4.86] | 7.5 us | 23.3x [23.3 – 23.3] |
| 16,384 | 1.4 us | 6.3 us | 4.38x [4.38 – 4.38] | 27.9 us | 19.4x [19.4 – 19.4] |
| 65,536 | 5.7 us | 25.2 us | 4.39x [4.39 – 4.40] | 109.0 us | 19.0x [19.0 – 19.0] |
| 262,144 | 27.6 us | 101.0 us | 3.66x [3.66 – 3.66] | 430.4 us | 15.6x [15.6 – 15.6] |
| 1,048,576 | 152.5 us | 404.0 us | 2.65x [2.64 – 2.66] | 1.71 ms | 11.2x [11.2 – 11.3] |
| 4,194,304 | 1.12 ms | 1.71 ms | 1.52x [1.51 – 1.53] | 6.85 ms | 6.10x [6.05 – 6.13] |
| 16,777,216 | 4.49 ms | 7.02 ms | 1.57x [1.55 – 1.57] | 30.99 ms | 6.91x [6.86 – 6.93] |
| overall |  |  | 2.89x [2.89 – 2.89] |  | 26.1x [26.1 – 26.1] |

With the output buffer 16 bytes behind its input ("apart" is the same kernel with the output clear of the input):

`random`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 305.5 ns | 307.8 ns | 1.01x [1.01 – 1.01] | 1.6 us | 5.13x [5.13 – 5.13] |
| 16,384 | 1.4 us | 1.5 us | 1.07x [1.07 – 1.07] | 6.3 us | 4.38x [4.38 – 4.38] |
| 65,536 | 5.7 us | 6.1 us | 1.06x [1.06 – 1.06] | 25.2 us | 4.40x [4.40 – 4.40] |
| 262,144 | 26.9 us | 42.3 us | 1.57x [1.57 – 1.58] | 114.4 us | 4.25x [4.25 – 4.26] |
| 1,048,576 | 135.4 us | 153.7 us | 1.13x [1.13 – 1.14] | 548.9 us | 4.05x [4.05 – 4.06] |
| 4,194,304 | 1.03 ms | 1.26 ms | 1.22x [1.21 – 1.23] | 3.77 ms | 3.66x [3.65 – 3.68] |
| 16,777,216 | 4.19 ms | 4.51 ms | 1.08x [1.07 – 1.08] | 15.76 ms | 3.76x [3.75 – 3.78] |
| overall |  |  | 1.15x [1.15 – 1.15] |  | 4.21x [4.21 – 4.21] |

No stall; the close placement is the faster of the two at 262,144.

What holds it at each size:

- **16 and 64:** the steps of a call.
- **256 to 65,536:** 0.08 to 0.09 ns per element. The loop needs three cycles per eight elements (0.073 ns per element) for its five adds and one permute on two ports.
- **262,144 and 1,048,576:** the third-level cache: 3.7x and 2.65x over the naive C.
- **4,194,304 and 16,777,216:** memory: 1.5x to 1.6x. Only 16,777,216 has an output of 24 MB or more and uses the cache-bypassing stores.

The candidates the design lists for this family: as in the `cumsum_f64` log. The prefix sum across the register is done with overlapping loads and one `vperm2f128`; non-temporal stores are kept from 24 MB; the overflow test is a compare of magnitudes, with the ARM64 kernel's limit of 2^64 because its proof for `float32` needs that limit; no instruction that is slow on other x86-64 processors is used.

**What differs from the ARM64 kernel for a caller:** the test raises no floating-point flag of its own. The limits are the same: running sums of about 1.8e19 and above take the slow path, and arrays of 2^29 elements or more take the plain loop throughout. Adjacent outputs come from eight separate running sums (four on ARM64), so non-negative input does not give a non-decreasing output. `out` must not overlap `x`.

Left untried: the shift by two elements in registers; two interleaved chains; the cache-bypassing stores between 2 MB and 24 MB of output.
