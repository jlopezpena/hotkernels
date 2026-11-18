# bincount_u8 attempts

This log holds two architectures. ARM64 (Apple M1 Pro) comes first. x86-64 (Intel Core i5-14600KF) follows under [x86-64](#x86-64), with its own numbering.

All runs on an Apple M1 Pro with `--unlocked` (the spec was not on `main` yet), while other
agents were timing other kernels on the same machine, so timings are noisier than usual.

## 1. Baseline: zero with `stp q`, one byte per iteration

- **Idea:** simplest correct version. Zero the 2,048 bytes with a 64-byte `stp q0, q0` loop, then load byte, load count, add, store.
- **check:** pass
- **compare:** n/a (first version). `run`: 0.88x [0.87 – 0.88] vs naive C overall; 0.85x above 256 elements (naive C unrolls by two and runs at 0.29 ns/byte), parity at 16 and 64 where zeroing dominates.
- **Kept:** yes
- **Notes:** starting point only.

## 2. Load eight input bytes at once, extract with `ubfx`

- **Idea:** one 64-bit load per eight elements instead of eight byte loads; indices come out of the register with `and`/`ubfx`/`lsr`. Still a single table (`counts` itself).
- **check:** pass
- **compare:** significantly faster, 1.370x [1.368 – 1.374]
- **Kept:** yes
- **Notes:** 1.44–1.50x from 256 elements up (0.234 ns/byte against 0.34), 1.03x at 16 and 1.11x at 64. Still about 0.75 cycles per byte, above the two-stores-per-cycle limit, which suggests increments are waiting on each other through memory.

## 3. Four 32-bit count tables on the stack, used for every n >= 8

- **Idea:** byte j of each input word increments table j % 4, so neighbouring equal values do not wait on each other through memory; tables summed into `counts` at the end. Chunked so the 32-bit counters cannot overflow.
- **check:** pass
- **compare:** significantly slower, 0.788x [0.786 – 0.789]
- **Kept:** no
- **Notes:** the fixed cost (zero 4 KB, sum 256 slots) is about 65 ns: 0.31x at n=16, 0.40x at 64, 0.62x at 256, 0.86x at 1,024, 0.96x at 4,096. From 16,384 up it is only 1.01–1.03x. A private micro-benchmark (1M bytes) showed why: on uniformly random bytes the single-table loop is not limited by memory dependencies at all (random 0.228 ns/byte, a repeating 0..255 ramp 0.221). Removing the stores gives 0.144 ns/byte and removing the count loads 0.16–0.19, which fits a limit of roughly 8 micro-ops per cycle (about 5 per element: extract, load, add, two for the store) and not a dependency chain. The four tables do matter for skewed data, which the judge does not time: all-equal bytes 2.11 -> 0.60 ns/byte, four distinct values 0.94 -> 0.32 ns/byte. Worth retrying only above a size threshold (see later attempts).

## 4. Pair table: 65,536 counters indexed by two bytes (measured privately, never run through `compare`)

- **Idea:** halve the loads and stores per element by counting 16-bit pairs in a 256 KB table of 32-bit counters on the stack, then fold rows and columns into the 256 counts.
- **check:** not run (private micro-benchmark with its own result check, which passed)
- **compare:** not run. Micro-benchmark, random bytes: 0.280 ns/byte at 1M and 0.275 at 16M, against 0.228 for the single table; 0.418 ns/byte at 65,536 because of the ~10 us fold.
- **Kept:** no
- **Notes:** the table is twice the 128 KB L1 data cache, and uniformly random pairs miss it about half the time. On cache-friendly data it is fast (0.122 ns/byte on a ramp), but the judge times random bytes. It would also need 256 KB of stack. Dead end for random input.

## 5. Zero `counts` with `dc zva`

- **Idea:** zeroing 2 KB with 16-byte stores is bounded by two stores per cycle (about 64 cycles, 20 ns, most of the time at n <= 64). `dc zva` clears a 64-byte line per instruction: 31 aligned lines with `dc zva`, the first and last 64 bytes with ordinary stores.
- **check:** pass
- **compare:** significantly slower, 0.955x [0.954 – 0.957]
- **Kept:** no
- **Notes:** 0.85x at 16, 0.82x at 64, 0.89x at 256. In a private micro-benchmark, 32 back-to-back `dc zva` on an aligned buffer took 119 ns per call with n=0 (about 12 cycles per line) against 20 ns for the `stp q0, q0` loop. `dc zva` is not a fast way to clear a small, hot, L1-resident buffer on this core. The `stp` loop already runs at the store-port limit, so the zeroing cost at small n looks like a floor.

## 6. Four stack tables above 8,192 elements, 16 bytes per step, loads/adds/stores grouped

- **Idea:** attempt 3 again, but only where its fixed cost is repaid, and with a tighter loop: `ldp` of two words per step (no post-index writeback, one compare-and-branch per 16 bytes) and each group of four bytes done as four loads, four adds, four stores, which is legal because the four go to different tables.
- **check:** pass (also passed with `CHUNK` temporarily set to 1,024 and the threshold to 16, to exercise the multi-chunk flush and accumulate path that needs n > 2^31 otherwise)
- **compare:** significantly faster, 1.026x [1.023 – 1.028]
- **Kept:** yes
- **Notes:** 1.04–1.05x from 16,384 up, unchanged below. Micro-benchmark at 1M random bytes: 0.212 ns/byte against 0.227 (about 5.5 cycles per 8 elements; the ~42 micro-ops per 8 elements put the floor near 5.25). Side benefit the judge does not time: all-equal input 2.11 -> 0.59 ns/byte, four distinct values 0.94 -> 0.30.

## 7. Small path: 16 bytes per step

- **Idea:** give the single-table path the same loop shape as the large path: `ldp` of two words, pointer bump and one compare-and-branch per 16 bytes instead of a post-indexed load and a branch per 8. Increments stay strictly in order (one table). One optional 8-byte step and up to 7 single bytes finish the tail.
- **check:** pass
- **compare:** significantly faster, 1.003x [1.001 – 1.006]; rerun because it was marginal: significantly faster, 1.004x [1.002 – 1.007]
- **Kept:** yes
- **Notes:** the gain is at 1,024 (1.03–1.04x) and 4,096 (1.02x) only; 16–256 are within noise (first run showed 0.97x at 16, the rerun 1.00x) and sizes from 16,384 up use the other path.

## 8. Lower the four-table threshold from 8,192 to 4,096

- **Idea:** let n=4,096 use the faster large-path loop.
- **check:** pass
- **compare:** no significant difference, 0.999x [0.997 – 1.001]
- **Kept:** no
- **Notes:** 0.98x at 4,096 (980 ns against 960 ns): the fixed cost of the large path (zero 4 KB of tables, sum them into `counts`, about 65 ns against 20 ns for zeroing `counts`) is not repaid there. 8,192 stays.

## 9. Other shapes for the large-path loop (measured privately, never run through `compare`)

- **Idea:** look for a loop body that gets past about 5.6 cycles per 8 elements.
- **check:** private micro-benchmark with its own result check, which passed for every variant
- **compare:** not run. 1M random bytes, two rounds each, kept version 0.217 ns/byte:
  - one `ldrb` per element instead of extracting from a 64-bit word: 0.264 ns/byte (slower);
  - half `ldrb`, half extracted from a 32-bit load: 0.245–0.247 (slower);
  - load-add-store per element instead of grouped by four: 0.218 (same);
  - 32 bytes per iteration: 0.217 (same).
- **Kept:** no
- **Notes:** more loads make it slower, more unrolling does nothing. Each element needs one counter load and one counter store whatever the loop shape, and the stores alone cost about 4 cycles per 8 elements at two per cycle. I found no way to count uniformly random bytes with fewer than one store per element that stays in the L1 cache (see attempt 4), so I stopped here.

## Not tried

- 16-bit counters in the stack tables (2 KB instead of 4 KB to zero, flushed every 262,128 bytes): would save about 20 ns per call on the large path, under 0.6% at 16,384 and less above. Too small to measure against the noise on a shared machine.
- Fixing the single-table small path for skewed input (all-equal bytes run at 2.1 ns/byte below 8,192 elements): the judge times uniformly random bytes only, and a multi-table small path costs more than it gains there (attempt 3).

# x86-64

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core), Linux, locked, pinned to one performance core by the judge.

What the compiler does with `naive.c` (`clang -O3 -march=native`): a call to `memset` and a loop unrolled eight times of a byte load and an `inc` on memory, 0.185 ns per byte, one cycle, up to 16,384 bytes. That is the limit of the scheme: one counter is loaded, incremented and stored per byte, and this processor makes one store per cycle unless two go to the same cache line. NumPy takes 0.8 ns per byte at 1,048,576.

**The naive C's time is not steady here.** Between runs of the judge it varied by up to 2x at single sizes (4,096 bytes: 759 ns in two runs, 1.7 us in a third; 64 bytes: 20 to 29 ns). Its counters are the output buffer and its loads follow its stores; which sizes are hit differs from process to process. The kernel's own times did not move.

## x86-64 1. Eight bytes per load, four tables from 32,768 bytes

- **Idea:** the design's candidate, the ARM64 scheme with an increment that addresses memory directly. One 8-byte load feeds eight `inc`, the byte taken with `movzx` from the low two bytes of the register and a shift by 16 between pairs. From 32,768 bytes the counts go into four tables of 256 counters below the stack pointer, byte i into table i mod 4, added up at the end, so that equal neighbouring bytes update different counters. Below that the counts go straight into the output, cleared with `rep stosb`.
- **check:** pass
- **compare:** not run (first version). Against the naive C in `run`: 1.07x overall in the final run (table below), 1.08x and 1.2x in two earlier ones, the difference being the naive C's (see above).
- **Kept:** yes
- **Notes:** three things were settled on the way to this version, by `run` and not by `compare`, there being no committed version yet. **Tables from 2,048 bytes:** 0.91x to 0.96x from 2,048 to 16,384, where clearing and adding four tables costs more than they save; they pay from 65,536 (1.16x to 1.27x). **Clearing 2 KB with a loop of 32-byte stores:** 22.8 ns for a call with 16 bytes against the naive C's 12; the first increments read counters whose zeros are still in the store buffer as parts of wider stores. `rep stosq` 15.4 ns, `rep stosb` 16.8 ns; `memset` in the C library does the same job in about 10. **A byte loop below the threshold** was 5% to 13% slower than the C's unrolled loop; the eight-bytes-per-load loop is level with it.

## x86-64: where it stands

Final `judge run`:

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 16.8 ns | 11.4 ns | 0.68x [0.67 – 0.68] | 474.6 ns | 28.3x [28.2 – 28.3] |
| 64 | 25.3 ns | 21.3 ns | 0.84x [0.83 – 0.84] | 515.0 ns | 20.3x [20.3 – 20.4] |
| 256 | 63.4 ns | 110.7 ns | 1.75x [1.74 – 1.75] | 682.9 ns | 10.8x [10.8 – 10.8] |
| 1,024 | 200.0 ns | 198.6 ns | 0.99x [0.99 – 0.99] | 1.3 us | 6.28x [6.27 – 6.28] |
| 4,096 | 758.5 ns | 760.1 ns | 1.00x [1.00 – 1.00] | 3.8 us | 4.97x [4.97 – 4.98] |
| 16,384 | 3.0 us | 3.0 us | 1.00x [1.00 – 1.00] | 13.2 us | 4.37x [4.37 – 4.37] |
| 65,536 | 12.8 us | 15.3 us | 1.20x [1.20 – 1.20] | 50.8 us | 3.97x [3.97 – 3.98] |
| 262,144 | 50.6 us | 60.4 us | 1.19x [1.19 – 1.19] | 208.2 us | 4.11x [4.11 – 4.12] |
| 1,048,576 | 202.9 us | 229.5 us | 1.13x [1.13 – 1.13] | 850.5 us | 4.19x [4.19 – 4.19] |
| 4,194,304 | 832.7 us | 931.0 us | 1.12x [1.12 – 1.12] | 6.32 ms | 7.59x [7.58 – 7.62] |
| 16,777,216 | 3.32 ms | 3.81 ms | 1.15x [1.15 – 1.15] | 26.17 ms | 7.89x [7.87 – 7.92] |
| overall |  |  | 1.07x [1.07 – 1.07] |  | 7.37x [7.36 – 7.37] |

With the output buffer 16 bytes behind its input ("apart" is the same kernel with the output clear of the input):

`random`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 757.4 ns | 758.8 ns | 1.00x [1.00 – 1.00] | 758.9 ns | 1.00x [1.00 – 1.00] |
| 16,384 | 3.0 us | 3.0 us | 1.00x [1.00 – 1.00] | 3.0 us | 1.00x [1.00 – 1.00] |
| 65,536 | 12.8 us | 12.7 us | 0.99x [0.99 – 0.99] | 16.1 us | 1.26x [1.26 – 1.26] |
| 262,144 | 50.6 us | 50.7 us | 1.00x [1.00 – 1.00] | 63.1 us | 1.25x [1.25 – 1.25] |
| 1,048,576 | 202.9 us | 202.5 us | 1.00x [1.00 – 1.00] | 232.1 us | 1.14x [1.14 – 1.14] |
| 4,194,304 | 811.4 us | 811.1 us | 1.00x [1.00 – 1.00] | 923.7 us | 1.14x [1.14 – 1.14] |
| 16,777,216 | 3.32 ms | 3.33 ms | 1.00x [1.00 – 1.01] | 3.78 ms | 1.14x [1.14 – 1.14] |
| overall |  |  | 1.00x [1.00 – 1.00] n.s. |  | 1.13x [1.13 – 1.13] |

What holds it: one counter update per byte, at one store per cycle. From 1,024 to 16,384 bytes the kernel and the naive C are the same loop at the same speed. At 16 and 64 bytes the kernel loses to the C library's `memset` (0.68x and 0.84x). From 65,536 the naive C slows to 0.22 to 0.23 ns per byte and the four tables do not.

**What the judge does not time: bytes that repeat.** A micro-benchmark outside the judge, 1,048,576 bytes, ns per byte, naive C then kernel: uniform random 0.220 and 0.194; 16 distinct values 0.206 and 0.194; runs of 1 to 8 equal bytes 0.286 and 0.204; all bytes equal **1.390 and 0.351**. In one table every update of the same counter waits for the one before it to be stored; in four tables it is every fourth. Images and text are closer to the last rows than to the first.

The candidates the design lists for this kernel (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| The same scheme with an increment that addresses memory directly | Kept |
| `rep stosq` to clear the tables | Used for the output on the short path (as `rep stosb`, see above); the four tables are cleared with a loop of 32-byte stores, which costs nothing there because the first increments come long after |
| The number of tables is retested | Four from 32,768 bytes, one below. Two and eight were not tried |
| Alignment of data, non-temporal stores, masked loads | Not tried: an 8-byte load per step and a 2 KB output |
| An output just behind the input | No cost for the kernel in the judge's close table |
| Instructions slow on other x86-64 processors | None used |

**Verdict for a person to confirm:** assembly, by a small margin on the judge's data (1.07x to 1.2x overall, 1.12x to 1.2x from 65,536 bytes) and a large one on repeated bytes, which the judge does not time. Shipping the C here would be defensible on the judge's figures alone.

Left untried: 32-bit counters, which halve the tables; a pair table of 65,536 entries (slower on ARM64).
