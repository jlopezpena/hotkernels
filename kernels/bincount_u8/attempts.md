# bincount_u8 attempts

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
