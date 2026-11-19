# cumsum_bool attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, pinned to one performance core by the judge. The runs are `--unlocked`: `spec.py` and `naive.c` are new in the same pull request and not on `main` yet. Neither was changed after the first line of assembly was written.

What the compiler does with `naive.c` (`clang -O3 -march=native`, clang 21.1.8): the scalar chain, one zero-extending load, one add and one store per element, one cycle each: 0.19 ns per element at every size up to 1,048,576. NumPy takes 1.9 ns per element at 4,096 and 3.5 at 65,536.

The output is eight bytes for every byte read, so beyond the first-level cache the kernel is a writer of memory with a small reader beside it.

## 1. Baseline: four elements per step from a table of 16 entries

- **Idea:** four mask bits index a table whose entry holds the running count within the group (P) and the group's total in every lane (C). With T the count before the group in every lane, the output vector is T + P, with the table load folded into the add, and the next T is T + C: a chain of one `vpaddq` per four elements. The bits of 32 elements come from `vpsllw` by 7 and one `vpmovmskb`; `pdep` (fast on this processor, slow on AMD before Zen 3) spreads them four to a byte, so each byte is a table index times 16 and the address scales it by 4 to the 64 bytes of an entry. Under 32 elements: four at a time with the bits from a 4-byte load and `pext` (the same caveat), then one at a time.
- **check:** pass
- **compare:** n/a (first version). `run`: 1.36x [1.36 – 1.37] over naive C overall, 28.0x over NumPy.
- **Kept:** yes
- **Notes:** 2.54x at 4,096 (0.074 ns per element, 12.7 cycles per block of 32), 1.25x at 16,384 and 65,536, 1.07x at 262,144, 0.78x at 1,048,576 in this run and 1.17x in the second table of the same run. The second table, with the output 16 bytes behind the input within a page, is 0.73x of the first at 4,096 and 0.67x at 16,384, and its own "apart" column has 16,384 at 1.6 us where the first table has 2.5 us: the same code at two placements of the output. Where a 32-byte store falls relative to the 64-byte lines decides the speed here.

## 2. Bring the output to a multiple of 32 bytes first

- **Idea:** up to three elements one at a time until `out` is a multiple of 32, from 35 elements up, so that no store of a block straddles two cache lines. The mask is a byte per element, so any head length is possible; an `out` that is not a multiple of 8 is left as it is.
- **check:** pass
- **compare:** significantly faster, 1.046x [1.038 – 1.054]
- **Kept:** yes
- **Notes:** 1.10x at 1,024, 1.36x at 65,536, 1.12x at 1,048,576; 0.93x to 0.95x at 16 and 64, which pay one more compare. 4,096 is unchanged at 0.073 ns per element.

## 3. The group's total from a broadcast load

- **Idea:** load the total as one 64-bit element with `vpbroadcastq` in place of a 32-byte table load folded into the add, in case two 32-byte loads per step were the limit.
- **check:** pass
- **compare:** significantly slower, 0.991x [0.986 – 0.995]
- **Kept:** no
- **Notes:** nothing changed from 256 elements up (4,096: 298.5 ns against 299.0); 0.86x at 16. The 32-byte loads are not the limit.

## 4. Eight elements per step: a table of byte counts, widened

- **Idea:** found with knock-out variants of attempt 2 in a micro-benchmark outside the judge, at 4,096 elements: the full step takes 12.8 cycles per block of 32, without its store 11.0, without the chain 9.4, without the output add 9.1, and with nothing but the index and the store 5.3. So the stores are not the limit and each table lookup with its add costs about half a cycle more than its share of the pipeline's width. Fewer, larger steps: the eight mask bits of a group index a table of 256 entries of eight bytes, the running count within the group; `vpmovzxbq` from memory widens each half to four 64-bit lanes; a second table of 256 entries holds the group's total, loaded with `vpbroadcastq`. Two adds give the two output vectors and one the next T. The bits come straight from `vpmovmskb`, so the block needs no `pdep`. Outside a block a group's bits come from an 8-byte load and `pext` (fast on this processor, slow on AMD before Zen 3). The tables are 4 KB where attempt 1's was 1 KB.
- **check:** pass
- **compare:** significantly faster, 1.080x [1.069 – 1.090]
- **Kept:** yes
- **Notes:** 1.17x to 1.19x from 256 to 4,096 (0.062 ns per element, 10.5 cycles per block), 1.18x at 16, 1.01x to 1.04x from 16,384 up. Beyond the first-level cache both versions take 0.10 ns per element: the speed at which this processor writes eight bytes per element to the second-level cache.

## 5. Stores past the caches from 24 MB of output

- **Idea:** from 3,145,728 elements the blocks store with `vmovntdq`, which does not read the line it is about to overwrite, followed by one `sfence`. It needs the output at a multiple of 32, which attempt 2 already arranges; an output that could not be brought there keeps the ordinary stores. The threshold is this processor's third-level cache, as in `cumsum_i64`: below it an output written with ordinary stores is left in a cache for the caller.
- **check:** pass. The judge's correctness cases do not reach this path, so it was tested outside the judge: 39 sizes around every path change, from 0 to 4,194,304, with three mask densities, three placements of the mask and seven of the output, two of them not multiples of 8, against the scalar loop, with a guard byte after the output: 2,457 runs, no failure.
- **compare:** significantly faster, 1.196x [1.183 – 1.209]
- **Kept:** yes
- **Notes:** 2.61x at 4,194,304 (1.70 ms to 648 us) and 2.60x at 16,777,216, nothing elsewhere. Far more than the 1.25x the same stores gave `cumsum_i64`: here the output is nearly all of the memory traffic.

## Where it stands

Final `judge run --record` (unlocked): 1.89x [1.88 – 1.91] over naive C overall, 39.0x over NumPy.

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 2.6 ns | 2.9 ns | 1.14x [1.13 – 1.15] | 1.1 us | 447x |
| 64 | 5.9 ns | 10.8 ns | 1.81x [1.80 – 1.82] | 1.2 us | 202x |
| 256 | 18.2 ns | 46.4 ns | 2.55x [2.54 – 2.55] | 1.6 us | 86.3x |
| 1,024 | 66.8 ns | 191.7 ns | 2.87x [2.86 – 2.88] | 2.8 us | 42.4x |
| 4,096 | 261.5 ns | 775.1 ns | 2.96x [2.96 – 2.97] | 7.6 us | 29.1x |
| 16,384 | 1.6 us | 3.1 us | 1.95x [1.95 – 1.95] | 26.6 us | 16.7x |
| 65,536 | 6.6 us | 12.4 us | 1.89x [1.89 – 1.89] | 233.7 us | 35.6x |
| 262,144 | 36.6 us | 49.9 us | 1.37x [1.33 – 1.41] | 1.01 ms | 27.7x |
| 1,048,576 | 261.5 us | 212.8 us | 0.81x [0.79 – 0.88] | 2.99 ms | 11.4x |
| 4,194,304 | 723.1 us | 1.90 ms | 2.63x [2.57 – 2.68] | 11.05 ms | 15.3x |
| 16,777,216 | 2.86 ms | 7.66 ms | 2.68x [2.57 – 2.73] | 42.76 ms | 14.9x |

(The table is from the run before the recorded one; the recorded run's figures are within 3% of it at every size.)

**1,048,576 elements.** The first table has the kernel slower than the naive C there. The second table of the same run has it at 174 us against 204 us, 1.17x faster, and so does a micro-benchmark of the two alone (174 us and 177 us at two placements against 207 us and 202 us). The first table times NumPy between the two, and NumPy allocates 8 MB for its result on every call; the kernel is timed straight after it, with its own output pushed out of the third-level cache.

The second table, output 16 bytes behind the input within a page: within 2% of the first at every size from 4,096 to 16,777,216 except 1,048,576, for the reason above.

Not tried:

- **A function hook's worth of work in the kernel for tiny arrays.** At 16 elements the kernel takes 2.6 ns and the call from Python 338 ns, which is `np.cumsum`'s wrapper; that is the binding's to remove, not the kernel's.
- **The byte counts computed in the register** (shifts and adds over 32 bytes, widened with `vpshufb`), which needs no table: by count it is about 29 vector instructions per block on three ports, no better than the 10.5 cycles the table version takes.
- **Stores past the caches below 24 MB.** In the micro-benchmark they would gain at 1,048,576 elements too. Not used there for the reason `cumsum_i64` gives: the caller gets an output that is in no cache.
