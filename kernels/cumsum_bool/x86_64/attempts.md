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
