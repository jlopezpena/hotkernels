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
