# any_rows_bool attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, pinned to one performance core by the judge. The runs are `--unlocked`: `spec.py` and `naive.c` are new in the same pull request and not on `main` yet. Neither was changed after the first line of assembly was written. Other judges were running on other cores during these runs, so sizes from 1,048,576 up are noisy.

What the compiler does with `naive.c` (`clang -O3 -march=native`): it vectorises the loop over a row 32 bytes wide and pays its entry and exit for every row. With rows of 22 it takes 0.082 ns per element at every size up to 1,048,576 (1.8 ns per row); with rows of 4, 0.20 ns per element (0.8 ns per row); with rows of 64, 0.032 ns per element. It divides `n` by `cols` first, which is most of its 1.5 ns when there is no row at all. NumPy takes 0.6 to 0.7 ns per element for rows of 22 and 700 ns for a call that has no row.

The timed sizes are element counts. With the default width of 22, 16 elements are no row, 64 are two rows and 4,096 are 186 rows; with a width of 64, both 16 and 64 elements are at most one row.

The kernel does the same work whatever the elements are, so the two timing distributions differ only by where their buffers were placed.

## 1. Baseline: one `vptest` per row

- **Idea:** no division; a row is taken while `cols` elements are left, and the count returned is how far `out` advanced. Rows of up to 32: one 32-byte load at the row's start, `vptest` against a vector whose first `cols` bytes are set, `setnz` to the output. The rows that start in the last 31 bytes of `m` cannot be loaded that way: the bits of the last 32 bytes go to a register through `vpcmpeqb` and `vpmovmskb`, and each such row is `cols` bits of it (`shrx`, `test`, `setnz`). With fewer than 32 elements the bits of all of them are gathered one byte at a time. Rows of 33 to 64: the first 32 bytes or'ed with the last 32. Longer rows: 32-byte chunks or'ed together, the last one ending at the row's end.
- **check:** pass
- **compare:** n/a (first version). `run`: 3.06x [3.05 – 3.07] over naive C over all six tables, 55.1x over NumPy. Rows of 22: 3.95x (sparse) and 3.85x (dense); rows of 4: 2.49x and 2.45x; rows of 64: 2.97x and 2.98x.
- **Kept:** yes
- **Notes:** rows of 22 at 4,096 elements: 65.6 ns, 0.35 ns per row, about 1.85 cycles where the loop is six instructions. Rows of 4: 0.30 ns per row. Rows of 64: 0.52 ns per row. No row (16 elements, width 22 or 64): 1.2 ns against the 1.5 of the reference, which divides. The table with the output 16 bytes behind the input is level with the other: 1.00x from 4,096 to 1,048,576.
