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

## 2. Four rows per step in the rows of up to 32

- **Idea:** four `vptest` and four `setnz` per step, the rows addressed as `rdi`, `rdi + cols`, `rdi + 2 * cols` and `rdi + 3 * cols`, so one add moves the position and one compare ends the loop for four rows.
- **check:** pass
- **compare:** significantly faster, 1.122x [1.115 – 1.129]
- **Kept:** yes
- **Notes:** rows of 22: 1.20x (1.43x at 256, 1.27x at 4,096, 1.14x from 65,536 up); rows of 4, which take the same path: 1.16x; rows of 64 unchanged. 4,096 elements in rows of 22 take 50.3 ns: 0.27 ns per row, 1.4 cycles. The four `vptest` need port 0, and the four `setnz` and the branch need port 0 or 6: nine instructions on two ports is 1.1 cycles per row at best, and that is the limit of one flag per row.

## 3. Rows of four: a row is a 32-bit lane

- **Idea:** a path of its own for `cols` = 4, chosen at entry. `vpcmpeqd` against zero marks the rows without a true element, eight per vector; two `vpackssdw` and one `vpacksswb` narrow four such vectors to 32 bytes, `vpermd` puts the lanes of the packs back in row order, and subtracting -1 turns -1 and 0 into 0 and 1. One 32-byte store per 32 rows. The rows left after the blocks are done by one more block that ends at the last row and stores some rows a second time; under 32 rows the same with steps of eight rows, and under eight rows one `cmp dword ptr` and `setne` per row.
- **check:** pass
- **compare:** significantly faster, 1.812x [1.789 – 1.835]
- **Kept:** yes
- **Notes:** rows of 4: 5.96x (sparse) and 5.95x (dense): 2.5x at 16 elements, 3.2x at 64, 9.0x at 1,024, 10.5x at 16,384, 7.8x at 65,536, 4.0x at 4,194,304 and above. 16,384 elements take 100.5 ns: 0.79 ns per block of 128 elements, about four cycles, which is the three packs and the `vpermd`, all four on port 5. The other tables are unchanged. Rows of 2 and of 8 could be done the same way with `vpcmpeqw` and `vpcmpeqq`, and rows of 1 are a copy; the judge times none of them, so `compare` could not have kept them and they still take the path of attempt 2.
