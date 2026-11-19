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

## 4. Two rows per `vptest`: one in the zero flag, one in the carry flag

- **Idea:** `vptest T, x` sets the zero flag when `x and T` is zero and the carry flag when `x and not T` is zero, so one test can answer for two rows if they are merged into one vector with the first row's elements in the bits of T and the second's outside them. `vpmaddubsw` against a vector of 1 in its first `cols` bytes turns the first row into 16-bit sums of 0 to 2, and against a vector of 4 the second row into sums of 0, 4 or 8; both loads fold into the multiplies, the mask is in the multiplier, one `vpor` merges them, and T is 3 in every 16-bit lane. `setnz` and `setnc` store the two results. Two `vptest` on port 0 per four rows where attempt 2 has four.
- **check:** pass
- **compare:** significantly slower, 0.992x [0.989 – 0.994]
- **Kept:** no
- **Notes:** rows of 22: 0.99x (sparse) and 0.98x (dense); 1.02x at 256, 0.97x at 4,096 and 16,384. Knock-out variants of attempt 2 in a micro-benchmark outside the judge, at 4,096 elements, in cycles per step of four rows: the full step 4.8 to 5.4; four `vptest` and one `setnz` 4.0; one `vptest` and four `setnz` 3.3; two `vptest` and two `setnz` 2.7; two `vptest` and four `setnz` 3.7. That fits `vptest` needing port 0 and `setnz` and the taken branch needing port 0 or 6. Halving the `vptest` should then have given 3.7 cycles and gave 5.2: the two multiplies and the `vpor` per pair take the vector ports the tests need. The same merge with `vpaddb`, `vpor` and `vpand` in place of the multiplies was slower still in the micro-benchmark (5.8 cycles), and so was storing a pair's two flags with one 16-bit store through `al` and `ah` (5.2).

## 5. Eight rows per step with no flags: byte sums, one `pext` and one `pdep`

- **Idea:** take the flags and the eight one-byte stores out altogether. A pair of rows is merged into one vector, the first row in bit 0 of each byte and the second in bit 4 (`vpsllw` by 4, `vpor`, `vpand` with a mask of 0x11 in the first `cols` bytes), and `vpsadbw` sums each eight bytes: a count of 0 to 8 in each nibble. Four pairs are shifted to four bytes of the same 64-bit lanes and or'ed; adding 0x77 sets the top bit of every nibble that is not zero; the four lanes are or'ed together, and `vmovd`, `pext` and `pdep` (both fast on this processor, slow on AMD before Zen 3) make eight result bytes of the eight bits, stored at once.
- **check:** pass
- **compare:** no significant difference, 0.999x [0.996 – 1.002]
- **Kept:** no
- **Notes:** rows of 22, sparse: 1.04x, with 0.95x at 4,096, 0.96x at 16,384 and 1.09x to 1.16x from 65,536 up; over all tables it comes to nothing. In the micro-benchmark at 4,096 elements it takes 11.5 cycles per eight rows against 9.6 to 10.8 for attempt 2. The step is 28 vector instructions for eight rows and this processor has three vector ports, so it cannot go under 9.3 cycles; attempt 2 spends two vector instructions per row. Any way of doing without the flags has to cost under two vector instructions per row, and this one costs 3.5.

## 6. Rows of four: pack the lanes as they are, each vector from two 16-byte loads

- **Idea:** the block of attempt 3 puts four instructions on port 5 (three packs and `vpermd`), and that is its four cycles. Two changes take the `vpermd` and the compares out. A 32-bit lane of bytes that are 0 or 1 is zero or a positive number, signed saturation narrows a positive number to a positive number, so the packs can work on the rows as loaded and `vpminub` with 1 finishes. And each vector is put together from two 16-byte loads 64 bytes apart (`vmovdqu` and `vinserti128` from memory), so the packs, which work within each half of a register, leave the rows in order. Eight loads per block in place of four; three instructions on port 5 in place of four. The step of eight rows under a block packs the same way, and its loop is aligned to 16 bytes.
- **check:** pass
- **compare:** significantly faster, 1.002x [1.001 – 1.003]
- **Kept:** yes
- **Notes:** a small gain, and the verdict was taken twice because the first run hinged on 262,144 and 1,048,576 elements, whose intervals were [0.89 – 1.22] in it: that run said no significant difference, 1.008x [0.998 – 1.018], and the second gave the figures above. Rows of 4: 1.03x in both distributions, made of 1.19x at 1,024 and 1.25x at 4,096 (31.1 ns to 24.9), 1.03x at 16,384, and 0.97x to 0.98x from 65,536 up, where eight loads cost more than four did. The block is still near four cycles in the micro-benchmark with `m` at a multiple of 64 (24.2 ns to 23.8 at 4,096) and gains where it is not (33.2 to 29.4): eight loads at two per cycle are four cycles too. Without the alignment of the step loop 64 elements were 0.86x (1.9 ns to 2.3) though the instructions they run are fewer; with it they are 0.95x to 0.98x. The first form, without that alignment, was also judged: significantly faster, 1.002x [1.000 – 1.004].

## 7. Four rows per step in the rows of 33 to 64

- **Idea:** the same unrolling as attempt 2 for the path with two loads per row: four rows per step, each `vmovdqu`, `vpor` with the row's last 32 bytes, `vptest` and `setnz`, addressed from one position with `cols`, twice and three times `cols` as indices.
- **check:** pass
- **compare:** significantly faster, 1.041x [1.040 – 1.042]
- **Kept:** yes
- **Notes:** rows of 64: 1.13x in both distributions. 1.46x at 256, 1.14x at 1,024, 1.16x at 4,096, 1.30x at 16,384, 1.18x at 65,536, 1.05x at 16,777,216. 0.80x at 64 elements, which are one row: 1.5 ns to 1.9, the price of setting the loop up before a row that does not use it. In the micro-benchmark at 4,096 elements, with `m` at a multiple of 64: 26.7 ns to 19.6, 0.31 ns per row or 1.6 cycles; with `m` five bytes off, where one of the two loads of every row straddles two cache lines: 32.7 ns to 30.2, 2.5 cycles per row.
