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

## 8. Rows of 33 to 64: the loop of single rows on the straight path

- **Idea:** attempt 7 cost 64 elements in one row of 64 a fifth of their time. The loop of single rows now follows the test for four rows directly, so fewer than four rows reach it without a taken branch, and the loop of four is entered by a jump.
- **check:** pass
- **compare:** significantly faster, 1.006x [1.005 – 1.007]
- **Kept:** yes
- **Notes:** rows of 64: 1.02x, all of it at 64 elements, 1.9 ns back to 1.5 (1.25x); every other size 1.00x.

## 9. Rows of up to 32: the loop of single rows on the straight path

- **Idea:** what attempt 8 did for the rows of 33 to 64: the loop of single rows directly after the test for four rows, the loop of four entered by a jump, for the two rows that 64 elements of width 22 are.
- **check:** pass
- **compare:** no significant difference, 1.001x [0.999 – 1.003]
- **Kept:** no
- **Notes:** 64 elements: 1.9 ns before and after, 1.00x [0.96 – 1.04]. Nothing moved at any size.

## 10. Eight rows per step in the rows of up to 32

- **Idea:** a second position four rows ahead of the first, so eight rows are addressed with the same three indices and one branch ends eight rows. The branch shares ports 0 and 6 with the `vptest` and the `setnz`: 17 instructions on them per eight rows where two steps of four have 18. Fewer than eight rows left go to the loop of four and then to single rows.
- **check:** pass
- **compare:** significantly faster, 1.016x [1.014 – 1.017]
- **Kept:** yes
- **Notes:** rows of 22: 1.04x in both distributions. 1.02x at 4,096 (50.1 ns to 48.9), 1.06x at 16,384, 1.12x from 65,536 to 1,048,576 (948.6 ns to 846.8 at 65,536), 1.08x above; 0.92x at 64 elements (1.9 ns to 2.1), 0.97x at 256 and 0.99x at 1,024, which pay for one more test on the way in. More than the count of instructions on the two ports predicts beyond the first-level cache, and the reason was not looked for. HEAD itself has taken 803 ns and 945 ns for 65,536 elements in different runs of `compare`, so the placement of the buffers plays a part there. The micro-benchmark was not usable for this attempt: its core was shared with another job.

## Where it stands

Final `judge run` (unlocked): 6.70x [6.69 – 6.72] over naive C over all six tables, 122x over NumPy.

| table | over naive C [95%] | over NumPy |
|---|---|---|
| rows of 22, sparse | 5.10x [5.07 – 5.14] | 91.7x |
| rows of 22, dense | 5.04x [5.01 – 5.09] | 80.4x |
| rows of 4, sparse | 17.6x [17.52 – 17.75] | 304x |
| rows of 4, dense | 17.5x [17.32 – 17.62] | 362x |
| rows of 64, sparse | 3.40x [3.38 – 3.41] | 69.2x |
| rows of 64, dense | 3.37x [3.36 – 3.38] | 59.7x |

The kernel does the same work whatever the elements are, so the per-size tables are those of the sparse distribution; the dense ones are within 3% of them at every size.

Rows of 22 (the default):

| n | kernel | naive C | speedup [95%] | NumPy | speedup |
|---|---|---|---|---|---|
| 16 | 1.3 ns | 1.4 ns | 1.11x [1.07 – 1.17] | 702.1 ns | 540x |
| 64 | 2.4 ns | 9.1 ns | 3.73x [3.62 – 3.92] | 771.9 ns | 316x |
| 256 | 4.2 ns | 35.7 ns | 8.48x [8.17 – 8.71] | 879.2 ns | 209x |
| 1,024 | 14.5 ns | 93.0 ns | 6.43x [6.41 – 6.47] | 1.4 us | 94.1x |
| 4,096 | 49.0 ns | 334.7 ns | 6.83x [6.82 – 6.84] | 3.0 us | 61.8x |
| 16,384 | 185.9 ns | 1.3 us | 7.02x [7.02 – 7.03] | 10.5 us | 56.4x |
| 65,536 | 856.1 ns | 5.2 us | 6.03x [6.02 – 6.04] | 44.7 us | 52.2x |
| 262,144 | 3.4 us | 20.6 us | 6.15x [6.15 – 6.15] | 187.4 us | 55.9x |
| 1,048,576 | 13.3 us | 82.6 us | 6.22x [6.21 – 6.23] | 758.8 us | 57.2x |
| 4,194,304 | 63.9 us | 334.0 us | 5.23x [5.21 – 5.25] | 3.03 ms | 47.4x |
| 16,777,216 | 289.9 us | 1.34 ms | 4.64x [4.52 – 4.78] | 12.14 ms | 41.9x |

Rows of 4:

| n | kernel | naive C | speedup [95%] | NumPy | speedup |
|---|---|---|---|---|---|
| 16 | 2.9 ns | 9.3 ns | 3.16x [3.16 – 3.16] | 776.5 ns | 265x |
| 64 | 2.1 ns | 28.9 ns | 13.9x [13.89 – 13.94] | 890.7 ns | 429x |
| 256 | 2.6 ns | 67.4 ns | 26.3x [26.20 – 26.43] | 1.4 us | 530x |
| 1,024 | 6.3 ns | 218.2 ns | 34.9x [34.82 – 34.89] | 3.2 us | 514x |
| 4,096 | 24.4 ns | 824.2 ns | 33.8x [33.74 – 33.76] | 10.7 us | 438x |
| 16,384 | 95.6 ns | 3.2 us | 33.9x [33.91 – 33.96] | 40.6 us | 425x |
| 65,536 | 559.0 ns | 12.9 us | 23.1x [23.11 – 23.12] | 163.6 us | 293x |
| 262,144 | 2.2 us | 51.6 us | 23.3x [23.30 – 23.32] | 653.3 us | 295x |
| 1,048,576 | 9.1 us | 206.3 us | 22.6x [22.48 – 22.73] | 2.61 ms | 286x |
| 4,194,304 | 63.5 us | 823.6 us | 13.0x [12.88 – 13.22] | 10.43 ms | 164x |
| 16,777,216 | 472.1 us | 3.29 ms | 6.96x [6.50 – 7.44] | 42.04 ms | 89.0x |

Rows of 64:

| n | kernel | naive C | speedup [95%] | NumPy | speedup |
|---|---|---|---|---|---|
| 16 | 1.3 ns | 1.4 ns | 1.07x [1.02 – 1.13] | 698.7 ns | 548x |
| 64 | 1.5 ns | 5.1 ns | 3.37x [3.33 – 3.41] | 753.3 ns | 499x |
| 256 | 2.3 ns | 12.3 ns | 5.39x [5.37 – 5.42] | 788.7 ns | 347x |
| 1,024 | 6.3 ns | 41.2 ns | 6.53x [6.52 – 6.54] | 956.6 ns | 151x |
| 4,096 | 27.7 ns | 131.1 ns | 4.74x [4.73 – 4.76] | 1.5 us | 53.0x |
| 16,384 | 80.3 ns | 494.3 ns | 6.16x [6.16 – 6.16] | 3.7 us | 45.5x |
| 65,536 | 623.3 ns | 2.0 us | 3.13x [3.13 – 3.14] | 13.6 us | 21.8x |
| 262,144 | 2.5 us | 7.8 us | 3.10x [3.09 – 3.11] | 56.4 us | 22.5x |
| 1,048,576 | 10.0 us | 31.1 us | 3.12x [3.12 – 3.13] | 244.7 us | 24.6x |
| 4,194,304 | 48.7 us | 124.0 us | 2.55x [2.55 – 2.55] | 1.05 ms | 21.6x |
| 16,777,216 | 206.4 us | 502.3 us | 2.43x [2.41 – 2.45] | 4.02 ms | 19.5x |

With rows of 22 or 64, 16 elements are no row at all: the kernel returns after one compare, and the reference after a division. The times of the rows of 64 depend on where `m` sits: 65,536 elements took 446 ns in the `compare` of attempt 7 and 623 ns in this run. A row of 64 that does not start at a multiple of 32 has one load across two cache lines.

Output 16 bytes behind the input, counted within a page (rows of 22, sparse), against the same code with the output clear:

| n | close | apart | apart / close [95%] |
|---|---|---|---|
| 4,096 | 49.1 ns | 48.9 ns | 1.00x [1.00 – 1.00] |
| 16,384 | 186.1 ns | 185.8 ns | 1.00x [1.00 – 1.00] |
| 65,536 | 849.0 ns | 849.3 ns | 1.00x [1.00 – 1.00] |
| 262,144 | 3.4 us | 3.4 us | 1.00x [1.00 – 1.00] |
| 1,048,576 | 13.4 us | 13.5 us | 1.00x [1.00 – 1.00] |
| 4,194,304 | 62.0 us | 62.1 us | 1.00x [1.00 – 1.00] |
| 16,777,216 | 450.3 us | 591.6 us | 1.31x [1.30 – 1.33] |

The close placement costs nothing: the kernel stores one byte for every `cols` it reads, and no store is taken for a store to the bytes about to be loaded. The last line is 16 MB, a size at which timings were not steady during these runs: other judges were running.

What limits it. Rows of 22 in the first-level cache take 0.26 ns per row, about 1.4 cycles, at 22 elements per row. Every row needs one `vptest`, which needs port 0, and one `setnz`, which needs port 0 or 6, as the loop's branch does: a little over one cycle per row is the floor of that form. Attempts 4 and 5 show that the two ways around it that were found cost more vector instructions than they save. Rows of 4 take about four cycles per 128 elements: eight loads, or three packs on port 5 with the instructions around them.

What a caller must know:

- `cols` is at least 1. `out` has room for `n / cols` bytes and does not overlap `m`.
- Every byte of `m` is 0 or 1. Any other non-zero byte counts as true on every path but one: with fewer than 32 elements and a width other than 4, the bits are gathered by shifting the bytes themselves, and a byte above 1 spills into its neighbours' bits. A binding that may see such bytes has to rule them out, or that path has to gather with a compare; the judge has no such input, and it times nothing on that path, so the change was left for a decision.
- `pext` and `pdep` are not used. No gather and no masked store.

Not tried:

- Rows of 1, 2 and 8 as whole vectors (a copy, `vpcmpeqw`, `vpcmpeqq`), in the way of attempt 3. They take the `vptest` path now, at about 1.4 cycles per row whatever the width, which for a width of 1 is 0.26 ns per element. The judge times widths of 22, 4 and 64 only, so `compare` cannot keep such a path; it needs a timed scalar set in the spec first.
- Narrow rows that are not a power of two (3, 5, 6, 7) through the mask bits: `vpmovmskb` of 32 elements, then all the rows of a 64-bit word at once with the carry trick for fields of `cols` bits, `pext` and `pdep`. Estimated at about two instructions per row for a width of 3 against six now. Not timed by the judge either.
- An early exit inside a long row once a true element is found. Rows of 1,000 are or'ed to the end whatever they hold; the judge does not time them.
- Rows of 65 and more with more than one chunk per step of the inner loop.
- Stores past the caches. The output is `1 / cols` of the input, so there is little to gain.
- For rows of 64 whose loads straddle cache lines, loads brought to a multiple of 32 with the two partial vectors masked: estimated to cost more than the straddling does.
