# any_rows_bool attempts (arm64)

All runs on an Apple M1 Pro (macOS), native, while another kernel run was timing in another
worktree, so timings from about 100,000 elements up are noisier than usual.

What the compiler does with `naive.c` (from the judge's figures): rows of 22 at 0.13 ns per
element, rows of 64 at 0.018, rows of 4 at 0.48 (the inner loop is entered and left per row).

## 1. Baseline: one row at a time

- **Idea:** the simplest correct kernel. `udiv` for the row count. Rows of 16 or more: 16-byte
  chunks or'ed together, the last one ending at the row's end, then `umaxv` and `str b` per row.
  Shorter rows: a byte at a time.
- **check:** pass
- **compare:** none (first version). `run`: 1.13x [1.13 – 1.13] over naive C overall; 2.98x on
  rows of 22 (sparse, 1,048,576), 1.00x on rows of 4, 0.50x on rows of 64.
- **Kept:** yes
- **Notes:** the per-row `umaxv` and one-byte store cost about 0.7 ns a row whatever its width.

## 2. Rows of 16 to 64: sixteen rows per block, reduced by a tree of `umaxp`

- **Idea:** each row is reduced to 16 bytes whose maximum is its result: rows of 16 to 32 as
  their first 16 bytes or'ed with their last 16 (two overlapping loads), rows of 33 to 64 as
  their first 32 or'ed with their last 32 (two `ldp`). Fifteen `umaxp` turn sixteen such rows
  into the sixteen result bytes, in order, and one `str q` stores them: no `umaxv`, no move to a
  general register, one store per sixteen rows. The rows left under a block take the loop of
  attempt 1.
- **check:** pass
- **compare:** significantly faster, 1.71x [1.71 – 1.72]; rows of 22 2.0x, rows of 64 2.5x
  (5.2x in the first-level cache), rows of 4 unchanged.
- **Kept:** yes
- **Notes:** rows of 64 at 64 and 256 elements (one and four rows) are slower, 0.77x and 0.92x:
  they never fill a block and pay for the extra tests at entry.

## 3. Rows of four: three `umaxp` per sixteen rows

- **Idea:** a path of its own for `cols` = 4. A row is four bytes, so 64 bytes are sixteen rows,
  and three `umaxp` (pairs, then pairs of pairs) leave one byte per row, in order, still 0 or 1:
  no compare and no narrowing. 32 rows per step (four `ldp`, six `umaxp`, one `stp`), then a
  block of sixteen, then one block that ends at the last row and stores some rows a second time;
  under sixteen rows in all, `ldr w`, `cmp`, `cset` and `strb` per row.
- **check:** pass
- **compare:** significantly faster, 3.20x [3.19 – 3.21] over all tables; the other widths
  unchanged.
- **Kept:** yes

## 4. Rows of 16 to 64: the last rows as one overlapping block, fewer than 16 rows one by one

- **Idea:** with at least 16 rows, the rows left under a block are done by one more block that
  ends at the last row (the positions step back by `msub`) and stores some rows a second time.
  With fewer than 16 rows in all, each row is reduced the same way as in a block, then `umaxv`
  and `str b`: no chunk loop and its branches.
- **check:** pass
- **compare:** significantly faster, 1.13x [1.13 – 1.13]; rows of 22 1.21x and rows of 64
  1.20x over each table, 2.2x to 3.3x at 64 and 256 elements.
- **Kept:** yes

## 5. Rows of 16 to 32: no chain of adds through the block

- **Idea:** in attempt 4 every row moved the position by `cols`, so the 32 loads of a block
  waited on a chain of sixteen dependent adds: one cycle a row. Now four positions four rows
  apart and three offsets (one, two and three `cols`, and the same plus `cols - 16` for each
  row's last 16 bytes) address all sixteen rows with register-offset loads; the four positions
  move once per block.
- **check:** pass
- **compare:** significantly faster, 1.047x [1.046 – 1.048] over all tables; rows of 22 1.15x,
  1.46x to 1.55x from 4,096 to 65,536 elements, about 1.04x beyond.
- **Kept:** yes
- **Notes:** rows of 22 now take 0.23 ns a row in the first-level cache, 0.74 cycles: the two
  loads a row on three load units. Rows of 64 (four loads a row) and rows of 4 (eight loads per
  32 rows) are at the same limit.

## 6. Rows of four: four to fifteen rows by 16-byte loads

- **Idea:** under sixteen rows in all, four rows per `ldr q`, two `umaxp` of the vector with
  itself and `str s`; the last four end at the last row and may store some rows a second time.
  Only under four rows (fewer than 16 elements) the scalar loop of attempt 3.
- **check:** pass
- **compare:** significantly faster, 1.015x [1.013 – 1.016]; rows of 4 at 16 elements 1.63x
  (4.1 ns to 2.6 ns), nothing else moved.
- **Kept:** yes

## 7. Rows of 16 to 32: prefetch four blocks ahead

- **Idea:** rows of 22 take 0.0155 ns a byte from 262,144 elements against 0.0105 in the
  first-level cache, more than rows of 4 and 64 lose there. Four `prfm pldl1keep` per block, one
  per four rows, 64 rows ahead, to start the second-level fetches earlier.
- **check:** pass
- **compare:** significantly slower, 0.974x [0.973 – 0.975]; rows of 22 0.92x, slower at every
  size from 1,024 up: the prefetches take load slots from a loop already limited by its loads.
- **Kept:** no

## 8. Rows of 16 to 64, fewer than 16 rows in all: groups of 1, 2, 4 and 8 rows

- **Idea:** in place of a `umaxv` and a one-byte store per row, the rows are taken by the bits
  of their count: one row (`umaxv`, `str b`), two (two `umaxv`, two `str b`), four (a tree of
  `umaxp`, `str s`), eight (a tree, `str d`). Each group's loads are independent of the others'.
  The smallest group first, so one row takes one taken branch.
- **check:** pass
- **compare:** significantly faster, 1.010x [1.009 – 1.011]; rows of 22 at 256 elements (11
  rows) 1.38x, rows of 64 at 256 elements (4 rows) 1.10x; rows of 22 at 64 elements (2 rows)
  0.92x, 2.0 ns to 2.1 ns.
- **Kept:** yes
- **Notes:** a first version took the groups largest first: one row of 64 went from 1.9 ns to
  2.9 ns behind three taken branches (0.997x, significantly slower). A pair of rows reduced by a
  `umaxp` and three more `umaxp` of the vector with itself is a chain of four; two `umaxv` side
  by side measured the same at two rows.

## 9. Non-temporal loads above 8 MB for rows of 4 and of 33 to 64

- **Idea:** `ldnp` in place of `ldp` in the main loops. Tried first everywhere: 1.16x at
  16,777,216 elements, but 0.88x at 1,048,576 (no significant difference over all). So the main
  loops are duplicated with `ldnp` and taken from 8,388,608 elements, larger than the 12 MB
  second-level cache. Rows of 16 to 32 cannot use it: `ldnp` has no register offset.
- **check:** pass
- **compare:** significantly faster, 1.010x [1.010 – 1.010]; rows of 4 and of 64 1.15x to 1.19x
  at 16,777,216, every other size unchanged.
- **Kept:** yes

## 10. Rows of 16 to 32 above 8 MB: non-temporal pairs, masked

- **Idea:** what attempt 9 did for rows of 4 and 64, for the path it could not reach. Each row is
  one `ldnp` of the 32 bytes at its start, the second 16 and'ed with a mask of the row's last
  `cols - 16` bytes (from a 32-byte constant). Four positions, one per row of four, each moved
  four rows after use. The pair reads up to 15 bytes past the row, so the loop stops while a
  whole block is left, and the ordinary loop does the rest.
- **check:** pass
- **compare:** significantly faster, 1.002x [1.001 – 1.003], and again 1.002x [1.002 – 1.003];
  rows of 22 1.05x and 1.06x at 16,777,216 elements, nothing else moved.
- **Kept:** yes
- **Notes:** a smaller gain than attempt 9 at the same size (1.17x there): the row of 22 costs
  more instructions in this form (`and` and a position add per row).

## 11. Rows of four: tested before the division, short matrices first

- **Idea:** `cols` = 4 is tested before the `udiv`, and its row count is `n >> 2`; under sixteen
  rows the path goes to the 16-byte loads of attempt 6 at once, not through the block tests.
- **check:** pass
- **compare:** significantly faster, 1.012x [1.011 – 1.012]; rows of 4 at 16 elements 1.33x
  (2.6 ns to 1.9 ns), rows of 22 at 64 elements 1.08x (one test fewer before its path).
- **Kept:** yes

## 12. Rows of 16 to 32 in the second-level cache: masked pairs with ordinary loads

- **Idea:** the form of attempt 10 (one pair of loads at the row's start, the second half
  masked, positions moved per row) with `ldp` in place of `ldnp`, from 131,072 elements, in case
  the pair at the row's start suits the second-level cache better than two overlapping loads.
- **check:** pass
- **compare:** significantly slower, 0.998x [0.997 – 0.999]; rows of 22 0.99x from 262,144 to
  4,194,304 and 0.95x at 16,777,216 (where it replaced `ldnp`).
- **Kept:** no
