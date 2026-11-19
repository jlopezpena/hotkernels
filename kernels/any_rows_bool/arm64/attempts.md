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
