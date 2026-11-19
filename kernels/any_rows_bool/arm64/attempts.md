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
