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
