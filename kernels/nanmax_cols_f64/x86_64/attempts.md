# nanmax_cols_f64 attempts (x86-64)

There is no `kernel.s` beside this log: on x86-64 the kernel ships as its naive C. No assembly has been tried, and no x86-64 figure has been taken: the kernel was written on an Apple M1 Pro. The one attempt is the C itself.

## 1. The naive C: 32 columns at a time, a row at a time

- **Idea:** as on ARM64: 32 columns carried together, rows in order, so each column keeps bottleneck's order and the row is vector code.
- **check:** not run on x86-64 here; `judge check --unlocked` runs in the Linux CI job for `nanstd_cols_f64`, and `delivery check` there covers all four
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** no figure.
