# nanstd_cols_f64 attempts (x86-64)

There is no `kernel.s` beside this log: on x86-64 the kernel ships as its naive C. No assembly has been tried, and no x86-64 figure has been taken: the kernel was written on an Apple M1 Pro.

The naive C rounds the square and the sum separately on x86-64 and fuses them on ARM64, to match bottleneck's builds for each ([docs/bottleneck-design.md](../../../docs/bottleneck-design.md#what-bottleneck-computes)). Whether the x86-64 bottleneck in use agrees is checked by `uv run judge check nanstd_cols_f64 --unlocked` in the Linux CI job and by the self-test of `install()`.
