# unique_i64 attempts (x86_64)

There is no `kernel.s` beside this log: on x86-64 the kernel ships as its naive C, compiled by clang for the x86-64-v3 level. The kernel was written on an Apple M1 Pro, and no x86-64 machine was at hand, so nothing has been tried or timed here yet. The judge checks the C on x86-64 in CI (`judge-linux`), and `delivery check` runs the binding with it there.

What the ARM64 log found, as reading material: the naive C costs one store per element; a 64-bit set built with vector shifts is faster for ranges of at most 64 values, and a byte table inside `out` for wider ones. AVX2 has `vpsllvq`, a shift of each 64-bit lane by its own count that gives zero for counts of 64 or more, which may serve the first. Any assembly is kept only if `judge compare` finds it significantly faster on x86-64 hardware.
