# fmax_reduce_f64 attempts (arm64)

All runs on an Apple M1 Pro (performance core), macOS, native. **Every judge command in this run takes `--unlocked`**, a departure from `AGENTS.md` decided by the maintainer: the kernel's `spec.py` on this branch differs from `main`. Until pull request #177 the locked reference failed the judge on ARM64 (issue 162: NumPy's ARM64 build treats zeros of both signs and NaNs of several bit patterns differently from the x86-64 build the spec described); #177 narrows the spec's domain to what the binding passes, and is not merged. `spec.py`, `naive.c` and `judge/` were not edited. Another kernel run and CI jobs shared the machine, so sizes from about 100,000 elements up are noisy.

**Where the order of the elements is relied on.** NumPy's ARM64 loop is the chain `m = fmaxnm(m, x[i])` from `x[0]` (`docs/fmax-references-design.md`). Within the spec's domain any order of `fmaxnm` gives the same bits: `fmaxnm` of `+0.0` and `-0.0` is `+0.0` in either order, and a zero maximum has no `-0.0` beside it anyway; an all-NaN run has one bit pattern, which every order returns (two quiet NaNs give the first operand, quieted, and `FPCR.DN` is clear on macOS); no NaN is signalling. Every attempt from 2 on reorders the elements, and reads some of them twice, on that ground.

## 1. Baseline: the scalar `fmaxnm` chain

- **Idea:** NumPy's own ARM64 rule, one `fmaxnm` per element from `x[0]`.
- **check:** pass
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** starting point; the chain is one `fmaxnm` latency per element.

## 2. Vectors: overlapping first and last loads up to 32 elements, eight accumulators beyond

- **Idea:** the order does not matter within the domain (see above), so: up to 32 elements no loop, the first and the last 2, 4, 8 or 16 elements as vectors that overlap in the middle, folded as a tree of `fmaxnm` and one `fmaxnmp`; the class is found from the bits of n - 1. From 33 elements eight two-lane accumulators start as the first 16 elements, take 16 elements per step, and the last step is the last 16 elements, overlapping the one before it.
- **check:** pass
- **compare:** significantly faster, 6.93x [6.85 – 7.02]
- **Kept:** yes
- **Notes:** 1.3 ns at 16 (5.2x), 4.4 ns at 64, 0.054 ns per element at 4,096 and 16,384: 2.8 cycles per 16 elements, the load limit of three 16-byte loads per cycle on this core. 0.095 ns per element from 65,536 (second-level cache), 110 us at 1,048,576, where NumPy takes 119 us.

## 3. `ldnp` in the main loop

- **Idea:** the non-temporal load pair, which gained about 10% at 16 MB in `sum_u8` on this core.
- **check:** pass
- **compare:** no significant difference, 1.000x [0.995 – 1.005]
- **Kept:** no
- **Notes:** 1.09x and 1.11x at 4,194,304, nothing at 16,777,216, 0.89x at 262,144 on `some_nan`; nothing below. The loop is at the load limit in the first-level cache and at the cache's own speed beyond it.

## 4. No loop for 33 to 64 elements

- **Idea:** the first 32 elements and the last 32 in straight-line code, as up to 32 elements, instead of the first step, one or two trips of the loop and the last step.
- **check:** pass
- **compare:** no significant difference, 1.001x [0.999 – 1.002]
- **Kept:** no
- **Notes:** 4.4 ns at 64 either way: 32 loads are 10.7 cycles at three per cycle, so the loop's branch and compare cost nothing there.

## 5. The loop on 64-byte boundaries

- **Idea:** start the loop at the first 64-byte boundary after the first step (overlapping it), so that no 32-byte `ldp` straddles two cache lines.
- **check:** pass
- **compare:** significantly slower, 0.998x [0.995 – 1.000]
- **Kept:** no
- **Notes:** 0.98x at 64, 0.99x at 256 (the extra elements read twice), nothing elsewhere. The judge's timed buffers do not straddle in a way that costs here.

## 6. Software prefetch one kilobyte ahead

- **Idea:** a `prfm pldl1strm` per step, for the sizes in the second-level cache (0.095 ns per element there against 0.054 in the first).
- **check:** pass
- **compare:** significantly slower, 0.916x [0.913 – 0.918]
- **Kept:** no
- **Notes:** 0.85x to 0.88x from 1,024 to 262,144, nothing from 4,194,304 up. The prefetch takes a load slot in a loop bound by loads, as in `argmax_f32`.

## 7. Two steps per trip of the loop

- **Idea:** 32 elements per trip into the same eight accumulators, halving the add, compare and branch per element.
- **check:** pass
- **compare:** no significant difference, 1.000x [0.998 – 1.001]
- **Kept:** no
- **Notes:** every size within 1%. The loop is bound by its loads (2.8 cycles per 16 elements against 2.67 for three loads per cycle), not by the instructions around them; for the same reason more accumulators were not tried as a separate attempt.

## 8. 9 to 16 elements as the fall-through of the dispatch

- **Idea:** runs of the binding are about 16 long; test bit 3 of n - 1 with `tbz` so that 9..16 reaches its code with no taken branch.
- **check:** pass
- **compare:** no significant difference, 0.999x [0.998 – 1.001]
- **Kept:** no
- **Notes:** 1.3 ns at 16 either way.

## 9. 9 to 32 elements in one path, with no branch on n

- **Idea:** four blocks of eight elements at addresses chosen with `csel` ([0, 8), [min(n - 8, 8), +8), [max(n - 16, 0), +8), [n - 8, n)), so that runs of mixed lengths around 16 do not mispredict the branch between 9..16 and 17..32.
- **check:** pass
- **compare:** significantly slower, 0.941x [0.938 – 0.944]
- **Kept:** no
- **Notes:** 2.5 ns at 16 against 1.3. Outside the judge, per call on 8,192 elements cut into runs (best of 400 passes): runs of 16 2.44 ns (1.22 before), runs of random length 12..20 2.37 ns (1.80 before), 2..32 2.36 ns (1.75 before). Twice the loads and an address that waits on `csel` cost more than the mispredicted branch.

## 10. `ld1` of four registers instead of `ldp`

- **Idea:** one instruction per 64 bytes in the loop and in the 9..16 path, fewer instructions to decode.
- **check:** pass
- **compare:** significantly slower, 0.945x [0.945 – 0.946]
- **Kept:** no
- **Notes:** 0.81x at 16, 0.89x to 0.90x from 1,024 to 16,384: on this core `ld1` of four registers loads more slowly than two `ldp`.

## Where it stands

Eight attempts in a row (3 to 10) gave nothing, and attempt 2 is at the limit of the core: three 16-byte loads per cycle while the data is in the first-level cache, the cache's own speed beyond it. Per call outside the judge (8,192 elements cut into runs, the kernel called once per run as the binding calls it, best of 400 passes), against `naive.c`: runs of 4, 8, 16, 32 elements 1.22, 1.22, 1.22, 2.28 ns (naive C 4.50, 7.08, 12.37, 21.81); random lengths 12..20 1.80 ns (11.95), 1..4 1.80 (4.43), 2..32 1.75 (12.67), 17..64 3.22 (26.57). Mixed lengths cost the mispredicted branch between the paths, about 0.6 ns.

**Outside the domain** (checked outside the judge against NumPy 2.5.5 on this machine, 2,490 arrays of each kind, 1 to 4,097 elements): zeros of both signs at a zero maximum and all-NaN runs of several quiet bit patterns give NumPy's bits on every array. `fmaxnm` gives `+0.0` for `+0.0` and `-0.0` in either order, and every `fmaxnm` in the kernel has the accumulator that started as `x[0]` as its first operand, and `fmaxnmp` takes lane 0 first, so an all-NaN run returns `x[0]`. A signalling NaN does not give NumPy's result: 440 of 2,490 arrays with one among numbers differ, 541 of 2,490 all-NaN arrays with one.
