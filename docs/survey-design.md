# The survey: design and findings

This is a proposal for review. It details phase 2 of [core-design.md](core-design.md): where NumPy is slow on ARM64, which operations are worth a kernel, and in what order to build them.

Unlike the other design documents, the measuring part is already done. The harness in [`survey/`](../survey) exists and the numbers below are real. What is proposed, and open to change, is the ranking and the order of work.

The survey was run before the judge existed, so it uses its own small harness. The timings are "naive timings": good enough to rank candidates, not good enough to defend a 10% difference. Every candidate is measured again by the judge before it ships.

## What the survey is

A command-line tool, `uv run survey`, that for each candidate operation times the NumPy statement a user would write against the simplest correct C loop and reports how far NumPy is behind. It answers two questions per candidate:

1. **Is NumPy slow here?** Measured: NumPy time divided by naive C time.
2. **Can assembly beat the C as well?** Partly measured (did clang vectorise the loop?), partly judgement (is there a known technique the compiler will not find?).

The survey is a development tool, like the judge. It is not part of the `hotkernels` package.

## Layout

```
survey/
  naive.c        naive C baselines for every candidate, with their timing loops
  candidates.py  the candidate list: NumPy statement, C kernel, input, expected result
  measure.py     builds the C, checks it against NumPy, times both
  render.py      prepares the numbers for the templates
  templates/     Jinja templates for the report and the progress lines
  cli.py         the `survey run` and `survey report` commands
  results/       recorded results, committed: one JSON file and one report per run
```

The baselines in `survey/naive.c` are survey code. A candidate that goes forward gets its own reviewed `naive.c` and `spec.py` under `kernels/`, as the kernel contract requires.

## Method

- **Machine.** Apple M1 Pro, macOS 27.2.1, NumPy 2.5.5, Apple clang 21.
- **Contenders.** For each candidate, the NumPy statement a user would write, against the simplest correct C loop compiled with `clang -O3 -mcpu=native`.
- **Timing.** C is timed inside C. NumPy is timed from Python and includes its call overhead. Samples are interleaved, the median of up to 11 is reported, and the thread asks for the performance cores. A reference loop timed before and after the run drifted by 0.6%.
- **Sizes.** 1, 16, 256, 4,096, 65,536, 1,048,576 and 16,777,216 elements.
- **Allocation.** Where NumPy allocates its result, the C side also mallocs and frees its output on every call. Otherwise the gap would partly be the allocator, which no kernel can remove.
- **Correctness.** Every naive C kernel was checked against NumPy on the timed input. All 65 variants agree.
- **Was the C vectorised?** Taken from clang's own loop-vectoriser remarks.

To reproduce:

```
uv run survey run
```

`uv run survey report <file.json>` regenerates the report of an existing result file.

Full tables: [apple-m1-pro.md](../survey/results/apple-m1-pro.md) (61 variants) and [apple-m1-pro-branchless.md](../survey/results/apple-m1-pro-branchless.md) (4 follow-up variants).

"Gap" below always means NumPy time divided by naive C time, as a geometric mean over sizes of 65,536 and up. A gap of 4x means the naive loop is four times faster than NumPy.

## What the numbers say

1. **Most of the phase 2 list is slower than a naive C loop.** Of the candidates named in the core design, only `argmax`/`argmin`, `any`/`all`, `count_nonzero` on booleans and `searchsorted` on integers are ahead of naive C. Most of the rest is behind, by 1.5x to 7x; `take` through `a[idx]` and `clip` on integers are level.
2. **The biggest gaps are often closed by the compiler alone.** `clip` on floats (7x), `astype` from `uint8` to `float32` (5.5x), `np.sum` on `uint8` (3.3x) and `count_nonzero` on non-boolean data (2.9x) are all loops that clang vectorises by itself. Assembly would tie the C, so these ship as C kernels (decided in review). **Phase 3 measured otherwise:** assembly beat the C on five of the six, see [kernel-findings.md](kernel-findings.md).
3. **The assembly targets are the loops clang leaves scalar.** Mask selection, `nonzero`, `argmax`, `cumsum`, `searchsorted`, `interp`, `histogram` and `bincount` are not vectorised by the compiler, and each has a known hand-written technique.
4. **Branch misprediction is a large part of NumPy's cost on masks.** `x[mask]` at 50% density costs NumPy 4.8 ns per element. A C loop with no data-dependent branch does it in 0.34 ns: 14x faster, before any SIMD.
5. **Where NumPy already has NEON, it is very good.** `np.max`, `np.any`, `np.all`, `count_nonzero` on booleans and `packbits` run at 0.01 to 0.05 ns per element, 10x to 50x ahead of naive C. These are not targets.
6. **Stepped slices are not a target.** Copying and adding over `a[::2]` tie naive C (gap 1.03x to 1.09x), and `np.sum` and `np.argmax` on a stepped slice are ahead of it.

## Shortlist

Ranked by how likely a hand-written kernel is to beat both NumPy and C, then by the size of the gap. The first two columns of numbers are measured. The last column is a judgement and has not been tested.

| # | Candidate | NumPy vs naive C | ns per element at 1M: NumPy / C | Does clang vectorise it? | Technique the compiler will not find |
|---|---|---|---|---|---|
| 1 | Mask selection, `a[mask]` | 1.6x to 1.8x behind the branchy loop; 14x behind the branchless loop at 50% density, 2.7x at 5% | f64, 50%: 4.75 / 0.34 (branchless) | No | Turn each block of mask bytes into a shuffle index and compact with a table lookup |
| 2 | `nonzero` on booleans | 1.5x behind at 5% density. At 50% NumPy beats the branchy loop but is 1.9x behind the branchless one | 50%: 0.72 / 0.38 (branchless) | No | Same compaction as 1, emitting indices |
| 3 | `argmax` / `argmin` on floats | NumPy is ahead of naive C (0.33x f32, 0.54x f64) | f32: 0.43 / 1.29; f64: 0.71 / 1.29 | No | Vector max with index tracking and a NaN check per block. NumPy has NEON here, but its float `argmax` is 8x slower than its own `np.max` (0.05) and 2.5x to 4x slower than its integer `argmax` (0.17), so there is room |
| 4 | `searchsorted` on floats (and `digitize`) | 1.7x to 4.7x behind up to 1M; ahead at 16M | f64: 142 / 84 per needle | No | Branchless search, several needles advanced together so cache misses overlap. NumPy's int64 path (39 ns per needle) shows what is reachable |
| 5 | `interp` | 3.4x to 5.3x behind from 4,096 up | 679 / 127 per point | Not the search | Same search core as 4, then the interpolation step in SIMD |
| 6 | `cumsum` | 3.7x behind on floats, 6.5x on int64 | f64: 3.61 / 0.97; i64: 2.18 / 0.34 | No | SIMD prefix sum. Exact for integers. For floats it reorders additions, so it ships in tolerance mode (accepted in review); in exact mode the C loop is already at the limit of one addition per element |
| 7 | `histogram`, uniform bins | 7.1x behind | 6.26 / 0.88 | No | Compute bin numbers in SIMD, count into several tables to avoid store-to-load stalls |
| 8 | Masked assignment, `x[mask] = v` | 1.2x behind; both NumPy and C cost 3.4 to 3.9 ns per element at 50% density | 3.85 / 3.40 | Not usefully | Load, select, store whole vectors. The compiler may not do this because it writes elements the source code does not write |
| 9 | `bincount` | 4.2x behind | 1.37 / 0.33 | No | Several count tables. C is already at about one cycle per element, so the gain over C is likely small |
| 10 | `isin` with a small test set | 5.7x behind for 8 test values | 7.86 / 1.08 | Yes, the inner compare | Compare against all test values held in registers. Gain over C unclear |

Also measured and worth keeping in mind:

- **`np.minimum.accumulate`** (1.4x behind). A SIMD prefix minimum is exact, unlike a float prefix sum. Low usage.
- **`np.add.at`** (2.4x behind). No SIMD technique; a C-level win only.
- **`np.take(a, idx)`** is 3.2x behind naive C, while `a[idx]` is level with it (1.07x). The function form is slow, not the gather.
- **`np.sum(mask)`** costs 0.21 ns per element; `np.count_nonzero(mask)` costs 0.03. Routing one to the other is a 7x win with no new kernel.

### C kernels

These have large gaps but clang vectorises the naive loop, so assembly is expected to tie C. They ship as C. The agent may still attempt assembly; it replaces the C only if the judge says it is significantly faster.

| Candidate | NumPy vs naive C | ns per element at 1M: NumPy / C |
|---|---|---|
| `np.clip` on float32 | 7.1x behind | 0.65 / 0.09 |
| `np.clip` on float64 | 3.5x behind | 0.65 / 0.18 |
| `astype`, uint8 to float32 | 5.5x behind | 0.32 / 0.06 |
| `np.sum` on uint8 | 3.3x behind | 0.19 / 0.06 |
| `count_nonzero` on int64 / float64 | 2.9x behind | 0.33 / 0.10 |
| `astype`, int64 to float64 | 1.8x behind | 0.34 / 0.18 |

`np.clip` on int64 is level with C, so the float gap is specific to NumPy's float loop.

### Not targets

NumPy is level with or ahead of naive C: `np.max`, `np.sum` on floats, `any`, `all`, `count_nonzero` on booleans, `packbits`, `argmax` on booleans and integers, `searchsorted` on int64, `np.where(c, a, b)`, `np.diff`, `np.dot` on int64, `np.array_equal`, `a[idx]`, `a[idx] = v`, `astype` from float64 to float32, `np.correlate`, and everything measured on stepped slices.

## Per-call overhead

NumPy's time on a one-element array, which is almost entirely fixed cost. The naive C kernel takes a few nanoseconds here, or about 25 ns when it has to allocate.

| Fixed cost | Calls |
|---|---|
| 12,000 to 18,000 ns | `np.histogram`, `np.isin` |
| 1,100 to 2,100 ns | `np.clip`, `np.diff` |
| 700 to 1,250 ns | `np.sum`, `np.max`, `np.cumsum`, `np.any`, `np.all`, `np.array_equal`, `np.take(..., out=)` |
| 400 to 650 ns | `np.argmax`, `np.argmin`, `np.searchsorted`, `np.interp`, `np.add.at`, `np.dot`, `np.where` |
| 140 to 310 ns | `np.nonzero`, `np.bincount`, `np.packbits`, `np.count_nonzero`, `astype`, `x[mask]`, `x[mask] = v` |
| 60 to 100 ns | `a[idx]`, `a[idx] = v` |

Two consequences:

- The floor for a call that goes straight into C is around 60 to 150 ns. Calls costing 500 ns or more spend most of that in Python-level wrappers, and a patched function with a direct path could remove it. That is a win in the binding, not in a kernel, and it is largest for `histogram`, `isin`, `clip` and the reductions.
- It sets the budget for phase 3: our own binding has to cost well under 100 ns per call, or small arrays get slower.

## Order of work

Every candidate above is tried; none is dropped in advance. The judge decides whether each ships as assembly, as C, or not at all. The order puts shared techniques together and the cheapest evidence first.

| Step | Candidates | Why here |
|---|---|---|
| 1 | `np.clip` on floats, as C | Proves the delivery path with a kernel that needs no assembly. `np.clip` is backed by a ufunc, so it also tests the inner-loop patching mechanism |
| 2 | Mask selection and `nonzero` | Largest gap the compiler does not close; one compaction technique, two kernels; `compress_f32` is already a judge fixture. Uses scratch output |
| 3 | `argmax` / `argmin` on floats | The harder case: beating NumPy where it already has NEON. Already a judge fixture |
| 4 | `searchsorted` on float64, then `interp` | One search core serves `searchsorted`, `digitize` and `interp` |
| 5 | `cumsum`: integers exact, then floats in tolerance mode | Integers first because they need no tolerance |
| 6 | `histogram`, `bincount` | Share the several-count-tables technique |
| 7 | Masked assignment | Only `np.putmask` and `np.copyto` are reached until the indexing gap is resolved |
| 8 | `isin` with a small test set | Least certain gain over C |
| 9 | Remaining C kernels: `astype`, `np.sum` on uint8, `count_nonzero` on int64 and float64 | No assembly expected; can be done at any point |

## How the kernels get built

The survey lands in its own pull request. The kernels follow in separate ones, because the judge refuses to run a kernel whose `spec.py` or `naive.c` differs from the main branch:

1. **This pull request:** the survey harness, the results and this document.
2. **Specs and baselines:** `kernels/<name>/spec.py` and `naive.c` for the candidates, reviewed by a person, since everything is measured against them.
3. **One pull request per kernel or kernel family:** the assembly from `/goal` runs against the judge, with its `attempts.md` and recorded result. A candidate where assembly does not beat the C ships the C.

## Known gap: indexing syntax

`a[mask]`, `x[mask] = v` and `x.astype(...)` are indexing syntax or array methods. They are not module functions and not ufuncs, so none of the three patching mechanisms in the core design obviously reaches them. Their function forms are reachable: `np.compress`, `np.extract`, `np.putmask` and `np.copyto`.

This limits what three shortlist entries are worth until it is resolved:

| Candidate | Reached today | Not reached |
|---|---|---|
| Mask selection | `np.compress(mask, a)`, `np.extract(mask, a)` | `a[mask]` |
| Masked assignment | `np.putmask(x, mask, v)`, `np.copyto(x, v, where=mask)` | `x[mask] = v` |
| `astype` C kernels | nothing | `x.astype(...)` |

The kernels are still worth building: the judge can measure them, and the function forms benefit at once. Whether the syntax can be reached is investigated during step 1 of the order of work. The gap is also recorded in [core-design.md](core-design.md) under phase 3.

## The baseline for mask selection

The branchy loop is the simplest C for `a[mask]`, but the branchless loop is just as short and 9x faster at 50% density, so the branchless loop is the honest baseline for mask selection and `nonzero`. It leaves a stray element just past the reported output, inside the buffer's capacity, and the NEON technique leaves up to a block of them. The judge's default output rule rejects that.

Decided in review: a spec may declare an output buffer as scratch, which lets the kernel write anywhere inside its capacity. The trade-offs, including what the stray last element means for the binding, are set out in [judge-design.md](judge-design.md#scratch-output-writing-inside-the-capacity). The judge implements it, with `compress_scratch_f32` as its fixture.

## Limits of this survey

- One machine, one run, one input distribution per candidate (two densities for the mask operations).
- At sizes up to 4,096 the C loop sees the same input on every repeat, so the branch predictor learns it. Ratios for the mask operations at small sizes flatter the C. The ranking uses sizes of 65,536 and up.
- NumPy's numbers include call overhead; the C numbers include no binding cost. At small sizes this flatters the C, as the judge design already notes.
- The last column of the shortlist is a judgement about what assembly can do. Nothing in this survey measures assembly.
- `histogram` was measured with 64 bins over [0, 1), where the naive bin calculation is exact. NumPy's rule for values on a bin edge needs care in a real kernel.
- The search candidates use random needles. Sorted or clustered needles behave differently and were not measured.

## Decisions made in review

| Question | Decision |
|---|---|
| Ship plain C where assembly only ties it? | Yes |
| Choose three kernels for phase 3? | No. All candidates are tried |
| Tolerance-mode float `cumsum`? | Acceptable. Float `cumsum` may reorder additions under a declared tolerance |
| Indexing syntax not reachable | Recorded as a known gap here and in the core design |
| May a kernel write inside its output capacity? | Yes, per buffer, when the spec declares it scratch. Strict by default; trade-offs in the judge design |
