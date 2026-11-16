# hotkernels: proof-of-concept design

This is a proposal for review. Nothing here is implemented yet.

It covers the proof of concept only: a trustworthy judge, a survey of where NumPy is slow, and three kernels delivered end to end. Everything beyond that is listed under [Later](#later) without a plan.

## What we are building

A Python package, `hotkernels`, that makes existing NumPy code faster with one line and no other changes:

```python
import hotkernels
hotkernels.install()
```

Behind that line sits a set of hand-written assembly kernels. Each kernel is written and tuned by a coding agent, and it is only shipped if an automated judge confirms it is correct against NumPy and faster than both NumPy and a naive C version compiled with `clang -O3`.

The first target is ARM64 NEON on Apple Silicon. x86-64 is planned and will be developed on a separate x86 machine, so the design keeps architectures apart from the start (see [Architectures](#architectures)).

Priorities, in order:

1. Real, measured speedups.
2. A user experience with no friction.
3. Learning to run coding agents against an objective judge. Understanding every line of the assembly comes later.

## Principles

1. **No friction.** One install command, one line to enable. No code changes, no configuration, no build step for the user.
2. **Correct by a declared standard.** Every kernel declares how its output is compared with NumPy's (see [Correctness policy](#correctness-policy)). Exact is the default.
3. **Never break anything.** If a kernel does not handle an input (dtype, memory layout, size, platform), the call goes to NumPy untouched. On an unsupported machine, `install()` does nothing.
4. **Only ship real wins.** A kernel must beat NumPy and the naive C baseline. Matching the compiler is a loss, because C is easier to maintain.
5. **Visible, reversible.** The user can see what was patched, how often each kernel ran, which kernels are inexact, and undo it all.

## What we know so far

One throwaway experiment was run on an M1 Pro: `out = max(0, a*x + y)` on float32, comparing NumPy, C and hand-written NEON.

| Finding | Number |
|---|---|
| Hand-written assembly vs `clang -O3` C | Tie (0.99x to 1.04x from 128 elements up) |
| One fused pass vs NumPy's three passes | 3.7x to 5.7x faster from 262,144 elements up |
| Cost of calling a kernel through `ctypes` | About 690 ns per call |
| Kernel time itself at 16 elements | About 2 ns |

Consequences:

- Simple elementwise maths is not worth writing in assembly. The targets must be operations the compiler cannot vectorise on its own.
- The binding layer matters as much as the kernels. A slow call path wipes out any win on small arrays.
- Fusing several operations into one pass is a large win, but it cannot be delivered transparently and is out of scope for the proof of concept.

## Kernel contract

A kernel is a directory:

```
kernels/<name>/
  spec.py          signature, dtypes, NumPy oracle, correctness mode
  naive.c          naive C reference: the simplest correct loop
  arm64/kernel.s   hand-written NEON
  x86_64/kernel.s  hand-written AVX (added later, on the x86 machine)
  attempts.md      log of what was tried and what the judge said
```

- `spec.py` and `naive.c` are shared by all architectures.
- `naive.c` has two jobs: it is the baseline the assembly must beat, and a second check on correctness alongside NumPy.
- `attempts.md` records every iteration, including failures, so later runs do not repeat dead ends.

## Correctness policy

Each kernel declares one of two modes in its spec.

- **`exact` (default).** Output must be byte-for-byte equal to NumPy's, including NaN, infinities and signed zero. This applies to every kernel that does no floating-point arithmetic: selection, search, counting, copying, comparison. For these, exactness costs nothing and a tolerance would only hide bugs.
- **`tolerance`.** For kernels that reorder floating-point arithmetic (sums, cumulative sums, dot products), where bit-identical output is often impossible at full speed. The bound is relative and expressed in units in the last place, so the same rule works for float32 and float64. A fixed absolute bound such as 1e-12 does not: float32 carries only about 7 significant digits.

`report()` marks every installed kernel that is not exact.

## Architectures

- The judge contains nothing architecture-specific. It builds and tests whichever kernel directory matches the machine it runs on.
- Timings are only ever taken on native hardware. x86-64 numbers come from a real x86 machine, never from Rosetta.
- x86-64 will need a runtime check for which instruction sets the CPU supports (for example AVX2 or AVX-512). This is noted here and not designed yet.

## Phases

Each phase has an exit criterion. The next phase does not start until it is met.

### Phase 1: the judge and the agent setup

The judge is the foundation. Agents can only iterate unattended if its verdict can be trusted. Once the judge, the kernel contract and `AGENTS.md` exist, writing a kernel is a `/goal` run with a kernel spec, not a separate phase.

**How:**

- Implement the kernel contract above.
- Correctness check. Run the kernel against NumPy under its declared mode across:
  - sizes 0, 1, every size up to 65, then powers of two and their neighbours, up to a few million
  - special values: NaN, infinities, signed zero, denormals, extreme integers
  - unaligned pointers
  - guard pages after the buffers, so a read or write past the end crashes instead of passing silently
- Speed check. Time each kernel inside C, not through Python, across a fixed size sweep. Report best-of-N against NumPy and the naive C baseline.
- Output a machine-readable result per kernel (pass or fail, timings per size) and a human-readable leaderboard.
- Write `AGENTS.md`: the kernel contract, the platform's assembly conventions, how to run the judge, and the single rule (keep a change only if the judge says correct and faster).
- Write a goal-spec template: what a `/goal` prompt for one kernel contains.

**Exit criterion:** the judge correctly rejects a set of deliberately broken kernels (off-by-one tail, wrong NaN handling, out-of-bounds read, wrong tie-breaking) and gives timings that are stable to within a few percent between runs.

### Phase 2: survey

Find out where NumPy is actually slow on ARM64 before choosing what to write. NumPy already has NEON code for some operations, so guessing is unreliable.

**How:**

- Write the naive C baseline for a list of candidates and measure NumPy against it. Candidates: `argmax`/`argmin`, boolean mask selection (`a[mask]`), `count_nonzero`, `nonzero`, `cumsum`, `searchsorted`, `clip`, `any`/`all`, `take`, `isin`, and operations on stepped slices such as `a[::2]`.
- Measure NumPy's fixed per-call overhead on tiny arrays separately, since that is a different kind of win.
- Rank candidates by how far NumPy is behind C, and by whether a known SIMD technique exists that the compiler will not find.

**Exit criterion:** a ranked shortlist with numbers, and three kernels chosen for phase 3.

### Phase 3: three kernels, end to end

Take the three chosen kernels from assembly to `install()`.

**How:**

- Prove the delivery path with the first kernel before writing the other two:
  - Build a real C extension module for the binding. No `ctypes`.
  - Establish which patching mechanisms work on current NumPy. There are three, and each reaches different calls:

    | Mechanism | Reaches | Does not reach |
    |---|---|---|
    | Replace module functions (`np.argmax`) | Function-style calls, including from other libraries | Method calls (`a.argmax()`), names imported before patching |
    | Replace NumPy's inner loops through its C API | Anything routed through a ufunc: operators, methods, reductions | Operations that are not ufuncs |
    | Decorator or own array type | Anything | Code the user has not opted in |

    The second mechanism is recalled from NumPy's C API and from how Intel's `mkl_umath` works. It has not been checked against NumPy 2.x. Verifying it is the first task of this phase.
  - Implement `install()`, `uninstall()`, a context manager, and `report()`.
  - Implement the fallback path and measure its cost. A call that falls back must cost almost nothing extra.
- Produce each kernel with `/goal` runs against the judge, logging every attempt in `attempts.md`.

**Exit criterion:**

- Three kernels beat both NumPy and the naive C baseline under the judge.
- An unmodified script runs faster after adding the one line, with output that meets each kernel's correctness mode, and `report()` shows the kernels being hit.
- The overhead of a patched call that falls back is measured and negligible.

## Non-goals

- Reimplementing all of NumPy.
- Beating BLAS at matrix multiplication. Apple's Accelerate already uses dedicated hardware for it.
- Multithreading. Single-core performance only, to keep measurements clean.
- Linux ARM64.

## Later

Mentioned so they are not forgotten. None of these is planned here.

- **x86-64 kernels.** Same contract and judge, developed and measured on the x86 machine.
- **Publishing.** Wheels on PyPI so installation needs no compiler, plus `python -m hotkernels script.py` for zero code changes. The release pipeline and its GitHub setup get their own design document.
- **More kernels.** Work down the survey shortlist.
- **Fusion.** Fusing a chain of operations into a single pass was the largest win measured. It first needs a way to identify which chains are worth fusing, which is a project of its own.

## Decisions made in review

| Question | Decision |
|---|---|
| Package and import name | `hotkernels` (`kernel-arena` is taken on PyPI) |
| Licence | MIT |
| Bit-identical always? | Per kernel: exact by default, declared tolerance for float arithmetic |
| How hands-on with the assembly? | Gains first, via `/goal` iteration; understanding later |
| Other platforms | x86-64 planned on a separate machine; Linux ARM64 out of scope |
| Agent loop as its own phase? | No; it is the judge, the contract and `AGENTS.md` |
