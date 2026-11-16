# kernel-arena: project plan

This is a proposal for review. Nothing here is implemented yet. Comment on the pull request and the plan will be revised before any code is written.

## What we are building

A Python package that makes existing NumPy code faster on Apple Silicon with one line and no other changes:

```python
import kernel_arena
kernel_arena.install()
```

Behind that line sits a set of hand-written ARM64 NEON assembly kernels. Each kernel is written and tuned by a coding agent, and it is only shipped if an automated judge confirms it is bit-identical to NumPy and faster than both NumPy and a plain C version compiled with `clang -O3`.

The project has three goals, in this order:

1. Learn how hardware-level optimisation works.
2. Learn how to run coding agents against an objective, automated judge.
3. Ship something that is pleasant to use.

## Principles

These are the rules every later decision is checked against.

1. **No friction.** One install command, one line to enable. No code changes, no configuration, no build step for the user.
2. **Never change results.** A kernel's output must be bit-identical to NumPy's, including NaN, infinities and signed zero. If it is not, the kernel is not installed.
3. **Never break anything.** If a kernel does not handle an input (dtype, memory layout, size, platform), the call goes to NumPy untouched. On a machine that is not ARM64, `install()` does nothing.
4. **Only ship real wins.** A kernel must beat NumPy and the `-O3` C baseline. Matching the compiler is a loss, because C is easier to maintain.
5. **Visible, reversible.** The user can see what was patched, how often each kernel ran, and undo it all.

## What we know so far

One throwaway experiment was run on an M1 Pro: `out = max(0, a*x + y)` on float32, comparing NumPy, C and hand-written NEON.

| Finding | Number |
|---|---|
| Hand-written assembly vs `clang -O3` C | Tie (0.99x to 1.04x from 128 elements up) |
| One fused pass vs NumPy's three passes | 3.7x to 5.7x faster from 262,144 elements up |
| Cost of calling a kernel through `ctypes` | About 690 ns per call |
| Kernel time itself at 16 elements | About 2 ns |

Three consequences for the plan:

- Simple elementwise maths is not worth writing in assembly. The targets must be operations the compiler cannot vectorise on its own.
- The binding layer matters as much as the kernels. A slow call path wipes out any win on small arrays.
- Fusing several operations into one pass is a large win, but it cannot be delivered transparently, because NumPy evaluates each operation before it sees the next. It is a separate, opt-in feature (phase 6).

## Non-goals

- Reimplementing all of NumPy.
- Beating BLAS at matrix multiplication. Apple's Accelerate already uses dedicated hardware for it.
- x86 support. Benchmarks under Rosetta are meaningless on the development machine.
- Multithreading. Single-core performance only, to keep measurements clean.

## Phases

Each phase has an exit criterion. The next phase does not start until it is met.

### Phase 1: the judge

The judge is the foundation. Agents can only run unattended if its verdict can be trusted.

**How:**

- Define a kernel contract. A kernel is a directory containing the assembly file, a plain C reference implementation, and a small spec: the function signature, supported dtypes, and the NumPy expression that defines the correct answer.
- Correctness check. Run the kernel against NumPy and require exact byte equality across:
  - sizes 0, 1, every size up to 65, then powers of two and their neighbours, up to a few million
  - special values: NaN, infinities, signed zero, denormals, extreme integers
  - unaligned pointers
  - guard pages after the buffers, so a read or write past the end crashes instead of passing silently
- Speed check. Time each kernel inside C, not through Python, across a fixed size sweep. Report best-of-N, against NumPy and the C baseline.
- Output a machine-readable result per kernel (pass or fail, timings per size) and a human-readable leaderboard.

**Exit criterion:** the judge correctly rejects a set of deliberately broken kernels (off-by-one tail, wrong NaN handling, out-of-bounds read, wrong tie-breaking) and gives timings that are stable to within a few percent between runs.

### Phase 2: survey

Find out where NumPy is actually slow on ARM64 before choosing what to write. NumPy already has NEON code for some operations, so guessing is unreliable.

**How:**

- Write the plain C baseline for a list of candidates and measure NumPy against it. Candidates: `argmax`/`argmin`, boolean mask selection (`a[mask]`), `count_nonzero`, `nonzero`, `cumsum`, `searchsorted`, `clip`, `any`/`all`, `take`, `isin`, and operations on stepped slices such as `a[::2]`.
- Measure NumPy's fixed per-call overhead on tiny arrays separately, since that is a different kind of win.
- Rank candidates by how far NumPy is behind C, and by whether a known NEON technique exists that the compiler will not find.

**Exit criterion:** a ranked shortlist with numbers, and three kernels chosen for phase 3.

### Phase 3: walking skeleton

Take one kernel all the way from assembly to `install()` before writing many kernels. This proves the delivery path early, since user experience is the main product requirement.

**How:**

- Write the first kernel by hand, without agents, to calibrate the contract and the judge.
- Build a real C extension module for the binding. No `ctypes`.
- Prove which patching mechanisms work on current NumPy. There are three, and each reaches different calls:

  | Mechanism | Reaches | Does not reach |
  |---|---|---|
  | Replace module functions (`np.argmax`) | Function-style calls, including from other libraries | Method calls (`a.argmax()`), names imported before patching |
  | Replace NumPy's inner loops through its C API | Anything routed through a ufunc: operators, methods, reductions | Operations that are not ufuncs |
  | Decorator or own array type | Anything, including fusion | Code the user has not opted in |

  The second mechanism is recalled from NumPy's C API and from how Intel's `mkl_umath` works. It has not been checked against NumPy 2.x. Verifying it is the first task of this phase.
- Implement `install()`, `uninstall()`, a context manager, and `report()`.
- Implement the fallback path and measure its cost. A call that falls back must cost almost nothing extra.

**Exit criterion:** an unmodified script runs faster after adding the one line, produces identical output, and `report()` shows the kernel being hit. The overhead of a patched call that falls back is measured and negligible.

### Phase 4: the agent loop

Turn kernel-writing into an automated loop.

**How:**

- Write the agent brief: the kernel contract, the target platform's assembly conventions, how to run the judge, and the single rule (keep a change only if the judge says correct and faster).
- Run the loop on the remaining two kernels from the shortlist. Each iteration: propose a change, run the judge, keep or revert, record the attempt.
- Keep a log of every attempt per kernel, including failures, so later agents do not repeat dead ends.
- Experiment with the loop itself: one agent against several in parallel on the same kernel, and different levels of guidance in the brief.

**Exit criterion:** at least three kernels beat both NumPy and `-O3` C, each produced or improved by the loop with no hand-editing of the assembly.

### Phase 5: package and broaden

Make it installable by someone else, then widen coverage.

**How:**

- Publish wheels for macOS ARM64 so `pip install kernel-arena` needs no compiler. On other platforms the package installs and `install()` is a no-op.
- Add `python -m kernel_arena script.py` for zero code changes.
- Test against a range of NumPy versions in CI. CI checks correctness only; shared runners are too noisy for timing.
- Run the test suites of a few NumPy-heavy libraries with kernels installed, to catch behaviour differences the judge missed.
- Work down the survey shortlist with the agent loop.

**Exit criterion:** a fresh machine can install the package from PyPI and see a speedup with one line, and the downstream test suites pass with kernels installed.

### Phase 6 (stretch): fusion

Fuse a chain of operations into a single pass. This was the largest win measured, but it is a compiler project and only worth starting if the earlier phases went well.

**How:** a decorator on a function, which traces the array expression, generates one fused kernel, and falls back to plain NumPy if anything is unsupported. The design is deliberately left open until phase 5 is done.

## Open questions for review

1. **Import name.** `kernel_arena`, or something shorter?
2. **Licence.** The repository has none yet. MIT or Apache-2.0 would be the usual choices.
3. **Bit-identical, always?** Some operations cannot be both fast and bit-identical. Should those be excluded, or offered behind an explicit opt-in such as `install(exact=False)`?
4. **How hands-on?** Read and understand every kernel the agents write, or treat the assembly as a black box and focus on the loop? The first is slower but teaches the assembly.
5. **Linux ARM64.** The same kernels would run on AWS Graviton with small syntax changes. In scope later, or never?
6. **Phase order.** The skeleton (phase 3) comes before the agent loop (phase 4) so delivery is proven early. If the agent loop is the more interesting part, the two can be swapped.
