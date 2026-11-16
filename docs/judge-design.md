# The judge: design

This is a proposal for review. Nothing here is implemented yet. It details phase 1 of [core-design.md](core-design.md).

## What the judge is

The judge is a command-line tool in this repository that takes one kernel and does two things:

1. **Decides whether it is correct.** Does it match NumPy on every input we can throw at it, without touching memory it does not own? This is an automatic pass or fail.
2. **Measures how fast it is.** How does it compare with the naive C version and with NumPy, and is the difference statistically significant? This is a report. Whether a kernel is a win worth shipping is decided by a person reading it.

A coding agent iterating on a kernel runs the judge after every change. Nobody reviews the assembly line by line, so the judge is the only thing standing between a subtly wrong kernel and a user's data. The correctness side has to be strict, and the timing side has to be honest about its own uncertainty.

The judge is a development tool. It is not part of the `hotkernels` package that users install.

## Commands

All commands run through `uv`, so there is nothing to set up beyond cloning the repository.

| Command | What it does | Time budget |
|---|---|---|
| `uv run judge check <kernel>` | Lint and correctness only | About 10 seconds |
| `uv run judge bench <kernel>` | Timing report only | About 1 minute |
| `uv run judge run <kernel>` | Correctness, then the timing report | About 1.5 minutes |
| `uv run judge compare <kernel> --against <git-ref>` | Is the working copy faster than an earlier version of the same kernel? | About 1 minute |
| `uv run judge selftest` | Proves the judge catches known-bad kernels | About 1 minute |
| `uv run judge leaderboard` | Regenerates the results table | Seconds |

The exit code reflects correctness only: 0 if the kernel is correct, non-zero if not. Speed never fails a run. `check` is fast on purpose: agents run it on every edit and run the slower commands only once `check` passes.

## Layout

```
judge/                 the tool (Python, plus a small amount of C and assembly)
  fixtures/            kernels used only to test the judge itself
kernels/<name>/        one directory per kernel (the contract from core-design.md)
  spec.py
  naive.c
  arm64/kernel.s
  attempts.md
results/               recorded results, committed
.judge/                build output and raw runs, not committed
```

The judge is written in Python because NumPy is the reference for correctness. It compiles kernels by calling `clang` directly with flags it controls. There is no build system in the proof of concept.

## The kernel spec

`spec.py` tells the judge how to call a kernel and what the right answer is.

**This format is provisional.** It is the smallest thing that covers the two fixture kernels. When the survey or a real kernel needs something it cannot express, the format changes. Nothing else in the design depends on its details.

A kernel that returns a single value:

```python
import numpy as np
from judge.spec import Kernel, In, Size

def oracle(x):
    return np.argmax(x)

KERNEL = Kernel(
    name="argmax_f32",
    returns="i64",
    args=[In("x", "f32", len="n"), Size("n")],
    oracle=oracle,
    mode="exact",
    min_n=1,   # np.argmax raises on empty input; the kernel is never called with n = 0
)
```

A kernel that writes a variable amount of output:

```python
import numpy as np
from judge.spec import Kernel, In, Out, Size

def oracle(x, mask):
    return x[mask]

KERNEL = Kernel(
    name="compress_f32",
    returns="usize",   # number of elements written
    args=[
        In("x", "f32", len="n"),
        In("mask", "bool", len="n"),
        Out("out", "f32", capacity="n", used="return"),
        Size("n"),
    ],
    oracle=oracle,
    mode="exact",
)
```

- `oracle` is a Python function over the inputs. Its result is the correct answer.
- `mode` is `"exact"` or a tolerance in units in the last place, as defined in the core design.
- A spec can also supply its own input generators when the standard ones are not enough.

From the signature the judge generates the C glue it needs to call the kernel. Kernel authors never write harness code.

### Why a Python file

The oracle is Python code, so a Python file per kernel exists either way. Putting the signature in the same file means one file and one format per kernel.

The alternatives considered:

| Format | Why not |
|---|---|
| TOML plus a separate Python oracle | Two files to keep in step for no gain. TOML on its own cannot hold the oracle beyond a one-line expression. |
| JSON | No comments, and the same two-file problem. |
| YAML | Needs an extra dependency, has well-known parsing surprises, and the same two-file problem. |

The cost of Python is that a spec is executable code, not inert data. That is acceptable here because specs are written or reviewed by a person and locked against agent edits (see [Keeping the agent honest](#keeping-the-agent-honest)).

## Stage 1: lint

Cheap static checks on the assembly source, run before anything is executed. A kernel fails if it:

- calls any external function, or makes a system call
- declares writable data
- uses a register the platform reserves (`x18` on Apple platforms)
- exports any symbol other than the kernel itself

A kernel is a pure function over the buffers it is given. These rules keep it that way.

## Stage 2: correctness

The kernel and `naive.c` are both run against the oracle. Checking `naive.c` as well matters: if it is wrong, the speed comparison is meaningless.

### Inputs

Every case is generated from a fixed seed, so a failure can always be reproduced.

- **Sizes.** Every size from `min_n` to 65, then each power of two up to about four million, together with one less and one more. Small sizes catch mistakes in leftover-element handling; the neighbours of powers of two catch mistakes at block boundaries.
- **Values.** Chosen per data type:
  - random data
  - all elements equal
  - sorted ascending and descending
  - floating point: NaN, both infinities, both signed zeros, denormals
  - integers: the minimum and maximum values
- **Positions.** For sizes up to 65, the interesting element (the maximum, a NaN, the only `True` in a mask) is placed at every possible index in turn. This catches a kernel that is right except in one lane of one vector.
- **Ties.** Duplicates of the winning value at several positions, since NumPy defines which one wins.
- **Alignment.** Each case runs with the buffers at several different offsets from an aligned address.
- **Fresh random cases.** After the fixed cases, a batch generated from a new seed on every run. The seed is printed. This stops a kernel from passing by fitting the fixed cases.

### Memory safety

Buffers are not ordinary allocations. The judge maps them so that:

- **Reads and writes past the end crash.** Each buffer ends exactly at the boundary of an inaccessible page, and a second arrangement starts it exactly after one. Any access outside the buffer, by even one byte, faults.
- **Writes to inputs crash.** Input buffers are mapped read-only for the duration of the call.
- **Writes past the used output are detected.** The unused part of an output buffer is filled with a known pattern and checked afterwards.

The rule is strict: a kernel may not read outside its buffers at all.

#### What the strict rule costs

A SIMD kernel handles several elements per step, say four. An array of 37 elements is nine full steps plus one element left over. There are two ways to deal with the leftover:

- **Stay inside the buffer.** Handle the last element with separate, slower code. This is extra code and extra branches.
- **Read past the end.** Load one more full step of four, and ignore the three values that are not part of the array. This is shorter and faster, because every step looks the same.

Reading past the end changes nothing, so it is harmless as long as the extra bytes belong to the program. Memory is handed out in fixed-size pages. If the array happens to end at the very edge of the last page the program owns, the extra read lands on memory it does not own and the program crashes. Code that uses this trick therefore has to check how close it is to a page edge first.

The trick only matters for tiny arrays, where the leftover is most of the work. For an array of a million elements the leftover is a rounding error. The strict rule is kept for the proof of concept because it is simple to state to an agent and simple to verify. It can be relaxed for specific kernels if the survey shows that tiny arrays are where the wins are.

### Calling convention

Assembly can corrupt its caller in ways that tests pass by luck. The judge calls each kernel through a small assembly wrapper that fills every register the kernel is required to preserve with known values, calls the kernel, and verifies they are unchanged, along with the stack pointer. This wrapper is the only architecture-specific code in the judge: one short file per architecture.

### Comparison

- **Exact mode:** the output bytes must equal the oracle's bytes.
- **Tolerance mode:** each value must be within the declared number of units in the last place. NaN and infinity must appear in exactly the same positions.
- The return value is compared the same way as output buffers.

### Crashes

Each kernel is checked in a separate process. If it crashes, the judge reports which case was running and fails the kernel; the judge itself keeps going.

### Failure output

Output is written for an agent to act on: short, and specific enough to reproduce. Only the first failure is shown.

```
FAIL argmax_f32 (arm64): wrong result
  case:     nan_at_index  n=37  nan_index=33  offset=4  seed=1234
  expected: 33
  got:      0
  rerun:    uv run judge check argmax_f32 --case nan_at_index:37:33:4:1234
```

## Stage 3: speed

Timing is best effort for now. The method below is careful enough to trust a clear result, and it gets stricter if the project starts finding results worth defending.

### What is timed

| Contender | How it is timed |
|---|---|
| The assembly kernel | Inside C, called in a loop through a function pointer |
| `naive.c` | The same loop; compiled by the judge with `clang -O3 -mcpu=native` |
| NumPy | The oracle function, called from Python |

The kernel directory cannot change compiler flags, so the C baseline cannot be weakened.

The NumPy number includes NumPy's own per-call overhead, because that is what a user experiences. The assembly number does not yet include the cost of our binding, which does not exist until phase 3. So at small sizes the comparison with NumPy flatters the kernel, and the comparison with naive C is the honest one. The report says this next to the NumPy column. Phase 3 measures end to end.

### Sizes

A fixed sweep: 16, 64, 256, 1,024, 4,096, 16,384, 65,536, 262,144, 1,048,576, 4,194,304 and 16,777,216 elements. This spans data that fits in the fastest cache through to data far larger than any cache.

### Method

- **Repeat count.** For each size the loop count is calibrated so one sample takes at least 10 ms.
- **Samples.** 30 samples per contender per size.
- **Interleaving.** Samples from the contenders are alternated, so slow drift in the machine affects all of them equally.
- **Varied inputs.** At small sizes the loop cycles through a pool of different inputs. Timing one input repeatedly lets the processor memorise its branch pattern, which makes branchy code look faster than it is on real data.
- **Input shapes.** Where speed depends on the data (how many mask elements are true, where the maximum sits), the spec lists a few named distributions and each is timed.
- **Performance cores.** The timing thread asks macOS for the highest scheduling priority class, which keeps it on the performance cores.

### What is reported

There are no pass or fail thresholds on speed. For each size, and overall, the judge reports:

- the median time of each contender
- the speedup of the kernel over naive C and over NumPy
- a 95% confidence interval for each speedup, computed by resampling the measured samples
- whether the speedup is statistically significant, meaning the interval does not include 1.0

The overall figure is the geometric mean of the per-size speedups, with its own interval.

```
argmax_f32 (arm64, Apple M1 Pro)        vs naive C              vs NumPy (includes call overhead)
        n      kernel      speedup   95% interval          speedup   95% interval
       16      3.1 ns       1.02x    0.97 – 1.06  n.s.      310x     ...
    4,096      210 ns       2.41x    2.36 – 2.47             3.9x    ...
  ...
  overall                   2.10x    2.05 – 2.16             ...
```

### Noise

Before and after the sweep the judge times a small fixed reference loop. If the two disagree by more than 3%, or the machine is in Low Power Mode, the report carries a prominent warning that the machine was noisy. The numbers are still shown; the reader decides whether to rerun.

### Comparing two versions of a kernel

During iteration the question is narrower: is this edit faster than what we had? `judge compare` builds the kernel from an earlier git revision alongside the working copy and times them interleaved in the same process. It reports one of three outcomes, using the same statistics as above:

- significantly faster
- significantly slower
- no significant difference

An agent in a `/goal` run keeps an edit only when `compare` says significantly faster. This is bookkeeping for the iteration loop, not a decision about shipping. Whether the finished kernel is worth shipping stays with a person.

## Results

- Every `judge run` writes its full report as JSON under `.judge/`, which is not committed.
- `judge run --record` also copies the report into `results/`, one file per kernel, architecture and machine. Recording is a deliberate step taken when a person decides the result is worth keeping. Recorded results are committed for now; this can be revisited.
- A recorded result holds the timings and intervals per size, the processor model, the operating system, compiler and NumPy versions, and the git revisions of the kernel, its spec and the judge.
- `judge leaderboard` turns the recorded results into `LEADERBOARD.md`: one row per kernel with its speedup over naive C and over NumPy.

The NumPy version is pinned in the lock file. Results are only comparable across the same NumPy version, since NumPy's own speed changes between releases.

## Keeping the agent honest

An agent told to make a number go up will sometimes do it the wrong way. The judge closes the obvious routes:

- **Limited write access.** During a kernel run, the only files allowed to change are that kernel's `<arch>/kernel.s` and `attempts.md`. The judge checks with git that the spec, `naive.c` and the judge itself match the main branch, and fails the run otherwise.
- **A human approves the reference.** `spec.py` and `naive.c` are reviewed by a person when a kernel is created, because everything else is measured against them.
- **Fixed compiler flags** for the baseline, as above.
- **Fresh random cases** on every correctness run, as above.

## Testing the judge

`judge selftest` runs the judge against fixture kernels that are wrong in known ways and confirms each one is caught for the right reason. This is the exit criterion for phase 1.

There are two fixture kernels, matching the two spec examples above: `argmax_f32` (returns a value) and `compress_f32` (writes variable-length output). Each has a correct, deliberately simple assembly version, plus broken variants:

| Broken variant | Must be caught by |
|---|---|
| Skips the last element when the size is not a multiple of the vector width | Correctness |
| Ignores NaN | Correctness |
| Returns the last of several tied maxima instead of the first | Correctness |
| Only correct when the buffer is aligned | Correctness |
| Reads one vector past the end of the input | Memory safety |
| Writes one element past the used output | Memory safety |
| Writes to an input buffer | Memory safety |
| Overwrites a register it must preserve | Calling convention |
| Uses `x18` | Lint |
| Correct but slower than naive C | Speed report: shown as significantly slower |

The second half of the exit criterion is repeatability: ten consecutive `judge bench` runs of the correct fixture on an idle machine agree to within 3% at every size.

The fixtures test the judge. They are not candidates for shipping; phase 2 decides which kernels are worth writing.

## What agents get

Phase 1 also delivers the two documents that turn the judge into an agent loop. Their content follows directly from this design, so they are written alongside the implementation and not designed separately:

- **`AGENTS.md`:** the kernel contract, the assembly conventions for the platform, the judge commands, how to read a failure, the files that may be edited, and the one rule: keep a change only if `judge check` passes and `judge compare` says it is significantly faster.
- **Goal-spec template:** the prompt for a `/goal` run on one kernel. It names the kernel, the stopping condition (a set number of attempts without a significant improvement), and requires every attempt to be logged in `attempts.md` with what was tried and what the judge said.

## Out of scope

- Hardware performance counters (cycles, cache misses). On macOS they need elevated privileges. Wall-clock time with the method above is enough for now.
- Timing in CI. Shared runners are too noisy. CI will run `judge check` and `judge selftest` without the speed fixture.
- Multi-dimensional or non-contiguous arrays. Kernels see flat, contiguous buffers; anything else falls back to NumPy in the binding.
- x86-64. The design leaves room for it (one wrapper file and one kernel directory per architecture), but nothing x86-specific is built in phase 1.

## Decisions made in review

| Question | Decision |
|---|---|
| Spec format | One Python file per kernel, holding the signature and the oracle. Provisional; changes when a kernel needs it to. |
| Win thresholds | None. The judge reports speedups with confidence intervals and significance; a person decides. |
| Memory rule | Still open. Proposed: strict for the proof of concept, no reads outside the buffer. |
| Committing results | Committed for now; revisit later. |
| NumPy timing | Best effort: timed from Python with call overhead, labelled as such. Stricter if results warrant it. |
| Fixture kernels | `argmax_f32` and `compress_f32`. |
