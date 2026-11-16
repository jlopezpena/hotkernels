# The judge: design

This is a proposal for review. Nothing here is implemented yet. It details phase 1 of [core-design.md](core-design.md).

## What the judge is

The judge is a command-line tool in this repository that takes one kernel and answers two questions:

1. **Is it correct?** Does it match NumPy on every input we can throw at it, without touching memory it does not own?
2. **Is it a win?** Is it faster than the naive C version and faster than NumPy?

A coding agent iterating on a kernel runs the judge after every change and keeps the change only if the judge accepts it. Nobody reviews the assembly line by line, so the judge is the only thing standing between a subtly wrong kernel and a user's data. It has to be strict, and its answers have to be repeatable.

The judge is a development tool. It is not part of the `hotkernels` package that users install.

## Commands

All commands run through `uv`, so there is nothing to set up beyond cloning the repository.

| Command | What it does | Time budget |
|---|---|---|
| `uv run judge check <kernel>` | Lint and correctness only | About 10 seconds |
| `uv run judge bench <kernel>` | Timing only | About 1 minute |
| `uv run judge run <kernel>` | Both, then a verdict | About 1.5 minutes |
| `uv run judge compare <kernel> --against <git-ref>` | Is the working copy faster than an earlier version of the same kernel? | About 1 minute |
| `uv run judge selftest` | Proves the judge rejects known-bad kernels | About 1 minute |
| `uv run judge leaderboard` | Regenerates the results table | Seconds |

The exit code is 0 for accept and non-zero for reject. `check` is fast on purpose: agents run it on every edit and run the slower commands only once `check` passes.

## Layout

```
judge/                 the tool (Python, plus a small amount of C and assembly)
  fixtures/            kernels used only to test the judge itself
kernels/<name>/        one directory per kernel (the contract from core-design.md)
  spec.toml
  naive.c
  arm64/kernel.s
  attempts.md
results/               accepted results, committed
.judge/                build output and raw runs, not committed
```

The judge is written in Python because NumPy is the reference for correctness. It compiles kernels by calling `clang` directly with flags it controls. There is no build system in the proof of concept.

## The kernel spec

`spec.toml` tells the judge how to call a kernel and what the right answer is. Two examples show the range the format has to cover.

A kernel that returns a single value:

```toml
name   = "argmax_f32"
oracle = "np.argmax(x)"
mode   = "exact"
min_n  = 1                      # np.argmax raises on empty input; the kernel is never called with n = 0

[signature]
returns = "i64"
args = [
  { name = "x", type = "f32*", role = "in", len = "n" },
  { name = "n", type = "usize" },
]
```

A kernel that writes a variable amount of output:

```toml
name   = "compress_f32"
oracle = "x[mask]"
mode   = "exact"

[signature]
returns = "usize"               # number of elements written
args = [
  { name = "x",    type = "f32*",  role = "in",  len = "n" },
  { name = "mask", type = "bool*", role = "in",  len = "n" },
  { name = "out",  type = "f32*",  role = "out", capacity = "n", used = "return" },
  { name = "n",    type = "usize" },
]
```

- `oracle` is a NumPy expression over the input names. Its result is the correct answer.
- `mode` is `"exact"` or `{ tolerance_ulp = 4 }`, as defined in the core design.
- The set of types and roles starts small and grows only when a new kernel needs it.

From the signature the judge generates the C glue it needs to call the kernel. Kernel authors never write harness code.

## Stage 1: lint

Cheap static checks on the assembly source, run before anything is executed. A kernel is rejected if it:

- calls any external function, or makes a system call
- declares writable data
- uses a register the platform reserves (`x18` on Apple platforms)
- exports any symbol other than the kernel itself

A kernel is a pure function over the buffers it is given. These rules keep it that way.

## Stage 2: correctness

The kernel and `naive.c` are both run against the NumPy oracle. Checking `naive.c` as well matters: if it is wrong, the speed comparison is meaningless.

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

The rule is strict: a kernel may not read outside its buffers at all, even where such a read would be harmless on real hardware. A strict rule is simple to state to an agent and simple to verify.

### Calling convention

Assembly can corrupt its caller in ways that tests pass by luck. The judge calls each kernel through a small assembly wrapper that fills every register the kernel is required to preserve with known values, calls the kernel, and verifies they are unchanged, along with the stack pointer. This wrapper is the only architecture-specific code in the judge: one short file per architecture.

### Comparison

- **Exact mode:** the output bytes must equal the oracle's bytes.
- **Tolerance mode:** each value must be within the declared number of units in the last place. NaN and infinity must appear in exactly the same positions.
- The return value is compared the same way as output buffers.

### Crashes

Each kernel is checked in a separate process. If it crashes, the judge reports which case was running and rejects the kernel; the judge itself keeps going.

### Failure output

Output is written for an agent to act on: short, and specific enough to reproduce. Only the first failure is shown.

```
REJECT argmax_f32 (arm64): wrong result
  case:     nan_at_index  n=37  nan_index=33  offset=4  seed=1234
  expected: 33
  got:      0
  rerun:    uv run judge check argmax_f32 --case nan_at_index:37:33:4:1234
```

## Stage 3: speed

### What is timed

| Contender | How it is timed |
|---|---|
| The assembly kernel | Inside C, called in a loop through a function pointer |
| `naive.c` | The same loop; compiled by the judge with `clang -O3 -mcpu=native` |
| NumPy | The oracle expression, called from Python |

The kernel directory cannot change compiler flags, so the C baseline cannot be weakened.

The NumPy number includes NumPy's own per-call overhead, because that is what a user experiences. The assembly number does not yet include the cost of our binding, which does not exist until phase 3. So at small sizes the comparison with NumPy flatters the kernel, and the comparison with naive C is the honest one. Phase 3 measures end to end.

### Sizes

A fixed sweep: 16, 64, 256, 1,024, 4,096, 16,384, 65,536, 262,144, 1,048,576, 4,194,304 and 16,777,216 elements. This spans data that fits in the fastest cache through to data far larger than any cache.

### Method

- **Repeat count.** For each size the loop count is calibrated so one sample takes at least 10 ms.
- **Samples.** 15 samples per contender per size. The minimum is the reported time; the gap between minimum and median is reported as the noise level.
- **Interleaving.** Samples from the contenders are alternated, so slow drift in the machine affects all of them equally.
- **Varied inputs.** At small sizes the loop cycles through a pool of different inputs. Timing one input repeatedly lets the processor memorise its branch pattern, which makes branchy code look faster than it is on real data.
- **Input shapes.** Where speed depends on the data (how many mask elements are true, where the maximum sits), the spec lists a few named distributions and each is timed.
- **Performance cores.** The timing thread asks macOS for the highest scheduling priority class, which keeps it on the performance cores.

### Refusing to time on a noisy machine

Before and after the sweep the judge times a small fixed reference loop. If the two disagree by more than 3%, or the noise level of any measurement exceeds 5%, the judge reports "machine too noisy" and gives no verdict. It also refuses to time in Low Power Mode. A wrong timing verdict is worse than no verdict.

### Verdict

A kernel is accepted as a win when all of these hold:

| Rule | Threshold |
|---|---|
| Faster than naive C overall | Geometric mean of the speedups across sizes is at least 1.10 |
| No size is a regression against naive C | No size slower than 0.95 times naive C |
| Faster than NumPy overall | Geometric mean of the speedups across sizes is at least 1.10 |

The thresholds are deliberately above the noise floor, so a kernel cannot be accepted on measurement luck.

### Comparing two versions of a kernel

During iteration the question is narrower: is this edit faster than what we had? `judge compare` builds the kernel from an earlier git revision alongside the working copy and times them interleaved in the same process. That detects differences of about 2%, much finer than comparing against numbers recorded earlier. An edit is kept when it is at least 2% faster overall and no size regresses beyond the noise level.

## Results

- `judge run` writes one JSON file per kernel, architecture and machine into `results/` when a kernel is accepted. It records the timings per size, the verdict, the processor model, the operating system, compiler and NumPy versions, and the git revisions of the kernel, its spec and the judge.
- `judge leaderboard` turns those files into `LEADERBOARD.md`: one row per kernel with its speedup over naive C and over NumPy.
- Raw runs and build output go to `.judge/`, which is not committed.

The NumPy version is pinned in the lock file. Results are only comparable across the same NumPy version, since NumPy's own speed changes between releases.

## Keeping the agent honest

An agent told to make a number go up will sometimes do it the wrong way. The judge closes the obvious routes:

- **Limited write access.** During a kernel run, the only files allowed to change are that kernel's `<arch>/kernel.s` and `attempts.md`. `judge run` checks with git that the spec, `naive.c` and the judge itself match the main branch, and rejects otherwise.
- **A human approves the reference.** `spec.toml` and `naive.c` are reviewed by a person when a kernel is created, because everything else is measured against them.
- **Fixed compiler flags** for the baseline, as above.
- **Fresh random cases** on every correctness run, as above.

## Testing the judge

`judge selftest` runs the judge against fixture kernels that are wrong in known ways and confirms each one is rejected for the right reason. This is the exit criterion for phase 1.

There are two fixture kernels, matching the two spec examples above: `argmax_f32` (returns a value) and `compress_f32` (writes variable-length output). Each has a correct, deliberately simple assembly version, plus broken variants:

| Broken variant | Must be rejected by |
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
| Correct but slower than naive C | Speed |

The second half of the exit criterion is repeatability: ten consecutive `judge bench` runs of the correct fixture on an idle machine agree to within 3% at every size.

The fixtures test the judge. They are not candidates for shipping; phase 2 decides which kernels are worth writing.

## What agents get

Phase 1 also delivers the two documents that turn the judge into an agent loop. Their content follows directly from this design, so they are written alongside the implementation and not designed separately:

- **`AGENTS.md`:** the kernel contract, the assembly conventions for the platform, the judge commands, how to read a rejection, the files that may be edited, and the one rule: keep a change only if `judge check` passes and `judge compare` says it is faster.
- **Goal-spec template:** the prompt for a `/goal` run on one kernel. It names the kernel, the stopping condition (accepted by `judge run`, or a set number of attempts without improvement), and requires every attempt to be logged in `attempts.md` with what was tried and what the judge said.

## Out of scope

- Hardware performance counters (cycles, cache misses). On macOS they need elevated privileges. Wall-clock time with the method above is enough for a verdict.
- Timing in CI. Shared runners are too noisy. CI will run `judge check` and `judge selftest` without the speed fixture.
- Multi-dimensional or non-contiguous arrays. Kernels see flat, contiguous buffers; anything else falls back to NumPy in the binding.
- x86-64. The design leaves room for it (one wrapper file and one kernel directory per architecture), but nothing x86-specific is built in phase 1.

## Open questions for review

1. **Oracle format.** A NumPy expression in `spec.toml` (proposed) is compact but limited to one line. The alternative is a small Python file per kernel, which is more flexible and more to maintain. Start with the expression and add the file only if a kernel needs it?
2. **Win thresholds.** 10% over naive C to accept a kernel, and 2% to keep an edit during iteration. Too strict, too loose?
3. **Strict memory rule.** No reads outside the buffer at all (proposed). Allowing harmless over-reads would permit some faster tricks, at the cost of a rule that is harder to check. Keep it strict for the proof of concept?
4. **Committing results.** Accepted results and the leaderboard are committed (proposed), which gives the repository a public scoreboard but mixes one machine's numbers into git history. Alternatively they stay local.
5. **NumPy timing.** The judge times NumPy from Python, overhead included, and treats naive C as the honest baseline until phase 3. Acceptable, or should the NumPy rule be dropped from the verdict until the binding exists?
6. **Fixture kernels.** `argmax_f32` and `compress_f32` as the two kernels the judge is tested on. Any preference for others?
