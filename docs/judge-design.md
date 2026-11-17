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
  <arch>/kernel.s      arm64 or x86_64
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
- `mode` is `"exact"` or a tolerance in units in the last place, as defined in the core design, optionally with the scale the unit is taken at.
- A spec can also supply its own input generators when the standard ones are not enough.

From the signature the judge generates the C glue it needs to call the kernel. Kernel authors never write harness code.

### What a spec can express

The first real kernels needed more than the two fixtures do. Each addition below exists because a surveyed kernel could not be written without it.

| Feature | What it does | Needed by |
|---|---|---|
| `Scalar("lo", "f64", -0.5)` | A scalar argument and its default value. Integers go in the integer argument registers and floats in the floating-point ones. The oracle receives the scalars after the arrays: `oracle(x, lo, hi)` | `clip` (bounds), `histogram` (bin count and range), masked assignment (the value) |
| `Kernel(scalar_sets=[ScalarSet("zero_lo", {"lo": 0.0, "hi": 1.0})])` | Further combinations of scalar values to check, and optionally time. See [Scalar sets](#scalar-sets) | `clip`, `histogram`, masked assignment |
| `In("test", "i64", len=8)` | An input whose length is fixed and does not follow `n` | `isin` (the set of test values) |
| `Out("counts", "i64", capacity=256)` | An output whose size is fixed | `bincount` |
| `Out("counts", "i64", capacity="nbins")` | An output whose size is the value of an integer scalar, so it changes with the scalar set | `histogram` |
| `In("a", "f64", prepare=np.sort)` | A function applied to every generated input, for kernels defined only on some inputs | `searchsorted` and `interp` (sorted data) |
| `InOut("x", "f64")` | A buffer the kernel modifies in place. It starts as an input and is compared in full with the oracle's result | masked assignment |
| `Kernel(numpy=fn)` | What is timed as NumPy when that is not the oracle. An in-place oracle has to copy its input; the statement a user writes does not | masked assignment |
| `mode={"tolerance_ulp": 4, "scale": fn}` | A tolerance measured at a magnitude the spec computes from the inputs, instead of at the size of the result. See [Comparison](#comparison) | float `cumsum` |
| `Kernel(check_max_n=65537)` | Caps the sizes used for correctness, for an oracle too slow to run on millions of elements per case. Timing still uses the full sweep | `interp` |
| No `arm64/kernel.s` | The kernel is its naive C. `check` verifies the C against NumPy and `run` reports C against NumPy | every kernel before its assembly exists, and kernels shipped as C |

Two limits follow from these and are deliberate:

- **A scalar is tested with the values the spec lists and no others.** The judge has value classes for arrays and none for scalars. A kernel's behaviour for a combination that is not in the default set or in `scalar_sets` is not verified.
- **An in-place buffer is timed on its own output.** After the first repeat the kernel sees the result of the previous one. That is only a fair timing for operations that give the same result when applied twice, which masked assignment is.

### Scalar sets

A scalar often chooses a path through the kernel. `clip` takes a two-operation path unless a bound is a zero of the wrong sign, and `histogram` trusts its first bin estimate only when the bin count and range pass a check at entry. A kernel tested with one value of its scalars is tested on one of those paths. The first version of the judge did that, and it hid two things in `clip` that were found by other means: the reference disagrees with NumPy for reversed and NaN bounds, and the kernel is fast for some bounds only.

A spec therefore lists further combinations:

```python
KERNEL = Kernel(
    name="clip_f64",
    returns="void",
    args=[In("x", "f64"), Out("out", "f64"), Size("n"), Scalar("lo", "f64", -0.5), Scalar("hi", "f64", 0.5)],
    oracle=lambda x, lo, hi: np.clip(x, lo, hi),
    scalar_sets=[
        ScalarSet("zero_lo", {"lo": 0.0, "hi": 1.0}, timed=True),
        ScalarSet("neg_zero_hi", {"lo": -1.0, "hi": -0.0}),
        ScalarSet("infinite", {"lo": float("-inf"), "hi": float("inf")}),
    ],
)
```

- **Sets, not value classes.** Scalars are rarely independent: a lower bound must not exceed the upper one, and a bin count belongs to a range. Classes drawn per scalar would mostly produce combinations outside the kernel's domain. A set names one valid combination; a scalar it leaves out keeps its default.
- **The default set** is every `Scalar`'s own value. It always exists and gets the full sweep of cases, exactly as before, so a kernel without scalar sets runs the cases it always ran.
- **Every further set gets a thinner sweep:** every value class at the same sizes (less the exact powers of two above 65,536), but one memory arrangement per case instead of five, three indices per positional class instead of all, and no pairings of classes. What has to be shown is that each path handles every size and every kind of value; the default sweep has already shown the rest. A set costs about 2,700 cases and under a second for a one-input kernel.
- **Fresh cases** draw their set at random from all of them.
- **A case names its set:** `n=17 arr=end seed=1 scalars=zero_lo x=zeros`. The field is left out for the default set.
- **The reference is checked on every set too.** A `naive.c` that disagrees with NumPy for a listed combination fails as `reference`, which is how a wrong reference for some bounds would now be caught, provided the combination is listed.
- **`timed=True`** adds a full timing sweep with that set, reported as its own table (`random, scalars=zero_lo`) and included in the overall figure. `compare` times it as well, so an edit that helps one set of scalars and hurts another shows both. Each timed set adds about half a minute to `run` and `compare`, so they should be few. The timing loop takes the scalars as arguments; they are not compiled into it.

What this does not give: a combination nobody listed is still untested, and the array data does not follow the scalars. `histogram` is checked with sixteen bin counts and ranges, but the judge's data is random and seldom lands within rounding distance of a bin edge, which is where a histogram goes wrong. Evidence about edges still comes from a differential test outside the judge.

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

The rules are the same everywhere. What counts as a call, a system call or a reserved register is listed under [Per architecture](#per-architecture).

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
- **Positions.** For sizes up to 65, the interesting element (the maximum, a NaN, a single negative zero, the only `True` in a mask) is placed at every possible index in turn. This catches a kernel that is right except in one lane of one vector. At larger sizes it is placed at the start, the middle and the end, which is what reaches code that only runs once a kernel is well into an array: `clip` with a zero lower bound redoes a chunk when it meets a negative zero among ordinary values, and nothing but a single one far into the array gets it there.
- **Ties.** Duplicates of the winning value at several positions, since NumPy defines which one wins.
- **Alignment.** Each case runs with the buffers at several different offsets from an aligned address.
- **Fresh random cases.** After the fixed cases, a batch generated from a new seed on every run. The seed is printed. This stops a kernel from passing by fitting the fixed cases.

### Memory safety

Buffers are not ordinary allocations. The judge maps them so that:

- **Reads and writes past the end crash.** Each buffer ends exactly at the boundary of an inaccessible page, and a second arrangement starts it exactly after one. Any access outside the buffer, by even one byte, faults.
- **Writes to inputs crash.** Input buffers are mapped read-only for the duration of the call.
- **Writes past the used output are detected.** The unused part of an output buffer is filled with a known pattern and checked afterwards. A spec can relax this for one buffer; see [Scratch output](#scratch-output-writing-inside-the-capacity).

The rule is strict: a kernel may not read outside its buffers at all.

#### What the strict rule costs

A SIMD kernel handles several elements per step, say four. An array of 37 elements is nine full steps plus one element left over. There are two ways to deal with the leftover:

- **Stay inside the buffer.** Handle the last element with separate, slower code. This is extra code and extra branches.
- **Read past the end.** Load one more full step of four, and ignore the three values that are not part of the array. This is shorter and faster, because every step looks the same.

Reading past the end changes nothing, so it is harmless as long as the extra bytes belong to the program. Memory is handed out in fixed-size pages. If the array happens to end at the very edge of the last page the program owns, the extra read lands on memory it does not own and the program crashes. Code that uses this trick therefore has to check how close it is to a page edge first.

The trick only matters for tiny arrays, where the leftover is most of the work. For an array of a million elements the leftover is a rounding error. The rule stays strict because it is simple to state to an agent and simple to verify. It can be relaxed for specific kernels later if the survey shows that tiny arrays are where the wins are.

Some architectures offer a third way: a masked load, which names the elements it wants and does not touch the others, so nothing outside the buffer is read and nothing faults. x86-64 has one and ARM64 NEON does not. Whether it satisfies the rule is [open issue 3](x86-64-design.md#open-issues) of the x86-64 design.

#### Scratch output: writing inside the capacity

*Decided in review of the survey.*

By default a kernel may write only the output it reports. A spec can relax this for one output buffer by declaring it as scratch:

```python
Out("out", "f32", capacity="n", used="return", scratch=True)
```

A kernel may then write anywhere inside the declared capacity of that buffer. Only the reported part is compared with NumPy; the rest holds unspecified values. Writing even one byte past the capacity is still a failure, as is every other memory rule.

**Why it is needed.** Mask selection (`a[mask]`) copies the selected elements to the output. The simple loop asks "is this element selected?" before each store, and on an unpredictable mask the processor guesses that answer wrong about half the time. The fast loop does not ask: it stores every element at the current output position, and moves the position forward only if the element was selected. An unselected element is overwritten by the next one. The survey measured this on ARM64 at 0.34 ns per element against 3.2 ns for the loop that asks. A vector kernel does the same with a whole block of elements per store.

**The last-element problem.** Overwriting only works while there is a next element. Whatever was stored last and not selected stays where it is, just past the reported output. With the scalar loop that is at most one stray element; with a vector kernel it is up to one block minus one. The reported count and the reported elements are correct. The slots after them are not empty, and not what they held before the call.

That has consequences, which is why scratch is opt-in per buffer and not the default:

| Consequence | What it means in practice |
|---|---|
| The tail of the buffer is garbage | The binding must hand the user exactly the reported elements and never a view of the whole buffer. Otherwise values the mask excluded become visible |
| The buffer must be big enough for the stray writes | A block store at output position `k` is safe only because `k` never runs ahead of the input position, so a capacity of `n` always suffices. A binding that counts the selected elements first and allocates exactly that many cannot use a scratch kernel on that buffer: the last block would run past the end |
| Capacity `n` costs memory | NumPy counts first and allocates only what is selected. Allocating `n` for a mask that selects 1% touches a hundred times more memory before the result is shrunk. The binding has to choose between that and a count pass followed by a strict tail |
| A caller's own buffer cannot be scratch | With `np.compress(mask, a, out=buf)` the buffer belongs to the user. Stray writes past the result would change data the user can see. The binding must use a strict kernel, or a temporary buffer, whenever the output is not its own |
| The judge checks less | For a scratch buffer the judge no longer detects a kernel that stores one element too many inside the capacity. A wrong count or a wrong reported element is still caught by the comparison, and a store past the capacity is still caught by the guard page |
| Two correct kernels can differ | The tail is unspecified, so byte-for-byte comparison covers the reported part only |

The assembly still needs care at the end of the input: the leftover elements that do not fill a block are handled one at a time, because a full block load would read past the input, which stays forbidden.

### Calling convention

Assembly can corrupt its caller in ways that tests pass by luck. The judge calls each kernel through a small assembly wrapper that fills every register the kernel is required to preserve with known values, calls the kernel, and verifies they are unchanged, along with the stack pointer. There is one such wrapper per architecture. What each one checks is listed under [Per architecture](#per-architecture).

### Comparison

- **Exact mode:** the output bytes must equal the oracle's bytes.
- **Tolerance mode:** each value must be within the declared number of units in the last place. NaN and infinity must appear in exactly the same positions.
- **Tolerance with a scale:** by default the unit in the last place is taken at the size of the value itself. That is wrong for a running sum: terms that cancel leave a small result carrying the rounding error of the large terms, and a reordered float `cumsum` then differs from NumPy's by millions of units of the result while being exactly as accurate. A spec can therefore supply a scale function with its tolerance, `mode={"tolerance_ulp": 4, "scale": fn}`, which returns per output element the magnitude the unit is taken at. For `cumsum` that is `(i + 1) * (|x[0]| + ... + |x[i]|)`, the standard bound on the error of adding in any order. The price is a weaker check where the bound is wide: for float32 beyond some tens of thousands of elements it no longer notices a single skipped element. Small sizes, where every length and alignment is tried, are not affected.
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
| `naive.c` | The same loop; compiled by the judge with `clang -O3`, tuned for the machine it runs on (`-mcpu=native` on ARM64, `-march=native` on x86-64) |
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
- **Performance cores.** The timing thread is kept on a performance core. On macOS it asks for the highest scheduling priority class, which does that.

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

Before and after the sweep the judge times a small fixed reference loop. If the two disagree by more than 3%, or the platform says the machine is throttled (Low Power Mode on macOS), the report carries a prominent warning that the machine was noisy. The numbers are still shown; the reader decides whether to rerun.

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

A third fixture, `compress_scratch_f32`, covers [scratch output](#scratch-output-writing-inside-the-capacity). It is the same operation as `compress_f32` with the output declared `scratch=True`. Its correct kernel and its naive C both leave a stray element past the reported output, and must pass. The same assembly is a broken variant of `compress_f32`, where it must fail, so the two fixtures together prove the rule applies only where the spec asks for it.

A fourth, `tally_u8`, covers [scalar sets](#scalar-sets). It counts elements by their remainder modulo a scalar bucket count, which also sizes its output. Both of its broken variants are correct for the default bucket count and wrong for another, so they are caught only if the other sets are really run and the output is really sized by the scalar of the case.

| Broken variant | Must be caught by |
|---|---|
| Skips the last element when the size is not a multiple of the vector width | Correctness |
| Ignores NaN | Correctness |
| Returns the last of several tied maxima instead of the first | Correctness |
| Only correct when the buffer is aligned | Correctness |
| Reads one vector past the end of the input | Memory safety |
| Writes one element past the used output | Memory safety |
| Writes one element past the capacity of a scratch output | Memory safety |
| Writes to an input buffer | Memory safety |
| Hard-codes the default value of a scalar | Correctness, in a non-default scalar set |
| Clears as many output elements as the default scalar value gives, when the case gives fewer | Memory safety, in a non-default scalar set |
| Overwrites a register it must preserve | Calling convention |
| Uses a register the platform reserves (`x18` on ARM64) | Lint |
| Uses an instruction outside the level (x86-64) | Lint |
| Returns without clearing the upper halves of the vector registers (x86-64) | Calling convention |
| Correct but slower than naive C | Speed report: shown as significantly slower |

The second half of the exit criterion is repeatability: ten consecutive `judge bench` runs of the correct fixture on an idle machine agree to within 3% at every size.

The fixtures test the judge. They are not candidates for shipping; phase 2 decides which kernels are worth writing.

## What agents get

Phase 1 also delivers the two documents that turn the judge into an agent loop. Their content follows directly from this design, so they are written alongside the implementation and not designed separately:

- **`AGENTS.md`:** the kernel contract, the assembly conventions for the platform, the judge commands, how to read a failure, the files that may be edited, and the one rule: keep a change only if `judge check` passes and `judge compare` says it is significantly faster.
- **Goal-spec template:** the prompt for a `/goal` run on one kernel. It sets out:
  - **The target.** A statistically significant overall speedup of at least 10% over naive C. This is a target given to the agent, not a threshold in the judge.
  - **Do not stop at the target.** After reaching 10% the agent keeps going until it has exhausted the optimisations it can think of.
  - **If the target cannot be reached.** The agent reports its best result and explains why 10% was not possible, for example that the compiler already vectorises the naive loop well, or that the operation is limited by memory speed.
  - **The log.** Every attempt goes into `attempts.md` with what was tried and what the judge said.
  - **The final report.** The best speedup with its confidence interval, the optimisations that worked, the ones that did not, and what was left untried.

## Per architecture

The stages above state rules. This section says how each platform enforces them. Both columns describe what is built. Why x86-64 is done this way is in [x86-64-design.md](x86-64-design.md#the-judge).

| | ARM64 on macOS | x86-64 on Linux |
|---|---|---|
| Kernel source | `kernels/<name>/arm64/kernel.s` | `kernels/<name>/x86_64/kernel.s`, GNU assembler in Intel syntax |
| Assembled by | The C compiler | GNU `as`, because it can be told which instructions exist |
| Naive C | `clang -O3 -mcpu=native` | `clang -O3 -march=native` |
| Lint: calls | `bl`, `blr` and their authenticated forms | `call` |
| Lint: system calls | `svc`, `hvc`, `smc` | `syscall`, `sysenter`, `int` |
| Lint: reserved register | `x18` | None |
| Instruction-set level | Not needed: every Apple Silicon Mac runs what the M1 runs | The kernel is assembled a second time with x86-64-v3 stated, and anything the assembler then refuses fails as lint. The development machine runs newer instructions, so a crash cannot be relied on to catch one |
| Must be preserved | `x19` to `x28`, `x29`, the stack pointer, the low 64 bits of `v8` to `v15` | `rbx`, `rbp`, `r12` to `r15`, the stack pointer. Also: the direction flag clear, the rounding mode and exception masks unchanged, the upper halves of the vector registers cleared before returning, which the wrapper asks the processor about |
| Read-only constants | `__TEXT,__const` or `__TEXT,__literal16` | `.rodata` |
| Performance cores | The highest scheduling priority class | The timing thread pinned to one performance core. The core is claimed through a lock file, so judges running at once take different cores |
| Object format | Mach-O: symbols start with `_`, libraries end in `.dylib` | ELF: plain symbol names, `.so`, and every kernel must declare that it needs no executable stack |
| Sign of a throttled machine | Low Power Mode | Turbo switched off, a power-saving governor or energy preference, a load average of 2 or more, or a core shared with another judge |
| Integer arguments | Eight in registers | Six in registers, a seventh and eighth on the stack |
| Leftover elements | A scalar loop, or a final vector that overlaps the previous one | The same, and possibly a masked load: [open issue 3](x86-64-design.md#open-issues) |

Everything else is the same on both: the cases, the guarded buffers, the comparison with NumPy, the statistics, the locked files, and the selftest's broken variants. The variant that uses a reserved register has no x86-64 counterpart. x86-64 has two of its own: one uses an instruction outside the level, and one returns without clearing the upper halves of the vector registers.

## Out of scope

- Hardware performance counters (cycles, cache misses). On macOS they need elevated privileges. Wall-clock time with the method above is enough for now.
- Timing in CI. Shared runners are too noisy. CI will run `judge check` and `judge selftest` without the speed fixture.
- Multi-dimensional or non-contiguous arrays. Kernels see flat, contiguous buffers; anything else falls back to NumPy in the binding.
- x86-64 was out of scope for phase 1. It is designed in [x86-64-design.md](x86-64-design.md) and summarised under [Per architecture](#per-architecture).

## Decisions made in review

| Question | Decision |
|---|---|
| Spec format | One Python file per kernel, holding the signature and the oracle. Provisional; changes when a kernel needs it to. |
| Win thresholds | None. The judge reports speedups with confidence intervals and significance; a person decides. |
| Memory rule | Strict: no reads outside the buffer. |
| Writes inside the output capacity | Allowed per buffer when the spec declares it `scratch`. Strict by default. The tail is then garbage, so the binding must return only the reported elements and must not use a scratch kernel on a caller's buffer or an exactly sized one. |
| Keeping edits during iteration | The agent keeps an edit when `judge compare` says it is significantly faster. |
| Agent target | At least 10% over naive C, then continue until options are exhausted. If 10% is not possible, report best effort and why. |
| Committing results | Committed for now; revisit later. |
| NumPy timing | Best effort: timed from Python with call overhead, labelled as such. Stricter if results warrant it. |
| Fixture kernels | `argmax_f32`, `compress_f32` and `compress_scratch_f32`. |
