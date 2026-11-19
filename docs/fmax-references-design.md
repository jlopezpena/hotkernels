# References for the fmax kernels where NumPy's builds differ

This is a proposal for review. It answers [issue 162](https://github.com/sofeikov/hotkernels/issues/162): `cumfmax_f64` (`np.fmax.accumulate` on float64) and `fmax_reduce_f64` (the maximum of one run, behind `np.fmax.reduceat`) have x86-64 assembly and ship on ARM64 as their naive C, and no ARM64 assembly can be written for them, because on ARM64 `uv run judge check` fails on the locked reference itself. Specs are a person's to change, so nothing here is built until the decisions at the end are taken.

The ARM64 figures are an Apple M1 Pro's, NumPy 2.5.5 (baseline NEON/ASIMD), `naive.c` compiled with the judge's flags. Nothing here was timed.

## Summary

- **The references describe NumPy's x86-64 build, and the oracle is the machine's NumPy.** On ARM64 they disagree on two kinds of input only: zeros of both signs where the maximum is a zero, and NaNs of more than one bit pattern where a result is a NaN. The judge generates the first (its `zeros` class) and never the second.
- **The binding already sends exactly those inputs to NumPy**, on both architectures. On everything it lets through, `naive.c` agrees with NumPy on ARM64 for both kernels: no disagreement in 54,900 arrays per call.
- **On ARM64 NumPy is one `fmaxnm` per element** for both calls, from `x[0]` on: a zero maximum is `+0.0` if any `+0.0` is among the zeros, and an all-NaN reduction returns `x[0]`. No lane structure shows.
- **Proposed: narrow the domain to what the binding passes** (option 3). Each spec says what its guard says, and an `In(..., prepare=...)` function, which the judge already has, makes every generated input satisfy it. One reference is then right on both architectures, no judge code changes, and an ARM64 kernel may use `fmaxnm` directly.

## What is wrong today

```
$ uv run judge check cumfmax_f64       # no arm64/kernel.s
FAIL cumfmax_f64 (arm64): reference
  naive.c is wrong: wrong value in out[2]
  case:     n=4 arr=end seed=1 x=zeros
$ uv run judge check fmax_reduce_f64
FAIL fmax_reduce_f64 (arm64): reference
  case:     n=5 arr=end seed=1 x=zeros
```

A reference failure stops the judge before it looks at a kernel, so `check`, `compare` and `run` fail for any `kernel.s`. Both specs say they reproduce NumPy's x86-64 build: `cumfmax_f64` "the later of two equal values is kept", with the operands reversed for `n == 2`; `fmax_reduce_f64` the four lanes of NumPy's AVX2 loop, whose own docstring says an AVX-512 or NEON build "may answer those two cases differently". The survey on this machine has NumPy 5.5x behind the naive C for `np.fmax.accumulate`, so an ARM64 kernel is worth having.

## The evidence

**Every judge case, by value class.** `judge/cases.py`'s own `fixed_cases` (up to 70,000 elements) and 400 `fresh_cases`, each input run through `naive.c` and the spec's oracle, bits compared:

| Kernel | Cases | Disagree | In class | With every zero made `+0.0` |
|---|---|---|---|---|
| `cumfmax_f64` | 20,296 | 389 | `zeros` only (389 of 414) | 0 |
| `fmax_reduce_f64` | 20,241 | 172 | `zeros` only (172 of 409) | 0 |

`neg_zero_at` (one `-0.0` among random values) never disagrees: the maximum there is not a zero. Every NaN the judge makes is `np.nan`, one bit pattern, so the NaN half of both specs (which NaN is returned among several) is checked by the judge on no architecture.

**Outside the judge.** Fifteen kinds of data, sizes 1 to 300 and 511, 1,000, 1,024, 4,097, 5,000, twelve arrays each (3,660 arrays per kind). NaNs "of several patterns" have random payloads and signs, all quiet. "Guarded" counts the arrays the binding sends to NumPy; "passed, disagree" those it lets through on which `naive.c` and NumPy still differ.

| Data | `fmax.reduce`: `naive.c` differs | guarded | `fmax.accumulate`: differs | guarded |
|---|---|---|---|---|
| random, ties, `inf_mix`, denormal | 0 | 0 | 0 | 0 |
| one `-0.0` among random values | 0 | 25 | 0 | 3,660 |
| zeros, both signs | 1,750 | 3,649 | 3,629 | 3,649 |
| zeros of both signs among negatives | 1,774 | 3,594 | 3,509 | 3,594 |
| only `+0.0` among negatives | 0 | 0 | 0 | 0 |
| only `-0.0` among negatives | 0 | 3,635 | 0 | 3,635 |
| zeros of both signs and NaNs | 1,832 | 3,646 | 3,614 | 3,646 |
| NaNs of one pattern, scattered or all | 0 | 0 | 0 | 0 |
| NaNs of several patterns, scattered | 1 | 1 | 1 | 3,553 |
| all NaN, several patterns | 2,748 | 3,648 | 12 | 3,648 |
| leading NaNs of several patterns, then numbers | 46 | 60 | 7 | 3,576 |
| **passed, disagree** | **0 of 54,900** | | **0 of 54,900** | |

On every one of these arrays NumPy's result equals the chain `m = fmaxnm(m, x[i])` from `m = x[0]`, with the running value first, and its running values for `accumulate`. So NumPy's ARM64 rule for `np.fmax.reduce` is: the largest number; if that is a zero, `+0.0` when any `+0.0` is present and `-0.0` otherwise; if every element is NaN, `x[0]`, sign and payload. It does not depend on where the elements sit, unlike the x86-64 rule. A model of NumPy's NEON loop (two lanes, eight vectors unrolled, the C library's `fmax` for the leftovers) gives the same bits, as it must, since `fmaxnm` differs from itself in operand order only on two NaNs. `np.fmax.reduceat(x, [0])` gave the bits of `np.fmax.reduce(x)` on every array.

## What the binding passes

From `hotkernels/src/hotkernels/_binding/hook_method.c`, the same on both architectures:

| Call | Sent to NumPy when |
|---|---|
| `np.fmax.accumulate` (`fmax_accumulate_left_to_numpy`) | `x` holds a signalling NaN, a `-0.0`, or NaNs of more than one bit pattern |
| `np.fmax.reduceat` (`fmax_runs`, `fmax_runs_ambiguous`) | `x` holds a signalling NaN; or some run's result is a zero and `x` holds a `-0.0`; or some run's result is a NaN and the NaNs of `x` are of more than one pattern |

Per run of `reduceat`, the kernel therefore sees: no signalling NaN; no `-0.0` if the run's maximum is a zero; one NaN pattern if the run is all NaN. On those inputs every correct NaN-ignoring maximum has the same bits, which is why both builds of NumPy agree there and why the C ships on ARM64 today.

## The options

| | 1. Per-architecture references | 2. One reference, oracle not NumPy's build | 3. Narrow the domain |
|---|---|---|---|
| What changes | `naive.c` gets an `#if defined(__aarch64__)` branch (the `fmaxnm` chain); each spec's text says what each build does | The oracle becomes a Python model of NumPy's x86-64 loop, or the judge skips classes per architecture | Each spec states the guard's condition; a `prepare` function makes every generated input satisfy it |
| Judge code | None if `naive.c` branches; a per-architecture `naive.c` lookup otherwise | A model oracle: none, but the oracle stops being NumPy. Skipping classes: a new spec field | None |
| ARM64 kernel may use | `fmaxnm` directly, and must give ARM64 NumPy's bits on zeros of mixed sign | Compare and select to keep the later of equals, the lane order of AVX2 for the reduction, the `n == 2` swap | `fmaxnm` directly, or anything that gives the maximum |
| x86-64 kernels | Unchanged, still bound to the AVX2 loop | Unchanged | Unchanged; checked on a subset of what they handle |
| An x86-64 judge whose NumPy runs AVX-512 | Still fails `fmax_reduce_f64`'s reference | Passes (model oracle) | Passes |
| Caller-facing behaviour | Unchanged. The two kernels give different bits on inputs the binding never passes. The ARM64 guard could later drop its zero and NaN-pattern tests | Unchanged | Unchanged |

**Option 1** keeps "the oracle is the machine's NumPy" intact and gives the simplest ARM64 kernel. Its cost is that both architectures' judges then enforce bits that no caller can observe, through a guard that already hides them, and the spec carries two descriptions of the same call. It does nothing for an x86-64 machine with AVX-512, where NumPy picks an eight-lane loop and `fmax_reduce_f64`'s reference fails the same way. Its one gain beyond option 3, a lighter guard on ARM64, is small: the read for signalling NaNs stays, and the zero and pattern tests ride in the same pass.

**Option 2** makes the judge enforce x86-64 NumPy's bits on ARM64, where the machine's NumPy gives other ones. The ARM64 kernel pays for that with work (a select per element, a fixed lane order) to reproduce bits that its own platform's NumPy does not produce and no caller sees. A model oracle also breaks the rule that the oracle is NumPy, which is what makes the judge's verdict mean "bit-identical to NumPy". Skipping classes per architecture is the same as option 3 done less precisely, and only on one architecture.

**Option 3** makes the spec say what the kernel is actually called with. The judge already has the means: `In(..., prepare=fn)`, "a function applied to every generated input, for kernels defined only on some inputs" ([judge-design.md](judge-design.md)), as `searchsorted_f64` sorts its array. A sketch, not proposed text:

```python
def _no_negative_zero(x):
    """Every -0.0 made +0.0: the binding sends an array that holds one to NumPy."""
    return np.where(x == 0, 0.0, x)                               # cumfmax_f64

def _no_negative_zero_at_a_zero_maximum(x):
    """Every -0.0 made +0.0 when the maximum is a zero, as the binding requires of a run."""
    return np.where(x == 0, 0.0, x) if np.fmax.reduce(x) == 0 else x   # fmax_reduce_f64
```

The NaN half needs no generated change: the judge makes one NaN pattern only. The spec states it all the same, as it states the signalling NaN today, so a later value class with payloads cannot reach these kernels unnoticed. Both distributions of both kernels supply their own inputs, which `prepare` does not touch, so every timed input is byte for byte what it was. On these domains `naive.c` as it stands is right on both architectures (the last column of the first table), so it need not change, and it stays the baseline of the x86-64 figures. The `n == 2` paragraph of `cumfmax_f64` and the lane description of `fmax_reduce_f64` stop being needed to fix any result; they can stay as notes on what NumPy's x86-64 build does outside the domain.

What it costs: the judge stops checking behaviour that the binding never relies on, so a guard that is loosened later must come with a spec change, and the two are kept in step by hand. The case descriptions keep their class names (`x=zeros`) for inputs that `prepare` changed, as `x=desc` already names a sorted array for the search kernels. The spec change re-locks both kernels; their x86-64 assembly handles a superset of the new domain, so `judge check` should pass there unchanged, which has to be confirmed on the x86-64 machine. Their simpler paths (the `vmaxpd` against minus infinity, the fallback to NumPy's four lanes in `fmax_reduce_f64`, the `n == 2` path in `cumfmax_f64`) become removable, by a later kernel run under the one rule, not by this change.

## Proposed

**Option 3, with the `prepare` functions above.** The binding has always decided what these kernels must get right; option 3 writes that into the spec, where the judge can enforce it, and nothing beyond it. It is the only option that needs no change to the judge or to `naive.c`, keeps the oracle NumPy, leaves every recorded timing valid, makes one reference right on both architectures and on an AVX-512 machine, and lets the ARM64 kernel be a plain `fmaxnm` loop. Options 1 and 2 both spend kernel work, or judge rules, on bits that a caller cannot see.

## Documents and files that change if accepted

| File | What |
|---|---|
| `kernels/cumfmax_f64/spec.py`, `kernels/fmax_reduce_f64/spec.py` | The domain in the docstring, a `prepare` on `x`. A person's change |
| `docs/kernel-findings.md` | "What a caller must know" and the constraints table: the guards become what the specs declare, on every architecture; the ARM64 rule found here |
| `docs/judge-design.md` | One more example in the `prepare` row: a domain that excludes inputs the binding never passes |
| `docs/second-shortlist-design.md` | The two kernels leave the list of ARM64 assembly that waits on an issue |
| `docsite/pages/start.md`, `hotkernels/README.md`, `docs/delivery-design.md` | Nothing: what a caller gets does not change. To be confirmed by grep in the pull request |

The spec change alters nothing a caller can observe, so its pull request takes the `no release note` label, with that reason. The ARM64 kernels that follow are kernel runs with release notes of their own.

## Out of scope

- The ARM64 assembly of either kernel ([issue 122](https://github.com/sofeikov/hotkernels/issues/122), [issue 127](https://github.com/sofeikov/hotkernels/issues/127)).
- Any change to the binding's guards, on either architecture.
- Simplifying the x86-64 kernels or `naive.c` to the narrowed domain.
- A judge value class with NaNs of several bit patterns, which no kernel needs today.
- Running `judge check` on Apple Silicon in CI, which would have caught this.

## Decisions still open

| | Question | Proposed |
|---|---|---|
| 1 | Which option? | 3 |
| 2 | The domain of `fmax_reduce_f64`: the guard's exact condition (no `-0.0` when the maximum is a zero) or the simpler "no `-0.0`" | The exact condition: it keeps the `neg_zero_at` cases, which the binding does pass |
| 3 | Do the x86-64 descriptions (later of equals, `n == 2`, the four lanes) stay in the spec docstrings as notes? | Yes, marked as outside the domain; they explain the x86-64 kernels' paths |
| 4 | Does `prepare` also make NaNs one pattern, against a future value class? | No; the docstring states it, and a new class is a judge change reviewed with the specs in view |
| 5 | Is a `prepare` that changes inputs enough, or should the judge gain a field that excludes value classes by name? | `prepare`: it states the exact condition, and excluding `zeros` would also drop the all-`+0.0` arrays |

## What changed in the building

Step 1, the two specs, is built in the pull request stacked on this proposal's, as proposed and ahead of its acceptance: option 3, the exact condition for `fmax_reduce_f64`, the x86-64 descriptions kept as notes outside the domain, and NaNs left as the judge makes them. Nothing under `judge/`, no `naive.c` and no `kernel.s` changed.

- **The two `prepare` functions are the sketch's**, with one guard added: `fmax_reduce_f64`'s returns an empty array unchanged, which `min_n=1` makes unreachable, rather than ask NumPy for the maximum of nothing.
- **The domains are the binding's conditions as one call sees them.** `cumfmax_f64`'s is exactly `fmax_accumulate_left_to_numpy`'s. `fmax_reduce_f64`'s is a run's: the binding tests a negative zero and the NaN patterns over the whole array, so what it passes for one run is a subset of the domain.
- **The timed inputs are unchanged.** Both kernels supply every timing input from a distribution, which `prepare` does not touch, so no recorded figure needed taking again.
- **On ARM64,** with no ARM64 assembly, `uv run judge check cumfmax_f64 --unlocked` passes 20,368 cases and `fmax_reduce_f64` 20,313, where both failed on the reference at their first `zeros` case before.
- **`docs/judge-design.md`** gains the two kernels in the `prepare` row, and the row now says that a distribution's inputs do not pass through `prepare`, which `judge/spec.py` already said.
- **On x86-64 the two checks are not yet run.** CI's Linux job does not run `judge check` for these kernels, so whether the x86-64 assembly passes under the narrowed specs (it should, the domains being subsets of the old ones) is still to be confirmed on the x86-64 machine.
- **`fmax_reduce_f64` has ARM64 assembly on the narrowed spec**, in the pull request stacked on the specs' ([issue 127](https://github.com/sofeikov/hotkernels/issues/127)). As this design expected, it is a plain `fmaxnm` reduction: eight accumulators, overlapping first and last vectors, the elements in any order, which the domain allows. `uv run judge check fmax_reduce_f64 --unlocked` passes its 20,313 cases, and the kernel is 8.23x [8.22 – 8.23] over the naive C across the judge's sizes on the Apple M1 Pro ([kernel-findings.md](kernel-findings.md#maximum-by-runs-npfmaxreduceat)). Outside the domain it gives NumPy's ARM64 bits for zeros of both signs and for all-NaN runs of several patterns (none of 4,980 arrays differed; the second because every `fmaxnm` keeps the accumulator that began as `x[0]` as its first operand), but not for a signalling NaN (440 of 2,490 arrays with one among numbers differed). So on ARM64 the binding's guards for negative zeros and NaN patterns are not needed by this kernel; they are kept, since they are shared with x86-64 and the domain is written from them. `cumfmax_f64` still ships on ARM64 as its C.
