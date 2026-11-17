# Working in this repository

hotkernels is a set of kernels for NumPy operations, hand-written assembly where it beats the compiler and plain C where it does not, checked and timed by an automated judge. Read [docs/core-design.md](docs/core-design.md) for the project, [docs/judge-design.md](docs/judge-design.md) for the judge [docs/delivery-design.md](docs/delivery-design.md) for how the kernels reach NumPy and [docs/release-design.md](docs/release-design.md) for how releases are made.

This file is for coding agents. The first half applies to any work here; the second half applies when you are writing or tuning a kernel.

## General rules

- Design changes go through a pull request with a document in `docs/` before any code is written.
- Run everything through `uv`: `uv run judge <command>`. There is no other setup.
- `uv run judge selftest` must pass before any change to `judge/` is proposed.
- `uv run delivery check` must pass before any change to `hotkernels/` or to a kernel that is installed is proposed. It runs every hooked NumPy call with and without hotkernels and compares them.
- Timings are only meaningful on native hardware. Never report numbers taken under emulation.
- A pull request that changes what a user of the package gets leaves a release note. See [Release notes](#release-notes).
- READMEs describe the present state only: what exists and how to use it. No phases, status, roadmap or development history. Plans belong in `docs/`, and history belongs in git. When a change makes a README inaccurate, update it in the same change.

## Python code standards

These apply to all Python in this repository.

- **Command lines use `typer`.** Never `argparse`.
- **Output uses the `logging` module.** Never a bare `print`. Each module gets `log = logging.getLogger(__name__)`; the CLI configures the handler.
- **Generated text uses Jinja templates**, wherever it is produced: reports, tables, Markdown, progress lines, generated C. Each package keeps its templates in its own `templates/` directory (`judge/templates/`, `survey/templates/`). Python passes values in; it does not assemble multi-line or column-aligned text with string formatting, and it does not build a report out of consecutive `print` or `log` calls.
- **Every module, class and function has a docstring**, including private and nested ones. Say what it does and anything a reader could not guess from the signature.
- **Docstrings describe every parameter and the return value**, in `Args:` and `Returns:` sections: what each one means, its unit where it has one, and any constraint on it. A one-line docstring is enough only for a function with no parameters and an obvious result, or a nested function a few lines long.
- **Every dataclass field has a comment** directly above it saying what it means.
- **No magic constants.** Numbers and strings with meaning get a named module-level constant, with a comment explaining the value where it is not obvious.
- **Comment the non-obvious.** Where code depends on something outside itself, such as a calling convention, a file format or a memory layout, explain it at the point of use.
- **Small functions over closures that mutate outer state.** Return values; do not use `nonlocal` counters.

## Release notes

A release gathers short notes left by the pull requests since the last one. You write the note for your pull request.

**When a note is needed.** Your pull request changes `hotkernels/`, or a kernel the package installs (its `kernels/<name>/arm64/kernel.s`, `naive.c` or `spec.py`), and a user of the package would notice: a call served by a new kernel, a faster kernel, a fixed result or crash, a supported Python or NumPy version gained or lost, different `report()` output. Changes that touch only `judge/`, `survey/`, `delivery/`, `docsite/`, `release/`, `docs/` or a README need no note. CI checks this with `uv run release check-note`; run it yourself before pushing.

**When it is not.** A change in those paths that no user would notice, such as a comment, a rename or a refactor with identical behaviour, gets the `no release note` label instead. Say in the pull request description why.

**Where it goes.** One file per pull request, `release-notes/upcoming/<PR number>.<type>.md`. The number is only known once the pull request exists, so open the pull request first and add the note in a second commit. The type is one of:

| Type | For |
|---|---|
| `kernel` | A NumPy call or element type that a kernel now serves |
| `speed` | A kernel or the binding got faster |
| `fix` | A wrong result, a crash, or a call that went to the wrong place |
| `compat` | A Python or NumPy version gained or lost, a new platform, a change to what `install()` installs by default |
| `change` | Anything else a user would notice |

If a pull request does two of these, pick the one a user cares about most. One note per pull request, never one per commit.

**How to write it.**

- One to three sentences, plain Markdown, no heading. It becomes one bullet in the release notes.
- Write for someone who calls NumPy, not for someone who works on this repository. Name the NumPy call and element type (`np.cumsum` on `int32`), not the kernel's file name or the hook that serves it.
- Say what changed for them, not how. "`np.clip(x, lo, hi, out=x)` gave wrong results on some data and now goes to NumPy", not "added an aliasing check in hook_loop.c".
- A speed figure is allowed if it comes from `uv run judge run` or `uv run delivery time` on native hardware. Give the size it was measured at. It will be re-measured at release.
- Do not mention the pull request number in the text; the release notes add it.

Example, `release-notes/upcoming/31.kernel.md`:

```markdown
`np.cumsum` on `int32` is served by a kernel: about 11x faster than NumPy at a million elements.
```

**Releases are not yours to make.** Do not run `uv run release prepare` or `uv run release tag`, change the package version, or push tags, unless you are asked to.

## Writing or tuning a kernel

### The one rule

Keep a change to a kernel only if both of these hold:

1. `uv run judge check <kernel>` passes.
2. `uv run judge compare <kernel> --against HEAD` reports `RESULT: significantly faster`.

If either fails, revert the change. Commit each kept change before trying the next, so `HEAD` is always the best version so far.

### Files you may edit

During a kernel run you may change exactly two files:

- `kernels/<name>/<arch>/kernel.s`
- `kernels/<name>/attempts.md`

When the run is over and its pull request is open, and the kernel is one the package installs, add the run's one release note: a `speed` note summarising it, as described in [Release notes](#release-notes). That is the only other file a run creates.

Everything else is locked: `spec.py`, `naive.c`, and all of `judge/`. The judge compares those against the `main` branch and fails the run if they differ. Do not pass `--unlocked`; it exists for people creating a kernel or changing the judge, and it marks the results as untrusted.

If you believe the spec, the naive reference or the judge is wrong, stop and say so. Do not work around it.

### Commands

| Command | Use it for | Time |
|---|---|---|
| `uv run judge check <kernel>` | After every edit. Lint and correctness. | About 5 seconds, up to 15 for a kernel with many scalar sets |
| `uv run judge compare <kernel> --against HEAD` | After `check` passes. Is this edit faster? | About 30 seconds |
| `uv run judge run <kernel>` | The full report against naive C and NumPy. | About 30 seconds |
| `uv run judge check <kernel> --case '<case>'` | Rerun one failing case. | Instant |

`compare` and `run` both run `check` first.

### Reading a failure

```
FAIL argmax_f32 (arm64): wrong result
  wrong return value
  case:     n=2 arr=mid2 seed=1 x=nan_at@1
  expected: 1
  got:      0
  rerun:    uv run judge check argmax_f32 --case 'n=2 arr=mid2 seed=1 x=nan_at@1'
```

Only the first failure is shown. The case line says how the input was built:

- `n` is the element count.
- `arr` is where the buffers sit in memory:
  - `end`: the buffer ends exactly at an inaccessible page.
  - `start`: the buffer starts exactly after one.
  - `mid1` to `mid3`: deliberately misaligned.
- `x=nan_at@1` means argument `x` is random data with a NaN at index 1. Other value classes are named the same way (`max_at`, `neg_zero_at`, `ties`, `denormal`, `all_true`, and so on); see `judge/cases.py`.
- `scalars=zero_lo`, when present, names the set of scalar arguments the call was made with. The sets and their values are listed in the kernel's `spec.py`; without the field the scalars had their default values. A kernel must be right for every set, and `compare` times every set marked `timed`.

The failure categories:

| Category | Meaning |
|---|---|
| `lint` | The source breaks a static rule (see below). Nothing was run. |
| `build` | The assembler rejected the file. |
| `wrong result` | The output differs from NumPy's. |
| `memory safety` | The kernel touched memory outside its buffers, wrote to an input, or wrote past the output it reported using. A crash with `SIGBUS` or `SIGSEGV` is almost always this. |
| `calling convention` | The kernel changed a register it must preserve. |
| `crash` | Any other crash, or the kernel did not finish. |

### What a kernel must be

A kernel is a pure leaf function over the buffers it is given. The judge enforces all of this:

- **No reads or writes outside the buffers, by even one byte.** This is strict: loading a full vector that extends past the end of an array is a failure, even though it would usually be harmless. Handle leftover elements inside the buffer, for example with a scalar loop, or with a final vector that overlaps the previous one and ends exactly at the end of the array.
- **Inputs are read-only.** Writing to an input buffer crashes. A buffer the spec declares `InOut` is the exception: the kernel reads and modifies it in place, and may store an element's own bits back to it.
- **Write only the output you report.** If the kernel returns a count, it must not write beyond that many elements. The one exception is an output the spec declares `scratch=True`: there the kernel may write anywhere inside the buffer's capacity, and only the reported elements have to be right. Writing past the capacity is always a failure. Check `spec.py` before relying on this; the default is strict.
- **No function calls, no system calls, no writable data.** Read-only constants are fine (`.section __TEXT,__const` or `__TEXT,__literal16`).
- **Export exactly one symbol:** the kernel itself.
- **Match NumPy exactly** unless the spec declares a tolerance. This includes NaN, infinities, signed zeros and which element wins a tie.

### ARM64 on macOS

- The symbol for C function `foo` is `_foo`. Declare it with `.globl _foo` and align with `.p2align 2`.
- Labels that start with `L` are local to the file. Use them for everything except the kernel symbol.
- Arguments arrive in `x0` to `x7` (integers and pointers) and `v0` to `v7` (floating point). The result goes in `x0` or `v0`. The two kinds are numbered separately: in `f(double *x, size_t n, double lo, double hi)` the pointer is `x0`, `n` is `x1`, `lo` is `d0` and `hi` is `d1`.
- A leaf kernel may use stack memory below `sp` if it restores `sp` before returning and keeps it 16-byte aligned while it is moved.
- You may freely use `x0` to `x17` and `v0` to `v7`, `v16` to `v31`.
- You must preserve `x19` to `x28`, `x29`, the stack pointer, and the low 64 bits of `v8` to `v15`. Prefer not to use them at all.
- Never use `x18`. The platform reserves it.
- `//` and `;` both start a comment.
- Unaligned vector loads and stores are allowed and cost nothing extra on this hardware. Buffers are not guaranteed to be aligned.
- Kernels are only ever called with `n >= min_n` from the spec.

### The attempts log

Record every attempt in `kernels/<name>/attempts.md`, including the ones you reverted. Later runs read this file so they do not repeat dead ends. One entry per attempt:

```markdown
## 7. Unroll the main loop to four vectors per iteration

- **Idea:** hide the latency of the compare by keeping four independent maxima.
- **check:** pass
- **compare:** significantly faster, 1.21x [1.19 – 1.23]
- **Kept:** yes
- **Notes:** gain is all above 1,024 elements; small sizes unchanged.
```

For a rejected attempt, say why in one line: failed `check` (and how), not significantly faster, or significantly slower.
