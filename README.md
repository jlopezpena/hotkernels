# hotkernels

Fast kernels for the NumPy operations where NumPy is slow, delivered as a zero-friction drop-in: hand-written assembly where it can actually win, plain C where the compiler is already enough.

```python
import hotkernels
hotkernels.install()
```

Kernels are written by coding agents iterating against a judge. The judge checks every kernel against NumPy and times it against NumPy and naive `clang -O3` C.

**On an Apple M1 Pro, all 22 kernels are faster than NumPy at every size measured. At a million elements, 17 of them are more than 5x faster and the best is 76x. Twenty return NumPy's result bit for bit.**

## Results

How many times faster each kernel is than NumPy, and than the same loop written in plain C and compiled with `clang -O3`:

<img src="docs/images/speedup-vs-numpy.svg" alt="Bar chart of each kernel's speedup over NumPy at 1,048,576 elements, from 29.0x for np.cumsum on float32 down to 1.83x for np.isin with test values far apart. Lighter bars show the more favourable of two timed cases. The table below has every value.">

| NumPy call | Elements | vs NumPy | vs `clang -O3` |
|---|---|---|---|
| `np.cumsum(x)` | float32 | 29.0x | 7.83x |
| `np.interp(x, xp, fp)` | float64 | 22.7x | 6.02x |
| `np.cumsum(x)` | float64 | 20.1x | 5.48x |
| `x[mask]` | float32 | 17.3x – 41.7x | 2.70x – 5.70x |
| `np.sum(x)` | uint8 | 15.7x | 5.07x |
| `np.cumsum(x)` | int64 | 12.5x | 1.95x |
| `x[mask]` | float64 | 11.0x – 22.5x | 1.67x – 4.10x |
| `np.histogram(x, bins, range)` | float64 | 9.70x | 3.59x |
| `np.searchsorted(a, v)` | float64 | 8.84x | 8.42x |
| `np.argmax(x)`, `np.argmin(x)` | float32 | 8.82x, 8.86x | 26.6x, 26.7x |
| `np.clip(x, lo, hi)` | float32 | 7.70x – 11.1x | 1.00x – 1.44x |
| `x[mask] = value` | float64 | 7.48x – 21.4x | 4.64x – 19.0x |
| `np.argmax(x)`, `np.argmin(x)` | float64 | 6.86x, 7.10x | 12.7x, 13.2x |
| `np.bincount(x)` | uint8 | 6.74x | 1.31x |
| `np.nonzero(mask)` | bool | 5.15x – 19.2x | 2.47x – 8.08x |
| `np.count_nonzero(x)` | float64 | 3.79x | 1.28x |
| `np.clip(x, lo, hi)` | float64 | 3.70x – 4.88x | 1.01x – 1.30x |
| `np.count_nonzero(x)` | int64 | 3.25x | 1.03x |
| `x.astype(np.float32)` | uint8 | 2.68x | ships as the C |
| `np.isin(x, test)` | int64 | 1.83x – 75.6x | 1.00x – 5.34x |

Apple M1 Pro, NumPy 2.5.5, Apple clang 21, arrays of 1,048,576 elements, where the fixed cost of a call no longer matters. A range covers the two cases the kernel is timed in: 5% and 50% of the elements selected for the mask kernels, test values close together or far apart for `isin`, and, for `clip`, bounds of (-0.5, 0.5) at the high end and bounds of (0, 1) on data that holds negative zeros at the low end. Results at every size from 16 to 16,777,216 elements, with confidence intervals, are in [docs/kernel-findings.md](docs/kernel-findings.md).

What stands behind the numbers:

- **Checked against NumPy, not against a description of it.** Every kernel is compared with NumPy's own output on inputs that include NaN, infinities, signed zeros, ties and denormals. Twenty of the 22 match bit for bit. The two float `cumsum` kernels regroup the additions, which is where their speed comes from, and stay within a declared error bound.
- **Memory-safe by test.** Each kernel runs with its buffers placed against inaccessible pages, so reading or writing one byte outside them crashes the check.
- **Assembly only where it wins.** A kernel keeps its assembly only if the judge finds it significantly faster than the compiled C. `astype` from `uint8` to `float32` did not pass that bar and ships as C.

What the numbers do not say:

- **The table times the kernels as native functions.** From Python a call also pays for NumPy's call path and, for `x[mask]` and `np.nonzero`, for allocating the result. Those figures are under [From Python](#from-python).
- **Two kernels are not installed.** `np.searchsorted` and `np.interp` take one length for both of their inputs and cannot serve a real call yet.
- **Float `np.cumsum` is installed only on request.** It is not bit-identical to NumPy and its output is not monotonic for non-negative input, so it needs `hotkernels.install(inexact=True)`.
- **Two results depend on the arguments.** `np.isin` is fast when the test values all lie between 0 and 65,534 or within 32,767 of the first one, and ties the C otherwise. `np.clip` with a lower bound of zero, as in `np.clip(x, 0, 1)`, is about 1.3x over the C in cache when the data holds no negative zeros and runs at the speed of the C when one occurs in the first few thousand elements; with an upper bound of `-0.0` it always runs at the speed of the C. Those cases are timed, and are the low ends of those rows.
- **Scalar arguments are verified for the values each spec lists.** `np.clip` is checked with twelve pairs of bounds and `np.histogram` with sixteen bin counts and ranges, not with every possible one.
- **One machine.** Every figure is from a single Apple M1 Pro.

## From Python

The same machine, timing one Python statement from call to result, on plain NumPy and after `hotkernels.install()`:

| Statement | Elements | 16 elements | 1,048,576 elements |
|---|---|---|---|
| `np.argmax(x)` | float32 | 14.4x | 8.98x |
| `np.clip(x, -0.5, 0.5)` | float32 | 1.01x | 11.2x |
| `np.clip(x, -0.5, 0.5)` | float64 | 1.00x | 5.01x |
| `x[mask]`, half selected | float32 | 2.49x | 31.0x |
| `x[mask]`, half selected | float64 | 2.52x | 15.8x |
| `x[mask] = 0.0`, half selected | float64 | 5.28x | 20.3x |
| `np.nonzero(mask)`, few selected | bool | 1.27x | 13.9x |
| `np.cumsum(x)` | int64 | 1.93x | 13.6x |
| `x.astype(np.float32)` | uint8 | 1.90x | 5.29x |
| `np.sum(x)` | uint8 | 32.9x | 15.6x |
| `np.count_nonzero(x)` | float64 | 3.56x | 3.46x |
| `np.histogram(x, 50, (-3.0, 3.0))` | float64 | 6.27x | 9.40x |
| `np.bincount(x)` | uint8 | 2.23x | 6.74x |
| `np.isin(x, eight_values)` | int64 | 72.8x | 18.2x |

At 16 elements most of the gain is NumPy's own call path being skipped, not the kernel. `np.clip` gains nothing there: its kernel sits beneath NumPy's Python wrapper, which is nearly all of a small call.

- **A call no kernel takes costs under a nanosecond more.** `x[3]` takes 60.3 ns on plain NumPy and 61.1 ns with hotkernels installed.
- **An unmodified program.** [examples/workload.py](examples/workload.py) uses eleven of these operations on two million elements. It runs in 45 ms on plain NumPy and 8.4 ms with the one line added, with the same results bit for bit.
- **Checked from Python.** 70,031 NumPy calls give the same result, exceptions and warnings with hotkernels as without, and NumPy's own test suite gives the same outcome for each of its 47,234 tests.
- **Taken on a machine in use.** The full table, with the load average of the run, is in [delivery/results/apple-m1-pro.md](delivery/results/apple-m1-pro.md).

## What is here

- **The judge** (`judge/`): checks a kernel for correctness and memory safety, and reports how fast it is.
- **Fixture kernels** (`judge/fixtures/`): `argmax_f32`, `compress_f32`, `compress_scratch_f32` and `tally_u8`, used to test the judge itself.
- **The survey** (`survey/`): measures NumPy against naive C per operation, to decide which kernels are worth writing. Recorded results are in `survey/results/`.
- **Kernels** (`kernels/`): 22 kernels for 15 NumPy operations, each with a spec, a naive C reference and, where assembly beats the C, an ARM64 version. Each kernel's `attempts.md` lists every optimisation tried, kept or not.
- **The package** (`hotkernels/`): what a user installs. A C extension points NumPy at the kernels, and `install()`, `uninstall()`, `enabled()` and `report()` control it. Twenty of the 22 kernels are installed.
- **The delivery tool** (`delivery/`): tests and times the package from Python. Recorded timings are in `delivery/results/`.
- **An example** (`examples/workload.py`): a NumPy program written without hotkernels in mind, used to show the effect of the one line.
- **Release tooling** (`release/`, `release-notes/`): the notes each pull request leaves, and the commands that turn them into a release. Publishing to PyPI runs in [`.github/workflows/release.yml`](.github/workflows/release.yml).
- **The documentation site** (`docsite/`): the pages published at [sofeikov.github.io/hotkernels](https://sofeikov.github.io/hotkernels/), with a technical report on how all of this works. Its figures are read from the files above when it is built.

Supported platform: macOS on Apple Silicon (ARM64).

## Using hotkernels

In this repository, `uv sync` builds the package. It needs the Xcode command line tools.

```python
import numpy as np
import hotkernels

hotkernels.install()            # NumPy now uses the kernels, in the whole process

x = np.random.standard_normal(1_000_000)
x[x > 0]                        # served by a kernel
np.clip(x, 0, 1)                # served by a kernel
x[::2].sum()                    # not something a kernel handles: NumPy does it, as before

print(hotkernels.report())      # what is installed, and how often each kernel ran
hotkernels.uninstall()          # NumPy is exactly as it was
```

- `hotkernels.install(inexact=True)` also installs the kernels that are not bit-identical to NumPy.
- `with hotkernels.enabled():` installs for the duration of a block.
- A kernel takes a call only when the array is a plain contiguous `ndarray` of its element type and every other argument is one it implements. Any other call goes to NumPy as it was written.
- On any other platform, `install()` does nothing and `report()` says why.

## Using the judge

Requires the Xcode command line tools and [uv](https://docs.astral.sh/uv/).

```bash
uv run judge selftest            # prove the judge catches known-bad kernels
uv run judge check argmax_f32    # lint and correctness for one kernel
uv run judge run argmax_f32      # correctness, then a timing report
```

## Using the survey

```bash
uv run survey run                 # measure every candidate; takes about 15 minutes
uv run survey run --only argmax   # measure the candidates whose name matches
uv run survey report survey/results/apple-m1-pro.json   # regenerate a report
```

## Using the delivery tool

```bash
uv run delivery check         # every hooked call, with and without hotkernels; about a minute
uv run delivery demo          # run examples/workload.py both ways and compare
uv run delivery time          # time statements from Python; add --record to keep the result
uv run delivery numpy-tests   # NumPy's own test suite, both ways; about ten minutes
```

## Using the documentation site

```bash
uv run docsite serve   # build the site and serve it at http://127.0.0.1:8000, rebuilding on change
uv run docsite build   # build it into build/site
```

Pages are Markdown under `docsite/pages/` and are listed in `docsite/site.toml`.

## Making a release

Each pull request that changes the package leaves a note in `release-notes/upcoming/` (see [AGENTS.md](AGENTS.md#release-notes)). To release:

```bash
uv run release prepare 0.2.0   # set the version and compile the notes into release-notes/0.2.0.md
```

Write the highlights at the top of that file, re-measure on native hardware, and merge it as a pull request. Then, on an up-to-date `main`:

```bash
uv run release tag             # tag the merged commit v0.2.0 and push the tag
```

The tag starts the release workflow, which builds and tests the wheels and waits for the maintainer to approve the `pypi` deployment before uploading. [docs/release-design.md](docs/release-design.md) has the details.

## Documentation

- [sofeikov.github.io/hotkernels](https://sofeikov.github.io/hotkernels/): getting started, and the technical report with interactive figures
- [docs/core-design.md](docs/core-design.md): what the project is building and why
- [docs/judge-design.md](docs/judge-design.md): how kernels are checked and timed
- [docs/survey-design.md](docs/survey-design.md): where NumPy is slow, and which kernels to build in what order
- [docs/kernel-findings.md](docs/kernel-findings.md): what each kernel achieves, how, and where assembly could not win
- [docs/delivery-design.md](docs/delivery-design.md): how the kernels reach unmodified NumPy code: the binding, `install()` and `report()`, and how they are tested
- [docs/release-design.md](docs/release-design.md): how a release is made and published, and who can publish
- [docs/distribution-design.md](docs/distribution-design.md): what a release contains: wheels, the source distribution, other platforms, `python -m hotkernels`
- [AGENTS.md](AGENTS.md): rules for coding agents working in this repository
- [kernels/README.md](kernels/README.md): how to add a kernel

## Licence

MIT
