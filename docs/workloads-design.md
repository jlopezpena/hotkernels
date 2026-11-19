# Workloads: timing hotkernels on code it did not write

This is a proposal for review. It adds a third level of timing, above the judge and above `delivery time`: whole programs and library functions that a user runs, written by other people, timed with and without `hotkernels.install()`.

A study of five whole applications, with a native profile of each, is in [applications-findings.md](applications-findings.md). It is separate from the runner proposed here.

## Summary

- **Every figure today comes from code written for hotkernels.** The judge times a kernel. `delivery time` times one statement that meets the kernel's conditions. `delivery demo` times [examples/workload.py](../examples/workload.py), which was written to use eleven hooked calls. None of them says what a user of scikit-image, pandas or scikit-learn gets.
- **A probe says: often nothing.** Of eleven library and benchmark-suite workloads, three were faster (1.4x to 5.7x) and eight were within 2% of NumPy. In most of the eight, a hooked call was reached and left to NumPy. See [What was verified](#what-was-verified).
- **The "Left to NumPy" counts are the useful output.** They show which condition of the binding real code does not meet: a strided column, a short row under `axis=1`, a weighted histogram, more than 64 sample points in `np.interp`. That is a list of work ranked by real use, which the survey could not give.
- **Proposed:** `uv run delivery workloads`. It runs each workload both ways, checks that the results are the same, and records the time, the speedup and `report()`'s counts per workload. Workloads come from two sources: code taken from published benchmark suites, and functions of downstream libraries called as their users call them.
- **Not proposed:** adopting `asv` or `pyperf`, or running NumPy's own `asv` suite. See [Considered and not proposed](#considered-and-not-proposed).

## What is wrong today

The figures in [README.md](../README.md) are true for the statement they name and for nothing else. A user does not write `np.histogram(x, 50, (-3.0, 3.0))` on a contiguous `float64` array. They call `skimage.exposure.equalize_hist(image)` or `pd.cut(series, 10)`, and the library makes the NumPy call with the arguments it chooses. Whether those arguments meet a kernel's conditions is not known for any library.

The core design names a speedup of an unmodified program as the aim ([core-design.md](core-design.md#phase-3-the-shortlisted-kernels-end-to-end)). The one unmodified program measured so far was written by this project.

## What was verified

A throwaway script ran each workload below on plain NumPy and after `install()`, best of seven, and read `report()` after one installed run. It is not in the repository.

- **Machine.** Apple M4 Max, macOS, CPython 3.14.8, NumPy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, scikit-image 0.26.0, SciPy 1.18.1. Native.
- **Load.** Load average 2.1. The figures rank the workloads; they are not good to a few percent.

| Workload | Data | NumPy | Installed | Faster by | Kernel calls | Left to NumPy |
|---|---|---|---|---|---|---|
| NPBench `azimint_naive` | 1,000,000 float64, 100 bins | 53.6 ms | 38.9 ms | 1.38x | `x[mask]` 100 | |
| NPBench `azimint_hist` | 1,000,000 float64, 100 bins | 7.57 ms | 4.59 ms | 1.65x | `np.histogram` 1 | `np.histogram` 1 (weighted) |
| `s[s > 0]`, pandas `Series` | 1,000,000 float64 | 3.54 ms | 0.63 ms | 5.65x | `x[mask]` 1, `np.nonzero` 1 | |
| `skimage.exposure.equalize_hist` | 1024 × 1024 float64 | 35.4 ms | 32.3 ms | 1.10x | `np.histogram` 1, `np.cumsum` 1 | `np.interp` 1 (256 sample points) |
| `skimage.exposure.equalize_hist` | 2048 × 2048 uint8 | 131.9 ms | 131.7 ms | 1.00x | `np.cumsum` 1 | `np.bincount` 1, `np.interp` 1 |
| `skimage.filters.threshold_otsu` | 2048 × 2048 uint8 | 4.42 ms | 4.41 ms | 1.00x | `np.argmax` 1 | `np.bincount` 1, `np.cumsum` 4 |
| `pd.cut(s, 10)` | 1,000,000 float64 | 14.8 ms | 14.7 ms | 1.01x | `x[mask]` 4 | `np.searchsorted` 1 |
| `pd.qcut(s, 10)` | 1,000,000 float64 | 29.8 ms | 29.8 ms | 1.00x | `x[mask]` 4, `x[mask] = v` 2 | `np.searchsorted` 1 |
| `KBinsDiscretizer.transform` | 200,000 × 8 float64 | 28.4 ms | 28.3 ms | 1.00x | | `np.searchsorted` 8 |
| `LogisticRegression.predict` | 200,000 × 8, 3 classes | 2.73 ms | 2.73 ms | 1.00x | | `np.argmax` 200,000 |
| `scipy.stats.percentileofscore` | 1,000,000 float64 | 0.66 ms | 0.67 ms | 0.98x | | `np.count_nonzero` 2 |

Calls with no count on either side are left out of the table. What the probe shows:

- **Code that uses mask selection gains, wherever it is written.** NPBench's `azimint_naive` and a pandas boolean filter reach `x[mask]` with no change.
- **A library's own spelling often misses the kernel by one condition.** scikit-learn searches each column of a C-ordered matrix, which is strided. `LogisticRegression.predict` takes `argmax(axis=1)` over rows of three, far under the type slot's 128 elements. `equalize_hist` interpolates over 256 sample points, and `np.interp`'s kernel takes at most 64.
- **Why some calls were left is not known yet.** The `np.searchsorted` in `pd.cut` and the `np.bincount` in `threshold_otsu` meet the conditions as written in [delivery-design.md](delivery-design.md#which-hook-serves-which-kernel) at first sight. Finding out is the first use of the tool.

## Design

### What a workload is

One Python file per workload under `delivery/workloads/`, holding:

| Name | Is |
|---|---|
| `SOURCE` | Where the code comes from: a URL and a version or commit, or the library and the function called |
| `LICENCE` | The licence of copied code, or `None` for a call into an installed library |
| `SIZES` | Named sizes, `S`, `M` and `L`, each a dictionary of parameters |
| `make_data(**size)` | Builds the inputs from a fixed seed |
| `run(*data)` | Runs the workload once and returns its results |

The sizes follow NPBench's presets: `S`, `M` and `L` are chosen so that plain NumPy takes about 10, 100 and 1,000 ms on the reference machine. A workload that cannot reach `L` in reasonable memory leaves it out.

A workload copied from a benchmark suite is copied unchanged, with its licence header, so that "not written for hotkernels" stays true. A wrapper may convert its inputs and outputs; it may not change the NumPy calls.

### Where workloads come from

**Published benchmark suites, copied.**

- [NPBench](https://github.com/spcl/npbench) (BSD-3-Clause; Ziogas et al., ICS 2021): 54 NumPy programs from HPC domains, with presets sized by NumPy's run time. Most are dense linear algebra and stencils, which no kernel serves. They are kept anyway as controls: a workload that reaches no hook measures what installing costs everyone else. The candidates that reach hooks are `azimint_naive` and `azimint_hist` (from pyFAI), and any others the first run finds.
- [numpy-benchmarks](https://github.com/serge-sans-paille/numpy-benchmarks): the scientific NumPy kernels Pythran is benchmarked on, and one of NPBench's own sources. Short programs written the way people write NumPy. Its licence is to be checked before anything is copied.
- [pyhpc-benchmarks](https://github.com/dionhaefner/pyhpc-benchmarks) (Unlicense): parts of the Veros ocean model, copied nearly verbatim. Mostly arithmetic on large arrays; a control, and a check of whether its masks reach the hook.

**Downstream libraries, called.** Functions of scikit-image, pandas, scikit-learn and SciPy, called with typical arguments. These are not copied; the libraries go in a new `workloads` dependency group in `pyproject.toml`, pinned by `uv.lock`. The first list is the probe's table. A library function is a good workload if its documentation or its tutorials use it on data of the sizes timed, and it reaches at least one hook, whether taken or left.

Each library gets a few workloads, not dozens. The aim is to cover each hooked NumPy call from at least two libraries, and each library's commonest spelling of it.

### The command

```
uv run delivery workloads [--only NAME] [--size S|M|L] [--record]
```

For each workload and size:

1. **Correctness.** Run once each way and compare the results with `delivery.check.describe`, as `delivery demo` does. Results that differ fail the workload, and it is not timed. A workload whose results go through an `inexact=True` kernel is out of scope until the tolerance can be stated for a whole program.
2. **Timing.** Alternating runs each way, the minimum of each side, as `delivery demo` does today. The number of runs comes from the size: enough that the total is about one second per side.
3. **Counts.** One more installed run on fresh counters, then `report()`: kernel calls and calls left to NumPy, per hooked call.

The output is a table per size, rendered from a Jinja template under `delivery/templates/`. `--record` writes it with the JSON beside it to `delivery/results/<machine>-workloads.md` and `.json`, with the load average, the library versions and the NumPy version, as `delivery time --record` does.

`delivery demo` stays as it is. `examples/workload.py` becomes one workload among the others, marked as written by this project.

### What the results are used for

- **README and site.** A short table, "Code we did not write", beside "From Python" in [README.md](../README.md): one line per workload at size `M`. It states the workloads that did not get faster as well as those that did.
- **Issues.** Each call left to NumPy that a workload reaches gets one issue, with the workload as its reproduction. The probe already gives five: strided `searchsorted` needles, `argmax(axis=1)` on short rows, weighted `np.histogram`, `np.interp` over more than 64 sample points, and the two calls whose reason is not known.
- **Fusion.** [core-design.md](core-design.md#later) says that fusion first needs a way to find which chains of calls are worth fusing. The workloads are where the chains are.

### Considered and not proposed

| Option | Why not |
|---|---|
| Use [asv](https://asv.readthedocs.io/) as the runner, as NumPy, SciPy, pandas, scikit-image and Astropy do | asv compares commits of one package, each in its own environment. Here the comparison is one environment with and without `install()`, and the delivery tool already does that with alternating runs in one process. asv would add a build matrix and a results website for no gain |
| Run NumPy's own asv suite with `install()` | Its benchmarks time one call each, on arguments chosen to test NumPy. That is what `delivery time` does with arguments chosen to reach the kernels. It does not add realism |
| Use [pyperf](https://pyperf.readthedocs.io/) for each timing | It runs each benchmark in fresh worker processes, which is better for noise. It would mean installing in each worker and reading `report()` back across processes. Worth doing if the alternating method proves too noisy for 10 ms workloads; not first |
| Mine usage from GitHub code or notebooks for "realistic" arguments, for example with [python-api-inspect](https://github.com/scientific-python/python-api-inspect) | It counts calls, not arguments, sizes or memory layouts, which are what decide whether a kernel is taken. A later step, to rank which library to add next |
| Copy library functions into the repository | The point is the library's own code at its current version. A copy would drift and would be code written, in the end, by this project |

## Documents that change

| Document | Change |
|---|---|
| `README.md` | The "Code we did not write" table; the command in "Using the delivery tool" |
| `docsite/pages/start.md` | One paragraph: which kinds of library call gain and which do not yet |
| `docs/delivery-design.md` | A link from "End to end" to this document |
| `docs/core-design.md` | The phase 3 exit criterion's "unmodified script" points here |

No release note: nothing in `hotkernels/` changes.

## Order of work

1. This document, reviewed.
2. The runner and three workloads (`azimint_naive`, the pandas boolean filter, `equalize_hist` on float64), recorded on one machine.
3. The rest of the probe's list, NPBench controls, and the issues for each call left to NumPy.
4. The README table and the site, from a run on an idle machine.

## Exit criterion

- `uv run delivery workloads` runs every workload at `S` and `M` with identical results both ways.
- At least ten workloads from at least three sources, with each hooked NumPy call reached by at least one of them.
- A recorded run on native hardware at a load average under 1.0, in `delivery/results/`.
- An issue for each kind of call left to NumPy that a workload reaches.

## Out of scope

- Changing the binding or a kernel to take more of these calls. Those are the issues this work raises.
- Timing on x86-64. The runner works there unchanged; its recorded run waits for the x86-64 package ([x86-64-design.md](x86-64-design.md)).
- Running the workloads in CI. Correctness at size `S` may go in CI later; timing never does.

## Decisions still open

1. **Where the libraries live.** A `workloads` dependency group in the root `pyproject.toml`, or a separate project under `delivery/workloads/` with its own lock file, so a library upgrade does not touch the main lock.
2. **Pinned or current library versions.** Pinned gives figures that can be repeated. Current shows what a user installing today gets, and catches a library that changes its spelling of a call. Proposed: pinned, updated by hand.
3. **Licences of copied code.** NPBench is BSD-3-Clause, but its `azimint` files carry pyFAI's CC-BY 3.0 header. Copying needs the notices kept and a line in the licence file.
4. **A summary figure.** A geometric mean over workloads, as NPBench reports, would be one number for the README. It would also hide that most workloads do not change. Proposed: no summary figure; the table only.
