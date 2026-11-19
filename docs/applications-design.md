# Applications: five NumPy programs, with and without hotkernels

This is a proposal for review. It asks one question of five programs that this project's kernels were not written for: does `hotkernels.install()` make the whole program faster, and if not, where does its time go.

## Summary

- **The judge and `delivery time` show that single calls get faster.** They do not show that a program does. A program gains only by the share of its time that is spent in calls a kernel takes.
- **Five applications are run both ways:** one written here for the purpose, and four that are other people's code. They were chosen by field, before any of them was timed, and the list is not changed by the results.
- **Each gets three measurements:** the time of a whole run both ways with the results compared bit for bit, `report()`'s counts of kernel calls and calls left to NumPy, and a profile of where the processor's time goes in native code.
- **The profile is the new part.** It says what share of a run is NumPy's loops, the BLAS, the Python interpreter, memory and other compiled code, by function. That is what explains a program that did not get faster, and what ranks the kernels not yet written.
- **The output is a report,** [applications-findings.md](applications-findings.md), with one section per application, and an issue for each gap it finds.

## How this differs from the workloads design

[workloads-design.md](workloads-design.md) times library functions and benchmark programs and reads `report()`. Its runner is in a pull request that is not merged. This study is of a different size of thing, whole programs that make thousands of NumPy calls of many kinds, and it adds the native profile, which `report()` cannot give: `report()` counts calls that reached a hook, and says nothing of the time spent in calls that have no hook at all. The two share the method of timing. When the workloads runner is merged, the applications can become workloads of it and the profile an option of its command.

## The applications

| Name | What it is | Source | Why it was chosen |
|---|---|---|---|
| `mlp_train` | Training a two-layer perceptron by mini-batch gradient descent, with accuracy measured each epoch | Written for this study | The commonest kind of ad hoc NumPy program: matrix products and whole-array arithmetic. Expected to reach few hooks |
| `min_char_rnn` | Andrej Karpathy's `min-char-rnn.py`, a character-level recurrent network | [gist](https://gist.github.com/karpathy/d4dee566867f8291f086), BSD, copied | A well-known program that makes many calls on arrays of about a hundred elements, where the time is in NumPy's call path. It calls `np.clip` |
| `trimesh_mesh` | Building a mesh, sampling points on its surface, merging vertices and computing its geometric properties | [trimesh](https://trimesh.org), called as a library | A large library written almost wholly in NumPy: grouping, sorting, cumulative sums and searches |
| `sklearn_metrics` | Scoring a classifier's predictions: ROC AUC, average precision, calibration curve, confusion matrix, log loss | [scikit-learn](https://scikit-learn.org) `sklearn.metrics`, called as a library | What follows every trained model. Sorting, cumulative sums, searches and counts on a million elements |
| `empyrical_stats` | The risk and return statistics of a set of daily return series: drawdown, Sharpe and Sortino ratios, volatility, rolling figures | [empyrical-reloaded](https://github.com/stefan-jansen/empyrical-reloaded), called as a library | Finance code on time series: cumulative products and running maxima, NaN-aware reductions |

A library is called through its documented functions with typical arguments. Copied code is copied unchanged with its licence; the wrapper may supply its input and bound its run, and may not change a NumPy call. `min-char-rnn.py` loops for ever and reads `input.txt`: the wrapper replaces its `while True:` with a count of iterations when it loads the file, and supplies the text.

## What is measured

**Time and results.** As `delivery demo` does: alternating runs on plain NumPy and after `install()`, the minimum of each side, and the results of every pair compared with `delivery.check.describe`. A size is chosen for each application so that one run takes between 0.1 and 2 seconds on plain NumPy. Only `install()` with its defaults is timed; `install(inexact=True)` is run once more and reported separately, because its results are not bit-identical.

**Counts.** One installed run on fresh counters: kernel calls and calls left to NumPy, per hooked call.

**Native profile.** The application is run in a loop in a child process for a few seconds, plain and installed, while the system's sampling profiler (`sample` on macOS) records the stack of every thread each millisecond. The function at the top of each stack is put in one of these classes:

| Class | Is |
|---|---|
| Python | The interpreter and its object code: the cost of the program's own Python and of making calls |
| NumPy | Functions of NumPy's extension modules, kept by name: `DOUBLE_multiply`, the sort, the call path |
| BLAS | Accelerate or OpenBLAS |
| hotkernels | The binding and the kernels |
| Memory and system | `memmove`, `memset`, the allocator, the kernel |
| Other compiled code | The extension modules of the library under test and of pandas and SciPy |

Samples of threads that are waiting are left out. The share of NumPy functions for which a kernel exists, exists for another element type, or does not exist bounds what the program could gain. The profile runs only on macOS; on another system the command says so and gives the times and counts alone.

**Python profile.** One plain run under `cProfile`, to tie the native functions to the lines of the program that call them.

## The command

```
uv run --group apps delivery apps [--only NAME] [--profile] [--record]
```

The libraries go in an `apps` dependency group, pinned by `uv.lock` and not installed by default. `--record` writes `delivery/results/<machine>-apps.md` and `.json`, with the load average and the versions of the libraries. Templates are under `delivery/templates/`.

## What the results are used for

- **The report.** For each application: the two times, the counts, the profile, and a paragraph that says why it gained or did not.
- **Issues.** One for each gap: a hooked call left to NumPy, with the reason found, and a NumPy function with a large share of an application's time and no kernel.
- **The next shortlist.** The profile's table of NumPy functions by time, across five applications, is evidence of a kind the survey did not have.

## Documents that change

| Document | Change |
|---|---|
| `docs/applications-findings.md` | New: the report |
| `README.md` | The command, in "Using the delivery tool" |
| `docs/workloads-design.md` | A link to this study |

No release note: nothing in `hotkernels/` changes.

## Order of work

1. This document.
2. The runner, the profile and the five applications.
3. A recorded run on an idle machine, and the report.
4. The issues.

## Out of scope

- Changing the binding or a kernel so that an application gains. Those are the issues.
- x86-64. The times and counts work there; the profile needs a sampler for Linux.
- Running the applications in CI.

## Decisions still open

1. **Whether the applications join the workloads runner** once it is merged, or stay a separate command.
2. **A profile on Linux.** `perf` gives the same table and needs permissions the judge's machine may not grant.

## What changed in the building

- **`min-char-rnn.py` needed three textual changes, not one.** The program is Python 2. Besides bounding its loop, the wrapper rewrites its `print` statements and `xrange` when it loads the file. The list is `REWRITES` in `delivery/apps/min_char_rnn.py`; no NumPy call is touched, and the copied file is unchanged.
- **Functions are classed by the file they are in.** `sample` names an image by its file name, so the profile maps every compiled file under `site-packages` to its package. The C maths library got a class of its own, since `exp`, `log` and `tanh` are neither NumPy's nor the system's in the sense that matters here.
- **The shares are of working threads, by the function at the top of the stack.** A thread that waits is left out, so a main thread waiting for the BLAS's workers is not counted and the workers are. Functions with fewer than five samples are not in `sample`'s table and not in the shares.
- **The sizes were set by hand** so that a run takes 0.2 to 1.3 seconds: 12,000 examples for the perceptron, 300 iterations of the network, an icosphere of 81,920 faces, a million predictions, 60 return series of twenty years.
