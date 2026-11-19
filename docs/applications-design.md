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

Samples of threads that are waiting are left out. The share of NumPy functions for which a kernel exists, exists for another element type, or does not exist bounds what the program could gain. The profile ran first only on macOS; `perf` on Linux was added later, see [Two sizes, a profile on Linux, and the record](#two-sizes-a-profile-on-linux-and-the-record-issue-193). Where no sampler may run, the command says why and gives the times and counts alone.

**Python profile.** One plain run under `cProfile`, to tie the native functions to the lines of the program that call them.

## The command

```
uv run --group apps delivery apps [--only NAME] [--size small|large] [--profile] [--record [--allow-busy]]
uv run delivery apps-report delivery/results/<machine>-apps.json
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
- x86-64 hardware. The times and counts work there, and the profile does through `perf`, but neither has been run on an x86-64 machine yet ([#193](https://github.com/sofeikov/hotkernels/issues/193)).
- Running the applications in CI.

## Decisions still open

1. **Whether the applications join the workloads runner** once it is merged, or stay a separate command.
2. **A profile on Linux.** Resolved: `perf record -g`, with the permissions it needs stated by the command when they are missing. See [below](#a-profile-on-linux).

## What changed in the building

- **`min-char-rnn.py` needed three textual changes, not one.** The program is Python 2. Besides bounding its loop, the wrapper rewrites its `print` statements and `xrange` when it loads the file. The list is `REWRITES` in `delivery/apps/min_char_rnn.py`; no NumPy call is touched, and the copied file is unchanged.
- **Functions are classed by the file they are in.** `sample` names an image by its file name, so the profile maps every compiled file under `site-packages` to its package. The C maths library got a class of its own, since `exp`, `log` and `tanh` are neither NumPy's nor the system's in the sense that matters here.
- **The shares are of working threads, by the function at the top of the stack.** A thread that waits is left out, so a main thread waiting for the BLAS's workers is not counted and the workers are. Functions with fewer than five samples are not in `sample`'s table and not in the shares.
- **The sizes were set by hand** so that a run takes 0.2 to 1.3 seconds: 12,000 examples for the perceptron, 300 iterations of the network, an icosphere of 81,920 faces, a million predictions, 60 return series of twenty years.

## Two sizes, a profile on Linux, and the record (issue #193)

The first recorded run left three things owed ([#193](https://github.com/sofeikov/hotkernels/issues/193)): each application ran at one size, the profile ran only on macOS, and the record was taken at a load average of 18.5. This section is the design for the first two and for how the third is taken. It was written before the code.

### Two sizes

A result at one size cannot say whether it is about the program or about the size: `min_char_rnn` works on arrays of a hundred elements and `sklearn_metrics` on a million. Every application now runs at two sizes, `small` and `large`, relative to the program:

| Application | Small | Large | What changes |
|---|---|---|---|
| `mlp_train` | 196 inputs, 64 hidden units | 784 inputs, 256 hidden units (the first size) | The weight matrix: 12,544 elements against 200,704. Examples, batches and epochs stay |
| `min_char_rnn` | the licence, 58 distinct characters (the first size) | a text of 1,000 distinct characters | The vocabulary, and with it two weight matrices and their gradients, 5,800 elements against 100,000. The program is unchanged; only its input is |
| `trimesh_mesh` | 5,120 faces, 12,500 samples | 81,920 faces, 200,000 samples (the first size) | One sixteenth of the mesh |
| `sklearn_metrics` | 50,000 and 15,000 predictions | 1,000,000 and 300,000 (the first size) | One twentieth of the predictions |
| `empyrical_stats` | 60 series of 5,040 days (the first size) | 6 series of 50,400 days | The same number of returns in series ten times longer |

**How a module declares them.** `SIZES` maps each of `small` and `large` to the line that describes the data, and `make_data(size)` builds the inputs of that size. Whatever `run` needs to know of the size travels in the tuple `make_data` returns. `delivery.apps.SIZE_NAMES` lists the two names, and every application has both.

**How a size is chosen.** `delivery apps` runs both by default, so a record always holds both; `--size small` or `--size large` runs one.

**Short runs.** A small size can run in 10 ms, below what seven rounds of one run time well. A round is now as many runs as last 100 ms, at least one, and the time reported is of one run. The number of runs per round is recorded.

**What the record holds.** One row per application and size, in the order of `NAMES` then `SIZE_NAMES`. A timing names its `size` and gives the line describing it as `data`; in the first record `size` held that line. The Markdown report has one table of times with a size column, and under each application one section per size. `delivery apps-report RECORD.json` writes the report again from a record; it reads the first record too, which ran each application at the size marked "the first size" above, and its report has been rewritten in the new layout with every figure unchanged.

### A profile on Linux

The profile needs a sampler that records the native stack of every working thread about once a millisecond, and a parser that reduces its output to counts by the function at the top of the stack. On Linux that is `perf`:

```
perf record -e cpu-clock -F 1000 -g -p PID -o perf.data -- sleep 6
perf script -i perf.data -F comm,tid,ip,sym,dso
```

- **`cpu-clock`,** a software event, samples a thread only while it runs. That is the same set of samples as `sample`'s after its waiting threads are left out, and it works where the processor's counters are not exposed, as in a virtual machine.
- **`-p PID -- sleep 6`** attaches to the child after it has built its data and run once, as `sample` does, and samples every thread it has or starts.
- **`perf script`** prints each sample as a line naming the thread followed by one line per frame, the top first. The parser counts the first frame of each sample. A frame names its file by path; the file name is what is classed, so the classes are those of the macOS profile. The Linux names are added: `libm` is the maths library, `libc`, the dynamic loader, `libstdc++` and the kernel (`[kernel.kallsyms]`) are memory and system, OpenBLAS, a reference BLAS and MKL are the BLAS.
- **Unlike `sample`,** perf's output is not cut at five samples, so every function counts.

**Permission.** perf may sample only what `kernel.perf_event_paranoid` allows. At 1 or below, or as root, the event takes user space and the kernel. At 2, the kernel's default, a user may sample its own processes in user space only: the command then samples `cpu-clock:u`, and the record says that the kernel's time is not in the shares. At 3 and above, a level some distributions set, or in a container whose security profile forbids `perf_event_open`, it may not sample at all. The command does not guess from the level: it tries perf on `true` with each event, and when neither works it says why, with perf's own message, the level, and the `sysctl` that allows it. It then gives the times and counts alone, as before.

**Why not `py-spy --native`.** py-spy reads a process's memory through `ptrace`, which most distributions allow only for a process's own children (Yama's `ptrace_scope` 1), so it would have to start the child itself and sample its data being built. It stops the process at each sample. Its native unwinding is supported on x86-64 Linux alone. And what it adds, the Python frames, `cProfile` already gives. perf needs no new dependency, and the paranoid level it needs is one `sysctl`.

### The record

The record is taken when nothing else is running: when `gh run list` shows no unfinished run, since this machine hosts the CI runners. `--record` asks `gh run list` and refuses while any of the last twenty runs is queued, waiting or in progress; `--allow-busy` records anyway. The number of unfinished runs and the load average are written in the record. A load average above 1.0 is not refused, since this desktop does not reach it ([#51](https://github.com/sofeikov/hotkernels/issues/51)); the record marks its times rough, as before.

The record itself is taken once every change that moves an application's time has been made, and the findings are brought up to date from it then.

### What changed in the building of the second round

- **perf leaves the kernel out silently.** At `perf_event_paranoid` 2 a user who is not root asking for `cpu-clock` gets it, with the kernel's samples dropped and no error. Trying the event cannot tell the two apart, so the event is chosen from the level and the user first (`cpu-clock` for root or at 1 and below, `cpu-clock:u` otherwise) and only then tried. A user with `CAP_PERFMON` at level 2 is sampled in user space although the kernel would be allowed.
- **The small `min_char_rnn` has 58 distinct characters,** counted from the licence when the module loads; the description of the size says so, so its line differs from the first record's.
- **Short rounds.** The small `trimesh_mesh` runs in about 13 ms and the small `mlp_train` in about 36 ms, so they are repeated in each round as designed.
- **Tests.** The repository had no tests of the delivery tool's own code. `tests/test_apps.py` tests the sizes, the record and its old layout, the refusal while CI runs, and the parsing and classing of perf's output on a capture and on hand-written x86-64 samples. CI runs it on the hosted Linux runner (`uv run --group dev pytest tests`).
- **What ran where.** The perf path ran end to end in a Docker container on this Apple M1 Pro, ARM64 Linux (Debian bookworm, `linux-perf` 6.1, kernel 6.12): as root with `--privileged`, which samples `cpu-clock`; as an unprivileged user at level 2, which samples `cpu-clock:u`; and in a container without `--privileged`, where perf is refused and the command says so. hotkernels installs nothing on ARM64 Linux, so there both sides were plain NumPy. No figure from the container is used anywhere. Nothing has run on x86-64 Linux; that is left to [#193](https://github.com/sofeikov/hotkernels/issues/193)'s follow-up on the x86-64 machine. Level 3 was not reached in the container (its `/proc/sys` is read-only) and is covered by a unit test only.
