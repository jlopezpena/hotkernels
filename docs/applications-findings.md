# Applications: what five NumPy programs gain from hotkernels

This is the report of the study designed in [applications-design.md](applications-design.md). Five programs that the kernels were not written for were run on plain NumPy and after `hotkernels.install()`, their results compared, and their time profiled. The full tables are in [delivery/results/apple-m1-pro-apps.md](../delivery/results/apple-m1-pro-apps.md).

## Summary

- **Two of five got measurably faster, by 8% and 12%.** Two more gained 2% to 3%, and one did not change. No application got slower by more than the noise, and every result was the same with and without hotkernels, bit for bit.
- **The kernels work where they are reached.** In each application the calls a kernel took got faster. They are a small share of each program's time.
- **The time is somewhere else,** and the profile says where: sorting and `np.unique` (half to three quarters of two applications), the BLAS and whole-array arithmetic that NumPy already vectorises (the two neural networks), a library that bypasses NumPy (half of the finance application), and accumulations along an axis of a matrix (the other half).
- **Four gaps would move these numbers,** each filed as an issue with its measurement: 2-D accumulation along axis 0 ([#187](https://github.com/sofeikov/hotkernels/issues/187)), a stable `argsort` ([#188](https://github.com/sofeikov/hotkernels/issues/188)), `np.unique` on labels ([#189](https://github.com/sofeikov/hotkernels/issues/189)) and in-place `np.clip` ([#186](https://github.com/sofeikov/hotkernels/issues/186)). By arithmetic on the measured shares, the first three would take three of the five applications to between 1.3x and 1.8x.
- **The figures were taken on a busy machine,** at a load average of 18.5, because the CI runners on it did not go quiet. The speedups are ratios of alternating runs and are stable; the profile's shares are rough. See [Limits](#limits).

## Results

Apple M1 Pro, CPython 3.12.7, NumPy 2.5.5. Seven alternating runs of each side, the best of each.

| Application | What it is | NumPy | `install()` | Faster by | Results | Kernel calls | Left to NumPy |
|---|---|---|---|---|---|---|---|
| `mlp_train` | Training a two-layer perceptron, written for this study | 226.3 ms | 208.7 ms | 1.08x | identical | 188 | 4,942 |
| `min_char_rnn` | Karpathy's `min-char-rnn.py`, copied | 287.7 ms | 289.5 ms | 0.99x | identical | 0 | 11,400 |
| `trimesh_mesh` | Mesh processing with trimesh | 180.2 ms | 160.6 ms | 1.12x | identical | 35 | 61 |
| `sklearn_metrics` | Scoring a classifier with scikit-learn | 1005.8 ms | 978.6 ms | 1.03x | identical | 98 | 300,161 |
| `empyrical_stats` | Risk statistics of return series with empyrical | 1331.1 ms | 1305.2 ms | 1.02x | identical | 1,020 | 1,080 |

Since this run, `np.argmax(m, axis=1)` is served and a Python call is counted once however many rows NumPy splits it into ([#190](https://github.com/sofeikov/hotkernels/issues/190), [argmax-rows-design.md](argmax-rows-design.md)). With that change `mlp_train` reports 190 kernel calls and 942 left to NumPy, and `sklearn_metrics` 99 and 161 (one run each, load average about 5). The times above are not affected beyond the noise: the call was 0.25% of `sklearn_metrics` and less of `mlp_train`.

The median of the seven runs is within 6% of the best on every row, on both sides. A difference of 1% to 3% is at the edge of what this run can show; the 0.99x of `min_char_rnn` is no change.

`install(inexact=True)` gave the same times to within 2% on every application. Only the results of `empyrical_stats` differed in that mode, as it says they may: it calls `np.cumsum` on `float64`.

Where the time went on plain NumPy, by the class of the function at the top of each sampled stack:

| Application | Python | NumPy | BLAS | Maths library | Memory and system | Other compiled code |
|---|---|---|---|---|---|---|
| `mlp_train` | 0.5% | 64.3% | 15.9% | 0.2% | 19.2% | 0.0% |
| `min_char_rnn` | 8.9% | 41.8% | 31.2% | 4.6% | 13.6% | 0.0% |
| `trimesh_mesh` | 5.4% | 80.9% | 0.9% | 3.0% | 9.4% | 0.4% |
| `sklearn_metrics` | 0.7% | 96.0% | 0.0% | 1.4% | 1.4% | 0.5% |
| `empyrical_stats` | 2.4% | 47.8% | 0.0% | 2.4% | 0.9% | 46.5% |

None of the five is held back by the Python interpreter or by input and output. Four spend most of their time inside NumPy. That is the time hotkernels could reach, and the sections below say why it mostly does not.

## `mlp_train`: 1.08x

A hand-written training loop: 12,000 examples of 784 `float64` inputs, 256 hidden units, batches of 128, two epochs. Source: [delivery/apps/mlp_train.py](../delivery/apps/mlp_train.py).

**What a kernel took.** One statement, the ReLU's backward pass, `dhidden[hidden <= 0] = 0`, 188 times on a 128 × 256 array. That is the whole gain.

**Where the rest is.**

| Share | Function | What it is in the program |
|---|---|---|
| 50.4% | `DOUBLE_subtract` | `params[name] -= LEARNING_RATE * gradients[name]` on the 784 × 256 weights |
| 16.6% | `memmove` | `x[batch]`, which gathers 128 rows of 784, and copies around the matrix products |
| 15.9% | Accelerate | the four matrix products of a step |
| 5.0% | `DOUBLE_add` | adding the penalty and the bias |
| 3.3% | `DOUBLE_multiply` | `LEARNING_RATE * gradient` |
| 2.6% | `array_assign_boolean_subscript` | the masked assignment a kernel takes |

**Why it does not gain more.** The matrix products are Accelerate's and the element-wise loops are NumPy's own vector loops; neither has a kernel and neither is slow for what it does. The weight update is the largest cost because of memory, not arithmetic: timed alone, with its arrays in cache, the statement takes 0.08 ms; inside the loop it takes 0.45 to 0.53 ms, since the gradient computation has pushed the weights out of the cache and the machine was busy. Its share was 26% to 37% in three earlier profiles at lower load. It is one pass over memory written as two, and only fusion would change that ([#191](https://github.com/sofeikov/hotkernels/issues/191)). Since measured otherwise: the second pass was not the cost. The in-place subtraction is slow because Accelerate's threaded matrix product has just read the weights on other cores, and a loop of the binding's own now takes it on ARM64; see [gradient-step-findings.md](gradient-step-findings.md).

**What was left to NumPy.** `np.argmax(scores, axis=1)` on 2,000 rows of ten, counted as 4,000 calls, and 940 `float64` sums, for which there is no kernel. Neither costs anything that shows in the profile ([#190](https://github.com/sofeikov/hotkernels/issues/190)). The `np.argmax` call is now served, 3.5x faster than NumPy, and counted as one call each time.

## `min_char_rnn`: no change

Andrej Karpathy's 112-line recurrent network, copied unchanged, run for 300 iterations on sequences of 25 characters with 100 hidden units.

**What a kernel took.** Nothing. The program reached hooks 11,400 times and every call was left to NumPy.

**Where the time is.**

| Share | What |
|---|---|
| 31.0% | Accelerate: 61,800 calls of `np.dot` on a 100 × 100 matrix and vectors of 100 |
| 18.9% | `DOUBLE_add`: accumulating gradients, `dWhh += np.dot(dhraw, hs[t-1].T)`, and Adagrad's memory |
| 13.6% | Memory and system: `bzero` for `np.zeros_like`, the allocator |
| 8.9% | The Python interpreter |
| 4.6% | `exp`, `tanh` and the floating-point status checks around them |
| about 23% | NumPy's call path, spread over dozens of functions with no one above 2% |

**Why it does not gain.** Every array has between 60 and 10,000 elements, so the program's time is the cost of making 100,000 calls and of the BLAS, not of any loop a kernel replaces. The one hooked call that matters is the gradient clipping, `np.clip(dparam, -5, 5, out=dparam)`, 1,500 times. hotkernels leaves it to NumPy because `out` is given. On a 100 × 100 array that call takes 9.1 µs; the same clip without `out` takes 1.4 µs with hotkernels ([#186](https://github.com/sofeikov/hotkernels/issues/186)). Taking it would be worth about 3% here. The binding now takes it, through a buffer of its own ([clip-in-place-design.md](clip-in-place-design.md)); the figures here are from before that.

The other calls left are `np.sum` on `float64` (8,100, in the softmax) and `np.random.choice`'s internals. The evidence this program gives on hooks for short arrays is in [#99](https://github.com/sofeikov/hotkernels/issues/99).

## `trimesh_mesh`: 1.12x

trimesh 5.2.1: build a mesh of 81,920 faces with merging of vertices, read eleven properties, sample 200,000 points on the surface, cut a section, subdivide.

**What a kernel took.** 35 calls of eight kinds. The one that pays is in `trimesh.sample.sample_surface`:

```python
weight_cum = np.cumsum(weights)
face_index = np.searchsorted(weight_cum, face_pick)
```

NumPy's `binsearch<double>` was 10.6% of the run, the largest single function. With hotkernels it is gone from the profile and `searchsorted_f64` is 1.5%. The rest are `x[mask]` on `float64` (15 calls), `np.nonzero` (9), `np.cumsum` on a mask (4), `np.clip` (3), `np.bincount`, `np.sum` on `uint8` and `m.any(axis=1)`.

**Where the rest is.**

| Share | Function | What it is in the program |
|---|---|---|
| 18.7% | `aquicksort`, `amerge_at`, `atimsort` on `uint64` and `int64` | nine `argsort` calls, in `np.unique` and `trimesh.grouping`: 26.3% of the run under cProfile |
| 7.6% | `memmove` | copies |
| 5.7% | `_contig_to_contig` | casts and copies between types |
| 5.1% | `blake2b` | trimesh hashing its arrays for its cache |
| 7.6% | `mapiter_get`, `npyiter_buffered_iternext` | indexing by integer arrays, `vertices[faces]` |

**Why it does not gain more.** A quarter of the program is sorting integer keys, and there is no sort kernel ([#188](https://github.com/sofeikov/hotkernels/issues/188)). Another eighth is gathering rows by index, `vertices[faces]`, which no kernel serves. Of the 61 calls left to NumPy, 19 are `x[mask]` on arrays no kernel covers, 19 are casts other than `uint8` to `float32`, and 11 are sums of types other than `uint8`; none is a large share.

## `sklearn_metrics`: 1.03x

scikit-learn 1.9.1: ROC and precision-recall curves, AUC, average precision, a calibration curve, Brier score, log loss, accuracy, F1 and a confusion matrix on a million binary predictions; a confusion matrix, classification report, log loss and top-3 accuracy on 300,000 predictions over ten classes.

**What a kernel took.** 98 calls: `x[mask]` on `float64` (53), `x[mask] = v` (24), `np.bincount` on `int64` (16), `np.nonzero`, `np.where(c, a, b)` and one `np.searchsorted`, the binning of `calibration_curve`. With hotkernels 2.4% of the run is inside kernels.

**Where the rest is.** Two things are three quarters of the run:

| Share under cProfile | Calls | What |
|---|---|---|
| 49.6% | 5 | `argsort` with `kind="mergesort"` on a million `float64` scores: `roc_curve`, `precision_recall_curve`, `roc_auc_score`, `average_precision_score` and `top_k_accuracy_score` each sort the scores again |
| 24.9% | 67 | `np.unique` on the labels: every metric validates its inputs afresh |
| 3.2% | 20 | `np.searchsorted` on `int64`, in the label encoder |
| 3.2% | 183 | reductions |

**Why it does not gain more.** Neither of the two has a kernel. A stable `argsort` of a million `float64` takes 117 ms; one twice as fast would make this application 1.33x faster ([#188](https://github.com/sofeikov/hotkernels/issues/188)). `np.unique` on a million labels with two distinct values takes 5.9 ms and hashes every element; done as the bincount kernels do it, the application would be about 1.25x faster ([#189](https://github.com/sofeikov/hotkernels/issues/189)).

**What was left to NumPy.** 300,161 calls, of which 300,000 are one statement, `np.argmax(proba, axis=1)`, counted once per row. It takes 2.5 ms, 0.25% of the run ([#190](https://github.com/sofeikov/hotkernels/issues/190)). It is now served, in 0.7 ms, and counted as one call. The rest are casts, `searchsorted` on `int64`, `np.bincount` with weights and `np.cumsum` on `float64`.

## `empyrical_stats`: 1.02x

empyrical-reloaded 0.7.3: twelve statistics, alpha and beta, cumulative returns, and a rolling drawdown and Sharpe ratio over a year's window, for each of 60 series of 5,040 daily returns.

**What a kernel took.** 1,020 calls: `x[mask]` (420), `x[mask] = v` (240), `np.clip` (120), `np.where(c, a, b)` (120) and `np.fmax.accumulate` (120). Each is fast and none is where the time is.

**Where the time is.**

| Share | Function | What it is in the program |
|---|---|---|
| 22.4% | `DOUBLE_multiply` | `cumprod(axis=0)` on a matrix of one window per column, in the rolling drawdown |
| 18.4% | `DOUBLE_fmax` | `np.fmax.accumulate(out, axis=0)` on the same matrix |
| 18.6% | `nanstd_one_float64` | bottleneck, in the rolling Sharpe ratio |
| 13.3% | `nanmin_one_float64` | bottleneck, the minimum of each window's drawdown |
| 7.6% | `nanmean_one_float64` | bottleneck |
| 6.6% | `nanmax_one_float64` | bottleneck |

**Why it does not gain.** Two reasons, each about half the run.

- **46.5% never reaches NumPy.** empyrical calls bottleneck for its NaN-aware reductions. hotkernels hooks NumPy, so these calls reach no hook and `report()` cannot show them ([#192](https://github.com/sofeikov/hotkernels/issues/192)). *Since then:* hotkernels hooks bottleneck's four functions as well, with kernels that give bottleneck's results bit for bit ([bottleneck-design.md](bottleneck-design.md)). With both this and the accumulation below served, `uv run --group apps delivery apps --only empyrical_stats` gave 1,287.8 ms on plain NumPy and 360.5 ms after `install()`, 3.57x, with identical results, at a load average of 9; in alternating runs at the same load, `install(bottleneck=False)` took 860 ms and `install()` 364 ms.
- **38% is accumulation down the columns of a matrix,** which hotkernels took only on one-dimensional arrays when this was recorded. NumPy walks each column with a stride of a row. The same work done a row at a time is six times faster even from Python: 0.69 ms against 4.29 ms for `fmax`, 0.65 ms against 4.34 ms for the product, on 253 × 4,789. At that speed the application would be about 1.45x faster ([#187](https://github.com/sofeikov/hotkernels/issues/187)). Since served by loops of the binding ([accumulate-rows-design.md](accumulate-rows-design.md)): 1.57x in a run of this application on the same machine at a load average of about 4, against 1.01x before the change in the same conditions.

## What the five say together

**Why a program does not get faster, in order of how much time it explains here:**

| Reason | Where | Can hotkernels reach it? |
|---|---|---|
| Sorting and `np.unique` | 75% of `sklearn_metrics`, 26% of `trimesh_mesh` | Yes, with new kernels: [#188](https://github.com/sofeikov/hotkernels/issues/188), [#189](https://github.com/sofeikov/hotkernels/issues/189) |
| Whole-array arithmetic NumPy already vectorises, bound by memory | 59% of `mlp_train`, 19% of `min_char_rnn` | Only by fusing statements: [#191](https://github.com/sofeikov/hotkernels/issues/191) |
| A library that calls something other than NumPy | 46.5% of `empyrical_stats` | Not through NumPy's hooks; since [#192](https://github.com/sofeikov/hotkernels/issues/192), through a hook in bottleneck itself |
| Accumulation along an axis of a matrix | 38% of `empyrical_stats` | Yes, in the binding: [#187](https://github.com/sofeikov/hotkernels/issues/187) |
| The BLAS | 31% of `min_char_rnn`, 16% of `mlp_train` | No, and it is not slow |
| NumPy's call path on short arrays | about 23% of `min_char_rnn` | Partly: [#99](https://github.com/sofeikov/hotkernels/issues/99) |
| A hooked call whose arguments miss a condition | `np.clip(..., out=x)`: about 4% of `min_char_rnn` | Yes: [#186](https://github.com/sofeikov/hotkernels/issues/186) |

**What this changes about the plan.**

- **The survey ranked calls by how slow NumPy is at each. These programs rank them by time spent.** The two orders differ. `argsort` and `np.unique` are on neither shortlist and are the largest items here. The calls the kernels serve best, mask selection and `np.searchsorted`, are real but small shares of real programs.
- **"Left to NumPy" counts are a poor guide to time.** The largest count in the study, 300,000, is 0.25% of its application. It was one Python call counted once per row, which `report()` no longer does ([#190](https://github.com/sofeikov/hotkernels/issues/190)). The call that costs most among those left, the 2-D accumulation, is counted 60 times. And the largest costs, sorting and bottleneck, are not counted at all, because they have no hook. Time has to come from a profile.
- **With the three kernels that are within reach,** [#187](https://github.com/sofeikov/hotkernels/issues/187), [#188](https://github.com/sofeikov/hotkernels/issues/188) and [#189](https://github.com/sofeikov/hotkernels/issues/189), the estimates above put `sklearn_metrics` near 1.8x, `empyrical_stats` near 1.45x and `trimesh_mesh` near 1.3x. These are arithmetic on measured shares, not measurements.

## Issues filed

| Issue | Gap | Evidence |
|---|---|---|
| [#186](https://github.com/sofeikov/hotkernels/issues/186) | `np.clip(x, lo, hi, out=x)` is left to NumPy | 9.1 µs against 1.4 µs on 100 × 100 |
| [#187](https://github.com/sofeikov/hotkernels/issues/187) | Accumulation along axis 0 of a 2-D array | 38% of `empyrical_stats`; 6x a row at a time |
| [#188](https://github.com/sofeikov/hotkernels/issues/188) | No kernel sorts | 49.6% of `sklearn_metrics`, 26.3% of `trimesh_mesh` |
| [#189](https://github.com/sofeikov/hotkernels/issues/189) | `np.unique` on labels | 24.9% of `sklearn_metrics` |
| [#190](https://github.com/sofeikov/hotkernels/issues/190) | `argmax(axis=1)` over short rows, and how it is counted | 300,000 calls counted for 2.5 ms |
| [#191](https://github.com/sofeikov/hotkernels/issues/191) | Fusion: `w -= lr * g` | the largest cost of `mlp_train` after the matrix products |
| [#192](https://github.com/sofeikov/hotkernels/issues/192) | Calls that go to bottleneck | 46.5% of `empyrical_stats` |
| [#193](https://github.com/sofeikov/hotkernels/issues/193) | The study itself: an idle machine, Linux, a second size | below |

## Limits

- **A busy machine.** The run started at a load average of 18.5. The speedups come from alternating runs and their medians agree with their minima, so the ratios hold. The profile's shares do not all hold: functions bound by memory are inflated when other processes compete for the cache. `DOUBLE_subtract` in `mlp_train` is the clear case.
- **One machine, one architecture.** On x86-64 NumPy has vector loops it lacks on ARM64, and the profile needs a sampler for Linux.
- **One size each.** A larger mesh or a longer return series would move the shares.
- **Five programs.** They were chosen by field before any was timed, and the list was not changed by the results. They are still five. Dense linear algebra, image processing and pandas-heavy code are represented by one program or by none.
- **Estimates of what a missing kernel is worth** assume the rest of the program is unchanged and take the kernel's speed from a stand-in measured in Python, not from a kernel.

## Running it

```bash
uv run --group apps delivery apps --profile            # all five, with profiles
uv run --group apps delivery apps --only trimesh_mesh  # one, times and counts only
uv run --group apps delivery apps --profile --record   # write delivery/results/<machine>-apps.md
```
