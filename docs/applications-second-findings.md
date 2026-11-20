# Applications, second set: what five more NumPy programs gain from hotkernels

This is the report of the study designed in [applications-second-design.md](applications-second-design.md). Five programs that no kernel was written or tuned for were run on plain NumPy and after `hotkernels.install()`, their results compared bit for bit, and their time profiled. The record is [delivery/results/apple-m1-pro-apps-second.md](../delivery/results/apple-m1-pro-apps-second.md); the conditions it was taken in are under [Limits](#limits).

## Summary

- **One of five gains measurably: the particle filter, 1.11x** on 100,000 particles and 1.08x on 2,000. pandas gains 1.07x on two million rows and 1.05x on a hundred thousand. scikit-image gains 2% to 3%, the ocean model and Astropy 1% or nothing. Every result was the same with and without hotkernels, bit for bit, at both sizes, and with `install(inexact=True)` too.
- **Where a kernel is reached it works.** The particle filter's resampling search (`np.searchsorted` on `float64`) was 11.0% of the program on NumPy and is nearly all of its gain. pandas' gain is its masks, `nonzero` and one stable `argsort`.
- **The time is somewhere hotkernels does not reach,** and the reasons are the first study's in a new order. First, compiled code that is not NumPy's: half of the pandas analysis is pandas' own hash tables and group-by loops, and over a third of the image pipeline is SciPy's filters and scikit-image's Cython. Second, NumPy's own vector loops: the ocean model is element-wise arithmetic and copies of slices, which only fusion would change. Third, calls with no kernel: the median and the quantiles, `np.arctan2`, normal random draws, the FFT, gathering rows by index.
- **Eight gaps are new and filed,** [#220](https://github.com/sofeikov/hotkernels/issues/220) to [#227](https://github.com/sofeikov/hotkernels/issues/227); four existing issues got new evidence. The largest within reach are the median of a large array (a third of the Astropy program, [#220](https://github.com/sofeikov/hotkernels/issues/220)) and `np.arctan2` (9.5% of the image pipeline and 7.3% of the particle filter, [#221](https://github.com/sofeikov/hotkernels/issues/221)). By arithmetic on the measured shares, the two would take Astropy to about 1.3x and the image pipeline to about 1.1x.
- **The first study's kernels do not carry over by themselves.** The first set reached 1.27x to 3.5x after kernels were written for its gaps. Five programs that no kernel was written for gain 1.00x to 1.11x. That is the measure of what a user gets today from code hotkernels has not seen, and why the profile, not the survey, should choose the next kernels.

## Results

Apple M1 Pro, CPython 3.12.7, NumPy 2.5.5, pandas 3.1.1, scikit-image 0.26.0, Astropy 8.0.3, SciPy 1.18.1, bottleneck 1.6.1, hotkernels at `7a277a8`. Taken with `uv run --group apps delivery apps --set second --profile --record` with no GitHub Actions run unfinished, at a load average of 3.41. Seven alternating rounds of each side, the best of each; the median of the seven is within 2.3% of the best on every row, on both sides.

| Application | Size | Data | NumPy | `install()` | Faster by | Results | Kernel calls | Left to NumPy |
|---|---|---|---|---|---|---|---|---|
| `particle_filter` | small | 2,000 particles, 100 steps | 20.2 ms | 18.8 ms | 1.08x | identical | 241 | 243 |
| `particle_filter` | large | 100,000 particles, 100 steps | 895.4 ms | 804.1 ms | **1.11x** | identical | 342 | 244 |
| `veros_ocean` | small | a grid of 81 × 81 × 11 | 28.7 ms | 28.6 ms | 1.00x | identical | 9 | 1 |
| `veros_ocean` | large | a grid of 204 × 204 × 26 | 491.1 ms | 488.1 ms | 1.01x | identical | 130 | 1 |
| `skimage_pipeline` | small | 4 images of 256 × 256 | 96.6 ms | 94.0 ms | 1.03x | identical | 1,715 | 11,681 |
| `skimage_pipeline` | large | 4 images of 1,024 × 1,024 | 1180.2 ms | 1152.8 ms | 1.02x | identical | 2,939 | 133,794 |
| `pandas_sales` | small | 100,000 sales | 30.8 ms | 29.4 ms | 1.05x | identical | 65 | 117 |
| `pandas_sales` | large | 2,000,000 sales | 477.6 ms | 446.5 ms | **1.07x** | identical | 65 | 117 |
| `astropy_timeseries` | small | 10 light curves of 1,000 points, 4 images of 256 × 256 | 206.2 ms | 205.7 ms | 1.00x | identical | 1,256 | 3,228 |
| `astropy_timeseries` | large | 10 light curves of 5,000 points, 4 images of 1,024 × 1,024 | 1534.9 ms | 1529.3 ms | 1.00x | identical | 4,261 | 9,233 |

`install(inexact=True)` gave the same results as plain NumPy on every row, and times within 2% of `install()`: the particle filter, whose cumulative sum of weights is `float64`, ran at 1.13x in that mode.

Where the time went, as shares of the samples of working threads, plain NumPy:

| Application | Size | Python | NumPy | BLAS | Maths library | Memory and system | Other compiled code | hotkernels, installed |
|---|---|---|---|---|---|---|---|---|
| `particle_filter` | small | 3.1% | 64.7% | 2.2% | 15.7% | 14.2% | 0.0% | 0.8% |
| `particle_filter` | large | 0.1% | 73.8% | 3.3% | 13.4% | 9.4% | 0.0% | 2.5% |
| `veros_ocean` | small | 0.4% | 82.5% | 2.6% | 1.1% | 13.4% | 0.0% | 0.4% |
| `veros_ocean` | large | 0.1% | 81.0% | 2.4% | 1.4% | 15.0% | 0.0% | 6.3% |
| `skimage_pipeline` | small | 18.4% | 32.7% | 0.0% | 11.4% | 10.0% | 27.5% | 0.4% |
| `skimage_pipeline` | large | 9.7% | 34.1% | 0.0% | 12.5% | 6.6% | 37.1% | 0.6% |
| `pandas_sales` | small | 21.8% | 33.6% | 0.0% | 0.3% | 5.6% | 38.7% | 0.7% |
| `pandas_sales` | large | 4.5% | 41.9% | 0.0% | 0.2% | 4.2% | 49.2% | 0.8% |
| `astropy_timeseries` | small | 11.6% | 64.9% | 0.2% | 4.1% | 8.5% | 10.8% | 0.1% |
| `astropy_timeseries` | large | 4.3% | 67.1% | 0.0% | 3.3% | 3.2% | 22.0% | 0.3% |

None of the five is held back by input and output, and only the small pandas and image runs by the Python interpreter. The sections below say where the NumPy share goes and why hotkernels takes little of it.

## `particle_filter`: 1.11x

A bootstrap particle filter tracking a target from range and bearing, written for this study: 100 steps of predict, weigh, estimate and, when the effective sample size falls below half, systematic resampling. The particles are one (particles × 4) array, as textbooks write it. Source: [delivery/apps/particle_filter.py](../delivery/apps/particle_filter.py).

**What a kernel took.** The resampling search, `np.searchsorted(cumulative, positions)` on 100,000 `float64` (42 calls): NumPy's `binsearch<double>` was 11.0% of the run, and that is the gain. Also `np.argmax(weights)` (100), the mask of particles above the speed limit, `particles[too_fast, 2:] *= ...` through `x[mask]` (100) and, on the large size, `log_weights += ...` through the in-place add (100).

**Where the rest is.**

| Share | Function | What it is in the program |
|---|---|---|
| 15.5% + 3.9% + 2.3% | `random_standard_normal`, `pcg64`, `random_standard_normal_fill` | `rng.standard_normal((n, 4))`, the process noise, each step |
| 9.7% + 9.4% + 5.4% | `_contig_to_contig`, `memmove`, `PyArray_TransferNDimToStrided` | copies of the particles' columns, and `particles[idx]` at each resampling |
| 7.3% | `atan2` | `np.arctan2(particles[:, 1], particles[:, 0])`, the predicted bearing |
| 4.6%, 2.8% | `DOUBLE_add`, `DOUBLE_multiply` | the arithmetic of the step |
| 2.5% | `DOUBLE_remainder` | wrapping the bearing error into [-π, π) with `%` |

**Why it does not gain more.** A fifth of the program draws normal random numbers, which no hook reaches: `Generator.standard_normal` is a method of a Cython class in `numpy.random`, and a kernel would have to give the same stream bit for bit ([#223](https://github.com/sofeikov/hotkernels/issues/223)). `np.arctan2` calls the C library's `atan2` per element, at 25 ns per element against 1.9 ns for `np.hypot` ([#221](https://github.com/sofeikov/hotkernels/issues/221)). The resampling gather `particles[idx]` takes 0.79 ms, twice `np.take` and sixteen times a copy, and no kernel serves it ([#224](https://github.com/sofeikov/hotkernels/issues/224)). `x % c` is a scalar loop ([#222](https://github.com/sofeikov/hotkernels/issues/222)). The cumulative sum before the search is `float64`, which only `install(inexact=True)` serves: 1.13x in that mode.

**Left to NumPy.** `np.sum` on `float64` (200, no kernel), `np.cumsum` on `float64` (42, the inexact kernel) and two casts. None is a visible share.

## `veros_ocean`: no change

Three routines of the Veros ocean model as pyhpc-benchmarks publishes them, copied unchanged: the equation of state (a polynomial in salinity, temperature and pressure), isoneutral mixing (slopes and mixing tensors from finite differences), and one step of turbulent kinetic energy (an implicit vertical solve through LAPACK's `dgtsv`, then advection with a superbee limiter). Source: [delivery/apps/vendor/pyhpc/](../delivery/apps/vendor/pyhpc/).

**What a kernel took.** `np.where(c, a, b)` on `float64` (9) and, on the large grid, the in-place adds and subtracts of the tendencies (121). Neither moves the time.

**Where the time is.** On the large grid: `DOUBLE_multiply` 25.6%, `memmove` 13.5%, `DOUBLE_add` 12.1%, `DOUBLE_tanh` 11.1%, `DOUBLE_subtract` 5.2%, `DOUBLE_divide` 4.5%, strided copies about 9%, the LAPACK solve 2.4%.

**Why it does not gain.** This is the program NumPy is best at: whole-array arithmetic on slices of 3-D and 4-D grids, through NumPy's own loops, for which hotkernels has no kernel and none would be bit-identical and much faster. An expression like `0.5 * (1.0 + np.tanh((-np.abs(sx) + iso_slopec) / iso_dslope))` makes five temporaries for one pass of arithmetic, so the program is bound by memory and by the number of passes, not by any one loop. Fusing such expressions is the only thing that would change it, and no hook can see an expression whole: [#191](https://github.com/sofeikov/hotkernels/issues/191) concluded so for `w -= lr * g`, and the evidence from this program was added there. It is the control of the set: a real program that reaches almost no hook, and on which installing costs nothing measurable.

## `skimage_pipeline`: 1.02x to 1.03x

scikit-image 0.26.0 on four noisy, unevenly lit images of blobs: adaptive histogram equalisation, Gaussian smoothing, Otsu's threshold, removal of small objects, labelling, six region properties, Canny edges, Sobel and HOG features.

**What a kernel took.** 2,939 calls on the large size: `x[mask]` on `float64` (1,926), `np.clip` (520), `np.sum` on `uint8` (473), `np.histogram` (4, the Otsu threshold) and `np.argmax`.

**Where the rest is.**

| Share | Function | What it is in the program |
|---|---|---|
| 37.1% in all | SciPy's `NI_Correlate1D`, `NI_LineBufferToArray`, `NI_Correlate`; scikit-image's Cython | `ndimage` filters behind `gaussian`, `sobel` and Canny's smoothing; HOG's histograms; Canny's non-maximum suppression; labelling |
| 9.5% + 1.5% | `atan2`, `fetestexcept` | `np.arctan2(g_row, g_col)` in HOG, on every pixel |
| 6.5% | `mapiter_get` | indexing by integer arrays; CLAHE also gathers each tile's mapping with `np.take_along_axis`, 87 ms of the run under cProfile |
| 9.7% | the Python interpreter | HOG normalises each block in Python: 63,504 calls of `_hog_normalize_block` |
| 3.1% | `DOUBLE_remainder` | `% 180` on HOG's orientations |
| 2.3% | `arr_bincount` | CLAHE's histogram of each tile, `np.bincount` through `np.apply_along_axis`, and `remove_small_objects` |

**Why it does not gain more.** Over a third of the pipeline is compiled code that is not NumPy's: SciPy's `ndimage` and scikit-image's Cython, which no NumPy hook reaches. Of what is NumPy's, the largest calls have no kernel: `np.arctan2` ([#221](https://github.com/sofeikov/hotkernels/issues/221)), the gathers ([#224](https://github.com/sofeikov/hotkernels/issues/224)) and `%` ([#222](https://github.com/sofeikov/hotkernels/issues/222)).

**Left to NumPy.** 133,794 calls, of which 128,017 are `np.sum` on 81-element `float64` blocks from HOG's Python loop. They are about a tenth of the run between them, and the cost is the call, not the sum ([#99](https://github.com/sofeikov/hotkernels/issues/99), where this program's figures were added). The rest are casts to `float32` from types other than `uint8` (2,461), `x[mask] = v` with an array `v` (1,320) and `np.count_nonzero` (1,096), none of them a visible share.

## `pandas_sales`: 1.07x

pandas 3.1.1 on a table of two million sales: filters, derived columns, group-by with several aggregations, a pivot table, daily resampling with a rolling mean, a join with a store table, `nlargest`, `rank`, `pd.qcut`, `pd.cut`, `drop_duplicates`, `value_counts` and quantiles.

**What a kernel took.** 65 calls: `np.nonzero` on masks (27), `x[mask]` (21), `np.sum` on `uint8` (6), `np.clip` on `int64`, `np.bincount` on `int64`, `np.unique` on `int64`, `np.cumsum` on `bool` and one stable `argsort`. That is the whole gain, 31 ms of 478.

**Where the rest is.**

| Share | Function | What it is in the program |
|---|---|---|
| 49.2% in all | `pandas._libs`: `kh_put_int64` 6.5%, `Int64HashTable.unique` 6.2%, `group_sum` 6.8%, `StringHashTable.unique` 3.6%, `Int8HashTable.unique` 2.8%, `kth_smallest_c` 2.6%, `value_count` 2.0% | factorising keys, group-by sums and medians, `nunique`, `value_counts`, `drop_duplicates` |
| 9.0% | `binsearch<double>` | `pd.cut` and `pd.qcut` |
| 6.8% | `introselect_<double>` | `np.quantile` in `pd.qcut` and `Series.quantile` |
| 5.2% + 2.6% + 2.2% | `aquicksort` on `int64` and `datetime64`, `amerge_at` | `argsort(kind="quicksort")` from `sort_values`, `rank`, `nlargest` and group keys |

**Why it does not gain more.** Half the program is pandas' own compiled code, which never calls NumPy ([#226](https://github.com/sofeikov/hotkernels/issues/226)). Of the NumPy half, the three largest calls are left for three different reasons:

- **`pd.cut` and `pd.qcut` pass a pandas `Index` as the needles** of `np.searchsorted`, and the binding never converts an argument, so both calls go to NumPy. Even given an `ndarray`, the kernel is 1.08x faster with six bin edges and two million needles: a binary search of six elements is the wrong algorithm there ([#225](https://github.com/sofeikov/hotkernels/issues/225)).
- **pandas sorts with `kind="quicksort"`,** which hotkernels does not serve ([#207](https://github.com/sofeikov/hotkernels/issues/207); 16 calls, 13% of the run under cProfile, and some of them on `datetime64`, which the sort kernels do not take).
- **Quantiles** select with NumPy's `introselect`, for which there is no kernel ([#220](https://github.com/sofeikov/hotkernels/issues/220)).

**Left to NumPy.** 117 calls: casts to `float32` (60), `x[mask]` on types without a kernel (24), `x[mask] = v` with an array (8), `np.bincount` with weights (6), the two searches, and five stable sorts whose number does not grow with the table, so most likely of the 200 stores or 30 categories, below the 256 elements the sort kernels start at.

## `astropy_timeseries`: no change

Astropy 8.0.3 on a small survey: sigma-clipped and biweight statistics of four images, and for ten light curves sigma clipping, a Lomb–Scargle periodogram by the fast method and its best period, the median absolute deviation, and Bayesian blocks of two.

**What a kernel took.** `np.argmax` on `float64` (4,010, most of them inside Bayesian blocks' loop), `x[mask]` (235), `x[mask] = v`, `np.cumsum` on `bool` and two stable `argsort`.

**Where the time is.**

| Share | Function | What it is in the program |
|---|---|---|
| 17.4% | `introselect_<double>` | `np.median` in `biweight_location` and `biweight_scale`, on each 1,024 × 1,024 image |
| 15.4% + 6.2% | bottleneck's `nanmedian_all_float64`, `nanstd_all_float64` | `sigma_clipped_stats` on each image, once per clipping iteration |
| about 24% | pocketfft (`pass8` 9.5%, `general_nd` 7.8%) and the casts and copies around it | `np.fft.ifft(..., axis=0)` in the fast periodogram's non-uniform FFT |
| 4.2% + 2.5% + 2.7% | `DOUBLE_multiply`, `DOUBLE_subtract`, `CDOUBLE_multiply` | the periodogram's arithmetic |

**Why it does not gain.** A third of the program is a median of a million elements, taken by NumPy's `introselect` or by bottleneck's `nanmedian`, and hotkernels serves `np.median` only up to 128 elements. A median takes 9.3 ms on 1,024 × 1,024 where a mean takes 0.21 ms; a selection by radix on the float key, as the sort kernels order floats, could be several times faster and is exact by construction ([#220](https://github.com/sofeikov/hotkernels/issues/220)). bottleneck is reached through hotkernels' own bottleneck hooks, but only along an axis of a matrix: Astropy reduces the whole image, `axis=None`, so `bn.nanstd` and `bn.nanmean` are left (55 and 4 calls; [#213](https://github.com/sofeikov/hotkernels/issues/213), where this was added), and `nanmedian` is not one of the hooked functions. Nearly a quarter is the FFT, which cannot be faster and bit-identical without reproducing pocketfft ([#227](https://github.com/sofeikov/hotkernels/issues/227)).

**Left to NumPy.** 9,233 calls: `np.cumsum` on `float64` (8,000), in Bayesian blocks' loop and on reversed views, `x[::-1]`, which are not contiguous; `np.bincount` with weights (840), in the periodogram's scattering step; and `np.median` on more than 128 elements (44). Bayesian blocks' loop also asks `inspect.signature` for the fitness function's arguments at every step (`_fitness_args`, 20,006 calls on the large size), which is Astropy's Python and no concern of NumPy's.

## What the five say together

**Why a program does not get faster, in order of how much time it explains here:**

| Reason | Where | Can hotkernels reach it? |
|---|---|---|
| Compiled code that is not NumPy's | 49% of `pandas_sales` (pandas), 37% of `skimage_pipeline` (SciPy, scikit-image), 22% of `astropy_timeseries` (bottleneck) | pandas: perhaps, by hooking its functions as bottleneck's are ([#226](https://github.com/sofeikov/hotkernels/issues/226)). bottleneck: partly, by extending its hooks ([#213](https://github.com/sofeikov/hotkernels/issues/213), [#220](https://github.com/sofeikov/hotkernels/issues/220)). SciPy and scikit-image's Cython: no |
| NumPy's own element-wise loops and copies, bound by memory | about 60% of `veros_ocean` in arithmetic and 20% in copies; about 10% of `particle_filter` | Only by fusion, which no hook can see ([#191](https://github.com/sofeikov/hotkernels/issues/191)) |
| Median and quantiles | 32.8% of `astropy_timeseries`, 6.8% of `pandas_sales` | Yes, with a kernel: [#220](https://github.com/sofeikov/hotkernels/issues/220) |
| Random draws | 22% of `particle_filter` | Only with a hook into `numpy.random` and an exact stream: [#223](https://github.com/sofeikov/hotkernels/issues/223) |
| The FFT | about 24% of `astropy_timeseries` | Not bit-identically: [#227](https://github.com/sofeikov/hotkernels/issues/227) |
| The C library's `atan2` | 9.5% of `skimage_pipeline`, 7.3% of `particle_filter` | Yes, as `hypot` was: [#221](https://github.com/sofeikov/hotkernels/issues/221) |
| Sorting with the default kind | about 10% of `pandas_sales` | Yes, if the stable kernel may answer: [#207](https://github.com/sofeikov/hotkernels/issues/207) |
| A hooked call with an argument it will not convert, or a shape its kernel is slow at | `searchsorted` from `pd.cut`, 9.0% of `pandas_sales` | Yes: [#225](https://github.com/sofeikov/hotkernels/issues/225) |
| Gathering by index | 6.5% of `skimage_pipeline`, about 4% of `particle_filter` | Yes, with a kernel: [#224](https://github.com/sofeikov/hotkernels/issues/224) |
| NumPy's call path on short arrays | about 10% of `skimage_pipeline` | Partly: [#99](https://github.com/sofeikov/hotkernels/issues/99) |
| `x % c` on floats | 3.1% of `skimage_pipeline`, 2.5% of `particle_filter` | Yes: [#222](https://github.com/sofeikov/hotkernels/issues/222) |

**Against the first study.** The first set's largest reasons were sorting, `np.unique` and accumulation along an axis, each of which a kernel now serves; this set's largest are compiled code of other libraries and NumPy's own arithmetic, which no kernel will serve. Two findings repeat. The "Left to NumPy" counts are a poor guide to time: the largest, 128,017 sums in HOG, is a tenth of its program, and the two calls of `searchsorted` that cost 9.0% are counted twice. And the calls worth most are often not hooked at all: the median, `atan2`, the random draws and the FFT appear in no count, only in the profile.

**What this changes about the plan.** With the two gaps a kernel can close and that are worth most, the median ([#220](https://github.com/sofeikov/hotkernels/issues/220)) and `atan2` ([#221](https://github.com/sofeikov/hotkernels/issues/221)), the estimates put `astropy_timeseries` near 1.3x, `skimage_pipeline` and `particle_filter` near 1.1x. These are arithmetic on measured shares, not measurements. pandas-heavy code, the most common way NumPy is used, stays near 1.1x unless hotkernels reaches into pandas itself ([#226](https://github.com/sofeikov/hotkernels/issues/226)).

## Issues filed

| Issue | Gap | Evidence |
|---|---|---|
| [#220](https://github.com/sofeikov/hotkernels/issues/220) | Median and quantiles of large `float64` arrays, `np.median`, `np.quantile`, `bn.nanmedian` | 32.8% of `astropy_timeseries`; 9.3 ms for a median of a million against 0.21 ms for a mean |
| [#221](https://github.com/sofeikov/hotkernels/issues/221) | `np.arctan2` on `float64` | 9.5% of `skimage_pipeline`, 7.3% of `particle_filter`; 25 ns per element |
| [#222](https://github.com/sofeikov/hotkernels/issues/222) | `x % c` on `float64` | 3.1% and 2.5%; 3.7 ns per element against 1.9 ns for `np.fmod` |
| [#223](https://github.com/sofeikov/hotkernels/issues/223) | Random draws through `numpy.random.Generator` | 22% of `particle_filter` |
| [#224](https://github.com/sofeikov/hotkernels/issues/224) | Gathering rows by an integer index | 0.79 ms against 0.05 ms for a copy; 6.5% of `skimage_pipeline` |
| [#225](https://github.com/sofeikov/hotkernels/issues/225) | `np.searchsorted` from `pd.cut`: an `Index` as needles, and six edges | 9.0% of `pandas_sales`; 1.08x even when taken |
| [#226](https://github.com/sofeikov/hotkernels/issues/226) | pandas' own compiled code | 49.2% of `pandas_sales` |
| [#227](https://github.com/sofeikov/hotkernels/issues/227) | The FFT along the first axis | about 24% of `astropy_timeseries` |

New evidence was added to [#207](https://github.com/sofeikov/hotkernels/issues/207) (pandas sorts with the default kind, and on `datetime64`), [#99](https://github.com/sofeikov/hotkernels/issues/99) (HOG's 128,017 short sums), [#213](https://github.com/sofeikov/hotkernels/issues/213) (Astropy reduces whole images with bottleneck) and [#191](https://github.com/sofeikov/hotkernels/issues/191) (the ocean model as a case for fusion).

## Limits

- **A desktop in use.** This desktop hosts the project's CI runners. The record waited until `gh run list` showed no unfinished run and started at a load average of 3.41; the desktop does not go below about 2 ([#51](https://github.com/sofeikov/hotkernels/issues/51)), so the record marks its times rough. A profile run taken earlier, at load averages of 8 to 22, gave the same speedups to within 0.01x and shares within about 1.5 points of the record's; the figures in the issues come from it and from timings of single calls at those loads, and the report quotes the record.
- **One machine, one architecture.** Nothing here ran on x86-64 ([#205](https://github.com/sofeikov/hotkernels/issues/205) covers both sets).
- **Five programs, chosen by field** before any was timed, and not changed by the results. The four libraries are called with typical arguments, not taken from a user's code; one program is written here.
- **Estimates of what a missing kernel is worth** assume the rest of the program is unchanged and take the kernel's speed from a stand-in or from the speed of a neighbouring function, not from a kernel.

## Running it

```bash
uv run --group apps delivery apps --set second --profile                          # all five at both sizes, with profiles
uv run --group apps delivery apps --only pandas_sales --size large                # one at one size, times and counts only
uv run --group apps delivery apps --set second --profile --record                 # write delivery/results/<machine>-apps-second.md
uv run delivery apps-report delivery/results/apple-m1-pro-apps-second.json        # write the report again from its JSON
```
