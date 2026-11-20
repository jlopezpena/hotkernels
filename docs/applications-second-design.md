# Applications, second set: five more NumPy programs, with and without hotkernels

This is a proposal for review. It asks the question of [applications-design.md](applications-design.md) of five more programs: does `hotkernels.install()` make the whole program faster, and if not, where does its time go and what would change that.

## Summary

- **The first five are answered.** After the gaps they found were served, four of them run 1.27x to 3.5x faster ([applications-findings.md](applications-findings.md)). They were chosen before any kernel was written for them; the kernels that made them faster were then written *for* them. Five programs that no kernel has yet been tuned against say whether that generalises.
- **The first study names its own blind spots:** "Dense linear algebra, image processing and pandas-heavy code are represented by one program or by none." The second set is chosen to fill them, and to add two fields the first did not touch: a physical simulation and astronomy.
- **Four are other people's code, one is written here.** They are chosen by field, before any of them is timed, and the list is not changed by the results.
- **The method and the tool are the first study's,** unchanged: alternating runs both ways with the results compared bit for bit, `report()`'s counts, a native profile by function, and `cProfile` to tie functions to lines. The applications are added to `delivery apps` as a second set, recorded apart from the first so the first record stays as it is.
- **The output is a report,** [applications-second-findings.md](applications-second-findings.md), one section per application, saying what was gained or why nothing was, and an issue for each gap found that is not already filed.

## The applications

| Name | What it is | Source | Why it was chosen |
|---|---|---|---|
| `particle_filter` | A bootstrap particle filter tracking a moving target from noisy range and bearing measurements, with systematic resampling | Written for this study | The commonest kind of ad hoc numerical algorithm after a training loop: random draws, whole-array arithmetic, a normalised weight vector, and resampling by a cumulative sum and a search |
| `veros_ocean` | Three routines of the Veros ocean model: the equation of state, isoneutral mixing and the turbulent kinetic energy step | [pyhpc-benchmarks](https://github.com/dionhaefner/pyhpc-benchmarks) at `d438ef8`, Unlicense, copied | Real simulation code, written by its authors for NumPy and benchmarked as such. Arithmetic on 3-D and 4-D grids through slices, `np.where`, `np.maximum`, a tridiagonal solve from LAPACK. The field the first set lacks most |
| `skimage_pipeline` | Segmenting and measuring objects in noisy images: smoothing, Otsu threshold, Canny edges, labelling, region properties, adaptive histogram equalisation, HOG features | [scikit-image](https://scikit-image.org), called as a library | Image processing. scikit-image is part NumPy, part Cython; a profile says which part a pipeline spends its time in |
| `pandas_sales` | Analysis of a table of sales: filters, group-by aggregations, a pivot table, daily resampling and rolling means, a join, binning, ranking, de-duplication, quantiles | [pandas](https://pandas.pydata.org), called as a library | pandas-heavy code, the most common way NumPy is reached by people who never import it. pandas calls its own compiled code for much of this; the profile says how much |
| `astropy_timeseries` | Light curves and images of an astronomical survey: sigma-clipped statistics, robust (biweight) statistics, a Lomb–Scargle periodogram and Bayesian blocks | [Astropy](https://www.astropy.org) `astropy.stats` and `astropy.timeseries`, called as a library | Astronomy, a field where NumPy is the language. These functions are written in NumPy, with a fast periodogram built on FFTs and `np.add.at` |

As in the first set, a library is called through its documented functions with typical arguments, and copied code is copied unchanged with its licence: the wrapper may supply input, copy inputs a routine changes in place, and bound a run; it may not change a NumPy call. `veros_ocean` is three files of pyhpc-benchmarks' `benchmarks/` directory, the NumPy implementation of each benchmark and its input generator, under `delivery/apps/vendor/pyhpc/`.

**Sizes.** Each application has the two sizes of the first set, `small` and `large`, relative to the program and set so that one run takes 10 ms to 2 s on plain NumPy:

| Application | Small | Large |
|---|---|---|
| `particle_filter` | 2,000 particles, 100 steps | 100,000 particles, 100 steps |
| `veros_ocean` | a grid of about 65,000 points | a grid of about 1,000,000 points |
| `skimage_pipeline` | 4 images of 256 × 256 | 4 images of 1,024 × 1,024 |
| `pandas_sales` | 100,000 sales | 2,000,000 sales |
| `astropy_timeseries` | 10 light curves of 1,000 points, images of 256 × 256 | 10 light curves of 5,000 points, images of 1,024 × 1,024 |

The sizes are fixed in the building and recorded; the table gives them as built (see [What changed in the building](#what-changed-in-the-building)).

**What is compared.** A pandas object or an Astropy table is reduced to NumPy arrays before it is returned, so that `delivery.check.describe` compares their bytes and not a `repr`.

## The second set in `delivery apps`

`delivery.apps` gains a second tuple of names, and the command an option that picks the set:

```
uv run --group apps delivery apps --set second [--only NAME] [--size small|large] [--profile] [--record [--allow-busy]]
```

- **`--set first`,** the default, is the five of the first study and writes `delivery/results/<machine>-apps.md` and `.json`, as now. Its record is not retaken here.
- **`--set second`** is the five above and writes `<machine>-apps-second.md` and `.json`. The record names its set, and `delivery apps-report` reads it to title the report.
- **`--only`** takes any application of either set.
- The `apps` dependency group gains scikit-image and Astropy, pinned by `uv.lock`. pandas and SciPy are already there.

## What the results are used for

- **The report.** For each application and size: the two times, whether the results agreed, the counts, the profile, and a paragraph that says why it gained or did not.
- **Issues.** One per gap not already filed: a hooked call left to NumPy with the reason, and a NumPy function, library or call path with a large share of a program's time and no kernel. Where a gap is already an issue, the report links it and adds the new evidence there as a comment.
- **A comparison with the first set:** whether the reasons a program does not gain, and their order, are the same in five more programs.

## Documents that change

| Document | Change |
|---|---|
| `docs/applications-second-findings.md` | New: the report |
| `docs/applications-findings.md` | A link to the second set |
| `delivery/apps/vendor/README.md` | The copied pyhpc-benchmarks files |
| `README.md` | `--set second` in "Using the delivery tool" |

No release note: nothing in `hotkernels/` changes.

## Order of work

1. This document.
2. The five applications, the `--set` option and the dependencies.
3. A recorded run on a machine as quiet as this desktop gets, and the report.
4. The issues.

Gaps the report finds are not served in this work. Each is an issue, to be designed and built as the first set's were.

## Out of scope

- Changing the binding or a kernel so that an application gains.
- x86-64 hardware: [#205](https://github.com/sofeikov/hotkernels/issues/205) covers both sets.
- Retaking the first set's record.

## Decisions still open

1. **Whether the two sets merge into one list** of ten once both are recorded, with one record per machine.

## What changed in the building

- **Two large sizes were made smaller.** The proposal had 200,000 particles and light curves of 20,000 points with 2,048 × 2,048 images. On plain NumPy they took 2.0 s and 10.6 s, over the 2 s a run may take: the fast Lomb–Scargle periodogram's frequency grid grows with the number of points, and 20 of them took 8.4 s alone. The large sizes are 100,000 particles, and 10 light curves of 5,000 points with images of 1,024 × 1,024; both take 0.9 to 1.6 s. The small sizes are as proposed, except that the Astropy program has 10 light curves, not 20, at both sizes.
- **Bayesian blocks run on two light curves, on the first 500 or 2,000 points of each.** Their cost grows as the square of the points, and on every curve they would be most of the program.
- **Astropy uses bottleneck,** which the `apps` group already installs for empyrical: its sigma clipping calls bottleneck's `nanmedian`, `nanstd` and `nanmean`, and the program measures it so, as a user with both installed would run it.
- **`--only` sets the run's set** from the application it names, so `--only pandas_sales --record` writes the second set's record, whatever `--set` says.
- **Tests** of the sets, of each set's record name and of the particle filter's resampling are in `tests/test_apps.py`.
