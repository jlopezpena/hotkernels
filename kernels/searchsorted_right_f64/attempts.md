# searchsorted_right_f64 attempts (arm64)

This log holds two architectures. ARM64 (Apple M1 Pro) comes first. x86-64 (Intel Core i5-14600KF) follows under [x86-64](#x86-64), with its own numbering.

The right side of the search: `np.searchsorted(a, v, side='right')`. The design
is [docs/searchsorted-right-design.md](../../docs/searchsorted-right-design.md).

All judge commands in this log were run with `--unlocked` (the spec is not on
`main` yet), on a machine in use (load average about 4).

This kernel is `searchsorted_f64` with the comparison turned round. Read
[its log](../searchsorted_f64/attempts.md) before trying anything here: every
idea in it was tried on the same loads and the same dependency chain. An
improvement that is kept in one of the two kernels is worth trying on the other.

## 1. First version: the left kernel with `a[i] <= key`

- **Idea:** take `searchsorted_f64` as it stands after its attempt 11 and change
  what decides a step. Every `fcmgt key, a[...]` (`a[i] < key`) becomes
  `fcmge key, a[...]` (`a[i] <= key`), 213 instructions in the four paths. The
  one-at-a-time search steps right on `b.ls` instead of `b.mi`. A NaN needle
  goes after every element, so both places that searched for the first NaN in
  `a` (`Lnan_search`, `Lnanfix`) store `n` instead.
- **check:** pass, 141,872 cases
- **compare:** not applicable (first version)
- **Kept:** yes
- **Notes:** nothing else differs: the paths, the group widths, the register
  use and every load address are the left kernel's. `fcmge` is false when
  either operand is NaN, as `fcmgt` is, so a NaN element stops a search as it
  does on the left, and a NaN needle leaves its base inside the array until
  the fix-up overwrites the result.

## What the judge does and does not check here

The two sides differ only for a needle that equals an element. The judge
produces such needles up to 5,000 elements (pairings of the `ties`, `equal`
and `zeros` classes up to 4,096, and the fresh random cases), and never above.

A mutation run put each part of the kernel back to the left side's comparison,
one at a time:

| Changed back | Caught by the judge |
|---|---|
| Sixteen-needle path: first step, `Lblock_4`, `Lblock_3`, `Llast` | Yes, each (`a=equal v=equal`, n = 16, 17 or 32) |
| 64-needle path: first step, first pass, later passes | Yes, each (n = 64 or 65) |
| One-at-a-time search (`b.ls` to `b.mi`) | Yes (n = 1) |
| NaN needle stored as 0 in `Lnan_search`, and in `Lnanfix` | Yes, each (`v=nan_all`) |
| Long-array path (n >= 1,048,576): first step (`Lbig_init`) | **No** |
| Long-array path: the probes (`Lbig_probe`) | **No** |

The two that pass the judge are caught by `uv run delivery check`, which
searches arrays of 1,048,576 and 1,048,579 elements for needles drawn from the
array, and for an array of one repeated value. The first step only shows a
wrong comparison when elements equal to the needle lie on both sides of
`a[p - 1]`, which is what the repeated value is for. The same two mutations of
`searchsorted_f64`, in the other direction, are caught by the same cases.

### The two again, after the judge drew needles from the array

[Issue 48](https://github.com/sofeikov/kernel-arena/issues/48) gave the judge a
value class drawn from another argument: the spec now declares
`In("v", ..., draws_from="a")`, and the judge searches, at every size and in
every scalar set, for needles drawn from `a`, with `a` holding repeated values
(`ties`) and, above 65,536 elements, random. The two mutations above were made
again on a copy of the kernel and checked with `uv run judge check
searchsorted_right_f64 --unlocked`, once with the spec on `main` and once with
the new one. The kernel was restored after each.

| Changed back | Spec on `main` | With needles drawn from `a` |
|---|---|---|
| `Lbig_init` (`fcmge` to `fcmgt`, two instructions) | Passes, 141,872 cases | **Fails**: `n=1048577 arr=mid1 seed=1 scalars=k128 a=ties v=from_a`, `out[2]` 1,048,576 for 1,048,577 |
| `Lbig_probe` (`fcmge` to `fcmgt`, two instructions) | Passes, 141,872 cases | **Fails**: the same case |

Both are found at the first long size in a set with 128 needles or more, which
is the first case that reaches `Lbig` with a needle equal to `a[p - 1]` and
equal elements on both sides of it. The unmutated kernel passes 145,599 cases.

## Numbers

`uv run judge run searchsorted_right_f64 --unlocked` (check: pass, 141,872
cases), Apple M1 Pro, NumPy 2.5.5, 30 samples per point, load average 3.8.
Indicative only.

64 needles per call:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 90.5 ns | 1.3 us | 14.5x [14.26 – 14.81] | 1.7 us | 18.7x [17.97 – 19.38] |
| 64 | 93.8 ns | 2.2 us | 23.7x [23.48 – 24.11] | 2.6 us | 27.7x [27.44 – 28.15] |
| 256 | 129.8 ns | 2.2 us | 17.0x [16.34 – 17.81] | 1.4 us | 10.9x [10.17 – 12.17] |
| 1,024 | 168.7 ns | 692.7 ns | 4.11x [4.03 – 4.25] | 1.3 us | 7.44x [7.38 – 7.47] |
| 4,096 | 210.7 ns | 924.7 ns | 4.39x [4.36 – 4.41] | 1.4 us | 6.43x [6.38 – 6.47] |
| 16,384 | 269.4 ns | 1.2 us | 4.28x [4.26 – 4.31] | 1.5 us | 5.48x [5.43 – 5.51] |
| 65,536 | 477.9 ns | 1.9 us | 3.92x [3.86 – 4.00] | 1.7 us | 3.47x [3.42 – 3.51] |
| 262,144 | 583.6 ns | 2.3 us | 3.93x [3.86 – 4.00] | 1.7 us | 2.99x [2.93 – 3.04] |
| 1,048,576 | 804.7 ns | 3.3 us | 4.07x [3.97 – 4.18] | 1.9 us | 2.41x [2.35 – 2.46] |
| 4,194,304 | 1.2 us | 5.1 us | 4.30x [4.25 – 4.38] | 2.6 us | 2.14x [2.13 – 2.15] |
| 16,777,216 | 1.5 us | 7.8 us | 5.10x [5.02 – 5.21] | 3.2 us | 2.09x [2.05 – 2.13] |
| overall | | | 6.30x [6.27 – 6.35] | | 5.59x [5.54 – 5.65] |

1,000 needles per call (`scalars=many`):

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.5 us | 19.4 us | 13.4x [13.17 – 13.52] | 17.2 us | 11.9x [11.36 – 12.19] |
| 64 | 1.5 us | 30.7 us | 21.1x [20.90 – 21.38] | 29.3 us | 20.1x [19.83 – 20.59] |
| 256 | 2.0 us | 44.2 us | 21.9x [21.58 – 22.34] | 41.9 us | 20.7x [20.30 – 20.88] |
| 1,024 | 2.7 us | 66.4 us | 24.7x [24.44 – 25.06] | 54.2 us | 20.2x [19.95 – 20.43] |
| 4,096 | 3.4 us | 68.4 us | 20.2x [19.79 – 20.75] | 44.2 us | 13.0x [11.53 – 13.68] |
| 16,384 | 4.4 us | 86.3 us | 19.6x [19.25 – 19.87] | 58.2 us | 13.2x [12.51 – 13.95] |
| 65,536 | 8.5 us | 127.8 us | 15.0x [14.42 – 15.30] | 71.3 us | 8.35x [8.00 – 8.67] |
| 262,144 | 10.9 us | 157.2 us | 14.4x [13.80 – 14.60] | 85.3 us | 7.83x [7.49 – 8.01] |
| 1,048,576 | 14.9 us | 198.4 us | 13.3x [13.01 – 13.82] | 108.5 us | 7.29x [7.02 – 7.47] |
| 4,194,304 | 26.1 us | 368.5 us | 14.1x [13.69 – 14.30] | 128.8 us | 4.93x [4.88 – 5.04] |
| 16,777,216 | 34.8 us | 557.0 us | 16.0x [15.63 – 16.38] | 147.9 us | 4.26x [4.18 – 4.32] |
| overall | | | 17.2x [17.07 – 17.30] | | 10.5x [10.33 – 10.60] |

NumPy is timed from Python and includes its per-call overhead.

With 64 needles the naive C is three times faster from 1,024 elements up than
it is at 256. clang compiles this reference with a branch per probe (`b.mi`
after the `fcmp`), where it compiles the left side's without one (`csel`). The
likely reason for the jump is that the judge cycles through fewer inputs as the
size grows, one from 4,096 elements, and 64 searches are few enough for the
branch predictor to learn; that was not measured. The kernel has no branch per
probe; from Python the two sides take the same time (`delivery time`: 2.26 ms
and 2.27 ms for a million needles against 1,000 elements). The figures for 1,000 needles are
the ones to read against naive C.

## Not tried

Nothing beyond the first version. The ideas listed as untried at the end of the
left kernel's log apply here unchanged.

## Ten-run means, locked

`uv run judge run searchsorted_right_f64`, locked, ten times, one run at a time, on a machine that was
not fully quiet (one-minute load average between 2.2 and 9.1 when a run started,
about 3 for most; a run the judge flagged as noisy was repeated). Check: pass,
141,872 cases. Mean of the ten, with the lowest and highest run in brackets. These
replace the indicative and single-pass figures above wherever they are quoted.

| Case | vs naive C, all sizes | at 1,048,576 | vs NumPy, all sizes | at 1,048,576 |
|---|---|---|---|---|
| 64 needles | 6.16x [6.03 – 6.28] | 4.00x [3.25 – 4.29] | 5.37x [5.25 – 5.53] | 2.37x [1.98 – 2.68] |
| 1,000 needles | 16.9x [16.8 – 17.1] | 12.8x [11.8 – 13.3] | 10.1x [9.84 – 10.2] | 6.89x [6.33 – 7.19] |

# x86-64

All runs on an Intel Core i5-14600KF, Linux, locked, pinned to one performance core by the judge. This kernel is `searchsorted_f64` as it stood after its x86-64 attempt 3, with two changes; its log has the measurements that led there.

## x86-64 1. The left kernel with `<=` for `<`, and n for a NaN needle

- **Idea:** side='right' advances when `a[lo + half - 1] <= needle`: `vcmplepd` in the vector paths and `cmovae` in the path for one to three needles. A NaN in `a` still compares false and is never passed, which is right for every needle that is not a NaN. A NaN needle goes after everything, so its answer is n and needs no search: when a NaN needle was seen, n is written over the answers of the NaN needles.
- **check:** pass
- **compare:** not run (first version)
- **Kept:** yes
- **Notes:** the two kernels are generated from one source and differ in the compare, the conditional move and the fix-up for NaN needles.

## x86-64: where it stands

Final `judge run`:

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 52.5 ns | 483.6 ns | 9.21x [9.00 – 9.94] | 1.3 us | 24.9x [24.8 – 25.8] |
| 64 | 74.4 ns | 1.1 us | 15.4x [15.4 – 15.5] | 2.0 us | 26.2x [26.1 – 26.3] |
| 256 | 98.1 ns | 358.3 ns | 3.65x [3.65 – 3.65] | 1.7 us | 17.4x [17.3 – 17.5] |
| 1,024 | 113.4 ns | 408.6 ns | 3.60x [3.60 – 3.61] | 1.0 us | 9.26x [9.24 – 9.27] |
| 4,096 | 134.3 ns | 493.2 ns | 3.67x [3.67 – 3.67] | 1.1 us | 8.21x [8.20 – 8.23] |
| 16,384 | 183.8 ns | 640.6 ns | 3.49x [3.48 – 3.49] | 1.2 us | 6.67x [6.66 – 6.68] |
| 65,536 | 248.6 ns | 840.4 ns | 3.38x [3.38 – 3.38] | 1.4 us | 5.59x [5.59 – 5.60] |
| 262,144 | 329.9 ns | 1.1 us | 3.30x [3.30 – 3.30] | 1.5 us | 4.58x [4.58 – 4.59] |
| 1,048,576 | 588.9 ns | 1.2 us | 2.03x [2.02 – 2.03] | 1.8 us | 2.98x [2.97 – 2.99] |
| 4,194,304 | 1.5 us | 1.9 us | 1.27x [1.27 – 1.27] | 2.7 us | 1.78x [1.77 – 1.78] |
| 16,777,216 | 2.8 us | 3.5 us | 1.25x [1.23 – 1.26] | 4.2 us | 1.51x [1.50 – 1.52] |
| overall |  |  | 3.46x [3.45 – 3.49] |  | 6.68x [6.67 – 6.70] |

`random, scalars=many`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 795.4 ns | 5.0 us | 6.35x [6.20 – 6.46] | 11.9 us | 14.9x [14.9 – 15.0] |
| 64 | 1.1 us | 15.8 us | 13.7x [13.7 – 13.8] | 21.6 us | 18.8x [18.7 – 18.9] |
| 256 | 1.5 us | 23.5 us | 15.8x [15.7 – 15.9] | 29.5 us | 19.8x [19.7 – 19.8] |
| 1,024 | 1.8 us | 33.1 us | 18.1x [18.0 – 18.1] | 38.2 us | 20.9x [20.9 – 20.9] |
| 4,096 | 2.1 us | 8.9 us | 4.20x [4.17 – 4.24] | 36.3 us | 17.1x [17.1 – 17.2] |
| 16,384 | 3.3 us | 20.5 us | 6.27x [6.09 – 6.37] | 49.2 us | 15.0x [15.0 – 15.1] |
| 65,536 | 4.6 us | 40.6 us | 8.87x [8.81 – 8.95] | 60.6 us | 13.3x [13.2 – 13.3] |
| 262,144 | 7.1 us | 56.4 us | 7.91x [7.83 – 7.95] | 72.6 us | 10.2x [10.2 – 10.2] |
| 1,048,576 | 12.6 us | 119.0 us | 9.46x [9.39 – 9.51] | 82.5 us | 6.56x [6.54 – 6.57] |
| 4,194,304 | 31.5 us | 214.2 us | 6.81x [6.72 – 6.91] | 103.6 us | 3.29x [3.27 – 3.31] |
| 16,777,216 | 55.4 us | 426.7 us | 7.70x [7.55 – 7.78] | 142.4 us | 2.57x [2.52 – 2.60] |
| overall |  |  | 8.74x [8.70 – 8.77] |  | 10.8x [10.7 – 10.8] |

All distributions: 5.50x [5.49 – 5.52] over naive C, 8.48x [8.47 – 8.49] over NumPy

With the output buffer 16 bytes behind its input (64 needles; "apart" is the same kernel with the output clear of the input):

`random`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 134.2 ns | 134.1 ns | 1.00x [1.00 – 1.00] | 493.1 ns | 3.67x [3.67 – 3.67] |
| 16,384 | 183.7 ns | 184.3 ns | 1.00x [1.00 – 1.00] | 641.7 ns | 3.49x [3.49 – 3.49] |
| 65,536 | 253.7 ns | 258.8 ns | 1.02x [1.02 – 1.02] | 837.4 ns | 3.30x [3.30 – 3.30] |
| 262,144 | 326.6 ns | 327.5 ns | 1.00x [1.00 – 1.00] | 1.1 us | 3.33x [3.33 – 3.33] |
| 1,048,576 | 447.1 ns | 622.6 ns | 1.39x [1.39 – 1.39] | 1.2 us | 2.71x [2.71 – 2.71] |
| 4,194,304 | 1.5 us | 1.5 us | 0.96x [0.95 – 0.96] | 2.0 us | 1.30x [1.29 – 1.31] |
| 16,777,216 | 2.8 us | 2.7 us | 0.94x [0.94 – 0.95] | 3.4 us | 1.19x [1.18 – 1.20] |
| overall |  |  | 1.04x [1.04 – 1.04] |  | 2.49x [2.48 – 2.49] |

Everything in the `searchsorted_f64` log under "where it stands" applies: what holds it, the candidates, and what differs from the ARM64 kernel for a caller. `vgatherqpd` is used in every path from four needles up and is untested on other x86-64 processors.
