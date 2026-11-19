# searchsorted_right_f64 attempts (arm64)

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
