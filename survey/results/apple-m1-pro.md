# Survey results: Apple M1 Pro

- NumPy 2.5.5, Python 3.12.7, macOS-27.2.1-arm64-arm-64bit
- Naive C: Apple clang version 21.0.0 (clang-2100.4.1.1), `-O3 -mcpu=native -std=c11 -Wall -Wextra`
- Reference loop drift over the run: 0.6%
- Git revision: `d2dea79`

Each ratio is NumPy time divided by naive C time. Above 1 means NumPy is slower than the naive loop.
`C SIMD` gives the vector width clang's loop vectoriser used for the naive kernel, or `no`. `alloc` marks statements where NumPy allocates
its result; the C side then mallocs and frees its output on every call as well.

| Candidate | NumPy statement | C SIMD | alloc | large-n gap | n=16 | n=4,096 | n=65,536 | n=1,048,576 | n=16,777,216 | C ns/elem at 1M | NumPy ns/elem at 1M | NumPy at n=1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clip_f32 out= | `np.clip(x, lo, hi, out=out)` | w4 |  | **7.16x** | 562x | 11.7x | 8.03x | 7.71x | 5.92x | 0.09 | 0.67 | 1312 ns |
| histogram_f64 64 bins | `np.histogram(x, bins=64, range=(0.0, 1.0))` | no | yes | **7.15x** | 385x | 12.8x | 7.31x | 7.14x | 6.99x | 0.88 | 6.26 | 18379 ns |
| clip_f32 | `np.clip(x, lo, hi)` | w4 | yes | **7.09x** | 56.5x | 10.5x | 7.61x | 7.49x | 6.25x | 0.09 | 0.65 | 1373 ns |
| cumsum_i64 out= | `np.cumsum(x, out=out)` | no |  | **6.54x** | 250x | 7.44x | 6.90x | 6.45x | 6.27x | 0.34 | 2.19 | 830 ns |
| cumsum_i64 | `np.cumsum(x)` | no | yes | **6.51x** | 36.6x | 7.32x | 6.76x | 6.40x | 6.38x | 0.34 | 2.18 | 798 ns |
| isin_i64 m=8 | `np.isin(x, test)` | w16 | yes | **5.65x** | 276x | 4.91x | 5.77x | 7.30x | 4.28x | 1.08 | 7.86 | 11953 ns |
| cast_u8_f32 | `x.astype(np.float32)` | w16 | yes | **5.50x** | 7.62x | 6.32x | 7.21x | 5.83x | 3.95x | 0.06 | 0.32 | 180 ns |
| interp_f64 | `np.interp(v, xp, fp)` | w2 | yes | **4.30x** | 9.68x | 4.30x | 3.38x | 5.33x | 4.40x | 127.22 | 678.55 | 485 ns |
| bincount_i64 256 bins | `np.bincount(x, minlength=256)` | no | yes | **4.15x** | 4.98x | 4.56x | 4.23x | 4.12x | 4.08x | 0.33 | 1.37 | 280 ns |
| cumsum_f64 out= | `np.cumsum(x, out=out)` | no |  | **3.84x** | 198x | 3.97x | 4.08x | 3.74x | 3.71x | 0.97 | 3.63 | 845 ns |
| cumsum_f32 out= | `np.cumsum(x, out=out)` | no |  | **3.72x** | 195x | 3.90x | 3.76x | 3.74x | 3.66x | 0.95 | 3.56 | 853 ns |
| cumsum_f64 | `np.cumsum(x)` | no | yes | **3.70x** | 29.6x | 3.87x | 3.72x | 3.71x | 3.67x | 0.97 | 3.61 | 790 ns |
| cumsum_f32 | `np.cumsum(x)` | no | yes | **3.68x** | 31.8x | 3.90x | 3.71x | 3.65x | 3.68x | 0.97 | 3.53 | 795 ns |
| take_f64 np.take out= | `np.take(a, idx, out=out)` | no |  | **3.59x** | 83.3x | 5.31x | 4.42x | 4.05x | 2.59x | 1.80 | 7.31 | 778 ns |
| clip_f64 out= | `np.clip(x, lo, hi, out=out)` | w2 |  | **3.59x** | 328x | 5.59x | 4.06x | 3.67x | 3.11x | 0.18 | 0.66 | 1155 ns |
| clip_f64 | `np.clip(x, lo, hi)` | w2 | yes | **3.53x** | 53.4x | 5.29x | 4.02x | 3.59x | 3.04x | 0.18 | 0.65 | 1157 ns |
| sum_u8 | `np.sum(x)` | w16 |  | **3.26x** | 351x | 8.10x | 3.50x | 3.14x | 3.14x | 0.06 | 0.19 | 1250 ns |
| take_f64 np.take | `np.take(a, idx)` | no | yes | **3.18x** | 21.0x | 3.85x | 3.64x | 3.58x | 2.48x | 1.68 | 6.01 | 545 ns |
| count_nonzero_i64 | `np.count_nonzero(x)` | w2 |  | **2.91x** | 58.5x | 4.41x | 3.42x | 3.31x | 2.18x | 0.10 | 0.33 | 150 ns |
| count_nonzero_f64 | `np.count_nonzero(x)` | w2 |  | **2.86x** | 53.9x | 2.98x | 2.99x | 3.01x | 2.59x | 0.13 | 0.38 | 141 ns |
| add_at_f64 4096 bins | `np.add.at(acc, idx, v)` | no |  | **2.40x** | 78.9x | 3.18x | 2.43x | 2.41x | 2.35x | 0.42 | 1.01 | 474 ns |
| cast_i64_f64 | `x.astype(np.float64)` | w2 | yes | **1.82x** | 8.43x | 3.08x | 1.93x | 1.89x | 1.64x | 0.18 | 0.34 | 174 ns |
| compress_f32 p=0.05 | `x[mask]` | no | yes | **1.81x** | 7.85x | 1.90x | 2.11x | 1.68x | 1.68x | 0.59 | 0.99 | 84 ns |
| compress_f32 p=0.5 | `x[mask]` | no | yes | **1.76x** | 9.53x | 5.94x | 2.04x | 1.65x | 1.63x | 3.04 | 5.00 | 198 ns |
| compress_f64 p=0.05 | `x[mask]` | no | yes | **1.66x** | 7.96x | 1.38x | 1.92x | 1.55x | 1.54x | 0.58 | 0.90 | 86 ns |
| compress_f64 p=0.5 | `x[mask]` | no | yes | **1.62x** | 9.10x | 5.90x | 1.87x | 1.47x | 1.55x | 3.24 | 4.75 | 196 ns |
| nonzero_bool p=0.05 | `np.nonzero(mask)[0]` | no | yes | **1.48x** | 10.6x | 1.02x | 1.52x | 1.45x | 1.45x | 0.58 | 0.85 | 306 ns |
| sum_bool | `np.sum(x)` | w16 |  | **1.39x** | 322x | 3.32x | 1.48x | 1.35x | 1.33x | 0.16 | 0.21 | 1218 ns |
| cummin_f64 | `np.minimum.accumulate(x)` | no | yes | **1.38x** | 9.09x | 1.41x | 1.37x | 1.38x | 1.39x | 2.59 | 3.57 | 249 ns |
| put_f64 a[idx]=v | `a[idx] = v` | no |  | **1.37x** | 8.96x | 1.66x | 1.57x | 1.50x | 1.10x | 1.03 | 1.54 | 62 ns |
| searchsorted_f64 | `np.searchsorted(a, v)` | no | yes | **1.28x** | 8.94x | 4.69x | 2.73x | 1.69x | 0.46x | 84.03 | 142.18 | 529 ns |
| maskfill_f64 x[mask]=0 | `x[mask] = 0.0` | w16 |  | **1.18x** | 43.3x | 3.98x | 1.28x | 1.13x | 1.12x | 3.40 | 3.85 | 248 ns |
| array_equal_f64 (equal) | `np.array_equal(x, y)` | no |  | **1.14x** | 110x | 1.80x | 1.19x | 1.10x | 1.12x | 0.35 | 0.38 | 760 ns |
| dot_i64 | `np.dot(x, y)` | no |  | **1.12x** | 83.4x | 1.41x | 1.14x | 1.10x | 1.12x | 0.34 | 0.37 | 424 ns |
| clip_i64 out= | `np.clip(x, lo, hi, out=out)` | w2 |  | **1.09x** | 586x | 4.19x | 1.29x | 1.01x | 1.00x | 0.18 | 0.18 | 2076 ns |
| copy_f64 [::2] out= | `np.copyto(out, xs[::2])` | w2 |  | **1.09x** | 29.6x | 1.21x | 1.16x | 1.03x | 1.09x | 0.40 | 0.42 | 211 ns |
| take_f64 a[idx] | `a[idx]` | no | yes | **1.07x** | 3.89x | 1.28x | 1.19x | 1.03x | 1.01x | 1.64 | 1.68 | 97 ns |
| where_f64 | `np.where(c, a, b)` | no | yes | **1.06x** | 16.2x | 1.65x | 1.06x | 1.07x | 1.05x | 0.45 | 0.48 | 397 ns |
| add_f64 [::2] | `xs[::2] + ys[::2]` | w2 | yes | **1.04x** | 14.1x | 1.61x | 1.08x | 1.06x | 1.00x | 0.62 | 0.66 | 388 ns |
| diff_f64 (control) | `np.diff(x)` | w2 | yes | **1.04x** | 46.3x | 3.00x | 1.10x | 1.02x | 1.00x | 0.17 | 0.17 | 1038 ns |
| add_f64 [::2] out= | `np.add(xs[::2], ys[::2], out=out)` | w2 |  | **1.03x** | 53.2x | 1.71x | 1.07x | 1.02x | 1.00x | 0.59 | 0.60 | 371 ns |
| copy_f64 [::-1] out= | `np.copyto(out, x[::-1])` | w2 |  | **1.03x** | 30.5x | 1.19x | 1.03x | 1.02x | 1.03x | 0.34 | 0.35 | 222 ns |
| clip_i64 | `np.clip(x, lo, hi)` | w2 | yes | **1.02x** | 77.8x | 3.83x | 1.15x | 0.93x | 1.01x | 0.19 | 0.18 | 2098 ns |
| cast_f64_f32 | `x.astype(np.float32)` | w4 | yes | **1.02x** | 7.72x | 1.70x | 1.04x | 1.00x | 1.01x | 0.14 | 0.14 | 183 ns |
| argmax_f64 [::2] | `np.argmax(xs[::2])` | no |  | **0.84x** | 76.5x | 0.99x | 0.82x | 0.85x | 0.85x | 1.32 | 1.12 | 645 ns |
| correlate_f64 k=8 | `np.correlate(x, k, 'valid')` | w2 | yes | **0.71x** | 8.11x | 0.74x | 0.71x | 0.72x | 0.71x | 1.46 | 1.05 | 331 ns |
| argmin_f64 | `np.argmin(x)` | no |  | **0.54x** | 55.1x | 0.64x | 0.55x | 0.54x | 0.54x | 1.30 | 0.70 | 643 ns |
| argmax_f64 | `np.argmax(x)` | no |  | **0.54x** | 54.8x | 0.64x | 0.54x | 0.55x | 0.54x | 1.29 | 0.71 | 543 ns |
| searchsorted_i64 | `np.searchsorted(a, v)` | no | yes | **0.36x** | 8.33x | 0.70x | 0.42x | 0.41x | 0.26x | 95.68 | 39.48 | 652 ns |
| argmax_f32 | `np.argmax(x)` | no |  | **0.33x** | 55.1x | 0.44x | 0.34x | 0.33x | 0.34x | 1.29 | 0.43 | 555 ns |
| sum_f64 [::2] | `np.sum(xs[::2])` | w2 |  | **0.29x** | 127x | 0.42x | 0.26x | 0.31x | 0.31x | 0.96 | 0.29 | 952 ns |
| argmax_i64 | `np.argmax(x)` | no |  | **0.25x** | 68.5x | 0.43x | 0.25x | 0.24x | 0.26x | 0.70 | 0.17 | 544 ns |
| nonzero_bool p=0.5 | `np.nonzero(mask)[0]` | no | yes | **0.25x** | 11.2x | 1.88x | 0.29x | 0.23x | 0.22x | 3.11 | 0.71 | 313 ns |
| count_nonzero_bool | `np.count_nonzero(x)` | w16 |  | **0.23x** | 39.2x | 0.50x | 0.24x | 0.22x | 0.23x | 0.16 | 0.03 | 149 ns |
| sum_f64 (control) | `np.sum(x)` | w2 |  | **0.20x** | 339x | 0.42x | 0.21x | 0.20x | 0.20x | 1.01 | 0.20 | 869 ns |
| sum_f32 (control) | `np.sum(x)` | w4 |  | **0.18x** | 300x | 0.41x | 0.20x | 0.18x | 0.18x | 0.96 | 0.17 | 844 ns |
| packbits | `np.packbits(x)` | no | yes | **0.09x** | 10.7x | 0.25x | 0.10x | 0.08x | 0.08x | 0.38 | 0.03 | 240 ns |
| all_bool (all True) | `np.all(x)` | no |  | **0.06x** | 101x | 0.62x | 0.08x | 0.04x | 0.05x | 0.38 | 0.02 | 710 ns |
| max_f32 (control) | `np.max(x)` | no |  | **0.05x** | 81.0x | 0.19x | 0.05x | 0.04x | 0.06x | 1.28 | 0.05 | 804 ns |
| any_bool (all False) | `np.any(x)` | no |  | **0.05x** | 103x | 0.60x | 0.06x | 0.03x | 0.05x | 0.38 | 0.01 | 720 ns |
| argmax_bool (find first) | `np.argmax(x)` | no |  | **0.02x** | 66.5x | 0.16x | 0.02x | 0.01x | 0.02x | 0.97 | 0.01 | 540 ns |
