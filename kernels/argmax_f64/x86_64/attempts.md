# argmax_f64 attempts (x86_64)

All runs on an Intel Core i5-14600KF, Linux, locked, pinned to one performance core by the judge. This kernel is generated from the same source as `argmax_f32`, whose log has the reasoning, the candidates and what differs from ARM64 for a caller. Here: `vmaxpd`, 32 elements per block, four per vector.

## 1. Block reduction with one rare branch, then the position from the remembered block

- **Idea:** as `argmax_f32`, x86-64 attempt 1.
- **check:** pass, first time
- **compare:** not run (first version)
- **Kept:** yes
- **Notes:** none beyond that log's.

## 2. An answer inside the array whatever happens to the data

- **Idea:** as `argmax_f32`, x86-64 attempt 2.
- **check:** pass
- **compare:** not run
- **Kept:** yes
- **Notes:** none.

## Where it stands

Final `judge run`:

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 5.5 ns | 10.1 ns | 1.83x [1.80 – 1.88] | 522.3 ns | 94.4x [92.8 – 97.2] |
| 64 | 10.8 ns | 39.1 ns | 3.62x [3.62 – 3.62] | 552.4 ns | 51.1x [51.0 – 51.1] |
| 256 | 23.2 ns | 188.0 ns | 8.10x [8.10 – 8.11] | 573.8 ns | 24.7x [24.7 – 24.8] |
| 1,024 | 72.9 ns | 778.8 ns | 10.7x [10.7 – 10.7] | 697.8 ns | 9.57x [9.57 – 9.58] |
| 4,096 | 239.5 ns | 3.1 us | 13.1x [13.1 – 13.2] | 1.2 us | 5.05x [5.03 – 5.07] |
| 16,384 | 966.9 ns | 12.6 us | 13.0x [13.0 – 13.0] | 3.2 us | 3.30x [3.29 – 3.30] |
| 65,536 | 4.2 us | 50.4 us | 11.9x [11.9 – 11.9] | 11.3 us | 2.68x [2.67 – 2.68] |
| 262,144 | 19.7 us | 201.9 us | 10.3x [10.2 – 10.3] | 43.8 us | 2.22x [2.22 – 2.23] |
| 1,048,576 | 91.4 us | 808.0 us | 8.84x [8.80 – 8.88] | 171.1 us | 1.87x [1.86 – 1.89] |
| 4,194,304 | 853.7 us | 3.34 ms | 3.92x [3.88 – 3.94] | 1.19 ms | 1.40x [1.38 – 1.41] |
| 16,777,216 | 4.20 ms | 13.34 ms | 3.18x [3.16 – 3.19] | 5.39 ms | 1.28x [1.28 – 1.29] |
| overall |  |  | 6.75x [6.74 – 6.77] |  | 6.03x [6.02 – 6.04] |
