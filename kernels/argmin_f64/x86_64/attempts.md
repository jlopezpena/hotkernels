# argmin_f64 attempts (x86_64)

All runs on an Intel Core i5-14600KF, Linux, locked, pinned to one performance core by the judge. This kernel is generated from the same source as `argmax_f32`, whose log has the reasoning, the candidates and what differs from ARM64 for a caller. Here: `vminpd` and "M is larger than a lane", 32 elements per block, four per vector.

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
| 16 | 5.6 ns | 10.2 ns | 1.82x [1.82 – 1.83] | 526.4 ns | 94.4x [94.2 – 94.8] |
| 64 | 10.2 ns | 39.1 ns | 3.84x [3.83 – 3.84] | 563.8 ns | 55.3x [55.2 – 55.4] |
| 256 | 24.0 ns | 188.0 ns | 7.84x [7.83 – 7.84] | 570.0 ns | 23.8x [23.7 – 23.8] |
| 1,024 | 69.0 ns | 778.7 ns | 11.3x [11.3 – 11.3] | 702.7 ns | 10.2x [10.2 – 10.2] |
| 4,096 | 249.2 ns | 3.1 us | 12.6x [12.6 – 12.6] | 1.2 us | 4.85x [4.85 – 4.85] |
| 16,384 | 967.4 ns | 12.6 us | 13.0x [13.0 – 13.0] | 3.2 us | 3.29x [3.29 – 3.29] |
| 65,536 | 4.2 us | 50.4 us | 11.9x [11.9 – 11.9] | 11.3 us | 2.66x [2.66 – 2.66] |
| 262,144 | 19.6 us | 202.0 us | 10.3x [10.3 – 10.3] | 43.8 us | 2.24x [2.23 – 2.24] |
| 1,048,576 | 91.0 us | 811.9 us | 8.92x [8.91 – 8.93] | 172.9 us | 1.90x [1.90 – 1.90] |
| 4,194,304 | 844.8 us | 3.35 ms | 3.96x [3.95 – 3.98] | 1.19 ms | 1.41x [1.41 – 1.42] |
| 16,777,216 | 4.17 ms | 13.39 ms | 3.21x [3.20 – 3.22] | 5.38 ms | 1.29x [1.29 – 1.30] |
| overall |  |  | 6.79x [6.79 – 6.79] |  | 6.07x [6.07 – 6.08] |
