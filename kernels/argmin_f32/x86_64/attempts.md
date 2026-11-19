# argmin_f32 attempts (x86_64)

All runs on an Intel Core i5-14600KF, Linux, locked, pinned to one performance core by the judge. This kernel is generated from the same source as `argmax_f32`, whose log has the reasoning, the candidates and what differs from ARM64 for a caller. Here: `vminps` and "M is larger than a lane", 64 elements per block.

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
| 16 | 4.3 ns | 10.4 ns | 2.39x [2.38 – 2.42] | 525.6 ns | 121x [121 – 123] |
| 64 | 7.2 ns | 39.8 ns | 5.53x [5.52 – 5.53] | 553.7 ns | 77.0x [76.9 – 77.1] |
| 256 | 16.1 ns | 188.0 ns | 11.7x [11.7 – 11.7] | 557.1 ns | 34.7x [34.6 – 34.7] |
| 1,024 | 39.2 ns | 778.7 ns | 19.9x [19.9 – 19.9] | 618.4 ns | 15.8x [15.8 – 15.8] |
| 4,096 | 133.2 ns | 3.1 us | 23.6x [23.6 – 23.6] | 855.7 ns | 6.43x [6.42 – 6.43] |
| 16,384 | 491.6 ns | 12.6 us | 25.6x [25.6 – 25.6] | 1.9 us | 3.83x [3.83 – 3.83] |
| 65,536 | 1.9 us | 50.4 us | 26.4x [26.4 – 26.4] | 5.8 us | 3.05x [3.05 – 3.05] |
| 262,144 | 9.4 us | 201.8 us | 21.5x [21.5 – 21.5] | 22.0 us | 2.34x [2.34 – 2.34] |
| 1,048,576 | 45.6 us | 812.1 us | 17.8x [17.7 – 17.8] | 87.3 us | 1.91x [1.91 – 1.92] |
| 4,194,304 | 188.5 us | 3.23 ms | 17.2x [17.0 – 17.2] | 361.3 us | 1.92x [1.90 – 1.93] |
| 16,777,216 | 1.98 ms | 12.83 ms | 6.47x [6.43 – 6.51] | 2.56 ms | 1.29x [1.28 – 1.30] |
| overall |  |  | 13.2x [13.2 – 13.2] |  | 7.50x [7.50 – 7.51] |
