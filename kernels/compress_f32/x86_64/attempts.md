# compress_f32 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, locked, pinned to one performance core by the judge.

What the compiler does with `naive.c` (`clang -O3 -march=native`): the scalar loop, one load, one store and one add of the mask byte per element, about 0.19 ns per element at every size up to 1,048,576. NumPy takes 3.4 ns per element for `x[mask]` at 1,048,576 with half selected.

## 1. Baseline: `vpermps` by a 256-entry table, 32 mask bytes per `vpmovmskb`

- **Idea:** the candidates from the design. One `vpcmpeqb` against zero and one `vpmovmskb` turn 32 mask bytes into 32 bits. Each byte of that word indexes a table of 256 permutations (8 KB, eight 32-bit lanes per entry); `vpermps` with the eight elements as its memory operand moves the selected ones to the front; all eight lanes are stored at the write position and `popcnt` advances it. A mask byte is true when it is not zero. 8 to 31 leftover elements take one table step per 8, the last 1 to 7 the scalar loop of the reference.
- **check:** pass (after one failure: the last group of a block was not masked, so five bits of the group before it reached the table offset and `vmovdqa` faulted on an unaligned entry)
- **compare:** not run (first version). `run`: 2.37x over naive C overall (half_selected 2.17x, few_selected 2.60x), 21.5x over NumPy.
- **Kept:** yes
- **Notes:** branch-free, so both distributions cost the same inside the first-level cache: 0.05 ns per element.

## 2. Take each group's table offset with `rorx`

- **Idea:** the mask word is shifted left by 5 once per block. A group's offset is then `rorx` by 8, 16 or 24 and one `and`, in place of a `mov`, an `and` and a destructive `shr` per group. One instruction slot fewer per group.
- **check:** pass
- **compare:** significantly faster, 1.008x [1.000 – 1.017]
- **Kept:** yes
- **Notes:** small, and at the edge of what the judge resolves. No size got slower.

## 3. Count output elements, not bytes

- **Idea:** the one chain that runs from group to group is the write position. Kept as a byte address it advanced through `lea r9, [r9 + r8*4]`, and a `lea` with a scaled index takes two cycles on this processor. Kept as an element count it advances through `add`, one cycle, and the store scales it in its address.
- **check:** pass
- **compare:** significantly faster, 1.018x [1.010 – 1.025]
- **Kept:** yes
- **Notes:** 1.06x at 1,024 with few selected, nothing elsewhere. The chain was not what held the loop. A micro-benchmark outside the judge, 1,024 elements in the first-level cache, with one thing removed at a time: 41.0 ns as it is, 41.1 ns without the store, 37.3 ns without the permute, 40.0 ns without `popcnt`. That is 1.70 cycles per group of eight whatever is taken out, so the loop is held by its instruction count: 35 instruction slots per block of 32 on a core that renames six per cycle.

## 4. Align the loads of x to 32 bytes

- **Idea:** what gave `sum_u8` 1.5x beyond the first-level cache. One extra table step over the first eight elements, restricted to those in front of the next 32-byte boundary, then blocks whose loads never straddle two cache lines.
- **check:** pass
- **compare:** significantly slower, 0.960x [0.952 – 0.968]
- **Kept:** no
- **Notes:** 1.02x to 1.05x from 16,384 to 1,048,576, and 0.71x at 64 and 0.85x at 256, where the head step is a large share of the call. The loads of x are not what limits this kernel beyond the first-level cache; see the next attempt.

## 5. Prefetch x and the mask

- **Idea:** `prefetcht0` 1,024 bytes ahead of x (two lines) and 256 ahead of the mask, once per block.
- **check:** pass
- **compare:** significantly slower, 0.980x [0.973 – 0.988]
- **Kept:** no
- **Notes:** 1.02x to 1.06x from 16,384 up, 0.81x to 0.96x below. A loop of its own for large arrays would keep the 2% to 6%; not pursued, because the micro-benchmark shows where the time goes there and it is not the reads. At 65,536 elements (data in the second-level cache), cycles per group of eight: 3.35 with half selected and 2.3 with 5% selected as the kernel is, 1.73 for both with the store removed, which is the speed inside the first-level cache. Beyond the first-level cache the kernel is held by the cache lines of the output: each has to be fetched before it is written and written back afterwards.

## 6. Blocks with at most four selected elements copied one by one

- **Idea:** the sparse path that paid on `compress_f64` and `nonzero_bool` (their attempt 2): a block of 32 with at most four selected elements skips the table, and `tzcnt` and `blsr` take four positions out of the mask word, each element copied with a scalar load and store.
- **check:** pass
- **compare:** significantly faster, 1.005x [1.001 – 1.009]
- **Kept:** no
- **Notes:** the verdict passes the rule and the change is still not worth having: few_selected 1.04x (1.07x to 1.09x from 65,536 to 1,048,576, but 0.81x at 1,024), half_selected 0.97x at every size, because every block pays the count and the branch. Four table steps for 32 `float32` cost 28 instruction slots and the four single copies 20, so there is little to win here; on `float64` the table path costs twice as much per block and the same change gave 1.24x.

## Where it stands

Final `judge run`:

`half_selected`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 2.4 ns | 4.1 ns | 1.72x [1.72 – 1.72] | 269.7 ns | 114x [114 – 114] |
| 64 | 4.5 ns | 14.4 ns | 3.17x [3.17 – 3.17] | 331.7 ns | 73.1x [73.0 – 73.2] |
| 256 | 15.4 ns | 47.6 ns | 3.09x [3.09 – 3.11] | 582.0 ns | 37.8x [37.8 – 37.9] |
| 1,024 | 60.9 ns | 192.5 ns | 3.16x [3.15 – 3.17] | 1.7 us | 28.2x [28.1 – 28.3] |
| 4,096 | 175.5 ns | 776.0 ns | 4.42x [4.42 – 4.42] | 5.9 us | 33.9x [33.8 – 34.0] |
| 16,384 | 1.3 us | 3.1 us | 2.37x [2.37 – 2.37] | 45.4 us | 34.8x [34.7 – 34.8] |
| 65,536 | 5.5 us | 13.1 us | 2.39x [2.39 – 2.40] | 212.6 us | 38.7x [38.7 – 38.8] |
| 262,144 | 23.5 us | 52.3 us | 2.22x [2.22 – 2.22] | 893.3 us | 38.0x [37.9 – 38.0] |
| 1,048,576 | 109.2 us | 207.2 us | 1.90x [1.89 – 1.91] | 3.57 ms | 32.6x [32.5 – 32.8] |
| 4,194,304 | 698.4 us | 1.05 ms | 1.50x [1.48 – 1.52] | 14.45 ms | 20.7x [20.5 – 20.9] |
| 16,777,216 | 3.68 ms | 4.72 ms | 1.28x [1.28 – 1.30] | 59.33 ms | 16.1x [16.1 – 16.2] |
| overall |  |  | 2.33x [2.32 – 2.33] |  | 36.8x [36.8 – 36.9] |

`few_selected`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 2.4 ns | 4.0 ns | 1.71x [1.70 – 1.71] | 219.1 ns | 92.8x [92.8 – 92.9] |
| 64 | 4.8 ns | 13.7 ns | 2.86x [2.85 – 2.86] | 264.4 ns | 55.3x [55.3 – 55.4] |
| 256 | 16.7 ns | 47.1 ns | 2.82x [2.82 – 2.83] | 323.8 ns | 19.4x [19.4 – 19.4] |
| 1,024 | 44.9 ns | 192.6 ns | 4.29x [4.27 – 4.30] | 595.0 ns | 13.3x [13.2 – 13.3] |
| 4,096 | 302.4 ns | 777.0 ns | 2.57x [2.57 – 2.57] | 1.6 us | 5.41x [5.40 – 5.42] |
| 16,384 | 902.9 ns | 3.1 us | 3.46x [3.46 – 3.46] | 5.4 us | 5.97x [5.96 – 5.98] |
| 65,536 | 3.7 us | 12.4 us | 3.34x [3.34 – 3.34] | 45.1 us | 12.1x [12.1 – 12.1] |
| 262,144 | 14.9 us | 50.0 us | 3.36x [3.36 – 3.36] | 212.8 us | 14.3x [14.3 – 14.3] |
| 1,048,576 | 68.7 us | 200.6 us | 2.92x [2.91 – 2.92] | 863.3 us | 12.6x [12.5 – 12.6] |
| 4,194,304 | 379.8 us | 864.7 us | 2.28x [2.24 – 2.31] | 3.48 ms | 9.17x [9.05 – 9.27] |
| 16,777,216 | 2.61 ms | 4.13 ms | 1.58x [1.57 – 1.59] | 14.34 ms | 5.49x [5.46 – 5.50] |
| overall |  |  | 2.73x [2.72 – 2.73] |  | 14.2x [14.2 – 14.2] |

All distributions: 2.52x [2.52 – 2.52] over naive C, 22.8x [22.8 – 22.9] over NumPy

With the output buffer 16 bytes behind its input (`half_selected`; "apart" is the same kernel with the output clear of the input):

`half_selected`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 210.6 ns | 202.1 ns | 0.96x [0.96 – 0.96] | 778.0 ns | 3.69x [3.69 – 3.69] |
| 16,384 | 1.4 us | 1.3 us | 0.96x [0.96 – 0.96] | 3.5 us | 2.58x [2.57 – 2.58] |
| 65,536 | 5.4 us | 5.5 us | 1.01x [1.01 – 1.02] | 12.9 us | 2.37x [2.37 – 2.37] |
| 262,144 | 22.6 us | 22.3 us | 0.98x [0.98 – 0.99] | 50.6 us | 2.24x [2.23 – 2.24] |
| 1,048,576 | 108.7 us | 106.7 us | 0.98x [0.98 – 0.98] | 208.9 us | 1.92x [1.92 – 1.93] |
| 4,194,304 | 750.7 us | 890.5 us | 1.19x [1.18 – 1.20] | 1.09 ms | 1.45x [1.44 – 1.45] |
| 16,777,216 | 3.61 ms | 3.64 ms | 1.01x [1.00 – 1.02] | 4.76 ms | 1.32x [1.31 – 1.33] |
| overall |  |  | 1.01x [1.01 – 1.01] |  | 2.11x [2.11 – 2.11] |

No stall: the write position falls behind the read position as elements are dropped, so the stores are only briefly at the addresses the loads are about to read.

What holds it at each size:

- **16 to 1,024:** the instruction count. 1.70 cycles per eight elements inside the first-level cache, 0.04 to 0.05 ns per element.
- **4,096 and up:** the cache-line traffic of the output (attempt 5). The judge's timing loop cycles through several inputs at small sizes, so 4,096 elements are already outside the first-level cache there.
- **4,194,304 and 16,777,216:** memory. 0.17 to 0.22 ns per element with half selected, 1.3x to 1.5x over the naive C.

The candidates the design lists for this family (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| 32 mask bytes to a 32-bit mask in one `vpmovmskb` | Kept (attempt 1) |
| `vpermd`/`vpermps` moves the selected elements to the front by index | Kept (attempt 1), with the elements as the instruction's memory operand |
| The index from a table | Kept: 256 entries of 32 bytes, so the load needs no widening. Entries of 8 bytes widened with `vpmovzxbd` were not tried: the widening and the permute both need port 5, which would hold the loop to two cycles per group |
| The index from `pext` with no table | Not tried. It needs a `pdep`, a multiply, a `pext`, a move into a vector register and a widening in place of one load, and the loop is held by its instruction count |
| `popcnt` advances the write pointer | Kept, as an element count (attempt 3) |
| Sparse masks: `tzcnt` and `blsr` per set bit | Rejected here (attempt 6), kept in `compress_f64` and `nonzero_bool` |
| A block all zeros or all ones recognised with one compare | Not tried. On the random masks the judge times, a block of 32 is all zeros 19% of the time at 5% selected and never at 50%, so the branch would be mispredicted about as often as it saves four steps. It would pay on masks with long runs, which the judge does not time |
| The switch by density can go | It has gone: one path for every density, and no weak spot near 10% |
| Masked loads and stores for leftover elements | Not tried: whether they satisfy the memory rule is open issue 3 of the design. The last 1 to 7 elements take the scalar loop |
| Non-temporal stores | Not tried. Every store is overwritten in part by the next one, so they would need a staging buffer and a branch per filled cache line |
| An output just behind the input | No cost in the judge's close table |
| Alignment of data | Rejected (attempt 4) |
| Alignment of code | The block loop starts on a 64-byte boundary from attempt 1. Not varied |
| Instructions slow on other x86-64 processors | None used: no `pext`, `pdep`, gather or masked store |

Left untried, with what each would be worth: the mask word stored below the stack pointer and read back a byte at a time, and the count taken from the table entry's cache line instead of `popcnt` (together about 7 of 35 instruction slots, inside the first-level cache only); a prefetching loop for large arrays (2% to 6%, attempt 5).

Found here and worth knowing on ARM64: nothing that carries over. The ARM64 kernel's density switch exists because its dense path costs more per element than this one.
