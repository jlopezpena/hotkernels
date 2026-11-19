# where_f64 attempts (arm64)

All runs on an Apple M1 Pro (performance cores), macOS, locked (`spec.py` and `naive.c` as on `main`). Other kernel runs were timing other kernels on the same machine at the same time, so the sizes from about 100,000 elements up are noisier than usual.

Every element is 17 bytes read and 8 written. The naive C is left scalar by clang on this machine (about 0.36 ns per element in the first-level cache), and NumPy is level with it.

## 1. Baseline: blocks of 16, cmtst and tbl to lane masks, bsl, overlapping last block

- **Idea:** sixteen bytes of c are loaded once and made 0xff or 0x00 by `cmtst`; eight `tbl` spread each pair of bytes over the two 64-bit lanes of a vector (the indices from `maskfill_f64`), and `bsl` takes each lane from a or b. Loads and stores in pairs (`ldp`/`stp`), half a block at a time. The elements left after the whole blocks are done by one more block that ends at the end of the buffers (out never overlaps an input, so writing some elements twice is harmless). Under 16 elements, a scalar loop with `csel`. Nothing but loads, stores, `tbl` and `bsl` touches the elements.
- **check:** pass
- **compare:** n/a (first version). `run`: 1.66x [1.66 – 1.67] over naive C overall (half_true 1.66x, mostly_true 1.67x), 4.83x over NumPy.
- **Kept:** yes
- **Notes:** 1.92x at 16, 2.6x to 2.7x from 256 to 4,096 (0.13 ns per element), 1.35x to 1.38x from 16,384 to 262,144 (0.26 ns per element), 1.04x to 1.17x from 1,048,576 up. Close placement: 1.00x of the clear placement at every size but 1,048,576, where the clear timing was the slow one.

## 2. Non-temporal loads (`ldnp`) for a and b

- **Idea:** as in `clip_f64` attempt 3: `ldnp` reads the same bytes as `ldp`, and its hint let that loop run at memory-copy speed beyond the first-level cache.
- **check:** pass
- **compare:** significantly faster, 1.060x [1.036 – 1.086]
- **Kept:** yes
- **Notes:** nothing up to 4,096; 1.14x to 1.15x at 16,384, 1.11x to 1.13x at 262,144, 1.26x at 1,048,576, 1.05x at 4,194,304, 1.02x to 1.04x at 16,777,216.

## 3. Non-temporal stores (`stnp`)

- **Idea:** the store-side hint, beside attempt 2's loads.
- **check:** pass
- **compare:** no significant difference, 1.004x [1.000 – 1.009]
- **Kept:** no
- **Notes:** every size 1.00x to 1.01x, 64 at 1.05x not significant. On this core `stnp` does not spare the read of the line it overwrites, or that read does not matter at the judge's sizes.

## 4. Two blocks (32 elements) per loop iteration

- **Idea:** half the loop overhead (four pointer additions, compare, branch) per element; the block is about 38 instructions, so the six of the loop are a sixth of them.
- **check:** pass
- **compare:** significantly slower, 0.873x [0.870 – 0.877]
- **Kept:** no
- **Notes:** nothing up to 4,096; **0.75x to 0.83x at every size from 16,384 up**, both distributions. The instruction count is not what limits the loop, and the longer body with offsets up to 224 bytes in each `ldnp` undoes most of attempt 2's gain beyond the first-level cache. Not understood.

## 5. Prefetch a and b 1,024 bytes ahead

- **Idea:** one `prfm pldl1strm` per block for each of a and b (a block is one 128-byte line of each), in a first loop that stops while the prefetched lines are still inside the buffers, then the plain loop for the last blocks.
- **check:** pass
- **compare:** significantly slower, 0.970x [0.967 – 0.974]
- **Kept:** no
- **Notes:** 0.94x at 1,024 and 4,096, 0.91x to 0.94x at 16,384 and 65,536, nothing from 262,144 up. The hardware prefetcher already has the streams, as `maskfill_f64` found (its attempt 5).

## 6. All sixteen vectors of a and b loaded before any store of the block

- **Idea:** the loads of the second half ahead of the stores of the first. To find the registers, four `tbl` index vectors in place of eight, with the second half's mask bytes moved down by one `ext`; `bif` builds each output in the register that held a.
- **check:** pass
- **compare:** significantly slower, 0.995x [0.992 – 0.997]
- **Kept:** no
- **Notes:** 1.06x at 16, 0.96x to 0.98x from 64 to 4,096, nothing from 16,384 up. The out-of-order window already overlaps the blocks; the extra `ext` costs more than the order gains.

## 7. Zero the output lines ahead with `dc zva`, so the stores need not read them

- **Idea:** a store to a line not in the cache first reads the line; that is 8 of the 33 bytes an element moves. `dc zva` (64-byte blocks on this machine, DCZID_EL0 = 4) allocates a block of out as zeros without reading it. Each block zeroed the two 64-byte blocks 256 and 320 bytes ahead of it in out, only while they ended inside out, so every zeroed byte is later overwritten.
- **check:** pass
- **compare:** significantly slower, 0.473x [0.458 – 0.489]
- **Kept:** no
- **Notes:** 0.64x at 256, falling to 0.30x to 0.38x from 16,384 up. `dc zva` is far too slow on this core to be used per line, as `bincount_u8` found for zeroing its tables.

## 8. Bring out to a multiple of 128 bytes (the cache line) from 48 elements

- **Idea:** as `where_f64` x86-64 attempt 4: the first block where out is, then the pointers move on by up to sixteen elements so that every later store of out falls within one 128-byte line.
- **check:** pass
- **compare:** significantly slower, 0.968x [0.965 – 0.971]
- **Kept:** no
- **Notes:** 0.79x at 64 and 0.92x at 256 (one block more), 0.97x to 0.99x up to 4,096, nothing from 16,384 up. Unaligned stores cost nothing measurable on this core.

## 9. Eight elements per loop iteration

- **Idea:** the opposite of attempt 4: half a block per iteration (8 bytes of c by `ldr d0`, four `tbl`, four `bsl`), to see whether a shorter body with smaller `ldnp` offsets helps beyond the first-level cache. The overlapping last block stays a whole one.
- **check:** pass
- **compare:** significantly slower, 0.970x [0.968 – 0.973]
- **Kept:** no
- **Notes:** 0.91x to 0.95x from 256 to 4,096 (the loop overhead per element doubles), 0.96x to 0.97x at 16,384 and 65,536, nothing from 262,144 up.

## 10. Load a's four vectors of each half block, then b's

- **Idea:** each stream's 64 bytes in two consecutive `ldnp` in place of a and b interleaved, in case the order of the streams matters to the non-temporal loads (attempt 4 showed the loop's shape does).
- **check:** pass
- **compare:** no significant difference, 1.001x [1.000 – 1.003]
- **Kept:** no
- **Notes:** every size 1.00x to 1.01x.

## Where it stands

Final `judge run` (locked), kernel as of attempt 2: 1.84x [1.83 – 1.84] over naive C over both distributions (half_true 1.84x [1.83 – 1.84], mostly_true 1.83x [1.83 – 1.84]), 5.36x [5.35 – 5.36] over NumPy. 1.9x at 16, 2.2x at 64, 2.6x to 2.7x from 256 to 4,096 (0.13 ns per element), 1.7x from 16,384 to 262,144 (0.21 ns per element), 1.4x to 1.5x at 1,048,576, 1.1x to 1.2x at 4,194,304 and 16,777,216 (0.32 to 0.34 ns per element, memory).

The close placement (out 16 bytes behind its input) costs nothing: 1.02x [1.02 – 1.03] of the clear placement overall. Its one slow size, 1,048,576 at 0.88x, came with its naive C 1.7 times slower than in the first table, so other work on the machine.

Eight attempts in a row (3 to 10) gave nothing. The kernel's time does not follow its instruction count (attempts 4, 6 and 9), stores and alignment cost nothing extra (3, 8), and the hardware prefetcher already has the streams (5). Beyond the first-level cache it moves memory.

What the kernel requires of its caller:

- out does not overlap c, a or b: the last block writes again some elements the block before it wrote, from the inputs.
- A byte of c is true when it is not zero (`cmtst`, and `cmp`/`csel` under 16 elements); the judge sends only 0 and 1.
- No buffer needs any alignment.

Not tried:

- **Stores past the caches for large arrays.** `stnp` gave nothing (attempt 3) and `dc zva` is slow (attempt 7); there is no other store on this core that skips the read of the line.
- **Skipping the loads of b for a block whose condition is all true.** A branch per block on random data (44% of blocks all true in mostly_true, none in half_true) mispredicts more than the loads cost, and the timing would depend on the data.
