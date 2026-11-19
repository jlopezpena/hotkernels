# argsort_f64 attempts (arm64)

There is no `kernel.s` beside this log: on ARM64 the kernel ships as its naive C, and this is the record of how that C came to be and of the assembly tried against it.

**Verdict: ships as its naive C.** The gain over NumPy is the algorithm, a radix sort where NumPy merges, and the loops that carry it are bound by memory. The one loop worth writing by hand, the scatter of a pass over packed words, came out of the compiler as the same seven instructions per element (attempt 5).

The attempts before the kernel had a spec were timed in a harness outside the repository, each version called through `ctypes` on the same arrays as `np.argsort(x, kind="stable")` and checked against it, on an Apple M1 Pro with other work running (load average 3 to 5). Their `compare` lines say so. The figures are for 1,048,576 uniform float64, where NumPy takes 120 to 125 ms.

## 1. Radix sort of (key, index) pairs, eight passes of eight bits

- **Idea:** a least-significant-digit radix sort is stable, and a stable sort has one right answer, so it can stand in for timsort exactly. Each element becomes a key that orders like NumPy orders the element (NaN last, all NaNs equal, `-0.0` equal to `+0.0`) and travels with its index. All eight counts are taken in one pass first, and a pass in which every key has the same digit is skipped.
- **check:** pass (against NumPy in the harness, on uniform, normal, tied and special values)
- **compare:** n/a: first version. 13.4 ms, 8.9x NumPy
- **Kept:** yes
- **Notes:** this is what the survey's `argsort_radix_f64` still is.

## 2. Digits of eleven bits: six passes

- **Idea:** fewer passes, each scattering to 2,048 places, which the first-level cache and the TLB of this core still follow.
- **check:** pass
- **compare:** n/a: timed in the harness. 11.5 ms against 13.4 ms, 1.17x
- **Kept:** yes
- **Notes:** on 0/1 scores and small integers the two tie; on normal data eleven bits won by 1.37x.

## 3. Sort only the bits that differ, and move one packed word per element

- **Idea:** a pass over the keys ORs and ANDs them, and only the window between the lowest and the highest bit that differs is sorted on. Once a digit is sorted on its bits are dead, so an element can travel as one 64-bit word, the key bits still to come above its index. A window too wide for that sheds its low bits in the first pass, which takes up to 13 bits as its digit; wider still, and the first pass writes pairs and the second packs.
- **check:** pass
- **compare:** n/a: timed in the harness. 8.6 ms against 11.5 ms, 1.34x; 2.5 ms on int64 in [0, 1000), which is one pass, where the pairs took 4.1 ms
- **Kept:** yes
- **Notes:** a first digit of 12 bits at most put a million uniform float64 on the pair path and cost 1.5 ms; 13 keeps them packed. Target digits of 10 or 12 bits were no better than 11.

## 4. A counting loop written out per number of passes, and a last pass that writes indices alone

- **Idea:** the pass that takes all the counts was 40% of the time: 3.5 ms of 8.6, against 0.75 ms for a scatter pass. Its inner loop ran over the passes with shifts, masks and tables read from memory. Written out once per number of passes, they stay in registers. And the last pass can drop the key and write the index straight into the result, if the buffers are chosen by the parity of the number of passes so that it lands there; that removes the pass that unpacked.
- **check:** pass
- **compare:** n/a: timed in the harness. 7.0 ms against 8.6 ms, 1.23x
- **Kept:** yes
- **Notes:** this is `naive.c` but for its key, which attempt 7 changed. The judge's figures are under [Result](#result).

## 5. The scatter loop of a packed pass in assembly

- **Idea:** the loop every pass over packed words spends its time in, `dst[count[(v >> shift) & mask]++] = v`, written by hand: four elements per iteration loaded as two pairs, the shifts and masks done ahead of the four count updates, and a second version that prefetches the source 512 bytes ahead.
- **check:** n/a: the loop alone, in the harness, against the compiler's on the same words
- **compare:** not significant. Over two runs 1.00x and 1.00x at 4,096 words, 1.03x and 1.03x at 65,536, 1.01x and 1.00x at 1,048,576, 1.00x and 1.02x at 4,194,304; with the prefetch 1.00x, 1.03x, 1.00x to 1.01x and 1.01x to 1.03x
- **Kept:** no
- **Notes:** clang compiles the C to the same seven instructions per element, unrolled three times. Each element's store depends on a count that the element before may have just written, so the updates cannot overlap, and beyond the second-level cache the pass runs at the speed of its 2,048 write streams (1.14 ns per element at four million). There is nothing here for assembly to take, and the rest of the kernel is planning code that runs once per call.

## 6. Rank the sign and exponent bits

- **Idea:** numbers of both signs have keys that differ in all 64 bits, so from about 16,000 elements they take the pair path. Their top twelve key bits, the sign and the exponent, take few distinct values (about forty for normal data). Marking which occur during the window pass and replacing them by their rank narrows the window by about six bits, which keeps more arrays on the packed path.
- **check:** pass (against NumPy in the harness)
- **compare:** n/a: timed in the harness against attempt 4. Normal data 1.00x at 4,096 elements, 1.05x to 1.11x from 16,384 to 262,144 and 0.93x at 1,048,576; uniform data 0.81x to 0.95x at every size; values spread over sixty decades 0.81x to 0.90x. With a first digit of up to 14 bits, which a million normal elements need to stay packed, 1.03x there and 0.88x on uniform data
- **Kept:** no
- **Notes:** the mark is a store per element in a pass that otherwise only reads, and the rank is a table lookup per element in two more. Data that gains nothing, which is any data of one sign, pays for both. A million normal elements are still one bit too wide to pack with a 13-bit first digit.

## 7. A key that subtracts the magnitude of a negative number

- **Idea:** found while timing attempt 6: whole numbers of both signs took 15.2 ms for a million, more than random ones. The key of a negative number was its bits inverted, which turns the zero low bits of a whole number into ones, while a positive number keeps them zero: every one of the 64 bits then differs between some two keys. The key 2^63 - magnitude orders the same way and keeps low zero bits zero, so only the high bits differ. It also gives `-0.0` the key of `+0.0` without a test of its own.
- **check:** pass
- **compare:** n/a: timed in the harness against attempt 4. Whole numbers from -1000 to 1000: 4.7 ms against 15.2 ms at 1,048,576, 3.26x, and 2.5x to 3.6x from 1,000 to 4,194,304. Uniform, normal, log-normal and special values: 1.00x to 1.05x
- **Kept:** yes
- **Notes:** the same holds for any data whose low bits are zero on both sides of zero: float32 values widened to float64, and values rounded to a few binary places.

## Result

`uv run judge run argsort_f64 --unlocked`, on an Apple M1 Pro. Unlocked because the kernel is not yet on `main`. NumPy is timed from Python.

RESULT_TABLE

## Not tried

- **Two count tables for a skewed digit.** When most keys share a digit, each count update waits for the one before it.
- **Insertion up to 48 elements.** Between 17 and 255 elements the kernel is no faster than NumPy, and the binding does not call it there.
