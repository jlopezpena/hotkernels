<p class="lede">Every optimisation in this report is an application of one of four facts about how a modern processor runs a loop. None of them needs assembly to understand. If you already think in terms of latency, ports and cache levels, skip to the next chapter; if those words mean nothing yet, this is the fifteen minutes that makes the rest readable.</p>

## The unit of time

The performance cores of an M1 Pro run at about 3.2 GHz. One clock **cycle** is therefore about 0.31 ns, and that is the natural unit for everything below.

The finished `argmax` kernel handles a float32 element in 0.028 ns. That is not a typo: it is eleven elements per cycle. The naive loop needs about 1.3 ns per element, or four cycles for one element. Both are "a loop over the array". The gap between them is what this chapter explains.

## Fact one: a register can hold several elements

A processor does arithmetic on **registers**, a few dozen named slots that each hold one value. Alongside the ordinary ones, every ARM64 chip has 32 **vector registers** of 128 bits each. The same 128 bits can be read as four float32 values, two float64 values, or sixteen bytes, and there are instructions that operate on all of those **lanes** at once: one add instruction, four additions.

This is SIMD (single instruction, multiple data), and on ARM the instruction set is called NEON. It is the reason element width matters so much in the results: a float32 kernel moves four elements per instruction where the float64 version of the same kernel moves two.

A vector loop has one awkward moment, at the end. An array of 37 float32 values is nine full vectors and one element left over. What to do with the leftover is a real design decision with a correctness trap in it. Try the three options:

<figure class="widget">
<div data-widget="lanes" data-n="37"></div>
<figcaption>A vector loop over an array, and three ways to finish it. Change the element count and type, then step through. "Read past the end" is the shortest code and the one the judge forbids: the extra bytes may not belong to the program.</figcaption>
</figure>

Reading past the end is tempting because it is harmless almost always. Memory is handed to a program in fixed-size **pages**, and an over-long read only fails when the array happens to end exactly at the edge of the last page the program owns. Then it crashes. The kernels here are never allowed to do it, and [the judge]([[ root ]]report/judge/#memory-safety-by-construction) enforces that by deliberately putting every array against such an edge. The standard answer in these kernels is the middle option: one final vector that ends exactly at the last element and overlaps the previous one. Processing a few elements twice is free for operations such as a maximum or a clip, where doing it again changes nothing.

## Fact two: instructions overlap, unless they wait for each other

A modern core does not execute instructions one after another. It looks hundreds of instructions ahead, and runs any instruction whose inputs are ready, several per cycle, on whichever execution unit is free. Two numbers describe every instruction:

- **Latency**: how many cycles until its result can be used by another instruction. A floating-point add on this core takes 3 cycles.
- **Throughput**: how many of them can be *started* per cycle. This core can start up to four vector operations per cycle, and three loads from memory.

The difference between those two is the single most important idea in this report. If each instruction needs the result of the one before, the instructions form a **dependency chain** and run one per latency: a chain of floating-point adds does one add every 3 cycles, however wide the machine is. If the instructions are independent, they overlap, and the same machine does four per cycle. That is a factor of twelve from the same instruction, decided entirely by whether the code makes each step wait for the last.

Summing an array into a single accumulator is exactly such a chain: `s = s + x[i]` cannot start until the previous `s` exists. The fix is to keep several accumulators and add them together at the end, so several chains run side by side. How many? This figure is the byte-sum kernel's actual tuning problem, with the measurements from its log:

<figure class="widget">
<div data-widget="chain"></div>
<figcaption>Summing with several accumulators. Each add into an accumulator takes three cycles to complete, so one accumulator handles one vector every three cycles. More accumulators run more chains side by side, until something else becomes the limit: here, the three loads per cycle the core can perform. The dots are measurements from the <code>sum_u8</code> kernel's log.</figcaption>
</figure>

The model in that figure is one line,

$$\text{cycles per vector} = \max\!\left(\frac{L}{k},\ \frac{1}{P}\right),$$

with latency $L = 3$, $k$ accumulators and $P = 3$ loads per cycle, and it predicts the three measured points (0.75, 0.37 and 0.333 cycles per 16 bytes) to within a percent. Most of the tuning in this project looks like this: find which of the two terms is binding, and push on that one. When the chain is binding, add independence. When the ports are binding, use fewer instructions per element. When neither is, the limit is memory, and the kernel is finished.

::: aside Why compilers often cannot do this for you
For integers, regrouping a sum is always safe: addition is associative. For floating-point it is not, because every add rounds, and `(a + b) + c` can differ from `a + (b + c)` in the last bit. A C compiler must produce exactly the result the source code specifies, so it may not split a floating-point sum across accumulators unless told that the rounding does not matter. This is the heart of the `cumsum` story in [chapter 6]([[ root ]]report/arithmetic/#cumsum-the-chain-you-cannot-break).
:::

## Fact three: branches are guesses

To look hundreds of instructions ahead, the processor has to know which instructions come next, and at every `if` that depends on a value it has not computed yet. So it **predicts** the outcome, keeps going on that assumption, and checks later. A correct guess costs nothing. A wrong one means throwing away everything started since the guess and beginning again: a dozen cycles or more, the cost of dozens of useful instructions.

Predictors are very good at patterns. A loop's "are we done yet?" branch is predicted almost perfectly. A branch that depends on *data* is another matter:

| Branch in a loop | Predictable? | Cost |
|---|---|---|
| "Have we reached the end of the array?" | Nearly always | Nothing |
| "Is this element a NaN?" on clean data | Nearly always | Nothing |
| "Is this mask element true?" on a mask that is 99% false | Mostly | Small |
| "Is this mask element true?" on a random mask | No: wrong half the time | Several cycles per element, on average |

That last row is the [mask surprise]([[ root ]]report/numpy/#the-mask-surprise) of the previous chapter: 3.2 ns per element for a loop whose actual work takes 0.34.

The cure is to turn decisions into arithmetic. Instead of "if selected, store and advance", write "store, then advance by the mask value, which is 0 or 1". Instead of "if greater, remember the index", compute both candidates and use a **conditional select**, an instruction that picks one of two values according to a previous comparison without changing which instruction runs next. The processor never has to guess, so it is never wrong. Nearly every kernel in this report is **branch-free in the data**: the sequence of instructions it executes depends on the length of the array and not on what is in it.

There is one honest exception. When a rare case would make the common case slower, the kernels branch on it deliberately: a NaN in an `argmax`, a negative zero in a `clip`. The branch is predicted correctly as long as the rare thing stays rare, and it buys a faster common path.

::: note A trap for benchmarks
A predictor can memorise a few thousand outcomes. Time a branchy loop on the *same* 4,096-element random mask over and over, and the predictor learns the mask, so the loop looks far faster than it would on fresh data. The judge cycles through a pool of different inputs at small sizes for this reason, and where it could not, the results chapter says so.
:::

## Fact four: memory has a speed limit

The processor is much faster than main memory, so it keeps recently used data in small, fast **caches**: on this chip a first-level cache of 128 KB that answers in a few cycles, a second level of 12 MB, and then main memory, which is slower again. A loop over an array that fits in the first-level cache runs at whatever speed the instructions allow. A loop over 100 MB runs at whatever speed memory can deliver, and nothing done to the instructions will change it.

This is why every result in this report is given per size, and why speedups shrink at the largest sizes. The same kernel shows three distinct plateaus:

<figure class="widget">
<div data-widget="series" data-per-element data-logy
     data-xlabel="elements (float32)" data-ylabel="nanoseconds per element"
     data-marks='[{"x": 32768, "label": "first-level cache, 128 KB"}, {"x": 3145728, "label": "second-level cache, 12 MB"}]'
     data-series='[
       {"label": "NumPy", "kind": "np", "points": {"x":[16,64,256,1024,4096,16384,65536,262144,1048576,4194304,16777216],"y":[611.8,663.6,720.8,1100,2400,7700,29400,112900,452200,1800000,7200000]}},
       {"label": "naive C", "kind": "cc", "points": {"x":[16,64,256,1024,4096,16384,65536,262144,1048576,4194304,16777216],"y":[10.3,54.4,299.7,1300,5400,21600,85700,336400,1360000,5380000,21720000]}},
       {"label": "argmax_f32 kernel", "kind": "hot", "points": {"x":[16,64,256,1024,4096,16384,65536,262144,1048576,4194304,16777216],"y":[1.7,4.1,12.6,33.4,117.7,458.2,3200,12500,51900,326900,1550000]}}
     ]'></div>
<figcaption>Time per element for <code>argmax</code> on float32, from 16 elements to 16.7 million (both axes logarithmic). NumPy's curve falls as its fixed cost per call is spread over more elements. Naive C is flat: it is limited by its own instructions at every size. The kernel reaches 0.028 ns per element inside the first-level cache and then steps up twice, each time the data outgrows a cache. Measurements from the kernel's final judge run.</figcaption>
</figure>

Three things in that picture recur throughout:

- **Small sizes are about the call, not the loop.** Below a few hundred elements the kernel's time per element rises again, because fixed work (deciding which path to take, handling the tail) is spread over few elements. Several kernels have separate straight-line code for short arrays for this reason.
- **Inside the first-level cache, instructions are the limit.** This is where clever code shows: the kernel is 47 times faster than naive C at 16,384 elements.
- **Beyond the caches, memory is the limit.** At 16.7 million elements the kernel runs at the same speed as the simplest possible vector pass over the data. It is three times slower per element than it was in cache, and in this run still 14 times faster than naive C, which was never fast enough to notice the memory.

The processor also **prefetches**: it notices a loop walking forward through memory and starts fetching ahead of it. Programs can request this explicitly too. The attempts logs record repeated tries at beating the hardware's own prefetcher with explicit requests. On a plain forward walk through data that fits the caches it made things slower every time; it paid only on arrays far larger than the second-level cache.

## What a compiler may not do

The compiler in this story, `clang -O3`, is not weak. On a loop like `out[i] = a * x[i] + y[i]` it produces vector code as good as anything written by hand. It loses on the operations in this report for reasons of principle, not of skill:

| The compiler may not... | Because | Which blocks |
|---|---|---|
| Reorder floating-point additions | The result could differ in the last bit from what the source specifies | A fast float `cumsum` or sum |
| Write to memory the source does not write | Another thread might be using that memory; the source said to leave it alone | Blending whole vectors for `x[mask] = v`; storing past the kept elements for `x[mask]` |
| Read memory the source does not read | It might not belong to the program | Loading a full vector at the end of an array |
| Assume the rare case is rare | It has no idea what your data looks like | Betting that there is no NaN, no negative zero, no element on a bin edge |
| Invent a different algorithm | It translates loops; it does not redesign them | Table-driven compaction, block maxima, batched search |
| Replace an operation with a proof that it is unnecessary | That needs mathematics about the specific problem | The `histogram` kernel's skipped corrections |

Every kernel in the next chapters works by doing something on that list, safely. Where "safely" needs an argument, the argument is written at the top of the kernel's source, and the judge is there to catch it being wrong.

## A dozen words of vocabulary

The following chapters occasionally name an instruction, because the attempts logs do and because it is shorter than describing it. These are all you need:

| Name | What it does, in NumPy terms |
|---|---|
| `fmax`, `fmin` | Lane-wise `np.maximum` and `np.minimum`: the result is NaN if *either* input is NaN |
| `fmaxv` | The maximum of the lanes of one vector, as a single number |
| `fcmgt` | Lane-wise `a > b`, giving all-ones or all-zeros per lane: a mask |
| `bsl`, `bit` | Lane-wise `np.where(mask, a, b)`, bit by bit |
| `csel` | The same choice for an ordinary register: pick one of two values by an earlier comparison. No branch |
| `tbl` | A lookup: output byte *i* is `table[index[i]]`. With the right index vector it is an arbitrary shuffle of bytes |
| `clz` | Count leading zeros of a 64-bit number: the position of its highest set bit |
| `uadalp` | Add neighbouring pairs of lanes and accumulate into lanes twice as wide |
| `umaxp` | The maximum of neighbouring pairs of lanes, packing two vectors into one |
| `prfm` | A prefetch hint: start fetching this address; never faults |
| FPSR | A status register in which floating-point instructions leave flags such as "an invalid operation happened" |

With that, on to the question the whole project turns on: if nobody is going to read the assembly, how do you know it is right?
