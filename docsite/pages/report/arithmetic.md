<p class="lede">The kernels in this chapter compute something: a running total, a count, a clamped value, a bin number. Here the interesting constraints are not about branches. They are about what floating-point arithmetic allows you to rearrange, and about how much of NumPy's exact behaviour can be reproduced by doing less work than NumPy does.</p>

## `cumsum`: the chain you cannot break

A running sum is the purest dependency chain there is. By definition

$$s_i = s_{i-1} + x_i,$$

so each output needs the one before it. The naive loop performs $n$ additions strictly one after another, and its speed is the latency of one add: 3 cycles, 0.93 ns, per element. A wider machine does not help. The loop never has two things it could do at once.

### Integers: shorten the chain, exactly

For integers there is a way out, because integer addition is associative and the additions can be regrouped freely with identical results.

Take a block of eight elements. The partial sums *within* the block, relative to the block's start, can be computed from those eight inputs alone. They do not need the running total, so they can be computed while the previous block is still finishing. Only one operation has to wait for the block before: adding the running total to the whole block at once.

<figure class="diagram">
<svg class="dg" viewBox="0 0 720 310" role="img" aria-label="Three blocks of eight elements. Inside each block, inputs lead to prefix sums and then to outputs, independently of the other blocks. A running total passes from block to block with one addition per block and is added to each block's outputs.">
<defs><marker id="dg-head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path class="dg-headfill" d="M0 0 10 5 0 10z"/></marker><marker id="dg-head-hot" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path class="dg-headfill-hot" d="M0 0 10 5 0 10z"/></marker></defs>
<text class="dg-label" x="130" y="22" text-anchor="middle">block k</text>
<rect class="dg-box" x="40" y="34" width="180" height="40" rx="6"/><text x="130" y="59" text-anchor="middle">8 inputs</text>
<path class="dg-arrow" d="M130 74 V98" marker-end="url(#dg-head)"/>
<rect class="dg-box" x="40" y="100" width="180" height="52" rx="6"/><text x="130" y="122" text-anchor="middle">prefix sums within</text><text x="130" y="140" text-anchor="middle">the block: no waiting</text>
<path class="dg-arrow" d="M130 152 V176" marker-end="url(#dg-head)"/>
<rect class="dg-box" x="40" y="178" width="180" height="40" rx="6"/><text x="130" y="203" text-anchor="middle">8 outputs</text>
<path class="dg-arrow dg-hot" d="M130 262 V222" marker-end="url(#dg-head-hot)"/>
<rect class="dg-box dg-hotbox" x="65" y="262" width="130" height="36" rx="18"/><text x="130" y="285" text-anchor="middle">running total</text>
<text class="dg-label" x="360" y="22" text-anchor="middle">block k+1</text>
<rect class="dg-box" x="270" y="34" width="180" height="40" rx="6"/><text x="360" y="59" text-anchor="middle">8 inputs</text>
<path class="dg-arrow" d="M360 74 V98" marker-end="url(#dg-head)"/>
<rect class="dg-box" x="270" y="100" width="180" height="52" rx="6"/><text x="360" y="122" text-anchor="middle">prefix sums within</text><text x="360" y="140" text-anchor="middle">the block: no waiting</text>
<path class="dg-arrow" d="M360 152 V176" marker-end="url(#dg-head)"/>
<rect class="dg-box" x="270" y="178" width="180" height="40" rx="6"/><text x="360" y="203" text-anchor="middle">8 outputs</text>
<path class="dg-arrow dg-hot" d="M360 262 V222" marker-end="url(#dg-head-hot)"/>
<rect class="dg-box dg-hotbox" x="295" y="262" width="130" height="36" rx="18"/><text x="360" y="285" text-anchor="middle">running total</text>
<text class="dg-label" x="590" y="22" text-anchor="middle">block k+2</text>
<rect class="dg-box" x="500" y="34" width="180" height="40" rx="6"/><text x="590" y="59" text-anchor="middle">8 inputs</text>
<path class="dg-arrow" d="M590 74 V98" marker-end="url(#dg-head)"/>
<rect class="dg-box" x="500" y="100" width="180" height="52" rx="6"/><text x="590" y="122" text-anchor="middle">prefix sums within</text><text x="590" y="140" text-anchor="middle">the block: no waiting</text>
<path class="dg-arrow" d="M590 152 V176" marker-end="url(#dg-head)"/>
<rect class="dg-box" x="500" y="178" width="180" height="40" rx="6"/><text x="590" y="203" text-anchor="middle">8 outputs</text>
<path class="dg-arrow dg-hot" d="M590 262 V222" marker-end="url(#dg-head-hot)"/>
<rect class="dg-box dg-hotbox" x="525" y="262" width="130" height="36" rx="18"/><text x="590" y="285" text-anchor="middle">running total</text>
<path class="dg-arrow dg-hot" d="M195 280 H293" marker-end="url(#dg-head-hot)"/><text class="dg-label dg-hottext" x="245" y="272" text-anchor="middle">one add</text>
<path class="dg-arrow dg-hot" d="M425 280 H523" marker-end="url(#dg-head-hot)"/><text class="dg-label dg-hottext" x="475" y="272" text-anchor="middle">one add</text>
<path class="dg-arrow dg-hot" d="M655 280 H716" marker-end="url(#dg-head-hot)"/>
</svg>
<figcaption>The dependency structure of the integer cumsum kernel. The work inside a block depends only on that block's inputs, so the blocks can be computed side by side. The chain that links them, in orange, is one addition per block of eight elements instead of one per element.</figcaption>
</figure>

The chain is now one add per eight elements. In cache the int64 kernel runs at about 0.13 ns per element, 1.90x faster than naive C overall and 12.5x faster than NumPy at a million elements. From about four million elements up it runs at the speed of a plain memory copy, which is the floor for anything that reads one array and writes another.

### Floats: exact and fast are mutually exclusive

For floating-point the same regrouping changes the answer. Every addition rounds, and `(a + b) + c` is not always `a + (b + c)`. NumPy's `cumsum` adds strictly left to right, so a bit-identical kernel must also add strictly left to right, one add after another.

That is not a limitation of this project's cleverness. It is a bound. The first version of the float `cumsum` specs demanded exact output, by mistake, and the agents' logs record what happened: naive C measured 0.95 to 0.97 ns per element, an exact assembly kernel measured the same, and the latency of one floating-point add on this core is 0.93 ns. Three numbers sitting on top of each other. No exact kernel can be faster than its own dependency chain.

::: takeaway
A float `cumsum` that matches NumPy bit for bit cannot be faster than about one add-latency per element. Any implementation that is faster is, necessarily, adding in a different order.
:::

So the project made a decision and stated it: float `cumsum` may regroup, under a declared error bound, and is clearly marked as inexact. The kernels use the integer scheme, with two independent running sums for float64 and four for float32, and they are limited by the number of operations per block and no longer by the latency of an add.

| | vs naive C, all sizes | vs NumPy at 1,048,576 |
|---|---|---|
| `cumsum_i64`, exact | 1.90x [1.89 – 1.93] | 12.5x |
| `cumsum_f64`, within tolerance | 4.46x [4.43 – 4.47] | 20.1x |
| `cumsum_f32`, within tolerance | 6.10x [6.02 – 6.11] | 29.0x |

### What "within tolerance" means here

A regrouped sum is not a worse sum. Both orders compute the same mathematical quantity with different rounding errors, and the standard bound for either is proportional to the number of terms times the sum of their magnitudes. The judge holds element $i$ to within 4 ulps of $(i+1)\sum_{j\le i}\lvert x_j\rvert$, as [chapter 3]([[ root ]]report/judge/#exact-or-within-a-stated-bound) described.

Measured against an exactly computed sum on 80,303 inputs per kernel, the kernels were at least as accurate as NumPy's sequential sum in most classes of input. On arrays of equal values, the worst case for sequential summation, NumPy's error was 7 to 16 times the kernel's. They were *less* accurate in two classes: sequences contrived so that every one of NumPy's partial sums happens to be exact (the kernel loses one unit per repetition of the pattern $-2^{24}+2,\ 1,\ 2^{24},\ 1$ in float32, where NumPy loses nothing), and magnitudes that grow quickly, where the kernel's error was up to about three times NumPy's and at most about 2.5 ulps. Small integers stored as floats come out exact in any order.

### Infinities and NaN still land where NumPy puts them

Rounding differences are tolerable. Differences in *kind* are not: if NumPy's output has an infinity at index 40,000, so must the kernel's. And a regrouped sum can overflow where the sequential one does not, or come back to a finite value where the sequential one stays infinite.

The kernel handles this with a proof instead of a test per element. It works in chunks of 1,024 elements and, alongside the outputs, accumulates the sum of the *squares* of every output in the chunk. Then:

- If that sum of squares is finite, every output in the chunk is finite and no larger than $\sqrt{\text{DBL\_MAX}}$, about $10^{154}$. Every input and every intermediate must have been finite too, since nothing finite comes out of an addition with a NaN or an infinity. The sequential sum differs from the regrouped one only by rounding, a relative $10^{-16}$ or so per term, so it cannot have been anywhere near overflow either. Both results are finite at every position of the chunk: nothing to reconcile.
- If it is not finite, the chunk is thrown away and recomputed one element at a time in NumPy's order.

One extra multiply-accumulate per vector buys a guarantee about every element. The price is that running sums above about $10^{153}$, though finite, take the slow path.

### The property that made it opt-in

There is one consequence of regrouping that no tolerance captures. With two interleaved running sums, adjacent outputs come from *different* accumulations. For non-negative input, mathematics says the running sum never decreases. The kernel's output can: on uniform random float32 data at a million elements, about one adjacent pair in six decreases, by up to about 5.

That breaks a common idiom. `np.searchsorted(np.cumsum(p), r)`, the standard way to sample from a discrete distribution, assumes a sorted array. So the float `cumsum` kernels are not installed by `hotkernels.install()`. They need `install(inexact=True)`, a decision the user makes knowingly.

## The loops the compiler already vectorises

The survey predicted that four operations would ship as plain C, because `clang` turns their naive loops into vector code unaided. Tried, hand-written assembly beat the compiler on all but one. The lesson is in *how*: in each case the compiler had vectorised the loop as written, and the win came from computing something different.

### Byte sums: do not widen until you must

Summing bytes into a 64-bit total, the compiler's vector code widens every byte to 64 bits *before* adding: 24 vector instructions for every 32 bytes. It is correct and it is general, and it is six cycles per 32 bytes at every array size.

The kernel uses one instruction, `uadalp`, that adds a vector of sixteen bytes, in pairs, into eight 16-bit lanes of an accumulator. One instruction per sixteen bytes. A 16-bit lane can overflow, so the kernel reasons about when: each step adds at most $2 \times 255 = 510$ to a lane, so 128 steps add at most 65,280, which fits. Every 128 steps the 16-bit lanes are folded into a 64-bit total.

Sixteen accumulators run side by side, which is exactly the tuning problem in the [chain figure]([[ root ]]report/machine/#fact-two-instructions-overlap-unless-they-wait-for-each-other): the kernel runs at three 16-byte loads per cycle, the most this core can load.

**5.85x over the compiled C overall, 8.9x in cache, and 15.7x over NumPy at a million bytes.**

### Counting non-zeros: narrow first, then test

To count non-zero int64 values the compiler compares each vector with zero and accumulates the comparison results: sixteen instructions for eight vectors.

The kernel observes that it does not need to know *which* elements are non-zero until the very end. A 64-bit value is non-zero exactly when the larger of its two 32-bit halves is non-zero, and the pairwise-maximum instruction computes that for two vectors at once while packing them into one. Applied three times, eight vectors of 64-bit elements become one vector of sixteen bytes, each byte non-zero exactly when its element was. Then one test and one accumulate. Nine instructions for eight vectors.

For float64 there is a further subtlety: `-0.0` is zero and NaN is not. A single bit test against a mask that ignores the sign bit gets both right, and the same narrowing follows.

**1.28x and 1.41x over the compiled C overall.** From 65,536 to about a million elements both kernel and compiler run at the rate the second-level cache delivers, and tie; the gain is in cache.

### `astype`: the one that stayed C

Converting uint8 to float32, the compiler's loop already stores one 16-byte vector per cycle, which is all the core can store. The assembly gained a little at the two smallest sizes (no entry checks) and at the largest (a different kind of load), and tied at the eight sizes between: 1.05x [0.99 – 1.08] in one run and 1.06x [1.00 – 1.07] in the next.

An interval that touches 1.0 is a tie, and on a tie the design ships C. The kernel is the reference loop, compiled. It is still 2.7x faster than NumPy at a million elements and 5.3x from Python.

## `clip`: two operations instead of four, when it is safe

The naive loop is `v < lo ? lo : (v > hi ? hi : v)`: two comparisons and two selections. The compiler vectorises exactly that, four operations per vector, and at four operations the loop is limited by the core's four vector ports.

Everyone knows the shorter way: `min(max(x, lo), hi)`. Two operations, limited only by how fast results can be stored. The trouble is that it is not the same function.

| Input | Compare and select (NumPy's result) | `fmin(fmax(x, lo), hi)` |
|---|---|---|
| `x = -0.0`, `lo = +0.0` | `-0.0`, because `-0.0 < +0.0` is false | `+0.0` |
| `x = +0.0`, `hi = -0.0` | `+0.0` | `-0.0` |
| A signalling NaN | Copied unchanged | Returned as a quiet NaN |

For reversed bounds or a NaN bound, the reference loop itself disagrees with NumPy, so the kernel is defined for ordered bounds only and the binding sends any other call to NumPy. Within that domain the kernel has three paths, and chooses once per call:

```mermaid How the clip kernels choose a path. The fast path is exact only because every case in which the two formulations differ has been ruled out before the loop or is detected after it.
flowchart TD
  start["clip(x, lo, hi)"] --> q0{"lo ≤ hi,<br/>neither NaN?"}
  q0 -->|"no"| numpy["Not this kernel:<br/>the binding sends the call to NumPy"]
  q0 -->|"yes"| q1{"lo is +0.0<br/>or hi is −0.0?"}
  q1 -->|"neither"| fast["<b>Fast path</b><br/>fmin(fmax(x, lo), hi)<br/>2 operations per vector"]
  q1 -->|"lo is +0.0"| zero["<b>Zero path</b><br/>the same 2 operations, plus 1<br/>that watches for −0.0"]
  q1 -->|"hi is −0.0"| exact["<b>Exact path</b><br/>compare and select<br/>4 operations per vector"]
  fast --> flag{"Did any operation<br/>raise 'invalid'?"}
  flag -->|"no"| done["done"]
  flag -->|"yes: a signalling NaN<br/>went through"| exact
  zero --> seen{"−0.0 in this chunk?"}
  seen -->|"no"| done
  seen -->|"yes"| redo["Redo the chunk exactly.<br/>Too many redos: finish<br/>on the exact path"]
  class fast,zero hot
  class numpy np
```

Three pieces of that deserve a word.

**The signed-zero case is settled at entry.** With `lo <= hi`, the two formulations can only disagree about a zero when `lo` is `+0.0` or `hi` is `-0.0`. For every other pair of bounds the kernel's header proves the fast path returns the same bits, case by case.

**The signalling NaN is caught after the fact.** A signalling NaN is a rare variety that floating-point operations are required to "quieten", changing one bit. The kernel does not test for it. It relies on the fact that `fmax` and `fmin` raise a flag in the processor's floating-point status register when they meet one, and for nothing else. After the loop the kernel reads the flag. If it is set, the whole array is done again on the exact path. This makes the kernel slightly less than a pure function, since it reads and restores processor state; the project lists it as a deliberate exception.

**A lower bound of zero gets a path of its own**, because `np.clip(x, 0, 1)` is the most common call there is. With `lo = +0.0`, the fast path is wrong for exactly one element value: `-0.0`. So this path runs the same two operations and spends a third watching for a negative zero in each chunk. A clean chunk is finished. A chunk with a negative zero is redone exactly, which is expensive (about 75 ns, mostly one mispredicted branch), so the kernel keeps a running credit: clean chunks earn, redone chunks cost, and when the credit runs out the rest of the array goes to the exact path for good.

**What it achieves**, over the compiled C and over NumPy:

| Bounds and data | vs C at 4,096 | vs C, all sizes | vs NumPy at 1M |
|---|---|---|---|
| float32, (−0.5, 0.5) | 1.55x | 1.38x [1.33 – 1.41] | 11.0x |
| float32, (0, 1), no negative zeros | 1.32x | 1.23x [1.21 – 1.24] | 10.0x |
| float32, (0, 1), a few negative zeros | 0.99x | 1.01x [1.00 – 1.02] | 7.62x |
| float64, (−0.5, 0.5) | 1.60x | 1.36x [1.31 – 1.38] | 5.06x |
| float64, (0, 1), no negative zeros | 1.34x | 1.22x [1.20 – 1.23] | 4.88x |

Each figure is the mean of ten runs on a machine that was not fully quiet, with the lowest and highest of the ten in brackets.

The gain over the compiler is modest and conditional, and the table says so. The gain over NumPy is large in every row, and most of it would be there with the plain C.

::: note Where rounded data hides negative zeros
`np.round` of a value between −0.5 and 0 is `-0.0`. Data that has been rounded is therefore full of negative zeros, and `np.clip(rounded, 0, 1)` runs at the speed of the compiled C, not faster. The judge times exactly this case, which is where the third row comes from.
:::

## `histogram`: replacing arithmetic with a proof

`np.histogram` with uniform bins is the most interesting kernel mathematically, because its speed comes from *not doing* part of what NumPy does, together with a proof that the omitted part would have done nothing.

**What NumPy computes.** For each element $v$ in the range $[lo, hi]$ with $N$ bins of width $w = hi - lo$:

1. Estimate the bin: $b = \lfloor (v - lo)/w \cdot N \rfloor$.
2. That estimate involves three rounded floating-point operations, so it can be off by one near a bin edge. NumPy therefore *corrects* it: compute the actual edges of bin $b$, and move down if $v$ is below the lower edge, up if $v$ is at or above the upper one.

Step 2 is most of the cost: two more multiplications, two additions and two comparisons per element, to fix an error that almost never occurs.

**The literal kernel** does all of this two elements at a time with exactly NumPy's operations in NumPy's order: fourteen vector operations per pair of elements, about 0.88 ns per element. Nothing cheaper computes the same thing step for step.

**The kernel that ships** asks a different question: for which elements is the correction *provably* unnecessary?

Write $u = 2^{-53}$ for the rounding unit, $M = \max(\lvert lo\rvert, \lvert hi\rvert)$, and $\tau = (v - lo)\,N/w$ for the exact real-valued position of $v$ in units of bins. The computed position $t$ is $\tau$ after three rounded operations, so

$$\lvert t - \tau \rvert \le 4uN .$$

The computed edges carry rounding error too. Expanding it, the comparison of $v$ with edge $k$ is certain to come out the way exact arithmetic says whenever $\tau$ is at least

$$D = 2.01\,uN + 1.01\,\frac{uMN}{w}$$

away from $k$. So if the *fractional part* of the computed position is at least $\delta = 4uN + D$ away from both 0 and 1, the first estimate is the right bin and neither correction can fire.

The kernel sets $\delta = 2^{-12}$ and checks once per call that the bin count and range make $4uN + D < 2^{-12}$ (it suffices that $N < 2^{31}$ and $MN \le 2^{39}\,w$). Then, per element, it computes only the estimate, looks at its fractional part, and:

- if the fraction lies in $[2^{-12},\ 1 - 2^{-12})$, which is true for 2,047 elements in 2,048 of a continuous distribution, counts it into bin $b$ directly;
- otherwise hands that one element to a scalar copy of the reference, corrections and all.

Four vector operations per pair instead of fourteen. **0.69 ns per element, 1.30x faster than the literal kernel, 3.48x faster than naive C overall, and 9.70x faster than NumPy at a million elements, with bit-identical counts.**

### Why this one was held back, and the bug that justified the caution

This kernel replaces operations with an inequality, and the inequality depends on the bin count and the range. When it was written the judge could test exactly one combination of those. It was not adopted until the judge had learned scalar sets and the spec listed sixteen.

Even then the judge's random data seldom lands within rounding distance of a bin edge, which is the only place a histogram goes wrong. So the evidence about edges came from differential tests outside the judge: 410 million elements against the reference, and 126 million against `np.histogram` itself with values placed on and within three ulps of NumPy's edges and around the $2^{-12}$ threshold, with no mismatch. For scale: a histogram *without* the corrections fails 2,086 of 3,000 rounds of those same inputs.

And still there was a bug. The entry check compares two products, $MN$ and $2^{39} w$. For ranges near the largest representable doubles, both products overflow to infinity, and two infinities compare as equal, so the check passed for ranges where the inequality is false. Elements near a bin edge could land in the neighbouring bin. It was found in review, by a third differential test aimed at the top of the float64 range, and fixed by requiring the products to be finite. After the fix, 6,000 ranges of magnitude $2^{960}$ and up gave no mismatch over 1.6 billion elements; before it, 28 of 3,000 did.

::: takeaway
A proof is only as good as the conditions it was checked under. The error bound was right. The code that tested whether the bound applied was wrong in a corner the tests did not reach. This is the strongest argument in the project for listing what the judge cannot see.
:::

## `bincount`: when the limit is one load and one store

Counting bytes into 256 bins is `counts[x[i]] += 1`: a load, an add and a store per element, to an address that depends on the data. There is no vector form of that on this hardware; scattered increments cannot be batched.

What can be improved is waiting. If two neighbouring bytes are equal, the second increment has to wait for the first to be stored before it can load the count. The kernel removes that dependency for large arrays by keeping *four* count tables on the stack and sending byte $j$ to table $j \bmod 4$, so neighbours never touch the same memory, then adding the four tables at the end. It also reads eight input bytes per load.

**1.24x over the compiled C, and 6.74x over NumPy.** The compiled C was already at about one cycle per element, and that is the honest size of what was left. A table of 65,536 entries indexed by pairs of bytes was slower.

That is the last of the kernels. Twenty-two native functions, each proven against NumPy on a flat buffer. None of them can be called from Python yet.
