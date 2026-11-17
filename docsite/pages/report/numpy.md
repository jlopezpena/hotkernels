<p class="lede">Before writing a single kernel, the project measured. The question was simple: for each everyday NumPy operation, how does the time NumPy takes compare with the time the most naive C loop takes for the same job? The answer decided what was worth building, and it was not what intuition suggested.</p>

## A call is a fixed cost plus a loop

To a good approximation, the time of a NumPy call on $n$ elements is

$$T(n) \approx a + b\,n$$

where $a$ is everything NumPy does before and after touching your data, and $b$ is the cost per element of the loop in the middle. The two have nothing to do with each other, and they are improved by entirely different means.

Take `np.argmax(x)` on a contiguous float32 array. This is roughly what happens:

```mermaid The path of np.argmax(x). Everything above the last box is the fixed cost; only the last box depends on the length of the array.
flowchart TD
  stmt["np.argmax(x)"]
  wrapper["<b>Python wrapper</b><br/>argument handling,<br/>dispatch to other array libraries"]
  method["<b>x.argmax()</b>, in C<br/>parse arguments, pick the axis"]
  prepare["<b>Prepare the data</b><br/>make a contiguous copy if the array is strided,<br/>look up the element type's argmax routine"]
  loop["<b>The loop</b><br/>one pass over n elements"]
  stmt --> wrapper --> method --> prepare --> loop
  class loop hot
```

On 16 elements, `np.argmax(x)` takes about 530 ns on the test machine. Calling the method directly, `x.argmax()`, takes about 170 ns, so the Python wrapper alone is some 360 ns. The loop over 16 floats takes about 10 ns in naive C and under 2 ns in the finished kernel. For a small array the loop is a rounding error; for a million elements the fixed cost is.

The survey measured $a$ for every candidate by timing it on a one-element array:

| Fixed cost per call | Calls |
|---|---|
| 12,000 to 18,000 ns | `np.histogram`, `np.isin` |
| 1,100 to 2,100 ns | `np.clip`, `np.diff` |
| 700 to 1,250 ns | `np.sum`, `np.max`, `np.cumsum`, `np.any`, `np.all` |
| 400 to 650 ns | `np.argmax`, `np.argmin`, `np.searchsorted`, `np.interp`, `np.where` |
| 140 to 310 ns | `np.nonzero`, `np.bincount`, `np.count_nonzero`, `astype`, `x[mask]`, `x[mask] = v` |
| 60 to 100 ns | `a[idx]`, `a[idx] = v` |

Two things follow. A call that goes more or less straight into C costs 60 to 150 ns, so anything costing 500 ns or more is spending most of its time in Python-level wrappers, and that part can be removed without writing a kernel at all. And whatever sits between NumPy and a kernel has to cost far less than 100 ns, or small arrays get slower. An early experiment that called a kernel through `ctypes` cost about 690 ns per call, for a kernel that ran in 2 ns. That number is why the project has a compiled binding, and why [chapter 7]([[ root ]]report/delivery/) exists.

The rest of this chapter, and most of the report, is about $b$.

## The yardstick: naive C

"Is NumPy slow here?" needs something to compare against. The yardstick used throughout is the simplest correct C loop for the operation, compiled with `clang -O3 -mcpu=native`. For `argmax` it is this, and it reproduces NumPy's rules exactly: the first maximum wins, and the first NaN wins over everything.

```c
int64_t argmax_f32(const float *x, size_t n) {
    float best = x[0];
    if (best != best) return 0;
    size_t best_i = 0;
    for (size_t i = 1; i < n; i++) {
        float v = x[i];
        if (v != v) return (int64_t)i;
        if (v > best) {
            best = v;
            best_i = i;
        }
    }
    return (int64_t)best_i;
}
```

Naive C does three jobs in this project. It is the baseline NumPy is measured against. It is a second, independent statement of what the right answer is. And it is what ships when hand-written assembly cannot beat it, because C is far easier to maintain.

Call the ratio of NumPy's time to naive C's time the **gap**. A gap of 4x means the dumbest possible loop is four times faster than NumPy. The gaps below are geometric means over arrays of 65,536 elements and up, where the fixed cost no longer matters.

## What the survey found

Sixty-one variants were timed: each candidate operation at the element types and input shapes people use. The result, in one sentence: **most of the operations people assume are optimised are slower than a naive loop, and the ones NumPy has optimised are superb.**

| Operation | Gap | ns per element at 1M: NumPy / naive C | Does the compiler vectorise the C? |
|---|---|---|---|
| `np.clip`, float32 | 7.1x behind | 0.65 / 0.09 | Yes |
| `np.histogram`, uniform bins | 7.1x behind | 6.26 / 0.88 | No |
| `np.cumsum`, int64 | 6.5x behind | 2.18 / 0.34 | No |
| `np.isin`, 8 test values | 5.7x behind | 7.86 / 1.08 | Partly |
| `astype`, uint8 to float32 | 5.5x behind | 0.32 / 0.06 | Yes |
| `np.interp` | 3.4x to 5.3x behind | 679 / 127 per point | No |
| `np.bincount` | 4.2x behind | 1.37 / 0.33 | No |
| `np.cumsum`, float64 | 3.7x behind | 3.61 / 0.97 | No |
| `np.sum`, uint8 | 3.3x behind | 0.19 / 0.06 | Yes |
| `np.count_nonzero`, int64 and float64 | 2.9x behind | 0.33 / 0.10 | Yes |
| `np.searchsorted`, float64 | 1.7x to 4.7x behind | 142 / 84 per needle | No |
| `x[mask]`, half selected | 14x behind a branch-free loop | 4.75 / 0.34 | No |
| `np.argmax`, float32 | NumPy 3x **ahead** | 0.43 / 1.29 | No |

And on the other side of the ledger, where NumPy already has hand-tuned vector code, it is 10 to 50 times *ahead* of naive C: `np.max`, `np.any`, `np.all`, `np.count_nonzero` on booleans and `np.packbits` all run at 0.01 to 0.05 ns per element. Float `np.sum`, `np.where(c, a, b)`, `np.diff`, fancy indexing with `a[idx]`, and everything on stepped slices like `a[::2]` are level with naive C or better. None of these is a target, and nothing in this project touches them.

::: takeaway The shape of the opportunity
NumPy's speed is uneven. A handful of loops have had years of attention and run at the limit of the hardware. Others are generic code paths that handle every element type, stride and corner case through one mechanism, and pay for the generality on every element.
:::

## Two kinds of gap

The last column of that table matters more than the first. A large gap says NumPy is slow. It does not say that *assembly* is needed to fix it.

```mermaid How each surveyed operation was sorted. The right-hand branch is where hand-written assembly was expected to matter.
flowchart TD
  start["An operation"] --> q1{"Is NumPy behind<br/>naive C?"}
  q1 -->|"no"| skip["Not a target<br/>np.max, np.any, float np.sum,<br/>a[idx], stepped slices"]
  q1 -->|"yes"| q2{"Does clang turn the naive loop<br/>into vector code on its own?"}
  q2 -->|"yes"| c["Expected: ship the C<br/>clip, astype, sum of bytes,<br/>count_nonzero"]
  q2 -->|"no"| asm["Assembly candidate<br/>masks, nonzero, argmax, cumsum,<br/>searchsorted, interp, histogram, bincount"]
  class asm hot
  class skip np
```

For `clip`, `astype`, byte sums and `count_nonzero`, the compiler looks at the three-line C loop and produces vector code by itself. The whole gap is closed by not being NumPy. The survey's prediction was that assembly would merely tie the compiler on these, so they would ship as C.

For the others the compiler produces a loop that handles one element at a time. Something in the loop defeats it: a result that depends on the previous iteration, a store whose position depends on the data, a search. These are the cases with a known hand-written technique that a compiler will not find, and they were the assembly candidates.

That prediction turned out to be wrong in an instructive way, which [chapter 6]([[ root ]]report/arithmetic/#the-loops-the-compiler-already-vectorises) comes back to: assembly beat the compiler on five of the six "C kernels". A loop the compiler has vectorised is not necessarily a loop it has vectorised well.

## The mask surprise

One finding deserves its own section, because it shows how much of the cost can be something other than arithmetic.

`x[mask]` with half the elements selected costs NumPy 4.75 ns per element. A C loop that copies each selected element costs about 3.2 ns. Neither number is reasonable for copying eight bytes. The same job, written so that no decision depends on the data, takes 0.34 ns:

```c
size_t compress(const double *x, const bool *mask, double *out, size_t n) {
    size_t k = 0;
    for (size_t i = 0; i < n; i++) {
        out[k] = x[i];      // store every element, selected or not
        k += mask[i];       // move on only if it was selected
    }
    return k;
}
```

That is a 14x difference with no vector instructions and no assembly: just the removal of an `if`. The processor guesses the outcome of every conditional branch before it knows the answer, and on a random mask it guesses wrong half the time. Each wrong guess throws away a dozen cycles of work. [The next chapter]([[ root ]]report/machine/#fact-three-branches-are-guesses) explains the mechanism, and [chapter 5]([[ root ]]report/selection/#boolean-masks-without-a-branch) shows where the idea goes when you add vectors to it.

It also changes what counts as a fair baseline. For mask selection and `nonzero` the yardstick is the branch-free loop above, because it is just as short as the branching one and nine times faster. A kernel that beat only the slow loop would have proved nothing.

## What an early experiment ruled out

Before any of this, one throwaway experiment set the direction. It computed `max(0, a*x + y)` on float32 three ways: NumPy, C, and hand-written vector assembly.

| Finding | Number |
|---|---|
| Hand-written assembly against `clang -O3` | A tie: 0.99x to 1.04x from 128 elements up |
| One fused pass against NumPy's three separate passes | 3.7x to 5.7x faster from 262,144 elements up |
| Cost of reaching a kernel through `ctypes` | About 690 ns per call |
| Time in the kernel itself at 16 elements | About 2 ns |

Simple elementwise arithmetic is not worth writing by hand; the compiler is already there. The targets had to be operations with some structure the compiler cannot see through. Fusing several operations into one pass is a large win, but it cannot be delivered to unmodified code, so it is out of scope. And the call path matters as much as the loop.

## The shortlist

The survey ended with a ranked list. Every candidate on it was tried; none was dropped in advance, and the judge decided the outcome of each.

| # | Candidate | The technique a compiler will not find |
|---|---|---|
| 1 | Mask selection, `x[mask]` | Turn each block of mask bytes into a shuffle pattern and compact with a table lookup |
| 2 | `np.nonzero` on booleans | The same compaction, emitting indices |
| 3 | `argmax` and `argmin` on floats | Block maxima, with the index recovered afterwards |
| 4 | `np.searchsorted` on floats | A branch-free search with many needles advancing together |
| 5 | `np.interp` | The same search, then the arithmetic two points at a time |
| 6 | `np.cumsum` | A prefix sum that does not wait on the previous element |
| 7 | `np.histogram` | Bin numbers computed in vectors |
| 8 | Masked assignment, `x[mask] = v` | Load, blend and store whole vectors |
| 9 | `np.bincount` | Several count tables, so neighbours do not wait on each other |
| 10 | `np.isin` with a small test set | All test values held in registers |
| C | `clip`, `astype`, `sum` of bytes, `count_nonzero` | None expected: ship the compiler's output |

Note `argmax` at number three, where NumPy is already *ahead* of naive C. It is on the list because NumPy's float `argmax` is eight times slower than its own `np.max` on the same data (0.43 against 0.05 ns per element). Finding where the maximum is should not cost eight times as much as finding what it is, so there was room, and it turned out to be the largest win over the compiler in the whole project.

To see why any of these techniques work, you need a small amount of knowledge about the machine. That is the next chapter.
