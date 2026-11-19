# The gradient step: what fusion would buy, and what hotkernels can reach

This answers [issue 191](https://github.com/sofeikov/hotkernels/issues/191). The applications study ([applications-findings.md](applications-findings.md)) found that in `mlp_train` the update of a 784 × 256 `float64` weight matrix,

```python
params[name] -= LEARNING_RATE * gradients[name]
```

takes 0.45 to 0.53 ms a step inside the training loop against 0.08 ms timed alone, and is half of the program's samples. The issue read it as one pass over memory done as two, which only fusing the multiply and the subtraction would fix, and which the hooks cannot reach because `LEARNING_RATE * gradients[name]` is evaluated before the subtraction is seen. This document measures where the time goes, what fusion would save, and what hotkernels can do about it, and proposes the one change that the evidence supports. Every figure is from an Apple M1 Pro, NumPy 2.5.5, Python 3.12.7, taken with `uv run --group apps delivery gradient-step` ([Running it](#running-it)).

## Summary

- **Two passes are not the problem.** Inside the loop the multiply into a fresh temporary takes 36 µs and the temporary costs 6 µs to take and give back, with no page faults. The subtraction takes 445 µs. Timed alone it takes 49 µs.
- **The subtraction is slow because Accelerate ran on other cores.** Above a size Accelerate multiplies matrices on several threads, and after `x @ w` has read `w` on other cores, NumPy's loop takes 0.78 ms to subtract in place from `w` instead of 0.05 ms. A read of `w`, or a write to it alone, is not slowed; a read followed by a write to the same line is. With `VECLIB_MAXIMUM_THREADS=1` the effect is gone, and with a 256 × 256 matrix, which Accelerate multiplies on one thread, it never appears.
- **A loop that prefetches each line for a store removes it.** The same subtraction with a `prfm pstl1keep` 4 KB ahead of its loads takes 57 µs after the product, and costs nothing when `w` is in the cache. It is exact: the same arithmetic in the same order.
- **That loop can be reached from the hooks hotkernels already has,** as the inner loop of `subtract` and `add` on `float64`, taken when the output is the first input: `w -= b`, `w += b`, and the in-place adds NumPy's temporary elision makes. In `mlp_train` the training went from 286 ms to 220 ms with it in the study, and with it built into hotkernels the whole application went from 260.5 ms to 152.9 ms (1.70x), the results bit for bit the same. This document proposes it, for ARM64, as a loop built into the binding ([Design](#design-an-in-place-add-and-subtract-of-the-bindings-own)).
- **Fusion would save little more.** A fused single pass with the same prefetch trains in 212 ms against the two passes' 220 ms, 4%. With Accelerate on one thread, where nothing slows the subtraction, the fused pass saves 35 µs a step, 6.6 ms of a 170 ms training.
- **A caching allocator gains nothing.** macOS already hands the 1.6 MB temporary back without page faults; a cache installed through NumPy's allocator interface (NEP 49) made both applications neither faster nor slower.
- **No hook can fuse the statement.** CPython gives an extension no way to see the operand the result of `lr * g` is about to be subtracted from, and a lazy result is unsound for a real `ndarray`. When the subtraction runs, the multiply is done and its temporary is about to be freed: there is nothing left to save ([below](#can-a-hook-fuse-the-statement)).
- **The accumulations of `min_char_rnn` are a different case.** Its arrays are small enough that Accelerate stays on one thread, so the prefetching loop changes nothing there. Fusing `dW += np.dot(a, b.T)` would save half of each such statement, up to about 80 ms of a 289 ms run, and is not reachable from a hook for the same reason; `mem += dparam * dparam` would save 0.2 ms.

## Where the time goes

The study runs `mlp_train`'s training loop statement for statement, calling the application's own `loss_and_gradients`, and replaces only the update of `w1`, the 784 × 256 matrix, by each of six ways of writing it. Five trainings per way, alternating; the update of `w1` is timed every step, and the page faults it takes are counted with `getrusage`.

| Way | Per step | Faults per step | Training, best | Identical |
|---|---|---|---|---|
| As written, `w -= lr * g` | 478 µs | 0.01 | 286 ms | yes |
| Product into a buffer that lives across steps, then `w -= buffer` | 484 µs | 0.00 | 279 ms | yes |
| Fused, one pass in C: `w[i] = w[i] - (lr * g[i])` | 106 µs | 0.02 | 219 ms | yes |
| NumPy's multiply, then a subtract that prefetches `w` for a store | 95.5 µs | 0.00 | 220 ms | yes |
| Fused, one pass, with the same prefetch | 62.6 µs | 0.00 | 212 ms | yes |

Load average 4.4 when the run started. The fused loops round the product before subtracting it, as NumPy's two calls do; a fused multiply-subtract instruction rounds once and would not be bit-identical, so the study's C is built with `-ffp-contract=off` and its object code has none.

The statement as written, in its four parts, median per step:

| Part | Time | Faults |
|---|---|---|
| `np.empty_like(g)`: taking the temporary | 5.2 µs | 0.00 |
| The multiply into it, its first touch | 36 µs | 0.00 |
| `w -= temporary` | 445 µs | 0.00 |
| Giving the temporary back | 1.1 µs | 0.00 |

Reusing the temporary gains nothing, because the temporary was never the cost. The subtraction is nine tenths of the statement, and every way that is fast has one thing in common: it is not NumPy's subtract loop.

## Why the subtraction is slow

The study times `w -= b` alone, after doing one thing to `w` before each call, with NumPy's loop, with a NEON loop of its own, and with that loop prefetching each 128-byte line of `w` for a store 4,096 bytes ahead of its loads. Median of 40 calls.

| `w` | Before each call | NumPy | Loop | Prefetch |
|---|---|---|---|---|
| 784 × 256 | nothing | 49 µs | 48 µs | 49 µs |
| 784 × 256 | `x @ w`, a matrix product, `x` 128 × 784 | 784 µs | 107 µs | 57 µs |
| 784 × 256 | `w @ v`, a matrix-vector product | 51 µs | 51 µs | 52 µs |
| 784 × 256 | `x @ w`, then `w.sum()` | 825 µs | 106 µs | 57 µs |
| 784 × 256 | `x @ w`, then `np.copyto(w, copy)` | 52 µs | 57 µs | 57 µs |
| 256 × 256 | `x @ w` | 17 µs | 18 µs | 19 µs |

With `VECLIB_MAXIMUM_THREADS=1` every row of the 784 × 256 matrix is 49 to 53 µs, NumPy's loop included.

What this shows:

- **Only a matrix product that Accelerate spreads over threads does it.** The matrix-vector product reads all of `w` and leaves no trace; the 256 × 256 product, which Accelerate keeps on the calling thread, leaves none either; one thread for all of Accelerate removes it.
- **Only a read followed by a write is slowed.** `w.sum()` after the product does not repair it, and is itself not slow; `np.copyto`, which writes `w` without reading it, is not slow and leaves `w` fast afterwards.
- **The order of the loop's memory operations decides it.** NumPy's loop and a plain C loop that clang vectorises are equally slow. The study's loop, which handles one cache line per step, is seven times faster without any prefetch, and with a prefetch for a store it is within 15% of the time with `w` untouched.

The explanation consistent with all of it, which the figures support but which was not observed directly: the threads of the product leave `w`'s lines in the caches of other cores. A load then fetches a line in a shared state, and the store that follows must ask again for the line alone, a second round trip per line that the core does not overlap with the next loads well. A prefetch for a store asks for the line alone the first time, so the store finds it ready. A write that does not read first, as `np.copyto` does, also asks for it once.

The gradient step meets exactly this. `forward` computes `x @ params["w1"]` with a batch of 128 rows, which Accelerate threads, and the update of `w1` is the next thing to write to it. The gradient `x.T @ dhidden` is threaded too and is written by the other cores, and `x.T @ dhidden + PENALTY * w1` adds into it in place, through NumPy's temporary elision: that add is slow for the same reason.

A user can avoid it today by setting `VECLIB_MAXIMUM_THREADS=1` before NumPy is imported: the training then takes 170 ms, faster than any way of writing the update on several threads, because the threaded products themselves gain nothing at these sizes. hotkernels cannot set that for the user: it is read by Accelerate when it starts, it changes every matrix product in the process, and the project leaves threading alone ([core-design.md](core-design.md#non-goals)).

## What fusing the statement would save

The table above holds the ceiling. With the prefetch in both, fusing saves 33 µs a step (95.5 µs to 62.6 µs) and 8 ms of a 220 ms training, 4%. On one Accelerate thread, where nothing slows the subtraction:

| Way, `VECLIB_MAXIMUM_THREADS=1` | Per step | Training, best |
|---|---|---|
| As written | 88 µs | 170 ms |
| Product into a reused buffer | 86 µs | 170 ms |
| Fused, one pass | 53 µs | 161 ms |

That is the gain fusion is worth on this machine: one pass over three arrays of 1.6 MB instead of two passes over five, about 35 µs, 188 times a training, 6.6 ms of 170 ms. The issue's ten-fold gap between the loop and the statement timed alone was the threads, not the second pass.

## The temporary and the allocator

`lr * g` makes a temporary of 1.6 MB, far above the size NumPy keeps in its own small cache of freed blocks (under 1,024 bytes), so every step asks the C library for it and gives it back. The question was whether each request gets fresh pages from the system and faults them in.

**On macOS it does not.** The statement takes 0.01 page faults a step, and a whole run of the application 308, under two a step. The system allocator keeps freed large blocks and their pages, and the temporary costs 5 µs to take and 1 µs to give back.

**What glibc would do on Linux,** from its documented behaviour and not measured here: a request of 1.6 MB is above its initial threshold for `mmap` (128 KB), so the first one is mapped and the free unmaps it; glibc then raises the threshold to the size freed, up to 32 MB on 64-bit systems, and later requests of that size come from the heap without new mappings. Its pages stay faulted in unless the top of the heap is trimmed.

**A caching allocator.** NumPy lets the memory of array data come from a handler that can be replaced (`PyDataMem_SetHandler`, NEP 49). The study builds one, `delivery/gradient_step/allocator.c`: up to 32 freed blocks of 64 KB or more, 64 MB in all, handed back to the next request of exactly their size, behind one mutex because NumPy may allocate without the interpreter lock. Five runs of each application on each allocator, alternating, best of each:

| Application | NumPy's allocator | Faults per run | Caching allocator | Faults per run | Identical |
|---|---|---|---|---|---|
| `mlp_train` | 288 ms | 308 | 291 ms | 3 | yes |
| `min_char_rnn` | 289 ms | 2 | 289 ms | 4 | yes |

The cache served 55,002 requests and passed 38 on, and removed the faults that remained; neither application got faster. Weighed as a feature of the package it would have costs as well: memory held after the arrays are gone, a lock on every allocation, and a single handler slot that other libraries using NEP 49 also want, with `install()` and `uninstall()` having to hand arrays allocated by one handler to the free of another (NumPy stores the handler in each array for that reason). With no gain to set against them, it is not proposed.

## Can a hook fuse the statement?

**What the interpreter does.** CPython 3.12 compiles the statement to

```text
LOAD_NAME   params
LOAD_NAME   name
COPY 2; COPY 2
BINARY_SUBSCR                  w = params[name]
LOAD_NAME   LEARNING_RATE
LOAD_NAME   gradients
LOAD_NAME   name
BINARY_SUBSCR                  g = gradients[name]
BINARY_OP   5 (*)              t = LEARNING_RATE * g
BINARY_OP   23 (-=)            w = w.__isub__(t)
SWAP 3; SWAP 2
STORE_SUBSCR                   params[name] = w
```

`BINARY_OP *` with a `float` on the left: `float.__mul__` declines an array, and the interpreter calls `ndarray`'s `nb_multiply` with the operands as written, `array_multiply(lr, g)` in `numpy/_core/src/multiarray/number.c`. That first tries `try_binary_elide` (`temp_elide.c`), which would reuse an operand whose only reference is the interpreter's stack: neither is one, since `lr` is a Python float and `g` is held by the dictionary. So it calls the `multiply` ufunc, which allocates `t` and runs `DOUBLE_multiply`. `BINARY_OP -=` then calls `nb_inplace_subtract`, `array_inplace_subtract(w, t)`, which calls the `subtract` ufunc as `subtract(w, t, out=w)` and runs `DOUBLE_subtract` with its output equal to its first input. `t`'s reference count drops to zero when the interpreter pops it, and its memory goes back to the allocator.

**What a hook sees.** The inner-loop hook sees two calls: a multiply of a scalar (a step of zero) by `g` into a new array, and a subtraction with output equal to the first input. A hook at `nb_multiply` would see `(lr, g)` and nothing else.

**Could the subtraction know its operand is a dead temporary?** Yes: at `nb_inplace_subtract`, `t` has a reference count of one, the test NumPy's elision makes. But by then the multiply has run, all of `t` is written and in the cache, and what is left is one pass reading `w` and `t` and writing `w`. There is nothing in that pass a fused kernel could leave out; the saving of fusion was the write and the reread of `t`, and both have happened.

**Could the multiply defer its work?** It would have to return something that is an `ndarray` and has no values yet, and compute them when the subtraction asks. That is not sound for a real `ndarray`: its data pointer is public C API, and any extension, a buffer-protocol consumer or NumPy itself can read the values without calling anything a hook could intercept. Returning a subclass changes `type(t)` and is visible to `__array_ufunc__` and to `isinstance` checks. It would only be sound if the hook knew that the very next operation is the in-place subtraction from an exact `ndarray`: that means reading the next instruction and the interpreter's value stack from `nb_multiply`. The instruction can be read (`PyFrame_GetCode`, `PyFrame_GetLasti`), but the operands on the stack cannot: they are in CPython's private frame layout, which changes between versions. NumPy's own elision gives the same reason for walking the C call stack instead (the comment above `NPY_MIN_ELIDE_BYTES` in `temp_elide.c`: "change cpython to give us access to the top of the stack"). And `sys.settrace` with opcode events, or a debugger, can run code between the two instructions. No such hook is proposed.

So fusion of a statement written with operators is out of reach of any hook, and the measured gain of having it is 4%.

## The accumulations of `min_char_rnn`

`min_char_rnn` makes 22,500 accumulations of an outer product per run, `dWhh += np.dot(dhraw, hs[t-1].T)` and its two siblings, and 1,500 Adagrad updates of the memory, `mem += dparam * dparam`, of which 300 are on a 100 × 100 matrix. Timed alone at those sizes, median of 2,000 calls:

| Statement | As written | What fusing leaves | Per run | Saved per run, at most |
|---|---|---|---|---|
| `dWhh += np.dot(dhraw, hs[t-1].T)`, 100 × 100 | 7.3 µs | 3.7 µs, the product alone | 22,500 | 82 ms |
| `mWhh += dWhh * dWhh`, 100 × 100 | 4.9 µs | 4.1 µs, one pass in C | 300 | 0.2 ms |

The first column of savings is an upper bound: two of the three outer products are 100 × 58, not 100 × 100, for the 58 characters of the text. Accelerate's rank-one update, `dger`, adds the outer product into the matrix in place in 2.0 µs, which is what a fused accumulation would cost, but rounds differently: it differs from the two-step result in the last bits.

Neither is slowed by the effect above: at 100 × 100 Accelerate does not thread, and with the prefetching loop installed `min_char_rnn` took the same time (294 ms against 293 ms in the prototype). The add is the second call of the statement and the product has already been made by then, so the fused accumulation is unreachable from the hooks for the same reason as the gradient step; a hook on `np.dot` would see the product and not the `+=`. The square is not worth fusing.

One thing the measurement turned up is a single-call gap and not fusion: `np.dot(a, b.T)` of two columns of 100 takes 3.7 µs for 10,000 products. It is outside this issue and is recorded for its own.

## An explicit function

A function such as `hotkernels.axpy(w, -lr, g)` would reach the fused loop, at the price of the user changing their code. [core-design.md](core-design.md) puts "no code changes" first among the principles, and the "Later" section there named fusion as a project of its own because it "cannot be delivered transparently". The measurement settles whether it is worth breaking that: with the prefetching loop installed, a fused `axpy` would make `mlp_train` 4% faster (220 ms to 212 ms), and on one Accelerate thread 5% (170 ms to 161 ms). That is less than either of the two changes that need no code: the in-place loop below (286 ms to 220 ms), or one environment variable (170 ms). It is not proposed.

## Design: an in-place add and subtract of the binding's own

### What it does

On ARM64, the inner loops of the `subtract` and `add` ufuncs for `float64` are hooked, as the loop of `clip` already is, through `PyUFunc_ReplaceLoopBySignature`. The hook takes a call when:

- the output is the first input, at the same address: `w -= b`, `w += b`, `np.subtract(w, b, out=w)`, and the in-place operation NumPy's elision substitutes for `t + b` when `t` is a temporary;
- all three operands step by one element (8 bytes), so a reduction, whose accumulator steps by zero, and every strided or broadcast operand go to NumPy;
- the second input is the output itself or does not overlap it;
- there are at least 131,072 elements (see [What changed in the building](#what-changed-in-the-building)).

It then runs a loop that handles one 128-byte line of the output per step, two elements per NEON instruction, and prefetches the line 4,096 bytes ahead for a store (`prfm pstl1keep`). No prefetch names an address beyond the end of the output: the last 4,096 bytes are handled without. Everything else goes to NumPy's own loop, which the hook keeps.

### Why it is exact

The loop subtracts or adds the same two elements in the same order as NumPy's loop, with the same instruction (`fsub`, `fadd`), so every result, NaN payloads and signed zeros included, is the same, and so are the floating-point flags that NumPy turns into warnings after the loop. A prefetch is a hint: it changes no value and cannot fault. `delivery check` holds it to NumPy bit for bit, with NaNs of different payloads in both operands, signed zeros, infinities, overflow under `np.errstate`, aliasing and every layout.

### Why it is a loop of the binding's own and not a judged kernel

The judge times a kernel alone, on buffers it has just written, and there the prefetching loop and NumPy's loop take the same time: 0.99x to 1.10x of NumPy from 1,024 to 4 million elements in the prototype. Its gain exists only after another core has read the output, which the judge's timing cannot arrange. A kernel in `kernels/` would therefore never pass `judge run`, and changing the judge to place another thread's reads before each timed call is a design of its own. The binding already has loops that the judge does not see, for the cases where what decides the speed lies outside a kernel's loop (`clip_i64`, `round_f64` and `median_f64`, for NumPy's call path). This is another: what decides its speed is the state of the caches when it is called. It is measured where that state arises, in the application and in `delivery gradient-step`, and held to NumPy's results by `delivery check`.

### What it costs elsewhere

Every `float64` add and subtract in a program reaches the hook. One that is not in place, `a + b`, pays for comparing two pointers and three steps before NumPy's loop runs; that is measured with `delivery time` and must be within the noise at every size. One that is in place on data already in the cache runs the loop of the binding at about NumPy's speed; the figures are below.

### Counting and reporting

The two loops are counted, as `add_inplace_f64` and `subtract_inplace_f64`, and shown by `report()` as loops built into the binding, marked `C` and left out of the count of kernels. Calls left to NumPy are not counted: every `float64` add and subtract in a program reaches the hook, and nearly all are not the calls the loop is for, so a count would bury the report's other rows. The column is blank for these two rows.

### Other architectures

The loop is not installed on x86-64. The effect it removes was found on the Apple M1 Pro with Accelerate, and no x86-64 machine was at hand to see whether a threaded BLAS leaves the same cost there. The question is recorded as [issue 204](https://github.com/sofeikov/hotkernels/issues/204).

### What is not proposed

- **Other in-place ufuncs.** `w *= 0.9` or `probs /= s` on an array a threaded product has just read would pay the same cost. The two operations of the gradient step are taken first; the rest wait for a program that needs them ([issue 202](https://github.com/sofeikov/hotkernels/issues/202)).
- **`float32`.** No application in the study trains in `float32` ([issue 203](https://github.com/sofeikov/hotkernels/issues/203)).
- **Setting Accelerate's threads.** See above.

## What changed in the building

**A minimum size.** The design took every in-place call of one element or more. `delivery time` found the loop 0.95x to 0.96x of NumPy at 4,096 elements, in the first-level cache, at a load average of about 5; timed in C at a load of about 15, 545 ns against 520 ns for the loop clang makes of `w[i] -= b[i]`. A version that loads a whole line before storing it was slower still (690 ns). From Python, on arrays alone and with no product before them, two alternating runs at a load average of 7 to 13:

| Elements | `w -= b` | `w += b` |
|---|---|---|
| 16,384 | 0.98x | 0.99x |
| 32,768 | 0.99x | 0.99x to 1.00x |
| 65,536 | 0.99x | 0.98x |
| 131,072 | 0.99x to 1.01x | 1.00x |
| 200,704 | 1.02x to 1.07x | 1.01x to 1.03x |
| 524,288 | 1.04x | 1.03x to 1.06x |
| 1,048,576 | 1.02x | 1.02x |

The cost the loop removes was not seen below a matrix of 784 × 256 (200,704 elements): a 256 × 256 product (65,536) runs on one thread. So the hooks take 131,072 elements (1 MB) or more, and smaller arrays go to NumPy's loop with nothing but the hook's test added. `uv run delivery time --only inplace` then has no row slower than NumPy, at a load average of 23: at 16 and 4,096 elements the calls are NumPy's, within 0.5% of plain NumPy, and at 1,048,576 the loop is 1.01x for both. `a + b`, `w -= 0.5` and `w -= b` on float32, which pass the hook to NumPy, are within -1.8% to +0.7% at every size.

**Three things the design did not say.** A dimension of one element may reach a loop with steps of zero, since its step means nothing; such a call is now below the minimum anyway. A reduction down the rows of a matrix, `np.add.reduce(m, axis=0)`, adds row after row into its result in place, and the loop takes those rows when they are long enough, with NumPy's result. And the report's line under the table now says why the two rows show no calls left to NumPy.

**In the application.** With the hooks built, `mlp_train`'s update of `w1` as written, with hotkernels installed, is the row "as written, hotkernels installed" of the study: 93 µs a step against 457 µs on plain NumPy, and the whole training 141 ms against 263 ms, at a load average of about 10, the trained weights identical. The training gains more than the update of `w1` alone, because the in-place add NumPy's elision makes in `x.T @ dhidden + PENALTY * w1`, into the product a threaded `x.T @ dhidden` has just written, is taken too. `uv run --group apps delivery apps --only mlp_train`, at a load average of about 17: 260.5 ms on plain NumPy, 152.9 ms after `install()`, 1.70x, results identical, where the applications study recorded 1.08x; the hooks took 188 calls each of `x += y` and `x -= y`. `min_char_rnn`, whose arrays are under the minimum, took no call and did not change (301.5 ms and 293.7 ms, within the noise of that load).

## Running it

```bash
uv run --group apps delivery gradient-step                            # the four measurements
VECLIB_MAXIMUM_THREADS=1 uv run --group apps delivery gradient-step   # with Accelerate on one thread
```

The study's code is in `delivery/gradient_step/`: `study.py` makes the measurements, `loops.c` holds the NEON loops it compares with NumPy's, and `allocator.c` the caching allocator. It runs on macOS on ARM64 only and takes about a minute.

## Limits

- **A machine that was not idle.** The figures were taken at a load average of 2 to 4.5, with other agents measuring on the same machine. Each table alternates the ways it compares, so its ratios hold; its absolute times are a little high.
- **One machine.** The effect belongs to a processor with several cores sharing work through their caches and to a BLAS that threads; the M1 Pro has two clusters of performance cores, and which cores Accelerate's threads ran on was not observed. Another Apple processor, another BLAS or x86-64 may show it more, less or not at all.
- **Two applications.** `mlp_train` was written for the applications study in the usual way of such a loop. How often other programs update an array in place right after a threaded product has read it is not known.
