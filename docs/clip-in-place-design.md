# `np.clip(x, lo, hi, out=x)`: clipping in place through a buffer of the binding's

This is a proposal for [issue 186](https://github.com/sofeikov/hotkernels/issues/186), and it settles the `clip` half of [issue 7](https://github.com/sofeikov/hotkernels/issues/7). The kernels and every figure here are ARM64's, measured on an Apple M1 Pro with NumPy 2.5.5, unless a line says otherwise.

## Summary

- **The idiom.** Training loops clip their gradients in place: `np.clip(g, -5, 5, out=g)`, or `g.clip(-5, 5, out=g)`. With hotkernels installed both reach the function or method hook, which declines any `out`, and then the clip loop hook, which declines an output that overlaps the input. NumPy does the work, and the call is 6x to 10x slower than the same clip without `out`, which the kernel takes.
- **The kernels are wrong in place on ARM64, on data a gradient can hold.** Not only for `lo = +0.0` and a negative zero, as issue 7 recorded, but for a signalling NaN with any bounds, from 96 float64 or 192 float32 elements. So the in-place call cannot be handed to the kernel as it is.
- **Proposed: the binding clips in place through a buffer of its own.** It runs the kernel on a run of the array into a buffer on the stack, copies the buffer back over that run, and moves on. Each kernel call sees two buffers that do not overlap, which is the contract the judge checks, so the result is the kernel's on both architectures without any argument about how a kernel behaves in place.
- **What still goes to NumPy.** An `out` that is any array other than the input itself, an output that overlaps the input only in part, a read-only array, int64, and everything the hooks decline today.

## Why the kernels are wrong in place

Each kernel computes every element of `out` from the same element of `x`, so element by element an in-place clip is well defined. What breaks it is that the ARM64 kernels read some of `x` a second time after they have written `out`.

`clip_f64` on ARM64 (`kernels/clip_f64/arm64/kernel.s`; `clip_f32` has the same structure with four lanes and thresholds twice as large) has three paths:

| Path | Taken for | Reads `x` again after storing to `out` | In place |
|---|---|---|---|
| Exact (`Lexact`, `Lsmall`, `Lscalar`) | fewer than 96 elements; `lo > hi`; `hi = -0.0`; an unusual FPCR; a stale invalid flag below 768 elements | Only the last block, which overlaps the one before it. It then reads elements it has already clipped, and clipping is idempotent bit for bit (compare and select, no arithmetic) | Right |
| Fast (`Lfast`) | every other pair of bounds from 96 elements, `lo = -5, hi = 5` among them | After the loop, if the invalid flag is set, which a signalling NaN does, the whole array is done again by `Lexact` from `x0`. In place, `x0` now holds the fast path's output, in which `fmax` has already quieted the signalling NaN | Wrong for any signalling NaN |
| Zero (`Lzero`) | `lo = +0.0` from 256 elements | A chunk that held a `-0.0` is done again by `Lzfix` from the chunk's start (`x10`), or the rest of the array by `Lexact` (`Lzgiveup`). In place, `fmax` has already turned the `-0.0` into `+0.0` | Wrong for any `-0.0` and any signalling NaN |

Measured on the M1 Pro by calling the kernels with `out == x` (`repro7.py`, the issue's reproduction extended): with bounds `(-5, 5)` and one signalling NaN in random data, `clip_f64` is wrong from 96 elements and `clip_f32` from 192; with bounds `(0, 1)` and one `-0.0`, wrong from 256 and 512. Plain random data, no NaN and no negative zero, comes out right at every size, which is why the fault is easy to miss.

The x86-64 kernels (`kernels/clip_f*/x86_64/kernel.s`) have one path: `vmaxpd` then `vminpd`, which return their second operand for a NaN, signalling or quiet, and for a tie, so each result is a pure, idempotent function of the element's bits. They too read some elements after storing them (the first vector overlaps the first aligned block, and the last block overlaps the one before), and by the argument above that re-read gives the same bits. So they are probably right in place. That is an argument from the source, not a test on the hardware; nothing here relies on it.

## Options

Issue 7 listed three.

1. **Leave in-place calls to NumPy.** What happens today. The idiom pays NumPy's call path, 1.4 µs for 100 float64 elements against 0.13 µs through the kernel, and NumPy's scalar loop on large arrays, about 0.65 ns per element against 0.13.
2. **Clip through a buffer of the binding's, then copy back.** One extra pass over data that is in the first-level cache. The kernels and their specs are unchanged, and the result is the kernel's by the contract the judge already checks.
3. **Make the kernels right in place.** The fast and zero paths would have to find a signalling NaN or a `-0.0` before they store, or keep a copy of each chunk to redo it from. That is a change to both ARM64 kernels and a new spec with an `InOut` buffer, and either way it adds work to the loop that the non-overlapping call pays for too.

**Proposed: option 2.** It is the only one that is right on both architectures without new kernels, and the copy is cheap next to what the in-place call costs today. Option 3 stays possible later and would make the copy unnecessary; it is a kernel change, judged by the judge.

## The binding

### Clipping in place

A helper, `hk_clip_in_place`, takes the kernel, the array's data, its element count and the bounds. It walks the array in runs of a fixed number of bytes. For each run it calls the kernel with the run as `x` and a buffer on the stack as `out`, then copies the buffer over the run. A run is read by the kernel before the copy writes it, and no run is read after its copy, so every element is clipped exactly once from its original value. The buffer is the size of a run, so the kernel never sees more than it can write.

The run length is a constant chosen by measurement: long enough that the kernel's fixed cost per call (two reads of the status register, about 3 ns) is small against the run, short enough that the buffer and the run stay in the first-level cache and the stack frame stays modest.

The kernels leave the invalid-operation flag set after NaN data, and a kernel that finds the flag already set on entry takes a slower path. The flags are cleared once after the last run, as after any clip kernel today; between runs they are left as they are, which costs speed only on data with NaNs and changes no result.

### Which calls take it

- **`np.clip` and `x.clip` (function and method hooks).** As today, with one addition: `out` may be given if it is the very object being clipped (`out is x`), and that array is writeable. The result is that same object, as NumPy returns it. Anything else given as `out`, including another array that views the same memory, goes on to NumPy as before.
- **The clip loop.** When the output pointer is exactly the input pointer and both steps are the element size, the loop clips in place through the buffer. An output that overlaps the input any other way still goes to NumPy's loop. This serves the in-place calls the two hooks above do not take, such as one with `dtype=` or `casting=`, or a NumPy scalar bound of another type, when NumPy hands the loop the array unbuffered.
- **int64** stays as it is: no in-place path. Its loop is the binding's own C, and its short arrays are not the gradient-clipping case.

Everything the hooks check today they still check: a plain, C-contiguous array in native byte order, bounds that are plain numbers or NumPy scalars of the array's type, `lo <= hi` and neither a NaN.

### Threads

The function and method hooks release the interpreter lock for large arrays, as `run_clip` does now. The buffer is on the stack, so two threads clipping at once do not share it. A thread that writes the array while it is being clipped sees what it would see under NumPy's loop, which also runs without the lock.

## Tests

`delivery check` compares each case's result with and without hotkernels, bit for bit. Today it holds `np.clip(x, lo, hi, out=x)` and `x.clip(lo, hi, out=x)` as spellings no kernel may take. They become spellings the kernel must take on a writeable, contiguous array, and new cases put the in-place form through the same values as the ordinary one:

- every value class of the judge, at every size of the value sweep, for each set of bounds in the spec that the sweep times, and the signalling and negative NaN data, for both the function and the method;
- the issue's shapes, 100 x 100, 100 x 60 and 100 x 1, contiguous, with integer and float Python bounds;
- the loop hook alone, through spellings the two hooks decline (`dtype=`, `casting=`);
- calls that must stay NumPy's: `out` a different array, `out` a view of the same memory shifted by one element (partial overlap), `lo > hi`, a NaN bound, a read-only array, a Fortran-ordered and a strided array.

Which elements a run boundary falls on does not matter to the result, but a case whose size is not a multiple of the run length, and one just above a run, are included.

## Exit criterion

- `delivery check` passes on ARM64 here and on x86-64 in CI.
- From Python, `np.clip(x, -5, 5, out=x)` at 100 x 100, 100 x 60 and 100 x 1 float64 is several times faster than NumPy, and at a million elements not slower than NumPy.
- The `min_char_rnn` application's clip calls are taken by the kernel.

## Out of scope

- Making the kernels right in place (option 3).
- `np.cumsum(x, out=x)`, the other half of issue 7.
- An `out` that is a different array from the input. It needs NumPy's rules for broadcasting and casting into the caller's buffer, which the loop hook already applies.
