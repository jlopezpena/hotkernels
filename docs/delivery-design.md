# Delivery: the binding and `install()`

This is a proposal for review. Nothing here is implemented.

It details the second half of phase 3 of [core-design.md](core-design.md). The first half produced 22 kernels that the judge finds correct and faster than NumPy ([kernel-findings.md](kernel-findings.md)). None of them can be called from Python. This document proposes how they reach unmodified NumPy code: the binding, `install()`, `report()`, and how all of it is tested.

The core design left two questions for the start of this phase: which patching mechanisms work on current NumPy, and whether indexing syntax and array methods can be reached at all. Both were answered with a throwaway probe before writing this, and the answers shape the design. They are in [What was verified](#what-was-verified).

## Summary

- **Everything can be reached**, including `a[mask]`, `x[mask] = v` and `x.astype(...)`, which the core design recorded as a known gap. It takes five kinds of hook. One is an API NumPy provides for the purpose. The other four change NumPy's own objects in place.
- **A hook costs about 1 ns on a call that does not take a kernel.** Measured: 0.5 to 1.4 ns per call, on calls that cost 59 ns and up.
- **End to end from Python, the speedups hold.** At a million elements: `np.argmax` 8.6x, `np.clip` 4.9x, `x[mask]` 9.8x to 16x, `x[mask] = v` 7.6x to 21x. The mask selection figure is lower than the judge's 22.5x, because a result has to be allocated.
- **A hooked function can also skip NumPy's Python wrapper.** `np.argmax` on 16 elements drops from 573 ns to 65 ns. That is a win of the binding, not of a kernel.
- **Two kernels cannot serve real calls yet.** `searchsorted_f64` and `interp_f64` take one length for both of their inputs. They need a second size before they are worth binding.
- **One recommendation changes an earlier decision.** The float `cumsum` kernels should not be installed by default. See [Decisions](#decisions).

## What the user gets

```python
import hotkernels
hotkernels.install()
```

The whole interface:

| Call | Does |
|---|---|
| `hotkernels.install(inexact=False)` | Applies every hook. Returns nothing. With `inexact=True` it also installs the kernels that are not bit-identical to NumPy |
| `hotkernels.uninstall()` | Removes every hook. NumPy is exactly as it was |
| `with hotkernels.enabled():` | `install()` on entry, `uninstall()` on exit if this block did the installing |
| `hotkernels.report()` | Returns what is installed and how often each kernel ran. Printing it gives a table |

Rules:

- **Importing does nothing.** NumPy is only changed by `install()`.
- **It covers the whole process**, every thread and every library that calls NumPy, not only the caller's own code. That is the point, and it is why the context manager is not a way to limit hotkernels to one thread.
- **`install()` never fails because of the environment.** On a machine, Python build or NumPy it does not support, it does nothing, and `report()` says why.
- **Calling it twice is harmless.** So is `uninstall()` without `install()`.
- **The binding never converts an argument and never raises an error of its own.** A call either meets a kernel's conditions exactly as it stands, or goes to NumPy with the caller's own arguments. Every error message and warning a user sees is NumPy's.

`print(hotkernels.report())` after running a script, as an illustration of the layout:

```
hotkernels: installed, 20 of 22 kernels (ARM64, NumPy 2.5.5)

NumPy call          Elements   Kernel       Result   Kernel calls   Left to NumPy
np.argmax           float32    assembly     exact          10,412               0
np.clip             float64    assembly     exact           2,000              17
x[mask]             float64    assembly     exact             640           1,203
x.astype(float32)   uint8      C            exact              12               0
np.cumsum           float64    assembly     within tolerance: not installed, needs install(inexact=True)
...
```

"Left to NumPy" counts calls that reached a hook and did not meet the kernel's conditions. It is how a user finds out that their arrays are strided, or of another type, and that nothing was gained.

## What was verified

A C extension of about 260 lines, written to be thrown away, applied each kind of hook to NumPy and recorded which calls arrived and what they cost. It linked four real kernels: `argmax_f32`, `clip_f64`, `compress_f64` and `maskfill_f64`. It is not in the repository.

- **Machine.** Apple M1 Pro, macOS 27.2.1, CPython 3.12.7, NumPy 2.5.5. Native.
- **Timing.** From Python, with and without the hook in alternating batches, the minimum of 11 to 15 batches. The machine was in use (load average about 3), so these are good to a nanosecond or two per call and a few percent at large sizes. The implementation measures again with the real binding.

### The five hooks

| Hook | What is changed | Provided by NumPy for this? |
|---|---|---|
| Inner loop | A ufunc's loop for one type, through `PyUFunc_ReplaceLoopBySignature` | Yes |
| Type slot | The `argmax` and `argmin` entries a NumPy type holds, through `PyDataType_GetArrFuncs` | The accessor is public API; writing through it is not a documented use |
| Function | The call pointer inside a NumPy function object such as `np.argmax` | No. Its position is given by the object's type, which is a CPython guarantee |
| Method | The C function pointer in `ndarray`'s method table | No |
| Indexing | The two pointers behind `a[i]` and `a[i] = v` in `ndarray`'s type | No |

The last three keep NumPy's objects and change one pointer inside them. The alternative is to replace the objects: assign a new function to `np.argmax`, or put a new method into `ndarray`'s dictionary. Both were tried. Replacing is worse in every respect measured:

| | Replace the name `np.argmax` | Hook the function object in place |
|---|---|---|
| `np.argmax(x)` | Reached | Reached |
| `from numpy import argmax` done before `install()` | Not reached | Reached |
| NumPy calling its own function (`np.nanargmax`) | Not reached | Reached |
| `np.argmax` is still the same object, pickles, keeps its signature | No | Yes |
| A library that registers `np.argmax` as a key for the `__array_function__` protocol | Breaks if it is imported after `install()` | Works |

### What each hook reaches

| Hook, tried on | Reached | Not reached |
|---|---|---|
| Type slot, float32 `argmax` | `np.argmax(x)`, `x.argmax()`, a function imported earlier, `np.nanargmax`, strided input (NumPy passes a contiguous copy), `axis=` on a 2-D array (one call per row or column) | Other types |
| Inner loop, float64 `clip` | `np.clip(x, 0, 1)`, `x.clip(0, 1)`, `out=`, in place, contiguous arrays of any shape (as one flat run) | `np.clip(x, 0, None)`, which NumPy sends to `np.maximum` |
| Function, `np.argmax` | Every call of the function, however it was imported or spelled | `x.argmax()` |
| Method, five methods | The method, a bound method saved earlier, and the NumPy function of the same name where that function calls the method: `np.searchsorted`, `np.digitize`, `np.clip`, `np.argmax` | `np.sum(x)`, which does not go through `x.sum()` |
| Indexing | `x[mask]`, `x[mask] = v`, `operator.getitem` | Subclasses of `ndarray`, whenever they were defined; `x.__getitem__(mask)` called by name |

Two details matter for the design:

- **The inner loop and the type slot are called after NumPy has done its own checking.** The type is settled and the memory layout is described by the arguments. A strided call shows up as a step that is not the element size and is handed back to NumPy's loop.
- **The method hook was run for the five methods that use CPython's fast calling convention** (`searchsorted`, `sum`, `astype`, `clip`, `argmax`). `nonzero`, `cumsum` and `compress` use the older tuple conventions. The same change applies to them with a wrapper of the matching shape. That was not run.

### What a hook costs

On calls that reach a hook and do not take a kernel, at 16 elements:

| Hook | Call | NumPy alone | With the hook | Difference |
|---|---|---|---|---|
| Function | `np.argmax(float64)` | 573.2 ns | 569.5 ns | Within noise |
| Inner loop | `np.clip(x[::2], -0.5, 0.5)` | 1,261.0 ns | 1,262.1 ns | +1.2 ns |
| Method | `x.astype(np.float32)` | 202.4 ns | 202.9 ns | +0.5 ns |
| Indexing | `x[3]` | 58.6 to 59.1 ns | 59.7 to 60.0 ns | +0.6 to +1.4 ns |
| Indexing | `x[1:5]` | 88.3 ns | 89.4 ns | +1.1 ns |

The type slot has no such row: the slot belongs to one type, so every call that arrives takes the kernel.

The indexing hook is the one that matters. It sits on every index operation on every array in the process, and adds 1% to 2% to the cheapest of them.

### End to end

Python statement to Python result, NumPy alone against the hooked call:

| Statement | Elements | NumPy | With the kernel | Faster by | Judge, kernel alone |
|---|---|---|---|---|---|
| `np.argmax(x)`, float32, type slot | 1,048,576 | 440.9 µs | 51.1 µs | 8.6x | 8.82x |
| `x.argmax()`, float32, type slot | 16 | 173.9 ns | 168.5 ns | 1.03x | |
| `np.argmax(x)`, float32, type slot | 16 | 562.8 ns | 555.8 ns | 1.01x | |
| `np.argmax(x)`, float32, function hook | 16 | 572.5 ns | 65.2 ns | 8.8x | |
| `np.clip(x, -0.5, 0.5)`, float64, inner loop | 1,048,576 | 661.5 µs | 135.1 µs | 4.9x | 4.88x idle, 3.65x under load |
| `np.clip(x, -0.5, 0.5)`, float64, inner loop | 16 | 1,167.0 ns | 1,154.6 ns | 1.01x | |
| `x[mask]`, float64, half selected | 16 | 249.0 ns | 159.4 ns | 1.6x | |
| `x[mask]`, float64, half selected | 4,096 | 9.66 µs | 0.97 µs | 10.0x | |
| `x[mask]`, float64, half selected | 1,048,576 | 4,814 µs | 301 µs | 16.0x | 22.5x |
| `x[mask]`, float64, few selected | 1,048,576 | 915.6 µs | 93.1 µs | 9.8x | 11.0x |
| `x[mask]`, float64, half selected | 16,777,216 | 76.95 ms | 6.30 ms | 12.2x | |
| `x[mask] = 0.0`, float64, half selected | 16 | 288.3 ns | 40.2 ns | 7.2x | |
| `x[mask] = 0.0`, float64, half selected | 1,048,576 | 3,799 µs | 179 µs | 21.2x | 21.4x |
| `x[mask] = 0.0`, float64, few selected | 1,048,576 | 817 µs | 108 µs | 7.6x | 7.48x |

What this says:

- **Kernels that write into memory the caller already has keep the judge's number.** `argmax`, `clip` and masked assignment land within a few percent of it.
- **Mask selection loses part of its number to allocation.** A scratch kernel needs room for all `n` elements. The probe allocated that, ran the kernel, and then either shrank the array in place or copied the result into one of the right size. Copying was faster: 301 µs against 555 µs at a million elements with half selected, and equal with few selected. The first number is the one in the table.
- **A deep hook does nothing for small arrays.** Through the type slot, `np.argmax` on 16 elements still costs 556 ns, nearly all of it NumPy's Python wrapper. The function hook removes the wrapper. This is the fixed cost the survey measured ([survey-design.md](survey-design.md#per-call-overhead)), and the function hook is the way to take it out.

## Design

### Which hook serves which kernel

Use the deepest hook that exists for an operation, because it reaches the most callers and needs the fewest checks. Add the function hook where the fixed cost of the call is worth removing.

| NumPy operation | Kernels | Hook | Takes the kernel when |
|---|---|---|---|
| `argmax`, `argmin` | 4 | Type slot, plus the function for small arrays | Always, for float32 and float64 |
| `clip` | 2 | Inner loop | Contiguous, scalar bounds with `lo <= hi`, output is not the input |
| `x[mask]` | 2 | Indexing | Both 1-D, contiguous, the same length; a plain `ndarray` indexed by a boolean `ndarray` |
| `x[mask] = v` | 1 | Indexing | As above, and `v` is a Python float |
| `nonzero` | 1 | Method | 1-D contiguous boolean array |
| `searchsorted` | 1 | Method | Both float64 and contiguous, `side='left'`, no `sorter`. Needs a kernel change first |
| `cumsum` | 3 | Method | Contiguous, no `axis`, `dtype` or `out`. The two float kernels only with `inexact=True` |
| `astype` | 1 | Method | Contiguous `uint8` to `float32`, other arguments at their defaults |
| `sum` | 1 | Method and function | Contiguous `uint8`, no other arguments |
| `count_nonzero` | 2 | Function | Contiguous, no `axis` |
| `interp` | 1 | Function | Three contiguous float64 arrays, no `left`, `right` or `period`. Needs a kernel change first |
| `histogram` | 1 | Function | Contiguous float64, an integer bin count, no weights or density |
| `bincount` | 1 | Function | Contiguous `uint8`, no weights |
| `isin` | 1 | Function | Contiguous int64, at most eight test values, flags at their defaults |

"Contiguous" for a kernel that works element by element or reduces the whole array means any shape. NumPy's inner loops already work that way: a contiguous 40 by 25 array arrived at the `clip` loop as one run of 1,000.

The function forms of the mask operations (`np.compress`, `np.extract`, `np.putmask`, `np.copyto`, `np.where(mask)`) can be hooked the same way. They are not needed for the exit criterion and come after it.

### Considered and not proposed

- **Replacing names instead of hooking objects.** Compared above.
- **`cumsum` through the inner loop of `add`.** NumPy runs `np.cumsum` as that loop with the output one element ahead of its first input, so the call can be recognised, and it would reach `axis=` and `np.add.accumulate` as well. Two things count against it. The loop is handed the array from its second element on, with the first already stored, and the kernels cannot start from a value. And the check would sit on every float64 addition in the process: +1.0 ns per call, measured.
- **`np.sum` on `uint8` through the inner loop.** NumPy converts the bytes to 64-bit integers in chunks before the loop sees them: 13 loop calls for 100,000 elements. The kernel would never see a byte.

### When a call takes a kernel

One set of checks, written once and shared by the function, method and indexing hooks:

- The array's type is exactly `ndarray`. Subclasses, masked arrays and memory-mapped arrays go to NumPy.
- Its element type is exactly the kernel's, in native byte order.
- It is contiguous. Alignment is not required; the kernels do not need it.
- It has at least the kernel's `min_n` elements.
- Every argument the kernel does not implement is absent or at its default.
- Scalars are inside the domain the kernel's spec declares. For `clip` that is `lo <= hi`, which a NaN bound also fails.
- An output the caller supplies is writable and does not overlap an input.
- The size is at or above any minimum that end-to-end timing sets for the entry point (see [Exit criterion](#exit-criterion)).

Anything else is passed to the original function pointer with the same arguments. Nothing is copied or converted on the way, which is why a fallback costs a nanosecond.

### Results and memory

- **A kernel never writes into a caller's buffer unless the operation is defined that way.** `clip` with `out=` and masked assignment are. Scratch kernels are not: [judge-design.md](judge-design.md#scratch-output-writing-inside-the-capacity) requires that a scratch kernel only ever sees a buffer the binding owns.
- **Scratch output.** Allocate `n` elements, run the kernel, copy the reported elements into an array of exactly that size, free the first. This is the faster of the two ways tried. With few elements selected the large buffer barely shows: 9.8x end to end against the judge's 11.0x for the kernel alone.
- **Result types match NumPy's.** `np.count_nonzero` returns a Python `int`, `np.sum` on `uint8` a `np.uint64`, `np.histogram` a pair of counts and edges. Edges are produced by calling the NumPy function that NumPy itself uses, so they are identical by construction.
- **`bincount_u8` always counts into 256 entries.** NumPy returns one entry more than the largest value present. The binding cuts the result to that length.

### Threads

- **Inner-loop and type-slot hooks inherit NumPy's behaviour.** The kernel runs where NumPy's loop ran, with the lock held or released as NumPy decided.
- **The other hooks release the interpreter lock above a size threshold**, as NumPy does. A kernel is a pure leaf function and touches no Python object.
- **Masked assignment keeps the lock.** The kernel rewrites elements the mask does not select with their own bits ([kernel-findings.md](kernel-findings.md#masked-assignment)). Holding the lock means no other Python thread can write one of them in between. It blocks other threads for about 180 µs per million elements.
- **Free-threaded Python is not supported.** `install()` does nothing there. Changing a shared function pointer while other threads run needs its own design.

### Counting

Each hook adds one to a per-kernel counter when it takes a kernel and to another when it passes the call on. `report()` reads them. The counters are atomic, because inner loops run with the lock released. Their cost is part of what the first step measures.

### Installing and uninstalling

- **Order.** Hooks go in one at a time, each recording the pointer it replaced in a table.
- **Each hook proves itself.** After applying a hook, `install()` makes one small call through NumPy's public interface with a known answer. If the kernel's counter did not move, or the answer is wrong, that hook is removed again and `report()` says so. This is the defence against a future NumPy that routes a call differently: the kernel is quietly not installed, and nothing breaks.
- **`uninstall()` restores in reverse order**, and only a pointer that is still ours. If something else has patched the same place since, it is left alone and reported.
- **Supported environment.** macOS on ARM64, CPython with the interpreter lock, NumPy 2. Anywhere else `install()` is a no-op.

### Package and build

```
hotkernels/                  the installable package, its own distribution
  pyproject.toml
  src/hotkernels/
    __init__.py              install, uninstall, enabled, report
    report.py                the report's data and its rendering
    templates/               the Jinja template for the report
    _binding/                the C extension
      module.c               hook table, install and uninstall, counters
      guard.h                the shared checks
      hook_*.c               one file per kind of hook
      entry/                 one file per NumPy operation
delivery/                    development tool: `uv run delivery check`, `uv run delivery time`
```

- **`kernels/` stays the only source.** The build takes `kernels/<name>/arm64/kernel.s` where it exists and `naive.c` where it does not, compiled with the flags the judge used. The C prototypes come from the specs, through the code the judge already has for it.
- **Two distributions.** `hotkernels` is what a user installs. The judge, the survey and the new `delivery` tool stay in the development distribution. The repository becomes a `uv` workspace holding both.
- **The package depends on NumPy and Jinja.** Jinja is there for the report, by the repository's rule on generated text, and is imported only when a report is rendered.
- **Installing needs a compiler for now.** Wheels are under "Later" in the core design, and so is `python -m hotkernels script.py`.

### Testing the binding

The judge proves a kernel right as a native function. It says nothing about the code between NumPy and the kernel, and that code decides, for every call, whether a kernel's conditions hold. A wrong check there is a wrong result in someone's program. It gets its own test, `uv run delivery check`:

- **Differential.** Every case is a Python call. It is run on plain NumPy and again with hotkernels installed, and the two must agree in result type, element type, shape and bytes (or the kernel's tolerance), in the exception raised, in the warnings issued, and in leaving the inputs unchanged.
- **Cases are a cross product**, per entry point: element types (the kernel's and its neighbours), shapes (0-d, empty, one element, 1-D, 2-D), layouts (contiguous, strided, reversed, misaligned, read-only, Fortran order, other byte order), array kinds (`ndarray`, a subclass, a masked array, a memory map, a list), every spelling of the call (positional, keyword, `out=`, `axis=`, method and function), and values from the judge's value classes.
- **Each case states whether a kernel should run**, and the counters are checked against it. A check that silently stops matching would otherwise pass every comparison while delivering nothing.
- **NumPy's own test suite**, run once on plain NumPy and once with the exact kernels installed, must give the same outcome for every test. This is the broadest evidence available for "never break anything", and it costs nothing to write.

`uv run delivery time` measures each entry point from Python, with and without hotkernels, at 16, 4,096 and 1,048,576 elements, for calls that take the kernel and calls that fall back.

What this does not cover: the binding runs without guard pages. A wrong length passed to a correct kernel is caught only if it changes a result.

## What the kernels need first

Binding the kernels to real calls shows where a kernel is narrower than the NumPy call it stands for. These are kernel or judge changes and go through their own pull requests.

| Kernel | Limit | Consequence | Proposed |
|---|---|---|---|
| `searchsorted_f64`, `interp_f64` | One `n` is the length of the sorted array and of the needles | Only a call where the two lengths happen to be equal can use the kernel | Give the judge a second size, then the specs and the assembly. Not bound until then |
| `clip_f32`, `clip_f64` | Input and output must not overlap | `np.clip(x, 0, 1, out=x)` goes to NumPy. The loop hook sees it as an output pointer equal to the input pointer | Let a spec declare an output that may be the input, check it, and see whether the kernels already pass |
| `isin_i64` | Exactly eight test values | Fewer than eight can be padded by repeating one, which does not change membership. More than eight go to NumPy | Add a case with repeated test values to the spec, since padding makes them the normal case |
| `histogram_f64` | A range must be given | For `range=None` the binding calls NumPy's `min` and `max` first, as NumPy does | Nothing |
| Every operation | One or two element types each | Other types fall back | More kernels; "Later" in the core design |

## Order of work

| Step | What | Why here |
|---|---|---|
| 1 | The package, the build, `install()`, `uninstall()`, `enabled()`, `report()`, counters, the self-test at install. `clip` through the inner loop and `argmax`/`argmin` through the type slot. `delivery check` and `delivery time` | Proves the path with the two hooks that need no checks on the Python side, and builds the tests everything else is held to |
| 2 | Function hook: `count_nonzero`, `sum`, `histogram`, `bincount`, `isin`, and the small-array path for `argmax` and `argmin` | The first hook into an object NumPy does not offer for it, on calls where a mistake is easiest to see |
| 3 | Method hook: `astype`, `sum`, `nonzero`, `cumsum` | Reaches methods; includes the two older calling conventions that were not probed |
| 4 | Indexing hook: `x[mask]`, `x[mask] = v` | Last, because it touches every index operation in the process. It can be left out without affecting the rest |
| 5 | The exit measurements; README updated | |

In parallel, as kernel work: the second size for `searchsorted_f64` and `interp_f64`, then their hooks; in-place `clip`. Also still open from [kernel-findings.md](kernel-findings.md#what-is-next): the locked re-run of every kernel and the leaderboard.

## Exit criterion

The core design's, made measurable:

1. **A demonstration script**, written against plain NumPy with the spellings people use (`x[mask]`, methods, functions), runs faster with the one line added, and `report()` shows every installed kernel being hit.
2. **`uv run delivery check` passes** for every entry point.
3. **NumPy's test suite gives the same outcome** with the exact kernels installed as without.
4. **A call that falls back costs at most 2 ns more than on plain NumPy**, for every hook, as measured by `delivery time`.
5. **No installed entry point is slower than NumPy at any size measured.** Where the binding loses at small sizes, the entry point gets a minimum size and smaller calls go to NumPy.
6. **Every kernel has a recorded verdict:** installed, installed only with `inexact=True`, or not bound, with the reason.

## Out of scope

- Wheels, PyPI and `python -m hotkernels`. "Later" in the core design.
- Free-threaded Python and sub-interpreters.
- x86-64 and Linux.
- Any array that is not a plain `ndarray`.
- New kernels and new element types.

## Decisions

Open, with a recommendation for each.

| # | Question | Recommendation |
|---|---|---|
| 1 | **Change NumPy's own objects in place** (function objects, `ndarray`'s method table)? Only the inner-loop hook is an interface NumPy offers for this, and it reaches `clip` alone. With the type slot as well, 3 of the 15 operations are reached | Yes. It is the only way to the one-line promise, each hook is one pointer with its original kept, and each proves itself at install |
| 2 | **Hook indexing?** It is the only way to `x[mask]` and `x[mask] = v`, the largest gains the compiler does not close. It adds about 1 ns to every index operation on every array | Yes, as the last step, and built so it can be dropped on its own |
| 3 | **Are kernels that are not bit-identical installed by default?** The core design accepted float `cumsum` in tolerance mode, marked in `report()`. Since then the findings showed its output is not monotonic for non-negative input: on a million uniform float32 values about one adjacent pair in six decreases. `np.searchsorted(np.cumsum(p), r)` is a common way to sample, and it assumes a sorted array | No. `install()` installs exact kernels only; `install(inexact=True)` adds the rest. One optional argument, and the default can never change a result |
| 4 | **Masked assignment rewrites unselected elements** (open question 1 in the findings) | Yes, with the interpreter lock held during the kernel |
| 5 | **`clip` reads and writes the floating-point status register** (open question 2 in the findings) | Yes, as the findings recommend. It has to be settled before `clip` is step 1 |
| 6 | **Remove NumPy's Python wrapper for small arrays** where a deeper hook already delivers the kernel (`np.argmax` at 16 elements: 573 ns to 65 ns)? It means a second hook and a second set of checks for the same operation | Yes for `argmax` and `argmin` in step 2. `clip` after the exit criterion: its wrapper has more arguments to get right |
| 7 | **Build system for the package** | `meson-python`: it compiles assembly sources without special handling and is what NumPy uses. Not yet tried in this repository |
| 8 | **Which NumPy versions?** Pin to tested versions, or rely on the self-test at install? | The self-test, plus `delivery check` run against the oldest and newest NumPy 2 releases. A pin would switch hotkernels off on every NumPy release until someone updates it |
