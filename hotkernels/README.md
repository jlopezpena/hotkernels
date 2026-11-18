# hotkernels

Faster NumPy on Apple Silicon and on x86-64 Linux, with one line and no other change to your code.

```python
import hotkernels
hotkernels.install()
```

`install()` points NumPy at hand-written assembly kernels, ARM64 NEON or x86-64 AVX2, for the calls where NumPy is slowest. Every kernel is checked against NumPy's own output, including NaN, infinities, signed zeros and ties, and is memory-safe by test. A call no kernel handles goes to NumPy exactly as it was written.

## Install

```bash
pip install hotkernels
```

Wheels are for macOS on Apple Silicon and for Linux on x86-64 (glibc), CPython 3.11 to 3.14, with NumPy 2.3 or a later NumPy 2. Another Python on one of those builds from source, which needs the Xcode command line tools on a Mac, or a C compiler and GNU binutils on Linux. On Linux the processor must have the x86-64-v3 instruction sets (AVX2, FMA, BMI1, BMI2, LZCNT, MOVBE, POPCNT), which Intel processors have had since 2013 and AMD's since 2015; on an older one, and on any other machine, `install()` does nothing and `report()` says why, so code that calls it stays portable.

The speed figures on this page were measured on an Apple M1 Pro. On x86-64 NumPy has vector loops of its own for some of these calls, so the gains differ: larger for some, close to none for `clip` and `astype`.

## What gets faster

`np.argmax`, `np.argmin`, `np.clip`, `x[mask]` (and `np.compress`, `np.extract`), `x[mask] = v` (and `np.putmask`, `np.copyto(..., where=mask)`), `np.nonzero` (and `np.where(mask)`), `np.where(mask, a, b)` on arrays of one shape, `np.cumsum` (on boolean masks too), `x.astype(np.float32)` from `uint8`, `np.sum` on `uint8` and on boolean masks, `np.count_nonzero`, `np.histogram`, `np.bincount`, `np.isin`, `np.searchsorted` (and `np.digitize`, which calls it) and `np.interp`, for the element types each kernel covers. At a million elements, from Python, `x[mask]` on `float32` is about 31x faster and `np.histogram` about 9x. On small arrays most of a NumPy call is NumPy's own call path, which hotkernels skips where it can: on 16 elements `np.clip(x, -0.5, 0.5)` is about 10x faster, when its bounds are plain numbers and there is no `out`, and `np.argmax(x)` about 14x.

## Using it

```python
import numpy as np
import hotkernels

hotkernels.install()            # NumPy now uses the kernels, in the whole process
x = np.random.standard_normal(1_000_000)
x[x > 0]                        # served by a kernel

print(hotkernels.report())      # what is installed, and how often each kernel ran
hotkernels.uninstall()          # NumPy is exactly as it was
```

- `hotkernels.install(inexact=True)` also installs `np.cumsum` on `float32` and `float64`, which are within a declared tolerance of NumPy rather than bit-identical.
- `with hotkernels.enabled():` installs for the duration of a block.

## Without changing your code

```bash
python -m hotkernels my_script.py --its --own --arguments
python -m hotkernels -m pytest
python -m hotkernels --report my_script.py   # print report() to standard error at the end
```

Documentation, the full results and how the kernels are made: [sofeikov.github.io/hotkernels](https://sofeikov.github.io/hotkernels/) and [github.com/sofeikov/hotkernels](https://github.com/sofeikov/hotkernels).
