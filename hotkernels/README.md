# hotkernels

Faster NumPy on Apple Silicon, with one line and no other change to your code.

```python
import hotkernels
hotkernels.install()
```

`install()` points NumPy at hand-written ARM64 kernels for the calls where NumPy is slowest. Every kernel is checked against NumPy's own output, including NaN, infinities, signed zeros and ties, and is memory-safe by test. A call no kernel handles goes to NumPy exactly as it was written.

## Install

```bash
pip install hotkernels
```

Wheels are for macOS on Apple Silicon, CPython 3.11 to 3.14, with NumPy 2.3 or a later NumPy 2. Another Python on an Apple Silicon Mac builds from source, which needs the Xcode command line tools. On any other machine the package installs without a compiler and `install()` does nothing, so code that calls it stays portable.

## What gets faster

`np.argmax`, `np.argmin`, `np.clip`, `x[mask]` (and `np.compress`, `np.extract`), `x[mask] = v` (and `np.putmask`, `np.copyto(..., where=mask)`), `np.nonzero` (and `np.where(mask)`), `np.cumsum`, `x.astype(np.float32)` from `uint8`, `np.sum` on `uint8`, `np.count_nonzero`, `np.histogram`, `np.bincount`, `np.isin`, `np.searchsorted` (and `np.digitize`, which calls it) and `np.interp`, for the element types each kernel covers. At a million elements, from Python, `x[mask]` on `float32` is about 31x faster and `np.histogram` about 9x. On small arrays most of a NumPy call is NumPy's own call path, which hotkernels skips where it can: on 16 elements `np.clip(x, -0.5, 0.5)` is about 10x faster, when its bounds are plain numbers and there is no `out`, and `np.argmax(x)` about 14x.

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
