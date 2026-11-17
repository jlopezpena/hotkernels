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

Wheels are for macOS on Apple Silicon, Python 3.11 to 3.14, with NumPy 2.

## What gets faster

`np.argmax`, `np.argmin`, `np.clip`, `x[mask]`, `x[mask] = v`, `np.nonzero`, `np.cumsum`, `x.astype(np.float32)` from `uint8`, `np.sum` on `uint8`, `np.count_nonzero`, `np.histogram`, `np.bincount` and `np.isin`, for the element types each kernel covers. At a million elements, from Python, `x[mask]` on `float32` is about 31x faster and `np.histogram` about 9x.

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

Documentation, the full results and how the kernels are made: [sofeikov.github.io/hotkernels](https://sofeikov.github.io/hotkernels/) and [github.com/sofeikov/hotkernels](https://github.com/sofeikov/hotkernels).
