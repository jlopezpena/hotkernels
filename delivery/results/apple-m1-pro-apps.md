# Applications with and without hotkernels: Apple M1 Pro

Written by `uv run --group apps delivery apps --profile --record`. Do not edit. What the figures mean is in [docs/applications-findings.md](../../docs/applications-findings.md).

- **Machine:** Apple M1 Pro, macOS-27.2.1-arm64-arm-64bit
- **Software:** Python 3.12.7, NumPy 2.5.5, trimesh 5.2.1, scikit-learn 1.9.1, empyrical-reloaded 0.7.3, pandas 3.1.1, scipy 1.18.1, hotkernels at ccd37d1
- **Load average when the run started:** 2.95, a busy machine: the times are rough
- **GitHub Actions runs unfinished when it started:** 0
- **Method:** 7 alternating rounds of each side, the best of each. A round is one run, or as many runs as last 100 ms where one is shorter; the times are of one run. Results compared bit for bit on every round.
- **Profile:** sampled with `sample`, every thread once a millisecond.

## Times

| Application | Size | Data | NumPy | `install()` | Faster by | Results | `install(inexact=True)` | Faster by | Kernel calls | Left to NumPy |
|---|---|---|---|---|---|---|---|---|---|---|
| `mlp_train` | small | 12,000 examples of 196 float64 inputs, 64 hidden units, batches of 128, 2 epochs | 32.5 ms | 26.1 ms | 1.24x | identical | 26.2 ms | 1.24x | 190 | 942 |
| `mlp_train` | large | 12,000 examples of 784 float64 inputs, 256 hidden units, batches of 128, 2 epochs | 311.5 ms | 141.4 ms | 2.20x | identical | 142.9 ms | 2.18x | 566 | 942 |
| `min_char_rnn` | small | 300 iterations, 100 hidden units, sequences of 25 characters, a vocabulary of 58 | 300.4 ms | 293.0 ms | 1.03x | identical | 292.5 ms | 1.03x | 1,500 | 9,900 |
| `min_char_rnn` | large | 300 iterations, 100 hidden units, sequences of 25 characters, a vocabulary of 1,000 | 1626.3 ms | 1623.9 ms | 1.00x | identical | 1582.4 ms | 1.03x | 1,500 | 9,900 |
| `trimesh_mesh` | small | 5,120 faces, 12,500 surface samples | 11.3 ms | 9.4 ms | 1.20x | identical | 9.5 ms | 1.20x | 39 | 70 |
| `trimesh_mesh` | large | 81,920 faces, 200,000 surface samples | 179.6 ms | 141.0 ms | 1.27x | identical | 141.0 ms | 1.27x | 49 | 71 |
| `sklearn_metrics` | small | 50,000 binary predictions, 15,000 predictions over 10 classes | 50.9 ms | 20.2 ms | 2.52x | identical | 20.2 ms | 2.52x | 170 | 162 |
| `sklearn_metrics` | large | 1,000,000 binary predictions, 300,000 predictions over 10 classes | 994.6 ms | 285.8 ms | 3.48x | identical | 286.0 ms | 3.48x | 174 | 162 |
| `empyrical_stats` | small | 60 series of 5,040 daily returns | 1274.0 ms | 361.4 ms | 3.52x | identical | 357.0 ms | 3.57x | 1,740 | 1,740 |
| `empyrical_stats` | large | 6 series of 50,400 daily returns | 1303.5 ms | 331.7 ms | 3.93x | identical | 327.3 ms | 3.98x | 162 | 186 |

The median of the 7 rounds, to show the noise, and the runs in each round:

| Application | Size | Runs per round | NumPy, best | NumPy, median | `install()`, best | `install()`, median |
|---|---|---|---|---|---|---|
| `mlp_train` | small | 3 | 32.5 ms | 33.4 ms | 26.1 ms | 26.9 ms |
| `mlp_train` | large | 1 | 311.5 ms | 314.9 ms | 141.4 ms | 143.6 ms |
| `min_char_rnn` | small | 1 | 300.4 ms | 300.8 ms | 293.0 ms | 295.9 ms |
| `min_char_rnn` | large | 1 | 1626.3 ms | 1653.0 ms | 1623.9 ms | 1667.0 ms |
| `trimesh_mesh` | small | 7 | 11.3 ms | 11.5 ms | 9.4 ms | 9.6 ms |
| `trimesh_mesh` | large | 1 | 179.6 ms | 185.3 ms | 141.0 ms | 143.8 ms |
| `sklearn_metrics` | small | 1 | 50.9 ms | 52.6 ms | 20.2 ms | 21.2 ms |
| `sklearn_metrics` | large | 1 | 994.6 ms | 1003.2 ms | 285.8 ms | 287.5 ms |
| `empyrical_stats` | small | 1 | 1274.0 ms | 1283.4 ms | 361.4 ms | 364.0 ms |
| `empyrical_stats` | large | 1 | 1303.5 ms | 1309.5 ms | 331.7 ms | 337.0 ms |

## `mlp_train`

Training a two-layer perceptron by mini-batch gradient descent. Source: written for this study.

### Small: 12,000 examples of 196 float64 inputs, 64 hidden units, batches of 128, 2 epochs

#### Calls in one run

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `np.argmax` | float64 | 2 | |
| `x[mask] = v` | float64 | 188 | |
| `x.astype(float32)` | | | 2 |
| `np.sum` | | | 940 |

#### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 2.7% | 2.7% |
| NumPy | 60.2% | 40.7% |
| BLAS | 16.6% | 29.4% |
| Maths library | 6.2% | 8.0% |
| hotkernels | 0.0% | 0.9% |
| Memory and system | 14.4% | 18.3% |
| Other compiled code | 0.0% | 0.0% |

The largest functions on plain NumPy (3,564 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 23.6% | NumPy | `array_assign_boolean_subscript` | `_multiarray_umath.cpython-312-darwin.so` |
| 16.6% | BLAS | `(functions without symbols)` | `libBLAS.dylib` |
| 11.0% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 9.9% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 5.4% | NumPy | `DOUBLE_maximum` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.6% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.0% | Maths library | `exp` | `libsystem_m.dylib` |
| 2.5% | NumPy | `_aligned_strided_to_contig_size8_srcstride0` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.5% | Maths library | `fetestexcept` | `libsystem_m.dylib` |
| 1.5% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 1.4% | NumPy | `DOUBLE_subtract` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.2% | NumPy | `NpyIter_AdvancedNew` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.1% | NumPy | `DOUBLE_square` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.1% | Memory and system | `mach_absolute_time` | `libsystem_kernel.dylib` |
| 1.0% | NumPy | `DOUBLE_pairwise_sum` | `_multiarray_umath.cpython-312-darwin.so` |

The largest after `install()` (3,519 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 29.4% | BLAS | `(functions without symbols)` | `libBLAS.dylib` |
| 13.7% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 10.7% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 6.7% | NumPy | `DOUBLE_maximum` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.6% | Maths library | `exp` | `libsystem_m.dylib` |
| 3.4% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.3% | Maths library | `fetestexcept` | `libsystem_m.dylib` |
| 2.0% | NumPy | `DOUBLE_subtract` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.6% | NumPy | `NpyIter_AdvancedNew` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.5% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 1.2% | NumPy | `DOUBLE_pairwise_sum` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.1% | Memory and system | `mach_absolute_time` | `libsystem_kernel.dylib` |
| 1.1% | NumPy | `DOUBLE_square` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.0% | Memory and system | `_tlv_get_addr` | `libdyld.dylib` |
| 1.0% | NumPy | `npyiter_buffered_reduce_iternext_iters2` | `_multiarray_umath.cpython-312-darwin.so` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 46.8% | 188 | `apps/mlp_train.py:70(loss_and_gradients)` |
| 22.0% | 1 | `apps/mlp_train.py:107(run)` |
| 18.6% | 190 | `apps/mlp_train.py:56(forward)` |
| 7.6% | 1,318 | `<method 'reduce' of 'numpy.ufunc' objects>` |
| 0.7% | 190 | `numpy/_core/_methods.py:115(_mean)` |
| 0.7% | 376 | `numpy/_core/fromnumeric.py:66(_wrapreduction)` |
| 0.6% | 376 | `numpy/_core/fromnumeric.py:2342(sum)` |
| 0.5% | 190 | `numpy/_core/_methods.py:73(_count_reduce_items)` |
| 0.4% | 564 | `<method 'sum' of 'numpy.ndarray' objects>` |
| 0.4% | 376 | `<built-in method numpy.arange>` |

### Large: 12,000 examples of 784 float64 inputs, 256 hidden units, batches of 128, 2 epochs

#### Calls in one run

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `np.argmax` | float64 | 2 | |
| `x[mask] = v` | float64 | 188 | |
| `x += y` | float64 | 188 | |
| `x -= y` | float64 | 188 | |
| `x.astype(float32)` | | | 2 |
| `np.sum` | | | 940 |

#### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 0.6% | 0.7% |
| NumPy | 76.0% | 15.3% |
| BLAS | 19.1% | 52.6% |
| Maths library | 0.5% | 0.8% |
| hotkernels | 0.0% | 10.7% |
| Memory and system | 3.7% | 19.8% |
| Other compiled code | 0.0% | 0.0% |

The largest functions on plain NumPy (3,485 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 54.6% | NumPy | `DOUBLE_subtract` | `_multiarray_umath.cpython-312-darwin.so` |
| 19.1% | BLAS | `(functions without symbols)` | `libBLAS.dylib` |
| 8.6% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 4.7% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 4.4% | NumPy | `DOUBLE_pairwise_sum` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.8% | Memory and system | `__bsdthread_ctl` | `libsystem_kernel.dylib` |
| 1.3% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 1.3% | NumPy | `array_assign_boolean_subscript` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.9% | NumPy | `DOUBLE_square` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.8% | NumPy | `simd_binary_scalar2_less_equal_f64` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.5% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 0.4% | Memory and system | `mach_absolute_time` | `libsystem_kernel.dylib` |
| 0.4% | NumPy | `DOUBLE_maximum` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.3% | NumPy | `random_standard_normal` | `_generator.cpython-312-darwin.so` |
| 0.2% | Maths library | `fetestexcept` | `libsystem_m.dylib` |

The largest after `install()` (3,692 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 52.6% | BLAS | `(functions without symbols)` | `libBLAS.dylib` |
| 11.4% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 7.4% | Memory and system | `__bsdthread_ctl` | `libsystem_kernel.dylib` |
| 5.3% | hotkernels | `subtract_inplace_f64` | `_binding.cpython-312-darwin.so` |
| 3.7% | hotkernels | `add_inplace_f64` | `_binding.cpython-312-darwin.so` |
| 3.6% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.4% | NumPy | `DOUBLE_pairwise_sum` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.2% | NumPy | `simd_binary_scalar2_less_equal_f64` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.6% | NumPy | `DOUBLE_square` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.5% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.4% | hotkernels | `maskfill_f64` | `_binding.cpython-312-darwin.so` |
| 0.9% | NumPy | `DOUBLE_maximum` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.7% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 0.6% | Memory and system | `mach_absolute_time` | `libsystem_kernel.dylib` |
| 0.5% | NumPy | `DOUBLE_subtract` | `_multiarray_umath.cpython-312-darwin.so` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 39.9% | 1 | `apps/mlp_train.py:107(run)` |
| 37.2% | 188 | `apps/mlp_train.py:70(loss_and_gradients)` |
| 17.7% | 190 | `apps/mlp_train.py:56(forward)` |
| 4.2% | 1,318 | `<method 'reduce' of 'numpy.ufunc' objects>` |
| 0.2% | 190 | `numpy/_core/_methods.py:115(_mean)` |
| 0.1% | 190 | `numpy/_core/_methods.py:73(_count_reduce_items)` |
| 0.1% | 564 | `<method 'sum' of 'numpy.ndarray' objects>` |
| 0.1% | 376 | `numpy/_core/fromnumeric.py:66(_wrapreduction)` |
| 0.1% | 376 | `numpy/_core/fromnumeric.py:2342(sum)` |
| 0.1% | 376 | `<built-in method numpy.arange>` |

## `min_char_rnn`

min-char-rnn.py, a character-level recurrent network. Source: https://gist.github.com/karpathy/d4dee566867f8291f086 (Andrej Karpathy), copied unchanged.

### Small: 300 iterations, 100 hidden units, sequences of 25 characters, a vocabulary of 58

#### Calls in one run

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `np.clip` | float64 | 1,500 | |
| `np.cumsum` | | | 600 |
| `x.astype(float32)` | | | 600 |
| `np.sum` | | | 8,100 |
| `np.searchsorted` | | | 600 |

#### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 9.3% | 9.1% |
| NumPy | 41.8% | 42.0% |
| BLAS | 29.0% | 28.4% |
| Maths library | 4.9% | 4.2% |
| hotkernels | 0.0% | 0.8% |
| Memory and system | 15.0% | 15.6% |
| Other compiled code | 0.0% | 0.0% |

The largest functions on plain NumPy (3,628 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 28.9% | BLAS | `(functions without symbols)` | `libBLAS.dylib` |
| 19.1% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 5.5% | Memory and system | `__bzero` | `libsystem_platform.dylib` |
| 3.6% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 3.4% | Memory and system | `_tlv_get_addr` | `libdyld.dylib` |
| 2.4% | Maths library | `fetestexcept` | `libsystem_m.dylib` |
| 1.7% | Memory and system | `_platform_memset` | `libsystem_platform.dylib` |
| 1.4% | NumPy | `PyUFunc_GenericFunctionInternal` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.3% | NumPy | `DOUBLE_clip` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.3% | Memory and system | `mach_absolute_time` | `libsystem_kernel.dylib` |
| 1.2% | Memory and system | `_platform_strcmp$VARIANT$Base` | `libsystem_platform.dylib` |
| 1.2% | NumPy | `PyArray_NewFromDescr_int` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.0% | NumPy | `ufunc_generic_fastcall` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.0% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.0% | Maths library | `exp` | `libsystem_m.dylib` |

The largest after `install()` (3,600 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 28.4% | BLAS | `(functions without symbols)` | `libBLAS.dylib` |
| 18.8% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 7.1% | Memory and system | `__bzero` | `libsystem_platform.dylib` |
| 4.0% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 3.1% | Memory and system | `_tlv_get_addr` | `libdyld.dylib` |
| 2.5% | Maths library | `fetestexcept` | `libsystem_m.dylib` |
| 1.7% | Memory and system | `mach_absolute_time` | `libsystem_kernel.dylib` |
| 1.4% | NumPy | `PyArray_NewFromDescr_int` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.3% | NumPy | `PyUFunc_GenericFunctionInternal` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.2% | Memory and system | `_platform_memset` | `libsystem_platform.dylib` |
| 1.1% | NumPy | `DOUBLE_tanh` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.1% | NumPy | `ufunc_generic_fastcall` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.0% | Memory and system | `_platform_strcmp$VARIANT$Base` | `libsystem_platform.dylib` |
| 1.0% | NumPy | `convert_ufunc_arguments` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.9% | Maths library | `exp` | `libsystem_m.dylib` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 81.1% | 300 | `min_char_rnn/min-char-rnn.py:27(lossFun)` |
| 5.4% | 1 | `min_char_rnn/min-char-rnn.py:1(<module>)` |
| 2.8% | 3 | `min_char_rnn/min-char-rnn.py:63(sample)` |
| 1.9% | 1,500 | `numpy/_core/_methods.py:96(_clip)` |
| 1.4% | 8,100 | `numpy/_core/fromnumeric.py:66(_wrapreduction)` |
| 1.2% | 8,100 | `numpy/_core/fromnumeric.py:2342(sum)` |
| 1.0% | 8,100 | `<method 'reduce' of 'numpy.ufunc' objects>` |
| 0.9% | 61,800 | `numpy/_core/multiarray.py:751(dot)` |
| 0.8% | 7,800 | `<built-in method numpy.array>` |
| 0.7% | 1,805 | `numpy/_core/numeric.py:97(zeros_like)` |

### Large: 300 iterations, 100 hidden units, sequences of 25 characters, a vocabulary of 1,000

#### Calls in one run

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `np.clip` | float64 | 1,500 | |
| `np.cumsum` | | | 600 |
| `x.astype(float32)` | | | 600 |
| `np.sum` | | | 8,100 |
| `np.searchsorted` | | | 600 |

#### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 0.8% | 1.5% |
| NumPy | 39.0% | 38.0% |
| BLAS | 29.5% | 26.3% |
| Maths library | 3.2% | 3.9% |
| hotkernels | 0.0% | 0.5% |
| Memory and system | 27.5% | 29.8% |
| Other compiled code | 0.0% | 0.0% |

The largest functions on plain NumPy (2,377 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 30.8% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 29.5% | BLAS | `(functions without symbols)` | `libBLAS.dylib` |
| 21.8% | Memory and system | `syscall_thread_switch` | `libsystem_kernel.dylib` |
| 2.8% | Maths library | `exp` | `libsystem_m.dylib` |
| 2.6% | Memory and system | `__bzero` | `libsystem_platform.dylib` |
| 2.3% | NumPy | `DOUBLE_clip` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.3% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.0% | NumPy | `DOUBLE_divide` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.0% | NumPy | `DOUBLE_sqrt` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.8% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 0.8% | Memory and system | `_tlv_get_addr` | `libdyld.dylib` |
| 0.5% | NumPy | `DYLD-STUB$$exp` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.5% | Memory and system | `_platform_memset` | `libsystem_platform.dylib` |
| 0.5% | NumPy | `_aligned_strided_to_contig_size8_srcstride0` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.4% | NumPy | `DOUBLE_exp` | `_multiarray_umath.cpython-312-darwin.so` |

The largest after `install()` (2,330 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 31.7% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 26.3% | BLAS | `(functions without symbols)` | `libBLAS.dylib` |
| 22.2% | Memory and system | `syscall_thread_switch` | `libsystem_kernel.dylib` |
| 3.6% | Memory and system | `__bzero` | `libsystem_platform.dylib` |
| 3.5% | Maths library | `exp` | `libsystem_m.dylib` |
| 2.1% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.4% | NumPy | `DOUBLE_sqrt` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.3% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 0.9% | NumPy | `DOUBLE_divide` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.8% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 0.6% | NumPy | `_aligned_strided_to_contig_size8_srcstride0` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.5% | NumPy | `DOUBLE_exp` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.5% | Memory and system | `mach_absolute_time` | `libsystem_kernel.dylib` |
| 0.5% | hotkernels | `clip_f64` | `_binding.cpython-312-darwin.so` |
| 0.4% | Memory and system | `_platform_memset` | `libsystem_platform.dylib` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 79.3% | 300 | `min_char_rnn/min-char-rnn.py:27(lossFun)` |
| 11.8% | 1 | `min_char_rnn/min-char-rnn.py:1(<module>)` |
| 3.2% | 3 | `min_char_rnn/min-char-rnn.py:63(sample)` |
| 2.7% | 1,500 | `numpy/_core/_methods.py:96(_clip)` |
| 0.6% | 1,805 | `numpy/_core/numeric.py:97(zeros_like)` |
| 0.4% | 7,800 | `<built-in method numpy.array>` |
| 0.4% | 8,100 | `<method 'reduce' of 'numpy.ufunc' objects>` |
| 0.3% | 8,100 | `numpy/_core/fromnumeric.py:66(_wrapreduction)` |
| 0.3% | 9,911 | `<built-in method numpy.zeros>` |
| 0.3% | 8,100 | `numpy/_core/fromnumeric.py:2342(sum)` |

## `trimesh_mesh`

Processing a triangle mesh: build, measure, sample, section, subdivide. Source: trimesh, called as a library.

### Small: 5,120 faces, 12,500 surface samples

#### Calls in one run

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `np.clip` | float64 | 3 | |
| `x[mask]` | float64 | 15 | |
| `np.nonzero` | bool | 9 | |
| `np.cumsum` | bool | 4 | |
| `np.sum` | uint8 | 1 | |
| `np.bincount` | int64 | 1 | |
| `np.searchsorted` | float64 | 1 | |
| `m.any(axis=1)` | bool | 1 | |
| `np.argsort, stable` | uint64 | 3 | |
| `np.round` | float64 | 1 | |
| `x[mask]` | | | 19 |
| `x[mask] = v` | | | 9 |
| `np.cumsum` | | | 1 |
| `x.astype(float32)` | | | 19 |
| `np.sum` | | | 11 |
| `np.unique` | | | 6 |
| `np.argsort, stable` | | | 4 |
| `np.round` | | | 1 |

#### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 10.0% | 11.8% |
| NumPy | 75.4% | 66.6% |
| BLAS | 0.2% | 0.2% |
| Maths library | 3.4% | 4.4% |
| hotkernels | 0.0% | 4.3% |
| Memory and system | 10.2% | 11.5% |
| Other compiled code | 0.7% | 1.1% |

The largest functions on plain NumPy (3,800 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 8.4% | NumPy | `binsearch<npy::floating_point_type<double, (NPY_TYPES)12>, (side_t)0>(` | `_multiarray_umath.cpython-312-darwin.so` |
| 8.0% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 7.9% | NumPy | `aquicksort_<npy::integral_type<unsigned long, (NPY_TYPES)8>, unsigned ` | `_multiarray_umath.cpython-312-darwin.so` |
| 6.1% | Python interpreter | `PyBlake2_blake2b_compress` | `libpython3.12.dylib` |
| 4.8% | NumPy | `_contig_to_contig` | `_multiarray_umath.cpython-312-darwin.so` |
| 4.3% | NumPy | `npyiter_buffered_iternext` | `_multiarray_umath.cpython-312-darwin.so` |
| 4.3% | NumPy | `amerge_at_<npy::integral_type<unsigned long, (NPY_TYPES)8>, unsigned l` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.3% | NumPy | `mapiter_get` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.5% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 2.4% | NumPy | `atimsort_<npy::integral_type<unsigned long, (NPY_TYPES)8>, unsigned lo` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.1% | NumPy | `PyArray_TransferNDimToStrided` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.0% | Maths library | `pow` | `libsystem_m.dylib` |
| 1.9% | NumPy | `aquicksort_<npy::integral_type<long, (NPY_TYPES)7>, long, false>(long*` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.8% | NumPy | `npyiter_goto_iterindex` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.8% | NumPy | `prepare_index_noarray` | `_multiarray_umath.cpython-312-darwin.so` |

The largest after `install()` (3,753 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 9.4% | NumPy | `aquicksort_<npy::integral_type<unsigned long, (NPY_TYPES)8>, unsigned ` | `_multiarray_umath.cpython-312-darwin.so` |
| 8.4% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 6.7% | Python interpreter | `PyBlake2_blake2b_compress` | `libpython3.12.dylib` |
| 5.6% | NumPy | `_contig_to_contig` | `_multiarray_umath.cpython-312-darwin.so` |
| 4.8% | NumPy | `npyiter_buffered_iternext` | `_multiarray_umath.cpython-312-darwin.so` |
| 4.0% | NumPy | `mapiter_get` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.1% | hotkernels | `argsort_u64` | `_binding.cpython-312-darwin.so` |
| 2.9% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 2.6% | Maths library | `pow` | `libsystem_m.dylib` |
| 2.5% | NumPy | `PyArray_TransferNDimToStrided` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.5% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.5% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.4% | NumPy | `prepare_index_noarray` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.1% | NumPy | `aquicksort_<npy::integral_type<long, (NPY_TYPES)7>, long, false>(long*` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.7% | NumPy | `VOID_compare` | `_multiarray_umath.cpython-312-darwin.so` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 18.5% | 9 | `<method 'argsort' of 'numpy.ndarray' objects>` |
| 7.9% | 1 | `trimesh/triangles.py:217(mass_properties)` |
| 7.8% | 71 | `<method 'reduce' of 'numpy.ufunc' objects>` |
| 7.0% | 1 | `trimesh/sample.py:20(sample_surface)` |
| 6.7% | 1 | `<method 'searchsorted' of 'numpy.ndarray' objects>` |
| 4.9% | 52 | `trimesh/caching.py:44(hash_fallback)` |
| 3.4% | 10 | `<method 'sort' of 'numpy.ndarray' objects>` |
| 2.2% | 1 | `trimesh/remesh.py:17(subdivide)` |
| 2.2% | 6 | `numpy/lib/_arraysetops_impl.py:350(_unique1d)` |
| 2.0% | 1 | `trimesh/triangles.py:106(angles)` |

### Large: 81,920 faces, 200,000 surface samples

#### Calls in one run

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `np.clip` | float64 | 3 | |
| `x[mask]` | float64 | 15 | |
| `np.nonzero` | bool | 9 | |
| `np.cumsum` | bool | 4 | |
| `np.sum` | uint8 | 1 | |
| `np.bincount` | int64 | 1 | |
| `np.searchsorted` | float64 | 1 | |
| `m.any(axis=1)` | bool | 1 | |
| `np.argsort, stable` | uint64 | 3 | |
| `x += y` | float64 | 10 | |
| `x -= y` | float64 | 1 | |
| `x[mask]` | | | 19 |
| `x[mask] = v` | | | 9 |
| `np.cumsum` | | | 1 |
| `x.astype(float32)` | | | 19 |
| `np.sum` | | | 11 |
| `np.unique` | | | 6 |
| `np.argsort, stable` | | | 4 |
| `np.round` | | | 2 |

#### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 5.0% | 6.6% |
| NumPy | 76.2% | 71.6% |
| BLAS | 0.4% | 1.1% |
| Maths library | 3.2% | 3.8% |
| hotkernels | 0.0% | 5.4% |
| Memory and system | 14.8% | 10.6% |
| Other compiled code | 0.4% | 0.8% |

The largest functions on plain NumPy (3,907 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 9.9% | NumPy | `binsearch<npy::floating_point_type<double, (NPY_TYPES)12>, (side_t)0>(` | `_multiarray_umath.cpython-312-darwin.so` |
| 7.6% | NumPy | `aquicksort_<npy::integral_type<unsigned long, (NPY_TYPES)8>, unsigned ` | `_multiarray_umath.cpython-312-darwin.so` |
| 6.4% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 5.3% | NumPy | `_contig_to_contig` | `_multiarray_umath.cpython-312-darwin.so` |
| 5.3% | NumPy | `amerge_at_<npy::integral_type<unsigned long, (NPY_TYPES)8>, unsigned l` | `_multiarray_umath.cpython-312-darwin.so` |
| 5.3% | Memory and system | `syscall_thread_switch` | `libsystem_kernel.dylib` |
| 4.7% | Python interpreter | `PyBlake2_blake2b_compress` | `libpython3.12.dylib` |
| 3.7% | NumPy | `npyiter_buffered_iternext` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.3% | NumPy | `mapiter_get` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.5% | Memory and system | `_dispatch_apply_invoke3` | `libdispatch.dylib` |
| 2.2% | Maths library | `pow` | `libsystem_m.dylib` |
| 2.1% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.0% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.0% | NumPy | `aquicksort_<npy::integral_type<long, (NPY_TYPES)7>, long, false>(long*` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.9% | NumPy | `atimsort_<npy::integral_type<unsigned long, (NPY_TYPES)8>, unsigned lo` | `_multiarray_umath.cpython-312-darwin.so` |

The largest after `install()` (3,489 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 10.4% | NumPy | `aquicksort_<npy::integral_type<unsigned long, (NPY_TYPES)8>, unsigned ` | `_multiarray_umath.cpython-312-darwin.so` |
| 8.9% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 7.1% | NumPy | `_contig_to_contig` | `_multiarray_umath.cpython-312-darwin.so` |
| 6.3% | Python interpreter | `PyBlake2_blake2b_compress` | `libpython3.12.dylib` |
| 5.0% | NumPy | `npyiter_buffered_iternext` | `_multiarray_umath.cpython-312-darwin.so` |
| 4.5% | NumPy | `mapiter_get` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.0% | NumPy | `aquicksort_<npy::integral_type<long, (NPY_TYPES)7>, long, false>(long*` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.0% | hotkernels | `argsort_u64` | `_binding.cpython-312-darwin.so` |
| 2.8% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.6% | NumPy | `VOID_compare` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.4% | NumPy | `PyArray_TransferNDimToStrided` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.4% | Maths library | `pow` | `libsystem_m.dylib` |
| 2.3% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.1% | NumPy | `npyiter_goto_iterindex` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.9% | NumPy | `prepare_index_noarray` | `_multiarray_umath.cpython-312-darwin.so` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 27.6% | 9 | `<method 'argsort' of 'numpy.ndarray' objects>` |
| 10.0% | 1 | `<method 'searchsorted' of 'numpy.ndarray' objects>` |
| 8.4% | 1 | `trimesh/triangles.py:217(mass_properties)` |
| 8.1% | 71 | `<method 'reduce' of 'numpy.ufunc' objects>` |
| 7.6% | 1 | `trimesh/sample.py:20(sample_surface)` |
| 4.9% | 52 | `trimesh/caching.py:44(hash_fallback)` |
| 3.2% | 10 | `<method 'sort' of 'numpy.ndarray' objects>` |
| 2.3% | 1 | `trimesh/triangles.py:106(angles)` |
| 2.1% | 6 | `numpy/lib/_arraysetops_impl.py:350(_unique1d)` |
| 2.0% | 1 | `trimesh/remesh.py:17(subdivide)` |

## `sklearn_metrics`

Scoring a classifier: ROC, precision-recall, calibration, confusion matrix. Source: scikit-learn (sklearn.metrics, sklearn.calibration), called as a library.

### Small: 50,000 binary predictions, 15,000 predictions over 10 classes

#### Calls in one run

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `np.argmax` | float64 | 1 | |
| `x[mask]` | float64 | 53 | |
| `x[mask] = v` | float64 | 24 | |
| `np.nonzero` | bool | 2 | |
| `np.bincount` | int64 | 16 | |
| `np.searchsorted` | float64 | 1 | |
| `np.where(c, a, b)` | float64 | 2 | |
| `np.unique` | int64 | 67 | |
| `np.argsort, stable` | float64 | 4 | |
| `x[mask]` | | | 4 |
| `x[mask] = v` | | | 23 |
| `np.nonzero` | | | 4 |
| `np.cumsum` | | | 7 |
| `x.astype(float32)` | | | 49 |
| `np.sum` | | | 47 |
| `np.bincount` | | | 2 |
| `np.isin` | | | 3 |
| `np.searchsorted` | | | 19 |
| `np.where(c, a, b)` | | | 2 |
| `m.any(axis=1)` | | | 1 |
| `np.argsort, stable` | | | 1 |

#### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 9.5% | 22.3% |
| NumPy | 84.9% | 48.5% |
| BLAS | 0.0% | 0.0% |
| Maths library | 1.2% | 3.2% |
| hotkernels | 0.0% | 17.3% |
| Memory and system | 4.0% | 7.3% |
| Other compiled code | 0.3% | 1.3% |

The largest functions on plain NumPy (4,002 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 25.5% | NumPy | `amerge_at_<npy::floating_point_type<double, (NPY_TYPES)12>, double, fa` | `_multiarray_umath.cpython-312-darwin.so` |
| 16.0% | NumPy | `npy_fnv1a` | `_multiarray_umath.cpython-312-darwin.so` |
| 11.1% | NumPy | `_object* unique_numeric<long, &unsigned long hash_integer<long>(long c` | `_multiarray_umath.cpython-312-darwin.so` |
| 10.6% | NumPy | `atimsort_<npy::floating_point_type<double, (NPY_TYPES)12>, double, fal` | `_multiarray_umath.cpython-312-darwin.so` |
| 6.3% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 2.2% | NumPy | `arr_bincount` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.0% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.9% | Memory and system | `_tlv_get_addr` | `libdyld.dylib` |
| 1.7% | NumPy | `binsearch<npy::floating_point_type<double, (NPY_TYPES)12>, (side_t)0>(` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.5% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 1.5% | NumPy | `mapiter_trivial_get` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.2% | NumPy | `binsearch<npy::integral_type<long, (NPY_TYPES)7>, (side_t)0>(char cons` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.2% | Maths library | `log` | `libsystem_m.dylib` |
| 1.0% | NumPy | `OUTLINED_FUNCTION_148` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.9% | NumPy | `DOUBLE_pairwise_sum` | `_multiarray_umath.cpython-312-darwin.so` |

The largest after `install()` (3,770 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 14.4% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 8.0% | NumPy | `atimsort_<npy::floating_point_type<double, (NPY_TYPES)12>, double, fal` | `_multiarray_umath.cpython-312-darwin.so` |
| 7.1% | hotkernels | `argsort_f64` | `_binding.cpython-312-darwin.so` |
| 4.8% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 4.8% | Memory and system | `_tlv_get_addr` | `libdyld.dylib` |
| 4.6% | hotkernels | `searchsorted_f64` | `_binding.cpython-312-darwin.so` |
| 4.3% | NumPy | `mapiter_trivial_get` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.3% | NumPy | `binsearch<npy::integral_type<long, (NPY_TYPES)7>, (side_t)0>(char cons` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.2% | Maths library | `log` | `libsystem_m.dylib` |
| 2.5% | hotkernels | `unique_values` | `_binding.cpython-312-darwin.so` |
| 2.3% | NumPy | `DOUBLE_pairwise_sum` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.7% | NumPy | `DOUBLE_subtract` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.5% | NumPy | `DOUBLE_logical_or` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.4% | hotkernels | `unique_i64` | `_binding.cpython-312-darwin.so` |
| 1.3% | NumPy | `LONG_add` | `_multiarray_umath.cpython-312-darwin.so` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 28.7% | 5 | `<method 'argsort' of 'numpy.ndarray' objects>` |
| 20.0% | 67 | `<built-in method numpy._core._multiarray_umath._unique_hash>` |
| 2.7% | 183 | `<method 'reduce' of 'numpy.ufunc' objects>` |
| 2.4% | 20 | `<method 'searchsorted' of 'numpy.ndarray' objects>` |
| 1.9% | 2 | `sklearn/utils/_array_api.py:885(_xlogy)` |
| 1.5% | 7 | `<method 'cumsum' of 'numpy.ndarray' objects>` |
| 1.2% | 2,124 | `python3.12/inspect.py:1828(_shadowed_dict)` |
| 1.2% | 177 | `sklearn/utils/validation.py:733(check_array)` |
| 1.1% | 4 | `sklearn/metrics/_ranking.py:924(confusion_matrix_at_thresholds)` |
| 1.1% | 6,043 | `<built-in method builtins.isinstance>` |

### Large: 1,000,000 binary predictions, 300,000 predictions over 10 classes

#### Calls in one run

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `np.argmax` | float64 | 1 | |
| `x[mask]` | float64 | 53 | |
| `x[mask] = v` | float64 | 24 | |
| `np.nonzero` | bool | 2 | |
| `np.bincount` | int64 | 16 | |
| `np.searchsorted` | float64 | 1 | |
| `np.where(c, a, b)` | float64 | 2 | |
| `np.unique` | int64 | 67 | |
| `np.argsort, stable` | float64 | 4 | |
| `x -= y` | float64 | 4 | |
| `x[mask]` | | | 4 |
| `x[mask] = v` | | | 23 |
| `np.nonzero` | | | 4 |
| `np.cumsum` | | | 7 |
| `x.astype(float32)` | | | 49 |
| `np.sum` | | | 47 |
| `np.bincount` | | | 2 |
| `np.isin` | | | 3 |
| `np.searchsorted` | | | 19 |
| `np.where(c, a, b)` | | | 2 |
| `m.any(axis=1)` | | | 1 |
| `np.argsort, stable` | | | 1 |

#### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 0.5% | 2.1% |
| NumPy | 96.0% | 65.7% |
| BLAS | 0.0% | 0.0% |
| Maths library | 1.1% | 4.6% |
| hotkernels | 0.0% | 23.8% |
| Memory and system | 2.1% | 2.4% |
| Other compiled code | 0.3% | 1.5% |

The largest functions on plain NumPy (4,254 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 36.5% | NumPy | `amerge_at_<npy::floating_point_type<double, (NPY_TYPES)12>, double, fa` | `_multiarray_umath.cpython-312-darwin.so` |
| 14.6% | NumPy | `npy_fnv1a` | `_multiarray_umath.cpython-312-darwin.so` |
| 11.4% | NumPy | `atimsort_<npy::floating_point_type<double, (NPY_TYPES)12>, double, fal` | `_multiarray_umath.cpython-312-darwin.so` |
| 10.3% | NumPy | `_object* unique_numeric<long, &unsigned long hash_integer<long>(long c` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.2% | NumPy | `mapiter_trivial_get` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.1% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.0% | NumPy | `arr_bincount` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.0% | NumPy | `binsearch<npy::floating_point_type<double, (NPY_TYPES)12>, (side_t)0>(` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.0% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 1.2% | NumPy | `binsearch<npy::integral_type<long, (NPY_TYPES)7>, (side_t)0>(char cons` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.1% | NumPy | `OUTLINED_FUNCTION_148` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.1% | Maths library | `log` | `libsystem_m.dylib` |
| 1.0% | NumPy | `OUTLINED_FUNCTION_152` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.8% | NumPy | `OUTLINED_FUNCTION_71` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.7% | NumPy | `DOUBLE_subtract` | `_multiarray_umath.cpython-312-darwin.so` |

The largest after `install()` (4,068 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 10.4% | NumPy | `atimsort_<npy::floating_point_type<double, (NPY_TYPES)12>, double, fal` | `_multiarray_umath.cpython-312-darwin.so` |
| 9.6% | hotkernels | `argsort_f64` | `_binding.cpython-312-darwin.so` |
| 7.2% | NumPy | `mapiter_trivial_get` | `_multiarray_umath.cpython-312-darwin.so` |
| 6.7% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 5.6% | hotkernels | `searchsorted_f64` | `_binding.cpython-312-darwin.so` |
| 5.3% | NumPy | `binsearch<npy::integral_type<long, (NPY_TYPES)7>, (side_t)0>(char cons` | `_multiarray_umath.cpython-312-darwin.so` |
| 4.5% | Maths library | `log` | `libsystem_m.dylib` |
| 3.4% | hotkernels | `unique_values` | `_binding.cpython-312-darwin.so` |
| 3.1% | NumPy | `DOUBLE_pairwise_sum` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.0% | hotkernels | `unique_i64` | `_binding.cpython-312-darwin.so` |
| 1.9% | NumPy | `DOUBLE_subtract` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.9% | NumPy | `npyiter_buffered_reduce_iternext_iters2` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.8% | NumPy | `DOUBLE_logical_or` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.8% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 1.8% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 49.6% | 5 | `<method 'argsort' of 'numpy.ndarray' objects>` |
| 24.9% | 67 | `<built-in method numpy._core._multiarray_umath._unique_hash>` |
| 3.2% | 20 | `<method 'searchsorted' of 'numpy.ndarray' objects>` |
| 3.2% | 183 | `<method 'reduce' of 'numpy.ufunc' objects>` |
| 2.4% | 2 | `sklearn/utils/_array_api.py:885(_xlogy)` |
| 1.9% | 7 | `<method 'cumsum' of 'numpy.ndarray' objects>` |
| 1.3% | 15 | `sklearn/utils/_array_api.py:1329(_bincount)` |
| 1.2% | 4 | `sklearn/metrics/_ranking.py:878(_sort_inputs_and_compute_classification_thresholds)` |
| 1.1% | 4 | `sklearn/metrics/_ranking.py:924(confusion_matrix_at_thresholds)` |
| 0.9% | 5 | `sklearn/metrics/_classification.py:626(multilabel_confusion_matrix)` |

## `empyrical_stats`

Risk and return statistics of daily return series. Source: empyrical-reloaded, called as a library.

### Small: 60 series of 5,040 daily returns

#### Calls in one run

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `np.clip` | float64 | 120 | |
| `x[mask]` | float64 | 420 | |
| `x[mask] = v` | float64 | 240 | |
| `np.where(c, a, b)` | float64 | 120 | |
| `fmax.accumulate` | float64 | 120 | |
| `np.cumprod axis=0` | float64 | 60 | |
| `fmax.acc axis=0` | float64 | 60 | |
| `np.unique` | int64 | 120 | |
| `bn.nanmean` | float64 | 240 | |
| `bn.nanstd` | float64 | 60 | |
| `bn.nanmin` | float64 | 120 | |
| `bn.nanmax` | float64 | 60 | |
| `x[mask]` | | | 300 |
| `x[mask] = v` | | | 180 |
| `np.cumsum` | | | 60 |
| `x.astype(float32)` | | | 360 |
| `np.where(c, a, b)` | | | 120 |
| `np.unique` | | | 60 |
| `bn.nanmean` | | | 300 |
| `bn.nanstd` | | | 120 |
| `bn.nanmin` | | | 180 |
| `bn.nanmax` | | | 60 |

#### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 1.9% | 8.9% |
| NumPy | 49.7% | 33.0% |
| BLAS | 0.0% | 0.0% |
| Maths library | 2.1% | 0.6% |
| hotkernels | 0.0% | 47.7% |
| Memory and system | 0.8% | 8.4% |
| Other compiled code | 45.5% | 1.5% |

The largest functions on plain NumPy (4,177 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 22.8% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 19.8% | NumPy | `DOUBLE_fmax` | `_multiarray_umath.cpython-312-darwin.so` |
| 17.5% | Other compiled code | `nanstd_one_float64` | `reduce.cpython-312-darwin.so` |
| 14.3% | Other compiled code | `nanmin_one_float64` | `reduce.cpython-312-darwin.so` |
| 7.3% | Other compiled code | `nanmean_one_float64` | `reduce.cpython-312-darwin.so` |
| 6.1% | Other compiled code | `nanmax_one_float64` | `reduce.cpython-312-darwin.so` |
| 2.0% | NumPy | `DOUBLE_subtract` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.9% | NumPy | `DOUBLE_divide` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.9% | Maths library | `fetestexcept` | `libsystem_m.dylib` |
| 1.6% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 1.0% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.6% | NumPy | `DOUBLE_isnan` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.6% | Memory and system | `_tlv_get_addr` | `libdyld.dylib` |
| 0.6% | NumPy | `introselect_<npy::floating_point_type<double, (NPY_TYPES)12>, false, d` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.5% | NumPy | `array_boolean_subscript` | `_multiarray_umath.cpython-312-darwin.so` |

The largest after `install()` (3,819 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 13.0% | hotkernels | `nanmin_cols_f64` | `_binding.cpython-312-darwin.so` |
| 9.2% | hotkernels | `nanstd_cols_f64` | `_binding.cpython-312-darwin.so` |
| 8.7% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 7.8% | NumPy | `DOUBLE_divide` | `_multiarray_umath.cpython-312-darwin.so` |
| 7.3% | NumPy | `DOUBLE_subtract` | `_multiarray_umath.cpython-312-darwin.so` |
| 6.6% | hotkernels | `fmax_row` | `_binding.cpython-312-darwin.so` |
| 5.9% | hotkernels | `nanmean_cols_f64` | `_binding.cpython-312-darwin.so` |
| 5.7% | hotkernels | `multiply_row` | `_binding.cpython-312-darwin.so` |
| 5.4% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 4.9% | hotkernels | `nanmax_cols_f64` | `_binding.cpython-312-darwin.so` |
| 4.8% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 3.4% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.4% | hotkernels | `fmax_accumulate_left_to_numpy` | `_binding.cpython-312-darwin.so` |
| 2.4% | Memory and system | `_tlv_get_addr` | `libdyld.dylib` |
| 2.1% | NumPy | `introselect_<npy::floating_point_type<double, (NPY_TYPES)12>, false, d` | `_multiarray_umath.cpython-312-darwin.so` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 18.7% | 180 | `<method 'accumulate' of 'numpy.ufunc' objects>` |
| 18.6% | 240 | `<method 'cumprod' of 'numpy.ndarray' objects>` |
| 16.9% | 180 | `<built-in method bottleneck.reduce.nanstd>` |
| 12.6% | 300 | `<built-in method bottleneck.reduce.nanmin>` |
| 7.1% | 540 | `<built-in method bottleneck.reduce.nanmean>` |
| 5.8% | 120 | `<built-in method bottleneck.reduce.nanmax>` |
| 3.9% | 180 | `empyrical/stats.py:263(drawdown_series)` |
| 3.7% | 240 | `empyrical/stats.py:123(cum_returns)` |
| 0.7% | 120 | `<built-in method builtins.sum>` |
| 0.5% | 64,620 | `<built-in method builtins.isinstance>` |

### Large: 6 series of 50,400 daily returns

#### Calls in one run

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `np.clip` | float64 | 12 | |
| `x[mask]` | float64 | 42 | |
| `x[mask] = v` | float64 | 24 | |
| `np.where(c, a, b)` | float64 | 12 | |
| `fmax.accumulate` | float64 | 12 | |
| `np.cumprod axis=0` | float64 | 6 | |
| `fmax.acc axis=0` | float64 | 6 | |
| `bn.nanmean` | float64 | 24 | |
| `bn.nanstd` | float64 | 6 | |
| `bn.nanmin` | float64 | 12 | |
| `bn.nanmax` | float64 | 6 | |
| `x[mask]` | | | 30 |
| `x[mask] = v` | | | 18 |
| `np.cumsum` | | | 6 |
| `x.astype(float32)` | | | 36 |
| `np.where(c, a, b)` | | | 12 |
| `np.unique` | | | 18 |
| `bn.nanmean` | | | 30 |
| `bn.nanstd` | | | 12 |
| `bn.nanmin` | | | 18 |
| `bn.nanmax` | | | 6 |

#### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 0.5% | 2.6% |
| NumPy | 51.5% | 34.7% |
| BLAS | 0.0% | 0.1% |
| Maths library | 2.7% | 0.4% |
| hotkernels | 0.0% | 54.9% |
| Memory and system | 0.4% | 5.4% |
| Other compiled code | 44.9% | 1.9% |

The largest functions on plain NumPy (4,342 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 23.4% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 20.7% | NumPy | `DOUBLE_fmax` | `_multiarray_umath.cpython-312-darwin.so` |
| 17.7% | Other compiled code | `nanstd_one_float64` | `reduce.cpython-312-darwin.so` |
| 13.4% | Other compiled code | `nanmin_one_float64` | `reduce.cpython-312-darwin.so` |
| 7.3% | Other compiled code | `nanmean_one_float64` | `reduce.cpython-312-darwin.so` |
| 6.2% | Other compiled code | `nanmax_one_float64` | `reduce.cpython-312-darwin.so` |
| 2.7% | Maths library | `fetestexcept` | `libsystem_m.dylib` |
| 2.0% | NumPy | `DOUBLE_divide` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.0% | NumPy | `DOUBLE_subtract` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.2% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.6% | NumPy | `DOUBLE_isnan` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.5% | NumPy | `introselect_<npy::floating_point_type<double, (NPY_TYPES)12>, false, d` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.4% | NumPy | `array_boolean_subscript` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.3% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 0.3% | Memory and system | `_tlv_get_addr` | `libdyld.dylib` |

The largest after `install()` (4,120 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 15.6% | hotkernels | `nanmin_cols_f64` | `_binding.cpython-312-darwin.so` |
| 10.0% | hotkernels | `nanstd_cols_f64` | `_binding.cpython-312-darwin.so` |
| 9.6% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 7.7% | hotkernels | `fmax_row` | `_binding.cpython-312-darwin.so` |
| 7.5% | NumPy | `DOUBLE_subtract` | `_multiarray_umath.cpython-312-darwin.so` |
| 7.3% | NumPy | `DOUBLE_divide` | `_multiarray_umath.cpython-312-darwin.so` |
| 6.6% | hotkernels | `nanmean_cols_f64` | `_binding.cpython-312-darwin.so` |
| 5.9% | hotkernels | `multiply_row` | `_binding.cpython-312-darwin.so` |
| 5.9% | hotkernels | `nanmax_cols_f64` | `_binding.cpython-312-darwin.so` |
| 4.6% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 4.3% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 3.1% | hotkernels | `fmax_accumulate_left_to_numpy` | `_binding.cpython-312-darwin.so` |
| 2.5% | NumPy | `DOUBLE_isnan` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.1% | NumPy | `introselect_<npy::floating_point_type<double, (NPY_TYPES)12>, false, d` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.1% | Memory and system | `_tlv_get_addr` | `libdyld.dylib` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 22.3% | 18 | `<method 'accumulate' of 'numpy.ufunc' objects>` |
| 20.4% | 24 | `<method 'cumprod' of 'numpy.ndarray' objects>` |
| 17.9% | 18 | `<built-in method bottleneck.reduce.nanstd>` |
| 14.1% | 30 | `<built-in method bottleneck.reduce.nanmin>` |
| 7.6% | 54 | `<built-in method bottleneck.reduce.nanmean>` |
| 6.2% | 12 | `<built-in method bottleneck.reduce.nanmax>` |
| 3.9% | 24 | `empyrical/stats.py:123(cum_returns)` |
| 3.9% | 18 | `empyrical/stats.py:263(drawdown_series)` |
| 0.8% | 12 | `<built-in method builtins.sum>` |
| 0.5% | 18 | `<method 'partition' of 'numpy.ndarray' objects>` |
