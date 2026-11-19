# Applications with and without hotkernels: Apple M1 Pro

Written by `uv run --group apps delivery apps --profile --record`. Do not edit. What the figures mean is in [docs/applications-findings.md](../../docs/applications-findings.md).

- **Machine:** Apple M1 Pro, macOS-27.2.1-arm64-arm-64bit
- **Software:** Python 3.12.7, NumPy 2.5.5, trimesh 5.2.1, scikit-learn 1.9.1, empyrical-reloaded 0.7.3, pandas 3.1.1, scipy 1.18.1, hotkernels at 6be1984
- **Load average when the run started:** 18.54, a busy machine: the times are rough
- **Method:** 7 alternating runs of each side, the best of each. Results compared bit for bit on every round.

## Times

| Application | Size | NumPy | `install()` | Faster by | Results | `install(inexact=True)` | Faster by | Kernel calls | Left to NumPy |
|---|---|---|---|---|---|---|---|---|---|
| `mlp_train` | 12,000 examples of 784 float64 inputs, 256 hidden units, batches of 128, 2 epochs | 226.3 ms | 208.7 ms | 1.08x | identical | 203.4 ms | 1.11x | 188 | 4,942 |
| `min_char_rnn` | 300 iterations, 100 hidden units, sequences of 25 characters | 287.7 ms | 289.5 ms | 0.99x | identical | 287.7 ms | 1.00x | 0 | 11,400 |
| `trimesh_mesh` | 81,920 faces, 200,000 surface samples | 180.2 ms | 160.6 ms | 1.12x | identical | 162.4 ms | 1.11x | 35 | 61 |
| `sklearn_metrics` | 1,000,000 binary predictions, 300,000 predictions over 10 classes | 1005.8 ms | 978.6 ms | 1.03x | identical | 977.5 ms | 1.03x | 98 | 300,161 |
| `empyrical_stats` | 60 series of 5,040 daily returns | 1331.1 ms | 1305.2 ms | 1.02x | identical | 1327.7 ms | 1.00x | 1,020 | 1,080 |

The median of the 7 runs, to show the noise:

| Application | NumPy, best | NumPy, median | `install()`, best | `install()`, median |
|---|---|---|---|---|
| `mlp_train` | 226.3 ms | 239.1 ms | 208.7 ms | 221.7 ms |
| `min_char_rnn` | 287.7 ms | 296.5 ms | 289.5 ms | 292.9 ms |
| `trimesh_mesh` | 180.2 ms | 190.9 ms | 160.6 ms | 166.5 ms |
| `sklearn_metrics` | 1005.8 ms | 1015.4 ms | 978.6 ms | 987.1 ms |
| `empyrical_stats` | 1331.1 ms | 1346.2 ms | 1305.2 ms | 1336.2 ms |

## `mlp_train`

Training a two-layer perceptron by mini-batch gradient descent. Source: written for this study.

### Calls in one run

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `x[mask] = v` | float64 | 188 | |
| `np.argmax` | | | 4,000 |
| `x.astype(float32)` | | | 2 |
| `np.sum` | | | 940 |

### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 0.5% | 0.8% |
| NumPy | 64.3% | 60.3% |
| BLAS | 15.9% | 14.8% |
| Maths library | 0.2% | 1.3% |
| hotkernels | 0.0% | 0.5% |
| Memory and system | 19.2% | 22.3% |
| Other compiled code | 0.0% | 0.0% |

The largest functions on plain NumPy (3,063 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 50.4% | NumPy | `DOUBLE_subtract` | `_multiarray_umath.cpython-312-darwin.so` |
| 16.6% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 15.9% | BLAS | `(functions without symbols)` | `libBLAS.dylib` |
| 5.0% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.3% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.6% | NumPy | `array_assign_boolean_subscript` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.3% | Memory and system | `__bsdthread_ctl` | `libsystem_kernel.dylib` |
| 0.6% | NumPy | `DOUBLE_pairwise_sum` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.6% | NumPy | `DOUBLE_maximum` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.5% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 0.4% | NumPy | `DOUBLE_square` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.4% | NumPy | `random_standard_normal` | `_generator.cpython-312-darwin.so` |
| 0.3% | NumPy | `NpyIter_AdvancedNew` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.3% | NumPy | `_aligned_strided_to_contig_size8_srcstride0` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.3% | Memory and system | `mach_absolute_time` | `libsystem_kernel.dylib` |

The largest after `install()` (2,995 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 50.7% | NumPy | `DOUBLE_subtract` | `_multiarray_umath.cpython-312-darwin.so` |
| 19.4% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 14.8% | BLAS | `(functions without symbols)` | `libBLAS.dylib` |
| 2.4% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.3% | Memory and system | `__bsdthread_ctl` | `libsystem_kernel.dylib` |
| 1.8% | NumPy | `DOUBLE_pairwise_sum` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.3% | NumPy | `DOUBLE_maximum` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.3% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.0% | NumPy | `DOUBLE_square` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.7% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 0.6% | Maths library | `exp` | `libsystem_m.dylib` |
| 0.6% | Maths library | `fetestexcept` | `libsystem_m.dylib` |
| 0.5% | NumPy | `NpyIter_AdvancedNew` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.5% | hotkernels | `maskfill_f64` | `_binding.cpython-312-darwin.so` |
| 0.4% | NumPy | `simd_binary_scalar2_less_equal_f64` | `_multiarray_umath.cpython-312-darwin.so` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 48.4% | 1 | `apps/mlp_train.py:95(run)` |
| 30.8% | 188 | `apps/mlp_train.py:58(loss_and_gradients)` |
| 16.1% | 190 | `apps/mlp_train.py:44(forward)` |
| 3.8% | 1,318 | `<method 'reduce' of 'numpy.ufunc' objects>` |
| 0.1% | 190 | `numpy/_core/_methods.py:115(_mean)` |
| 0.1% | 190 | `numpy/_core/_methods.py:73(_count_reduce_items)` |
| 0.1% | 376 | `numpy/_core/fromnumeric.py:66(_wrapreduction)` |
| 0.1% | 564 | `<method 'sum' of 'numpy.ndarray' objects>` |
| 0.1% | 376 | `numpy/_core/fromnumeric.py:2342(sum)` |
| 0.1% | 376 | `<built-in method numpy.arange>` |

## `min_char_rnn`

min-char-rnn.py, a character-level recurrent network. Source: https://gist.github.com/karpathy/d4dee566867f8291f086 (Andrej Karpathy), copied unchanged.

### Calls in one run

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `np.clip` | | | 1,500 |
| `np.cumsum` | | | 600 |
| `x.astype(float32)` | | | 600 |
| `np.sum` | | | 8,100 |
| `np.searchsorted` | | | 600 |

### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 8.9% | 9.4% |
| NumPy | 41.8% | 42.8% |
| BLAS | 31.2% | 30.0% |
| Maths library | 4.6% | 4.4% |
| hotkernels | 0.0% | 0.0% |
| Memory and system | 13.6% | 13.4% |
| Other compiled code | 0.0% | 0.0% |

The largest functions on plain NumPy (3,294 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 31.0% | BLAS | `(functions without symbols)` | `libBLAS.dylib` |
| 18.9% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 5.5% | Memory and system | `__bzero` | `libsystem_platform.dylib` |
| 3.6% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 3.1% | Memory and system | `_tlv_get_addr` | `libdyld.dylib` |
| 2.4% | Maths library | `fetestexcept` | `libsystem_m.dylib` |
| 1.9% | NumPy | `DOUBLE_clip` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.7% | NumPy | `PyArray_NewFromDescr_int` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.5% | Memory and system | `_platform_memset` | `libsystem_platform.dylib` |
| 1.5% | NumPy | `ufunc_generic_fastcall` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.2% | Memory and system | `mach_absolute_time` | `libsystem_kernel.dylib` |
| 1.1% | NumPy | `PyUFunc_GenericFunctionInternal` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.1% | NumPy | `PyUFunc_DefaultLegacyInnerLoopSelector` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.9% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.9% | Maths library | `exp` | `libsystem_m.dylib` |

The largest after `install()` (3,237 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 30.0% | BLAS | `(functions without symbols)` | `libBLAS.dylib` |
| 19.6% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 5.1% | Memory and system | `__bzero` | `libsystem_platform.dylib` |
| 4.2% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 3.1% | Memory and system | `_tlv_get_addr` | `libdyld.dylib` |
| 2.7% | Maths library | `fetestexcept` | `libsystem_m.dylib` |
| 2.0% | NumPy | `DOUBLE_clip` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.6% | Memory and system | `mach_absolute_time` | `libsystem_kernel.dylib` |
| 1.5% | Memory and system | `_platform_memset` | `libsystem_platform.dylib` |
| 1.4% | NumPy | `PyArray_NewFromDescr_int` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.2% | NumPy | `PyUFunc_GenericFunctionInternal` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.1% | NumPy | `ufunc_generic_fastcall` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.0% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.0% | NumPy | `DOUBLE_tanh` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.0% | NumPy | `DOUBLE_sqrt` | `_multiarray_umath.cpython-312-darwin.so` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 80.9% | 300 | `min_char_rnn/min-char-rnn.py:27(lossFun)` |
| 5.4% | 1 | `min_char_rnn/min-char-rnn.py:1(<module>)` |
| 2.9% | 3 | `min_char_rnn/min-char-rnn.py:63(sample)` |
| 1.9% | 1,500 | `numpy/_core/_methods.py:96(_clip)` |
| 1.5% | 8,100 | `numpy/_core/fromnumeric.py:66(_wrapreduction)` |
| 1.2% | 8,100 | `numpy/_core/fromnumeric.py:2342(sum)` |
| 1.0% | 8,100 | `<method 'reduce' of 'numpy.ufunc' objects>` |
| 0.9% | 61,800 | `numpy/_core/multiarray.py:751(dot)` |
| 0.9% | 7,800 | `<built-in method numpy.array>` |
| 0.7% | 1,805 | `numpy/_core/numeric.py:97(zeros_like)` |

## `trimesh_mesh`

Processing a triangle mesh: build, measure, sample, section, subdivide. Source: trimesh, called as a library.

### Calls in one run

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
| `x[mask]` | | | 19 |
| `x[mask] = v` | | | 9 |
| `np.cumsum` | | | 1 |
| `x.astype(float32)` | | | 19 |
| `np.sum` | | | 11 |
| `np.round` | | | 2 |

### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 5.4% | 6.2% |
| NumPy | 80.9% | 78.0% |
| BLAS | 0.9% | 0.8% |
| Maths library | 3.0% | 3.3% |
| hotkernels | 0.0% | 1.5% |
| Memory and system | 9.4% | 9.8% |
| Other compiled code | 0.4% | 0.4% |

The largest functions on plain NumPy (3,599 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 10.6% | NumPy | `binsearch<npy::floating_point_type<double, (NPY_TYPES)12>, (side_t)0>(` | `_multiarray_umath.cpython-312-darwin.so` |
| 7.9% | NumPy | `aquicksort_<npy::integral_type<unsigned long, (NPY_TYPES)8>, unsigned ` | `_multiarray_umath.cpython-312-darwin.so` |
| 7.6% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 6.3% | NumPy | `amerge_at_<npy::integral_type<unsigned long, (NPY_TYPES)8>, unsigned l` | `_multiarray_umath.cpython-312-darwin.so` |
| 5.7% | NumPy | `_contig_to_contig` | `_multiarray_umath.cpython-312-darwin.so` |
| 5.1% | Python interpreter | `PyBlake2_blake2b_compress` | `libpython3.12.dylib` |
| 3.9% | NumPy | `npyiter_buffered_iternext` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.7% | NumPy | `mapiter_get` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.3% | NumPy | `aquicksort_<npy::integral_type<long, (NPY_TYPES)7>, long, false>(long*` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.2% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.2% | NumPy | `atimsort_<npy::integral_type<unsigned long, (NPY_TYPES)8>, unsigned lo` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.0% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.0% | Maths library | `pow` | `libsystem_m.dylib` |
| 1.9% | NumPy | `prepare_index_noarray` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.9% | NumPy | `VOID_compare` | `_multiarray_umath.cpython-312-darwin.so` |

The largest after `install()` (3,524 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 9.0% | NumPy | `aquicksort_<npy::integral_type<unsigned long, (NPY_TYPES)8>, unsigned ` | `_multiarray_umath.cpython-312-darwin.so` |
| 8.1% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 6.6% | NumPy | `amerge_at_<npy::integral_type<unsigned long, (NPY_TYPES)8>, unsigned l` | `_multiarray_umath.cpython-312-darwin.so` |
| 5.9% | NumPy | `_contig_to_contig` | `_multiarray_umath.cpython-312-darwin.so` |
| 5.8% | Python interpreter | `PyBlake2_blake2b_compress` | `libpython3.12.dylib` |
| 4.4% | NumPy | `mapiter_get` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.9% | NumPy | `npyiter_buffered_iternext` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.7% | NumPy | `atimsort_<npy::integral_type<unsigned long, (NPY_TYPES)8>, unsigned lo` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.5% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.4% | NumPy | `PyArray_TransferNDimToStrided` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.4% | NumPy | `aquicksort_<npy::integral_type<long, (NPY_TYPES)7>, long, false>(long*` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.3% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.1% | NumPy | `VOID_compare` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.1% | Maths library | `pow` | `libsystem_m.dylib` |
| 1.9% | NumPy | `npyiter_goto_iterindex` | `_multiarray_umath.cpython-312-darwin.so` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 26.3% | 9 | `<method 'argsort' of 'numpy.ndarray' objects>` |
| 10.4% | 1 | `<method 'searchsorted' of 'numpy.ndarray' objects>` |
| 8.6% | 1 | `trimesh/triangles.py:217(mass_properties)` |
| 8.3% | 1 | `trimesh/sample.py:20(sample_surface)` |
| 8.2% | 71 | `<method 'reduce' of 'numpy.ufunc' objects>` |
| 5.2% | 52 | `trimesh/caching.py:44(hash_fallback)` |
| 3.2% | 10 | `<method 'sort' of 'numpy.ndarray' objects>` |
| 2.3% | 1 | `trimesh/triangles.py:106(angles)` |
| 2.2% | 1 | `trimesh/remesh.py:17(subdivide)` |
| 2.1% | 6 | `numpy/lib/_arraysetops_impl.py:350(_unique1d)` |

## `sklearn_metrics`

Scoring a classifier: ROC, precision-recall, calibration, confusion matrix. Source: scikit-learn (sklearn.metrics, sklearn.calibration), called as a library.

### Calls in one run

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `x[mask]` | float64 | 53 | |
| `x[mask] = v` | float64 | 24 | |
| `np.nonzero` | bool | 2 | |
| `np.bincount` | int64 | 16 | |
| `np.searchsorted` | float64 | 1 | |
| `np.where(c, a, b)` | float64 | 2 | |
| `np.argmax` | | | 300,000 |
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

### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 0.7% | 0.6% |
| NumPy | 96.0% | 93.7% |
| BLAS | 0.0% | 0.0% |
| Maths library | 1.4% | 1.2% |
| hotkernels | 0.0% | 2.4% |
| Memory and system | 1.4% | 1.8% |
| Other compiled code | 0.5% | 0.3% |

The largest functions on plain NumPy (4,302 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 37.8% | NumPy | `amerge_at_<npy::floating_point_type<double, (NPY_TYPES)12>, double, fa` | `_multiarray_umath.cpython-312-darwin.so` |
| 14.4% | NumPy | `npy_fnv1a` | `_multiarray_umath.cpython-312-darwin.so` |
| 10.9% | NumPy | `atimsort_<npy::floating_point_type<double, (NPY_TYPES)12>, double, fal` | `_multiarray_umath.cpython-312-darwin.so` |
| 10.5% | NumPy | `_object* unique_numeric<long, &unsigned long hash_integer<long>(long c` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.2% | NumPy | `mapiter_trivial_get` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.9% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.9% | NumPy | `arr_bincount` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.8% | NumPy | `binsearch<npy::floating_point_type<double, (NPY_TYPES)12>, (side_t)0>(` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.4% | Maths library | `log` | `libsystem_m.dylib` |
| 1.3% | NumPy | `binsearch<npy::integral_type<long, (NPY_TYPES)7>, (side_t)0>(char cons` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.3% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 1.1% | NumPy | `OUTLINED_FUNCTION_148` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.9% | NumPy | `array_boolean_subscript` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.7% | NumPy | `DOUBLE_pairwise_sum` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.7% | NumPy | `OUTLINED_FUNCTION_71` | `_multiarray_umath.cpython-312-darwin.so` |

The largest after `install()` (4,269 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 38.1% | NumPy | `amerge_at_<npy::floating_point_type<double, (NPY_TYPES)12>, double, fa` | `_multiarray_umath.cpython-312-darwin.so` |
| 14.2% | NumPy | `npy_fnv1a` | `_multiarray_umath.cpython-312-darwin.so` |
| 11.3% | NumPy | `atimsort_<npy::floating_point_type<double, (NPY_TYPES)12>, double, fal` | `_multiarray_umath.cpython-312-darwin.so` |
| 11.2% | NumPy | `_object* unique_numeric<long, &unsigned long hash_integer<long>(long c` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.5% | NumPy | `mapiter_trivial_get` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.0% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.9% | hotkernels | `searchsorted_f64` | `_binding.cpython-312-darwin.so` |
| 1.7% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 1.4% | NumPy | `binsearch<npy::integral_type<long, (NPY_TYPES)7>, (side_t)0>(char cons` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.2% | Maths library | `log` | `libsystem_m.dylib` |
| 1.1% | NumPy | `OUTLINED_FUNCTION_148` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.7% | NumPy | `DOUBLE_subtract` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.7% | NumPy | `DOUBLE_pairwise_sum` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.7% | NumPy | `OUTLINED_FUNCTION_71` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.6% | NumPy | `DOUBLE_logical_or` | `_multiarray_umath.cpython-312-darwin.so` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 49.6% | 5 | `<method 'argsort' of 'numpy.ndarray' objects>` |
| 24.9% | 67 | `<built-in method numpy._core._multiarray_umath._unique_hash>` |
| 3.2% | 20 | `<method 'searchsorted' of 'numpy.ndarray' objects>` |
| 3.2% | 183 | `<method 'reduce' of 'numpy.ufunc' objects>` |
| 2.2% | 2 | `sklearn/utils/_array_api.py:885(_xlogy)` |
| 1.9% | 7 | `<method 'cumsum' of 'numpy.ndarray' objects>` |
| 1.3% | 15 | `sklearn/utils/_array_api.py:1329(_bincount)` |
| 1.2% | 4 | `sklearn/metrics/_ranking.py:878(_sort_inputs_and_compute_classification_thresholds)` |
| 1.1% | 4 | `sklearn/metrics/_ranking.py:924(confusion_matrix_at_thresholds)` |
| 0.9% | 5 | `sklearn/metrics/_classification.py:626(multilabel_confusion_matrix)` |

## `empyrical_stats`

Risk and return statistics of daily return series. Source: empyrical-reloaded, called as a library.

### Calls in one run

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `np.clip` | float64 | 120 | |
| `x[mask]` | float64 | 420 | |
| `x[mask] = v` | float64 | 240 | |
| `np.where(c, a, b)` | float64 | 120 | |
| `fmax.accumulate` | float64 | 120 | |
| `x[mask]` | | | 300 |
| `x[mask] = v` | | | 180 |
| `np.cumsum` | | | 60 |
| `x.astype(float32)` | | | 360 |
| `np.where(c, a, b)` | | | 120 |
| `maximum.accumulate` | | | 60 |

### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 2.4% | 2.5% |
| NumPy | 47.8% | 48.7% |
| BLAS | 0.0% | 0.0% |
| Maths library | 2.4% | 2.1% |
| hotkernels | 0.0% | 0.0% |
| Memory and system | 0.9% | 0.8% |
| Other compiled code | 46.5% | 46.0% |

The largest functions on plain NumPy (4,587 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 22.4% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 18.6% | Other compiled code | `nanstd_one_float64` | `reduce.cpython-312-darwin.so` |
| 18.4% | NumPy | `DOUBLE_fmax` | `_multiarray_umath.cpython-312-darwin.so` |
| 13.3% | Other compiled code | `nanmin_one_float64` | `reduce.cpython-312-darwin.so` |
| 7.6% | Other compiled code | `nanmean_one_float64` | `reduce.cpython-312-darwin.so` |
| 6.6% | Other compiled code | `nanmax_one_float64` | `reduce.cpython-312-darwin.so` |
| 2.2% | Maths library | `fetestexcept` | `libsystem_m.dylib` |
| 2.1% | NumPy | `DOUBLE_subtract` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.0% | NumPy | `DOUBLE_divide` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.5% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 0.9% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.7% | Memory and system | `_tlv_get_addr` | `libdyld.dylib` |
| 0.5% | NumPy | `introselect_<npy::floating_point_type<double, (NPY_TYPES)12>, false, d` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.4% | NumPy | `array_boolean_subscript` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.4% | NumPy | `DOUBLE_isnan` | `_multiarray_umath.cpython-312-darwin.so` |

The largest after `install()` (4,635 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 22.8% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 18.8% | NumPy | `DOUBLE_fmax` | `_multiarray_umath.cpython-312-darwin.so` |
| 18.2% | Other compiled code | `nanstd_one_float64` | `reduce.cpython-312-darwin.so` |
| 13.7% | Other compiled code | `nanmin_one_float64` | `reduce.cpython-312-darwin.so` |
| 7.6% | Other compiled code | `nanmean_one_float64` | `reduce.cpython-312-darwin.so` |
| 6.2% | Other compiled code | `nanmax_one_float64` | `reduce.cpython-312-darwin.so` |
| 2.0% | NumPy | `DOUBLE_subtract` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.0% | NumPy | `DOUBLE_divide` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.0% | Maths library | `fetestexcept` | `libsystem_m.dylib` |
| 1.7% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 1.0% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.7% | Memory and system | `_tlv_get_addr` | `libdyld.dylib` |
| 0.6% | NumPy | `DOUBLE_isnan` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.5% | NumPy | `introselect_<npy::floating_point_type<double, (NPY_TYPES)12>, false, d` | `_multiarray_umath.cpython-312-darwin.so` |
| 0.2% | NumPy | `array_boolean_subscript` | `_multiarray_umath.cpython-312-darwin.so` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 19.8% | 180 | `<method 'accumulate' of 'numpy.ufunc' objects>` |
| 18.4% | 240 | `<method 'cumprod' of 'numpy.ndarray' objects>` |
| 16.5% | 180 | `<built-in method bottleneck.reduce.nanstd>` |
| 12.4% | 300 | `<built-in method bottleneck.reduce.nanmin>` |
| 7.0% | 540 | `<built-in method bottleneck.reduce.nanmean>` |
| 5.7% | 120 | `<built-in method bottleneck.reduce.nanmax>` |
| 3.8% | 240 | `empyrical/stats.py:123(cum_returns)` |
| 3.8% | 180 | `empyrical/stats.py:263(drawdown_series)` |
| 0.7% | 120 | `<built-in method builtins.sum>` |
| 0.5% | 64,620 | `<built-in method builtins.isinstance>` |
