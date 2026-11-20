# Applications, second set, with and without hotkernels: Apple M1 Pro

Written by `uv run --group apps delivery apps --set second --profile --record`. Do not edit. What the figures mean is in [docs/applications-second-findings.md](../../docs/applications-second-findings.md).

- **Machine:** Apple M1 Pro, macOS-27.2.1-arm64-arm-64bit
- **Software:** Python 3.12.7, NumPy 2.5.5, trimesh 5.2.1, scikit-learn 1.9.1, empyrical-reloaded 0.7.3, pandas 3.1.1, scipy 1.18.1, scikit-image 0.26.0, astropy 8.0.3, hotkernels at 7a277a8
- **Load average when the run started:** 3.41, a busy machine: the times are rough
- **GitHub Actions runs unfinished when it started:** 0
- **Method:** 7 alternating rounds of each side, the best of each. A round is one run, or as many runs as last 100 ms where one is shorter; the times are of one run. Results compared bit for bit on every round.
- **Profile:** sampled with `sample`, every thread once a millisecond.

## Times

| Application | Size | Data | NumPy | `install()` | Faster by | Results | `install(inexact=True)` | Faster by | Kernel calls | Left to NumPy |
|---|---|---|---|---|---|---|---|---|---|---|
| `particle_filter` | small | 2,000 particles, 100 steps | 20.2 ms | 18.8 ms | 1.08x | identical | 18.4 ms | 1.10x | 241 | 243 |
| `particle_filter` | large | 100,000 particles, 100 steps | 895.4 ms | 804.1 ms | 1.11x | identical | 792.8 ms | 1.13x | 342 | 244 |
| `veros_ocean` | small | a grid of 81 × 81 × 11 points | 28.7 ms | 28.6 ms | 1.00x | identical | 28.6 ms | 1.01x | 9 | 1 |
| `veros_ocean` | large | a grid of 204 × 204 × 26 points | 491.1 ms | 488.1 ms | 1.01x | identical | 487.4 ms | 1.01x | 130 | 1 |
| `skimage_pipeline` | small | 4 images of 256 × 256 | 96.6 ms | 94.0 ms | 1.03x | identical | 94.1 ms | 1.03x | 1,715 | 11,681 |
| `skimage_pipeline` | large | 4 images of 1,024 × 1,024 | 1180.2 ms | 1152.8 ms | 1.02x | identical | 1154.1 ms | 1.02x | 2,939 | 133,794 |
| `pandas_sales` | small | 100,000 sales in 200 stores over 365 days | 30.8 ms | 29.4 ms | 1.05x | identical | 29.3 ms | 1.05x | 65 | 117 |
| `pandas_sales` | large | 2,000,000 sales in 200 stores over 365 days | 477.6 ms | 446.5 ms | 1.07x | identical | 447.1 ms | 1.07x | 65 | 117 |
| `astropy_timeseries` | small | 10 light curves of 1,000 points, 4 images of 256 × 256 | 206.2 ms | 205.7 ms | 1.00x | identical | 205.7 ms | 1.00x | 1,256 | 3,228 |
| `astropy_timeseries` | large | 10 light curves of 5,000 points, 4 images of 1,024 × 1,024 | 1534.9 ms | 1529.3 ms | 1.00x | identical | 1530.2 ms | 1.00x | 4,261 | 9,233 |

The median of the 7 rounds, to show the noise, and the runs in each round:

| Application | Size | Runs per round | NumPy, best | NumPy, median | `install()`, best | `install()`, median |
|---|---|---|---|---|---|---|
| `particle_filter` | small | 5 | 20.2 ms | 20.6 ms | 18.8 ms | 18.9 ms |
| `particle_filter` | large | 1 | 895.4 ms | 898.7 ms | 804.1 ms | 811.5 ms |
| `veros_ocean` | small | 3 | 28.7 ms | 29.4 ms | 28.6 ms | 29.2 ms |
| `veros_ocean` | large | 1 | 491.1 ms | 495.2 ms | 488.1 ms | 489.5 ms |
| `skimage_pipeline` | small | 1 | 96.6 ms | 96.8 ms | 94.0 ms | 94.2 ms |
| `skimage_pipeline` | large | 1 | 1180.2 ms | 1183.6 ms | 1152.8 ms | 1156.6 ms |
| `pandas_sales` | small | 3 | 30.8 ms | 31.2 ms | 29.4 ms | 29.6 ms |
| `pandas_sales` | large | 1 | 477.6 ms | 478.0 ms | 446.5 ms | 447.1 ms |
| `astropy_timeseries` | small | 1 | 206.2 ms | 207.4 ms | 205.7 ms | 206.3 ms |
| `astropy_timeseries` | large | 1 | 1534.9 ms | 1561.4 ms | 1529.3 ms | 1560.5 ms |

## `particle_filter`

A bootstrap particle filter tracking a target from range and bearing. Source: Written for this study.

### Small: 2,000 particles, 100 steps

#### Calls in one run

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `np.argmax` | float64 | 100 | |
| `x[mask]` | float64 | 100 | |
| `np.searchsorted` | float64 | 41 | |
| `np.cumsum` | | | 41 |
| `x.astype(float32)` | | | 2 |
| `np.sum` | | | 200 |

#### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 3.1% | 1.6% |
| NumPy | 64.7% | 64.9% |
| BLAS | 2.2% | 3.2% |
| Maths library | 15.7% | 16.0% |
| hotkernels | 0.0% | 0.8% |
| Memory and system | 14.2% | 13.4% |
| Other compiled code | 0.0% | 0.0% |

The largest functions on plain NumPy (3,785 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 10.6% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 9.9% | NumPy | `random_standard_normal` | `_generator.cpython-312-darwin.so` |
| 8.6% | NumPy | `_contig_to_contig` | `_multiarray_umath.cpython-312-darwin.so` |
| 6.3% | Maths library | `atan2` | `libsystem_m.dylib` |
| 5.8% | NumPy | `PyArray_TransferNDimToStrided` | `_multiarray_umath.cpython-312-darwin.so` |
| 5.7% | NumPy | `binsearch<npy::floating_point_type<double, (NPY_TYPES)12>, (side_t)0>(` | `_multiarray_umath.cpython-312-darwin.so` |
| 5.3% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 4.1% | Maths library | `log` | `libsystem_m.dylib` |
| 2.9% | NumPy | `__pyx_f_5numpy_6random_6_pcg64_pcg64_uint64` | `_pcg64.cpython-312-darwin.so` |
| 2.8% | NumPy | `DOUBLE_remainder` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.5% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.2% | BLAS | `(functions without symbols)` | `libBLAS.dylib` |
| 1.9% | Maths library | `exp` | `libsystem_m.dylib` |
| 1.8% | NumPy | `DYLD-STUB$$memmove` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.5% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |

The largest after `install()` (3,728 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 12.3% | NumPy | `random_standard_normal` | `_generator.cpython-312-darwin.so` |
| 10.2% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 8.3% | NumPy | `_contig_to_contig` | `_multiarray_umath.cpython-312-darwin.so` |
| 7.6% | Maths library | `atan2` | `libsystem_m.dylib` |
| 6.7% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 5.4% | NumPy | `PyArray_TransferNDimToStrided` | `_multiarray_umath.cpython-312-darwin.so` |
| 4.6% | NumPy | `__pyx_f_5numpy_6random_6_pcg64_pcg64_uint64` | `_pcg64.cpython-312-darwin.so` |
| 3.2% | NumPy | `DOUBLE_remainder` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.2% | BLAS | `(functions without symbols)` | `libBLAS.dylib` |
| 2.7% | Maths library | `exp` | `libsystem_m.dylib` |
| 2.7% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.2% | Maths library | `log` | `libsystem_m.dylib` |
| 2.2% | NumPy | `random_standard_normal_fill` | `_generator.cpython-312-darwin.so` |
| 1.7% | Maths library | `hypot` | `libsystem_m.dylib` |
| 1.6% | NumPy | `DYLD-STUB$$memmove` | `_multiarray_umath.cpython-312-darwin.so` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 83.9% | 1 | `apps/particle_filter.py:98(run)` |
| 7.5% | 41 | `<method 'searchsorted' of 'numpy.ndarray' objects>` |
| 1.9% | 41 | `<method 'cumsum' of 'numpy.ndarray' objects>` |
| 1.1% | 41 | `apps/particle_filter.py:81(systematic_resample)` |
| 1.1% | 300 | `<method 'reduce' of 'numpy.ufunc' objects>` |
| 1.0% | 100 | `<method 'argmax' of 'numpy.ndarray' objects>` |
| 0.9% | 300 | `numpy/_core/fromnumeric.py:66(_wrapreduction)` |
| 0.5% | 200 | `numpy/_core/fromnumeric.py:2342(sum)` |
| 0.3% | 182 | `numpy/_core/fromnumeric.py:48(_wrapfunc)` |
| 0.3% | 100 | `numpy/_core/fromnumeric.py:1259(argmax)` |

### Large: 100,000 particles, 100 steps

#### Calls in one run

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `np.argmax` | float64 | 100 | |
| `x[mask]` | float64 | 100 | |
| `np.searchsorted` | float64 | 42 | |
| `x += y` | float64 | 100 | |
| `np.cumsum` | | | 42 |
| `x.astype(float32)` | | | 2 |
| `np.sum` | | | 200 |

#### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 0.1% | 0.0% |
| NumPy | 73.8% | 68.3% |
| BLAS | 3.3% | 4.6% |
| Maths library | 13.4% | 14.3% |
| hotkernels | 0.0% | 2.5% |
| Memory and system | 9.4% | 10.3% |
| Other compiled code | 0.0% | 0.0% |

The largest functions on plain NumPy (3,856 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 15.5% | NumPy | `random_standard_normal` | `_generator.cpython-312-darwin.so` |
| 11.0% | NumPy | `binsearch<npy::floating_point_type<double, (NPY_TYPES)12>, (side_t)0>(` | `_multiarray_umath.cpython-312-darwin.so` |
| 9.7% | NumPy | `_contig_to_contig` | `_multiarray_umath.cpython-312-darwin.so` |
| 9.4% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 7.3% | Maths library | `atan2` | `libsystem_m.dylib` |
| 5.4% | NumPy | `PyArray_TransferNDimToStrided` | `_multiarray_umath.cpython-312-darwin.so` |
| 4.6% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.9% | NumPy | `__pyx_f_5numpy_6random_6_pcg64_pcg64_uint64` | `_pcg64.cpython-312-darwin.so` |
| 3.3% | BLAS | `(functions without symbols)` | `libBLAS.dylib` |
| 2.8% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.5% | NumPy | `DOUBLE_remainder` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.5% | Maths library | `exp` | `libsystem_m.dylib` |
| 2.3% | NumPy | `random_standard_normal_fill` | `_generator.cpython-312-darwin.so` |
| 1.7% | NumPy | `DYLD-STUB$$memmove` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.7% | Maths library | `log` | `libsystem_m.dylib` |

The largest after `install()` (3,799 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 16.7% | NumPy | `random_standard_normal` | `_generator.cpython-312-darwin.so` |
| 11.1% | NumPy | `_contig_to_contig` | `_multiarray_umath.cpython-312-darwin.so` |
| 10.3% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 7.7% | Maths library | `atan2` | `libsystem_m.dylib` |
| 5.4% | NumPy | `PyArray_TransferNDimToStrided` | `_multiarray_umath.cpython-312-darwin.so` |
| 4.9% | NumPy | `__pyx_f_5numpy_6random_6_pcg64_pcg64_uint64` | `_pcg64.cpython-312-darwin.so` |
| 4.6% | BLAS | `(functions without symbols)` | `libBLAS.dylib` |
| 3.3% | NumPy | `random_standard_normal_fill` | `_generator.cpython-312-darwin.so` |
| 3.0% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.0% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.9% | Maths library | `exp` | `libsystem_m.dylib` |
| 2.8% | NumPy | `DOUBLE_remainder` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.2% | NumPy | `DOUBLE_square` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.9% | NumPy | `_aligned_strided_to_contig_size8_srcstride0` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.9% | NumPy | `DYLD-STUB$$memmove` | `_multiarray_umath.cpython-312-darwin.so` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 85.6% | 1 | `apps/particle_filter.py:98(run)` |
| 10.7% | 42 | `<method 'searchsorted' of 'numpy.ndarray' objects>` |
| 1.6% | 42 | `<method 'cumsum' of 'numpy.ndarray' objects>` |
| 0.8% | 100 | `<method 'argmax' of 'numpy.ndarray' objects>` |
| 0.6% | 300 | `<method 'reduce' of 'numpy.ufunc' objects>` |
| 0.4% | 42 | `apps/particle_filter.py:81(systematic_resample)` |
| 0.1% | 43 | `<built-in method numpy.zeros>` |
| 0.1% | 42 | `<built-in method numpy.arange>` |
| 0.1% | 300 | `numpy/_core/fromnumeric.py:66(_wrapreduction)` |
| 0.0% | 100 | `numpy/_core/fromnumeric.py:1259(argmax)` |

## `veros_ocean`

Three routines of the Veros ocean model: equation of state, isoneutral mixing, turbulent kinetic energy. Source: https://github.com/dionhaefner/pyhpc-benchmarks at d438ef8, NumPy implementations copied unchanged.

### Small: a grid of 81 × 81 × 11 points

#### Calls in one run

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `np.where(c, a, b)` | float64 | 9 | |
| `np.where(c, a, b)` | | | 1 |

#### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 0.4% | 0.3% |
| NumPy | 82.5% | 82.2% |
| BLAS | 2.6% | 2.9% |
| Maths library | 1.1% | 1.1% |
| hotkernels | 0.0% | 0.4% |
| Memory and system | 13.4% | 13.1% |
| Other compiled code | 0.0% | 0.0% |

The largest functions on plain NumPy (3,823 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 21.1% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 12.5% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 12.1% | NumPy | `DOUBLE_tanh` | `_multiarray_umath.cpython-312-darwin.so` |
| 11.2% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 7.0% | NumPy | `_contig_to_contig` | `_multiarray_umath.cpython-312-darwin.so` |
| 4.7% | NumPy | `DOUBLE_subtract` | `_multiarray_umath.cpython-312-darwin.so` |
| 4.4% | NumPy | `PyArray_TransferNDimToStrided` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.8% | NumPy | `_aligned_strided_to_contig_size8` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.5% | NumPy | `DOUBLE_divide` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.6% | NumPy | `_aligned_contig_to_strided_size8` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.6% | BLAS | `(functions without symbols)` | `libLAPACK.dylib` |
| 2.1% | NumPy | `DOUBLE_minimum` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.5% | NumPy | `_aligned_strided_to_contig_size8_srcstride0` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.2% | NumPy | `raw_array_assign_array` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.0% | NumPy | `DYLD-STUB$$memmove` | `_multiarray_umath.cpython-312-darwin.so` |

The largest after `install()` (3,826 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 22.9% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 12.4% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 11.0% | NumPy | `DOUBLE_tanh` | `_multiarray_umath.cpython-312-darwin.so` |
| 10.7% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 6.2% | NumPy | `_contig_to_contig` | `_multiarray_umath.cpython-312-darwin.so` |
| 4.8% | NumPy | `PyArray_TransferNDimToStrided` | `_multiarray_umath.cpython-312-darwin.so` |
| 4.0% | NumPy | `DOUBLE_subtract` | `_multiarray_umath.cpython-312-darwin.so` |
| 4.0% | NumPy | `_aligned_strided_to_contig_size8` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.5% | NumPy | `DOUBLE_divide` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.9% | BLAS | `(functions without symbols)` | `libLAPACK.dylib` |
| 2.3% | NumPy | `_aligned_contig_to_strided_size8` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.9% | NumPy | `DOUBLE_minimum` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.8% | NumPy | `_aligned_strided_to_contig_size8_srcstride0` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.5% | NumPy | `DYLD-STUB$$memmove` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.5% | NumPy | `DOUBLE_negative` | `_multiarray_umath.cpython-312-darwin.so` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 35.3% | 1 | `isoneutral_mixing/isoneutral_numpy.py:24(isoneutral_diffusion_pre)` |
| 22.3% | 1 | `equation_of_state/eos_numpy.py:63(gsw_dHdT)` |
| 13.9% | 16 | `isoneutral_mixing/isoneutral_numpy.py:115(dm_taper)` |
| 7.3% | 3 | `turbulent_kinetic_energy/tke_numpy.py:61(_adv_superbee)` |
| 7.3% | 1 | `turbulent_kinetic_energy/tke_numpy.py:137(integrate_tke)` |
| 3.5% | 1 | `turbulent_kinetic_energy/tke_numpy.py:30(solve_tridiag)` |
| 2.4% | 47 | `<built-in method numpy.array>` |
| 1.1% | 1 | `turbulent_kinetic_energy/tke_numpy.py:9(solve_implicit)` |
| 1.1% | 3 | `turbulent_kinetic_energy/tke_numpy.py:41(_calc_cr)` |
| 1.0% | 3 | `turbulent_kinetic_energy/tke_numpy.py:57(limiter)` |

### Large: a grid of 204 × 204 × 26 points

#### Calls in one run

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `np.where(c, a, b)` | float64 | 9 | |
| `x += y` | float64 | 86 | |
| `x -= y` | float64 | 35 | |
| `np.where(c, a, b)` | | | 1 |

#### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 0.1% | 0.0% |
| NumPy | 81.0% | 75.6% |
| BLAS | 2.4% | 2.5% |
| Maths library | 1.4% | 1.2% |
| hotkernels | 0.0% | 6.3% |
| Memory and system | 15.0% | 14.4% |
| Other compiled code | 0.0% | 0.0% |

The largest functions on plain NumPy (3,864 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 25.6% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 13.5% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 12.1% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 11.1% | NumPy | `DOUBLE_tanh` | `_multiarray_umath.cpython-312-darwin.so` |
| 5.2% | NumPy | `DOUBLE_subtract` | `_multiarray_umath.cpython-312-darwin.so` |
| 4.5% | NumPy | `DOUBLE_divide` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.7% | NumPy | `_contig_to_contig` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.3% | NumPy | `_aligned_contig_to_strided_size8` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.4% | BLAS | `(functions without symbols)` | `libLAPACK.dylib` |
| 2.0% | NumPy | `PyArray_TransferNDimToStrided` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.8% | NumPy | `_aligned_strided_to_contig_size8_srcstride0` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.6% | NumPy | `_aligned_strided_to_contig_size8` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.5% | NumPy | `DOUBLE_minimum` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.5% | Memory and system | `_kernelrpc_mach_vm_deallocate_trap` | `libsystem_kernel.dylib` |
| 1.4% | NumPy | `DOUBLE_negative` | `_multiarray_umath.cpython-312-darwin.so` |

The largest after `install()` (3,864 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 27.3% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 12.9% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 11.0% | NumPy | `DOUBLE_tanh` | `_multiarray_umath.cpython-312-darwin.so` |
| 7.9% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 4.0% | hotkernels | `add_inplace_f64` | `_binding.cpython-312-darwin.so` |
| 3.9% | NumPy | `DOUBLE_divide` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.5% | NumPy | `_contig_to_contig` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.2% | NumPy | `_aligned_contig_to_strided_size8` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.8% | NumPy | `DOUBLE_subtract` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.5% | BLAS | `(functions without symbols)` | `libLAPACK.dylib` |
| 2.1% | NumPy | `PyArray_TransferNDimToStrided` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.0% | NumPy | `_aligned_strided_to_contig_size8_srcstride0` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.9% | NumPy | `DOUBLE_negative` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.8% | hotkernels | `subtract_inplace_f64` | `_binding.cpython-312-darwin.so` |
| 1.6% | NumPy | `DOUBLE_minimum` | `_multiarray_umath.cpython-312-darwin.so` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 29.9% | 1 | `isoneutral_mixing/isoneutral_numpy.py:24(isoneutral_diffusion_pre)` |
| 26.8% | 1 | `equation_of_state/eos_numpy.py:63(gsw_dHdT)` |
| 13.9% | 16 | `isoneutral_mixing/isoneutral_numpy.py:115(dm_taper)` |
| 6.4% | 47 | `<built-in method numpy.array>` |
| 5.7% | 1 | `turbulent_kinetic_energy/tke_numpy.py:137(integrate_tke)` |
| 5.5% | 3 | `turbulent_kinetic_energy/tke_numpy.py:61(_adv_superbee)` |
| 3.5% | 1 | `turbulent_kinetic_energy/tke_numpy.py:30(solve_tridiag)` |
| 1.3% | 1 | `equation_of_state/eos_numpy.py:253(run)` |
| 1.1% | 10 | `turbulent_kinetic_energy/tke_numpy.py:5(where)` |
| 1.1% | 3 | `turbulent_kinetic_energy/tke_numpy.py:57(limiter)` |

## `skimage_pipeline`

Segmenting and measuring objects in images: CLAHE, Otsu, labels, region properties, Canny, HOG. Source: scikit-image (skimage.exposure, filters, morphology, measure, feature), called as a library.

### Small: 4 images of 256 × 256

#### Calls in one run

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `np.argmax` | float64 | 4 | |
| `np.clip` | float64 | 40 | |
| `x[mask]` | float64 | 1,411 | |
| `np.sum` | uint8 | 256 | |
| `np.histogram` | float64 | 4 | |
| `x[mask]` | | | 154 |
| `x[mask] = v` | | | 1,049 |
| `np.nonzero` | | | 260 |
| `np.cumsum` | | | 20 |
| `x.astype(float32)` | | | 1,376 |
| `np.sum` | | | 7,992 |
| `np.count_nonzero` | | | 554 |
| `np.bincount` | | | 260 |
| `np.unique` | | | 4 |
| `np.round` | | | 12 |

#### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 18.4% | 19.3% |
| NumPy | 32.7% | 30.7% |
| BLAS | 0.0% | 0.2% |
| Maths library | 11.4% | 11.6% |
| hotkernels | 0.0% | 0.4% |
| Memory and system | 10.0% | 9.9% |
| Other compiled code | 27.5% | 27.9% |

The largest functions on plain NumPy (3,551 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 9.7% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 7.5% | Maths library | `atan2` | `libsystem_m.dylib` |
| 7.0% | Other compiled code | `NI_Correlate1D` | `_nd_image.cpython-312-darwin.so` |
| 5.3% | NumPy | `mapiter_get` | `_multiarray_umath.cpython-312-darwin.so` |
| 5.2% | Memory and system | `_tlv_get_addr` | `libdyld.dylib` |
| 4.9% | Other compiled code | `__pyx_fuse_1__pyx_pw_7skimage_7feature_13_hoghistogram_5hog_histograms` | `_hoghistogram.cpython-312-darwin.so` |
| 4.0% | Other compiled code | `NI_LineBufferToArray` | `_nd_image.cpython-312-darwin.so` |
| 2.9% | Other compiled code | `NI_Correlate` | `_nd_image.cpython-312-darwin.so` |
| 2.3% | NumPy | `DOUBLE_remainder` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.1% | Other compiled code | `NI_ArrayToLineBuffer` | `_nd_image.cpython-312-darwin.so` |
| 1.9% | Other compiled code | `__pyx_pw_7skimage_7feature_9_canny_cy_5__pyx_fuse_1_nonmaximum_suppres` | `_canny_cy.cpython-312-darwin.so` |
| 1.9% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 1.7% | NumPy | `arr_bincount` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.7% | NumPy | `NpyIter_AdvancedNew` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.6% | Other compiled code | `__pyx_f_9_ni_label_label_line_with_neighbor` | `_ni_label.cpython-312-darwin.so` |

The largest after `install()` (3,589 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 10.6% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 7.6% | Maths library | `atan2` | `libsystem_m.dylib` |
| 6.3% | Other compiled code | `NI_Correlate1D` | `_nd_image.cpython-312-darwin.so` |
| 5.5% | NumPy | `mapiter_get` | `_multiarray_umath.cpython-312-darwin.so` |
| 5.2% | Other compiled code | `__pyx_fuse_1__pyx_pw_7skimage_7feature_13_hoghistogram_5hog_histograms` | `_hoghistogram.cpython-312-darwin.so` |
| 5.1% | Other compiled code | `NI_LineBufferToArray` | `_nd_image.cpython-312-darwin.so` |
| 5.0% | Memory and system | `_tlv_get_addr` | `libdyld.dylib` |
| 3.2% | Other compiled code | `NI_Correlate` | `_nd_image.cpython-312-darwin.so` |
| 2.3% | NumPy | `DOUBLE_remainder` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.1% | Other compiled code | `NI_ArrayToLineBuffer` | `_nd_image.cpython-312-darwin.so` |
| 2.0% | Other compiled code | `__pyx_pw_7skimage_7feature_9_canny_cy_5__pyx_fuse_1_nonmaximum_suppres` | `_canny_cy.cpython-312-darwin.so` |
| 1.8% | Other compiled code | `__pyx_f_9_ni_label_label_line_with_neighbor` | `_ni_label.cpython-312-darwin.so` |
| 1.7% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 1.6% | NumPy | `arr_bincount` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.6% | Maths library | `fetestexcept` | `libsystem_m.dylib` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 11.6% | 4 | `skimage/feature/_hog.py:48(hog)` |
| 9.4% | 3,600 | `skimage/feature/_hog.py:7(_hog_normalize_block)` |
| 8.2% | 40 | `<built-in method scipy.ndimage._nd_image.correlate1d>` |
| 4.4% | 1,024 | `numpy/_core/einsumfunc.py:634(einsum_path)` |
| 4.0% | 512 | `skimage/measure/_moments.py:201(moments_central)` |
| 3.5% | 1,024 | `numpy/_core/einsumfunc.py:445(_parse_einsum_input)` |
| 3.4% | 16 | `numpy/lib/_shape_base_impl.py:60(take_along_axis)` |
| 3.1% | 9,136 | `<method 'reduce' of 'numpy.ufunc' objects>` |
| 2.8% | 8,008 | `numpy/_core/fromnumeric.py:66(_wrapreduction)` |
| 2.3% | 12 | `scipy/ndimage/_measurements.py:43(label)` |

### Large: 4 images of 1,024 × 1,024

#### Calls in one run

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `np.argmax` | float64 | 4 | |
| `np.clip` | float64 | 520 | |
| `x[mask]` | float64 | 1,926 | |
| `np.sum` | uint8 | 473 | |
| `np.histogram` | float64 | 4 | |
| `x += y` | float64 | 12 | |
| `x[mask]` | | | 127 |
| `x[mask] = v` | | | 1,320 |
| `np.nonzero` | | | 477 |
| `np.cumsum` | | | 20 |
| `x.astype(float32)` | | | 2,461 |
| `np.sum` | | | 128,017 |
| `np.count_nonzero` | | | 1,096 |
| `np.bincount` | | | 260 |
| `np.unique` | | | 4 |
| `np.round` | | | 12 |

#### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 9.7% | 10.1% |
| NumPy | 34.1% | 32.3% |
| BLAS | 0.0% | 0.0% |
| Maths library | 12.5% | 11.9% |
| hotkernels | 0.0% | 0.6% |
| Memory and system | 6.6% | 7.2% |
| Other compiled code | 37.1% | 37.9% |

The largest functions on plain NumPy (3,839 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 9.8% | Other compiled code | `NI_LineBufferToArray` | `_nd_image.cpython-312-darwin.so` |
| 9.5% | Maths library | `atan2` | `libsystem_m.dylib` |
| 7.8% | Other compiled code | `NI_Correlate1D` | `_nd_image.cpython-312-darwin.so` |
| 6.5% | NumPy | `mapiter_get` | `_multiarray_umath.cpython-312-darwin.so` |
| 6.5% | Other compiled code | `__pyx_fuse_1__pyx_pw_7skimage_7feature_13_hoghistogram_5hog_histograms` | `_hoghistogram.cpython-312-darwin.so` |
| 4.2% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 3.8% | Other compiled code | `NI_Correlate` | `_nd_image.cpython-312-darwin.so` |
| 3.1% | NumPy | `DOUBLE_remainder` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.1% | Other compiled code | `NI_ArrayToLineBuffer` | `_nd_image.cpython-312-darwin.so` |
| 3.0% | Memory and system | `_tlv_get_addr` | `libdyld.dylib` |
| 2.3% | NumPy | `arr_bincount` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.5% | Other compiled code | `__pyx_pw_7skimage_7feature_9_canny_cy_5__pyx_fuse_1_nonmaximum_suppres` | `_canny_cy.cpython-312-darwin.so` |
| 1.5% | Maths library | `fetestexcept` | `libsystem_m.dylib` |
| 1.4% | Other compiled code | `__pyx_f_9_ni_label_label_line_with_neighbor` | `_ni_label.cpython-312-darwin.so` |
| 1.3% | Other compiled code | `__pyx_fuse_2__pyx_f_9_ni_label_fused_write_line` | `_ni_label.cpython-312-darwin.so` |

The largest after `install()` (3,872 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 9.8% | Other compiled code | `NI_LineBufferToArray` | `_nd_image.cpython-312-darwin.so` |
| 9.3% | Maths library | `atan2` | `libsystem_m.dylib` |
| 8.3% | Other compiled code | `NI_Correlate1D` | `_nd_image.cpython-312-darwin.so` |
| 6.6% | NumPy | `mapiter_get` | `_multiarray_umath.cpython-312-darwin.so` |
| 6.4% | Other compiled code | `__pyx_fuse_1__pyx_pw_7skimage_7feature_13_hoghistogram_5hog_histograms` | `_hoghistogram.cpython-312-darwin.so` |
| 4.1% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 4.0% | Other compiled code | `NI_Correlate` | `_nd_image.cpython-312-darwin.so` |
| 3.3% | Other compiled code | `NI_ArrayToLineBuffer` | `_nd_image.cpython-312-darwin.so` |
| 3.1% | NumPy | `DOUBLE_remainder` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.0% | Memory and system | `_tlv_get_addr` | `libdyld.dylib` |
| 1.7% | Other compiled code | `__pyx_f_9_ni_label_label_line_with_neighbor` | `_ni_label.cpython-312-darwin.so` |
| 1.7% | NumPy | `arr_bincount` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.6% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 1.5% | Other compiled code | `__pyx_pw_7skimage_7feature_9_canny_cy_5__pyx_fuse_1_nonmaximum_suppres` | `_canny_cy.cpython-312-darwin.so` |
| 1.3% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 19.6% | 4 | `skimage/feature/_hog.py:48(hog)` |
| 16.7% | 40 | `<built-in method scipy.ndimage._nd_image.correlate1d>` |
| 16.4% | 63,504 | `skimage/feature/_hog.py:7(_hog_normalize_block)` |
| 5.1% | 16 | `numpy/lib/_shape_base_impl.py:60(take_along_axis)` |
| 4.3% | 128,467 | `numpy/_core/fromnumeric.py:66(_wrapreduction)` |
| 4.0% | 130,089 | `<method 'reduce' of 'numpy.ufunc' objects>` |
| 3.6% | 127,954 | `numpy/_core/fromnumeric.py:2342(sum)` |
| 3.1% | 8 | `<built-in method scipy.ndimage._nd_image.correlate>` |
| 2.8% | 12 | `scipy/ndimage/_measurements.py:43(label)` |
| 2.4% | 4 | `skimage/feature/_canny.py:103(canny)` |

## `pandas_sales`

Analysing a table of sales: filters, group-by, pivot, resample, rolling, merge, rank, binning. Source: pandas, called as a library.

### Small: 100,000 sales in 200 stores over 365 days

#### Calls in one run

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `np.clip` | int64 | 2 | |
| `x[mask]` | float64 | 21 | |
| `x[mask] = v` | float64 | 2 | |
| `np.nonzero` | bool | 27 | |
| `np.cumsum` | bool | 2 | |
| `np.sum` | uint8 | 6 | |
| `np.bincount` | int64 | 2 | |
| `np.where(c, a, b)` | float64 | 1 | |
| `np.unique` | int64 | 1 | |
| `np.argsort, stable` | int64 | 1 | |
| `x[mask]` | | | 24 |
| `x[mask] = v` | | | 8 |
| `np.nonzero` | | | 4 |
| `np.cumsum` | | | 3 |
| `x.astype(float32)` | | | 60 |
| `np.bincount` | | | 6 |
| `np.searchsorted` | | | 2 |
| `np.where(c, a, b)` | | | 1 |
| `m.any(axis=1)` | | | 2 |
| `np.unique` | | | 2 |
| `np.argsort, stable` | | | 5 |

#### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 21.8% | 21.8% |
| NumPy | 33.6% | 28.3% |
| BLAS | 0.0% | 0.0% |
| Maths library | 0.3% | 0.0% |
| hotkernels | 0.0% | 0.7% |
| Memory and system | 5.6% | 8.1% |
| Other compiled code | 38.7% | 41.2% |

The largest functions on plain NumPy (3,731 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 11.6% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 7.4% | NumPy | `binsearch<npy::floating_point_type<double, (NPY_TYPES)12>, (side_t)0>(` | `_multiarray_umath.cpython-312-darwin.so` |
| 6.1% | NumPy | `introselect_<npy::floating_point_type<double, (NPY_TYPES)12>, false, d` | `_multiarray_umath.cpython-312-darwin.so` |
| 4.7% | Other compiled code | `__pyx_pw_6pandas_5_libs_9hashtable_14Int64HashTable_29_unique` | `hashtable.cpython-312-darwin.so` |
| 3.8% | Other compiled code | `__pyx_fuse_0__pyx_pw_6pandas_5_libs_7groupby_67group_sum` | `groupby.cpython-312-darwin.so` |
| 3.5% | Other compiled code | `kh_put_int64` | `hashtable.cpython-312-darwin.so` |
| 3.4% | NumPy | `aquicksort_<npy::integral_type<long, (NPY_TYPES)7>, long, false>(long*` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.4% | Other compiled code | `__pyx_pw_6pandas_5_libs_9hashtable_15StringHashTable_21_unique` | `hashtable.cpython-312-darwin.so` |
| 2.7% | Memory and system | `_tlv_get_addr` | `libdyld.dylib` |
| 2.6% | Other compiled code | `__pyx_pw_6pandas_5_libs_9hashtable_13Int8HashTable_27_unique` | `hashtable.cpython-312-darwin.so` |
| 2.4% | Other compiled code | `__pyx_fuse_9__pyx_f_6pandas_5_libs_5algos_kth_smallest_c` | `algos.cpython-312-darwin.so` |
| 1.9% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 1.9% | Python interpreter | `PyType_IsSubtype` | `libpython3.12.dylib` |
| 1.7% | NumPy | `aquicksort_<npy::datetime_type<long, (NPY_TYPES)21>, long, false>(long` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.7% | Other compiled code | `__pyx_pw_6pandas_5_libs_5algos_229take_2d_axis0_float64_float64` | `algos.cpython-312-darwin.so` |

The largest after `install()` (3,670 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 11.7% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 8.1% | NumPy | `binsearch<npy::floating_point_type<double, (NPY_TYPES)12>, (side_t)0>(` | `_multiarray_umath.cpython-312-darwin.so` |
| 6.4% | NumPy | `introselect_<npy::floating_point_type<double, (NPY_TYPES)12>, false, d` | `_multiarray_umath.cpython-312-darwin.so` |
| 5.0% | Other compiled code | `__pyx_pw_6pandas_5_libs_9hashtable_14Int64HashTable_29_unique` | `hashtable.cpython-312-darwin.so` |
| 4.4% | Other compiled code | `__pyx_fuse_0__pyx_pw_6pandas_5_libs_7groupby_67group_sum` | `groupby.cpython-312-darwin.so` |
| 4.2% | Other compiled code | `kh_put_int64` | `hashtable.cpython-312-darwin.so` |
| 3.9% | Memory and system | `_tlv_get_addr` | `libdyld.dylib` |
| 3.6% | NumPy | `aquicksort_<npy::integral_type<long, (NPY_TYPES)7>, long, false>(long*` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.5% | Other compiled code | `__pyx_pw_6pandas_5_libs_9hashtable_15StringHashTable_21_unique` | `hashtable.cpython-312-darwin.so` |
| 2.8% | Other compiled code | `__pyx_pw_6pandas_5_libs_9hashtable_13Int8HashTable_27_unique` | `hashtable.cpython-312-darwin.so` |
| 2.6% | Other compiled code | `__pyx_fuse_9__pyx_f_6pandas_5_libs_5algos_kth_smallest_c` | `algos.cpython-312-darwin.so` |
| 2.3% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 2.0% | NumPy | `aquicksort_<npy::datetime_type<long, (NPY_TYPES)21>, long, false>(long` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.8% | Python interpreter | `PyType_IsSubtype` | `libpython3.12.dylib` |
| 1.7% | Other compiled code | `__pyx_f_6pandas_5_libs_7missing_is_matching_na` | `missing.cpython-312-darwin.so` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 9.3% | 15 | `pandas/core/algorithms.py:727(factorize_array)` |
| 6.5% | 8 | `pandas/core/groupby/ops.py:374(_call_cython_op)` |
| 5.9% | 20 | `<method 'argsort' of 'numpy.ndarray' objects>` |
| 5.0% | 2 | `<method 'searchsorted' of 'numpy.ndarray' objects>` |
| 4.6% | 2 | `<method 'partition' of 'numpy.ndarray' objects>` |
| 2.7% | 9,967 | `<built-in method builtins.isinstance>` |
| 2.5% | 69 | `pandas/core/array_algos/take.py:130(_take_nd_ndarray)` |
| 2.1% | 1 | `pandas/core/groupby/generic.py:890(nunique)` |
| 1.8% | 1 | `pandas/core/arrays/datetimes.py:147(f)` |
| 1.7% | 6 | `pandas/core/algorithms.py:473(unique_with_mask)` |

### Large: 2,000,000 sales in 200 stores over 365 days

#### Calls in one run

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `np.clip` | int64 | 2 | |
| `x[mask]` | float64 | 21 | |
| `x[mask] = v` | float64 | 2 | |
| `np.nonzero` | bool | 27 | |
| `np.cumsum` | bool | 2 | |
| `np.sum` | uint8 | 6 | |
| `np.bincount` | int64 | 2 | |
| `np.where(c, a, b)` | float64 | 1 | |
| `np.unique` | int64 | 1 | |
| `np.argsort, stable` | int64 | 1 | |
| `x[mask]` | | | 24 |
| `x[mask] = v` | | | 8 |
| `np.nonzero` | | | 4 |
| `np.cumsum` | | | 3 |
| `x.astype(float32)` | | | 60 |
| `np.bincount` | | | 6 |
| `np.searchsorted` | | | 2 |
| `np.where(c, a, b)` | | | 1 |
| `m.any(axis=1)` | | | 2 |
| `np.unique` | | | 2 |
| `np.argsort, stable` | | | 5 |

#### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 4.5% | 4.6% |
| NumPy | 41.9% | 37.8% |
| BLAS | 0.0% | 0.0% |
| Maths library | 0.2% | 0.0% |
| hotkernels | 0.0% | 0.8% |
| Memory and system | 4.2% | 4.6% |
| Other compiled code | 49.2% | 52.1% |

The largest functions on plain NumPy (4,098 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 9.0% | NumPy | `binsearch<npy::floating_point_type<double, (NPY_TYPES)12>, (side_t)0>(` | `_multiarray_umath.cpython-312-darwin.so` |
| 6.8% | NumPy | `introselect_<npy::floating_point_type<double, (NPY_TYPES)12>, false, d` | `_multiarray_umath.cpython-312-darwin.so` |
| 6.5% | Other compiled code | `kh_put_int64` | `hashtable.cpython-312-darwin.so` |
| 6.2% | Other compiled code | `__pyx_pw_6pandas_5_libs_9hashtable_14Int64HashTable_29_unique` | `hashtable.cpython-312-darwin.so` |
| 5.2% | NumPy | `aquicksort_<npy::integral_type<long, (NPY_TYPES)7>, long, false>(long*` | `_multiarray_umath.cpython-312-darwin.so` |
| 4.4% | Other compiled code | `__pyx_fuse_0__pyx_pw_6pandas_5_libs_7groupby_67group_sum` | `groupby.cpython-312-darwin.so` |
| 3.6% | Other compiled code | `__pyx_pw_6pandas_5_libs_9hashtable_15StringHashTable_21_unique` | `hashtable.cpython-312-darwin.so` |
| 2.8% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 2.8% | Other compiled code | `__pyx_pw_6pandas_5_libs_9hashtable_13Int8HashTable_27_unique` | `hashtable.cpython-312-darwin.so` |
| 2.6% | NumPy | `aquicksort_<npy::datetime_type<long, (NPY_TYPES)21>, long, false>(long` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.6% | Python interpreter | `PyType_IsSubtype` | `libpython3.12.dylib` |
| 2.6% | Other compiled code | `__pyx_fuse_9__pyx_f_6pandas_5_libs_5algos_kth_smallest_c` | `algos.cpython-312-darwin.so` |
| 2.4% | Other compiled code | `__pyx_pf_6pandas_5_libs_7groupby_74group_sum` | `groupby.cpython-312-darwin.so` |
| 2.2% | NumPy | `amerge_at_<npy::integral_type<long, (NPY_TYPES)7>, long, false>(long*,` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.0% | Other compiled code | `__pyx_fuse_3__pyx_f_6pandas_5_libs_9hashtable_value_count` | `hashtable.cpython-312-darwin.so` |

The largest after `install()` (4,100 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 9.7% | NumPy | `binsearch<npy::floating_point_type<double, (NPY_TYPES)12>, (side_t)0>(` | `_multiarray_umath.cpython-312-darwin.so` |
| 7.4% | NumPy | `introselect_<npy::floating_point_type<double, (NPY_TYPES)12>, false, d` | `_multiarray_umath.cpython-312-darwin.so` |
| 6.8% | Other compiled code | `kh_put_int64` | `hashtable.cpython-312-darwin.so` |
| 6.3% | Other compiled code | `__pyx_pw_6pandas_5_libs_9hashtable_14Int64HashTable_29_unique` | `hashtable.cpython-312-darwin.so` |
| 5.6% | NumPy | `aquicksort_<npy::integral_type<long, (NPY_TYPES)7>, long, false>(long*` | `_multiarray_umath.cpython-312-darwin.so` |
| 4.8% | Other compiled code | `__pyx_fuse_0__pyx_pw_6pandas_5_libs_7groupby_67group_sum` | `groupby.cpython-312-darwin.so` |
| 3.9% | Other compiled code | `__pyx_pw_6pandas_5_libs_9hashtable_15StringHashTable_21_unique` | `hashtable.cpython-312-darwin.so` |
| 3.0% | Other compiled code | `__pyx_pw_6pandas_5_libs_9hashtable_13Int8HashTable_27_unique` | `hashtable.cpython-312-darwin.so` |
| 2.9% | NumPy | `aquicksort_<npy::datetime_type<long, (NPY_TYPES)21>, long, false>(long` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.8% | Python interpreter | `PyType_IsSubtype` | `libpython3.12.dylib` |
| 2.8% | Other compiled code | `__pyx_pf_6pandas_5_libs_7groupby_74group_sum` | `groupby.cpython-312-darwin.so` |
| 2.7% | Memory and system | `_platform_memmove` | `libsystem_platform.dylib` |
| 2.6% | Other compiled code | `__pyx_fuse_9__pyx_f_6pandas_5_libs_5algos_kth_smallest_c` | `algos.cpython-312-darwin.so` |
| 2.4% | Other compiled code | `__pyx_fuse_3__pyx_f_6pandas_5_libs_9hashtable_value_count` | `hashtable.cpython-312-darwin.so` |
| 2.0% | NumPy | `hwy::N_NEON::detail::Recurse<(hwy::N_NEON::detail::RecurseMode)0, hwy:` | `_multiarray_umath.cpython-312-darwin.so` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 16.7% | 15 | `pandas/core/algorithms.py:727(factorize_array)` |
| 12.6% | 20 | `<method 'argsort' of 'numpy.ndarray' objects>` |
| 10.9% | 8 | `pandas/core/groupby/ops.py:374(_call_cython_op)` |
| 8.4% | 2 | `<method 'searchsorted' of 'numpy.ndarray' objects>` |
| 7.4% | 2 | `<method 'partition' of 'numpy.ndarray' objects>` |
| 4.9% | 1 | `pandas/core/groupby/generic.py:890(nunique)` |
| 3.9% | 69 | `pandas/core/array_algos/take.py:130(_take_nd_ndarray)` |
| 3.2% | 1 | `pandas/core/arrays/datetimes.py:147(f)` |
| 2.8% | 6 | `pandas/core/algorithms.py:473(unique_with_mask)` |
| 2.1% | 1 | `pandas/core/algorithms.py:1123(value_counts_arraylike)` |

## `astropy_timeseries`

Survey statistics: sigma clipping, biweight, Lomb–Scargle periodograms, Bayesian blocks. Source: Astropy (astropy.stats, astropy.timeseries), called as a library.

### Small: 10 light curves of 1,000 points, 4 images of 256 × 256

#### Calls in one run

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `np.argmax` | float64 | 1,010 | |
| `x[mask]` | float64 | 230 | |
| `x[mask] = v` | float64 | 12 | |
| `np.cumsum` | bool | 2 | |
| `np.argsort, stable` | float64 | 2 | |
| `x[mask] = v` | | | 30 |
| `np.cumsum` | | | 2,000 |
| `x.astype(float32)` | | | 128 |
| `np.sum` | | | 60 |
| `np.bincount` | | | 840 |
| `np.where(c, a, b)` | | | 60 |
| `np.unique` | | | 2 |
| `np.round` | | | 10 |
| `np.median` | | | 44 |
| `bn.nanmean` | | | 4 |
| `bn.nanstd` | | | 50 |

#### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 11.6% | 12.6% |
| NumPy | 64.9% | 65.0% |
| BLAS | 0.2% | 0.1% |
| Maths library | 4.1% | 4.3% |
| hotkernels | 0.0% | 0.1% |
| Memory and system | 8.5% | 6.9% |
| Other compiled code | 10.8% | 11.0% |

The largest functions on plain NumPy (3,761 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 7.7% | NumPy | `introselect_<npy::floating_point_type<double, (NPY_TYPES)12>, false, d` | `_multiarray_umath.cpython-312-darwin.so` |
| 7.3% | Other compiled code | `nanmedian_all_float64` | `reduce.cpython-312-darwin.so` |
| 7.1% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 6.0% | NumPy | `pocketfft::detail::cfftp<double>::pass3<false, pocketfft::detail::cmpl` | `_pocketfft_umath.cpython-312-darwin.so` |
| 5.9% | NumPy | `pocketfft::detail::cfftp<double>::pass8<false, pocketfft::detail::cmpl` | `_pocketfft_umath.cpython-312-darwin.so` |
| 5.3% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 5.3% | NumPy | `_aligned_contig_cast_double_to_cdouble` | `_multiarray_umath.cpython-312-darwin.so` |
| 5.2% | NumPy | `pocketfft::detail::general_nd<pocketfft::detail::pocketfft_c<double>, ` | `_pocketfft_umath.cpython-312-darwin.so` |
| 4.4% | Memory and system | `_tlv_get_addr` | `libdyld.dylib` |
| 4.2% | NumPy | `CDOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.9% | NumPy | `DOUBLE_subtract` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.3% | NumPy | `_contig_to_strided_size16` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.0% | Other compiled code | `nanstd_all_float64` | `reduce.cpython-312-darwin.so` |
| 2.9% | NumPy | `_aligned_contig_to_strided_size8` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.1% | NumPy | `npyiter_buffered_reduce_iternext_iters2` | `_multiarray_umath.cpython-312-darwin.so` |

The largest after `install()` (3,801 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 8.4% | NumPy | `introselect_<npy::floating_point_type<double, (NPY_TYPES)12>, false, d` | `_multiarray_umath.cpython-312-darwin.so` |
| 7.6% | Other compiled code | `nanmedian_all_float64` | `reduce.cpython-312-darwin.so` |
| 7.5% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 6.5% | NumPy | `pocketfft::detail::cfftp<double>::pass3<false, pocketfft::detail::cmpl` | `_pocketfft_umath.cpython-312-darwin.so` |
| 6.4% | NumPy | `pocketfft::detail::cfftp<double>::pass8<false, pocketfft::detail::cmpl` | `_pocketfft_umath.cpython-312-darwin.so` |
| 5.7% | NumPy | `_aligned_contig_cast_double_to_cdouble` | `_multiarray_umath.cpython-312-darwin.so` |
| 5.5% | NumPy | `pocketfft::detail::general_nd<pocketfft::detail::pocketfft_c<double>, ` | `_pocketfft_umath.cpython-312-darwin.so` |
| 5.0% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.9% | Memory and system | `_tlv_get_addr` | `libdyld.dylib` |
| 3.7% | NumPy | `DOUBLE_subtract` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.4% | NumPy | `_contig_to_strided_size16` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.3% | NumPy | `CDOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.2% | Other compiled code | `nanstd_all_float64` | `reduce.cpython-312-darwin.so` |
| 2.7% | NumPy | `_aligned_contig_to_strided_size8` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.3% | NumPy | `CDOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 12.9% | 30 | `astropy/timeseries/periodograms/lombscargle/implementations/utils.py:327(nufft1_lra)` |
| 11.9% | 30 | `numpy/fft/_pocketfft.py:58(_raw_fft)` |
| 7.5% | 60 | `astropy/timeseries/periodograms/lombscargle/implementations/utils.py:217(chebyshev_polynomials)` |
| 5.6% | 5,006 | `python3.12/inspect.py:2393(_signature_from_function)` |
| 5.1% | 44 | `<method 'partition' of 'numpy.ndarray' objects>` |
| 4.8% | 50 | `<built-in method bottleneck.reduce.nanmedian>` |
| 4.8% | 10,012 | `python3.12/inspect.py:2497(_signature_from_callable)` |
| 3.8% | 10,012 | `python3.12/inspect.py:3025(__init__)` |
| 3.4% | 15,018 | `python3.12/inspect.py:2739(__init__)` |
| 3.1% | 312 | `<method 'reduce' of 'numpy.ufunc' objects>` |

### Large: 10 light curves of 5,000 points, 4 images of 1,024 × 1,024

#### Calls in one run

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `np.argmax` | float64 | 4,010 | |
| `x[mask]` | float64 | 235 | |
| `x[mask] = v` | float64 | 12 | |
| `np.cumsum` | bool | 2 | |
| `np.argsort, stable` | float64 | 2 | |
| `x[mask] = v` | | | 30 |
| `np.cumsum` | | | 8,000 |
| `x.astype(float32)` | | | 128 |
| `np.sum` | | | 60 |
| `np.bincount` | | | 840 |
| `np.where(c, a, b)` | | | 60 |
| `np.unique` | | | 2 |
| `np.round` | | | 10 |
| `np.median` | | | 44 |
| `bn.nanmean` | | | 4 |
| `bn.nanstd` | | | 55 |

#### Where the time goes

Share of the samples of working threads, by the class of the function at the top of the stack.

| Class | Plain NumPy | After `install()` |
|---|---|---|
| Python interpreter | 4.3% | 3.9% |
| NumPy | 67.1% | 66.2% |
| BLAS | 0.0% | 0.1% |
| Maths library | 3.3% | 3.7% |
| hotkernels | 0.0% | 0.3% |
| Memory and system | 3.2% | 3.4% |
| Other compiled code | 22.0% | 22.3% |

The largest functions on plain NumPy (4,036 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 17.4% | NumPy | `introselect_<npy::floating_point_type<double, (NPY_TYPES)12>, false, d` | `_multiarray_umath.cpython-312-darwin.so` |
| 15.4% | Other compiled code | `nanmedian_all_float64` | `reduce.cpython-312-darwin.so` |
| 9.5% | NumPy | `pocketfft::detail::cfftp<double>::pass8<false, pocketfft::detail::cmpl` | `_pocketfft_umath.cpython-312-darwin.so` |
| 7.8% | NumPy | `pocketfft::detail::general_nd<pocketfft::detail::pocketfft_c<double>, ` | `_pocketfft_umath.cpython-312-darwin.so` |
| 6.2% | Other compiled code | `nanstd_all_float64` | `reduce.cpython-312-darwin.so` |
| 4.2% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.6% | NumPy | `_aligned_contig_cast_double_to_cdouble` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.1% | NumPy | `_contig_to_strided_size16` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.8% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 2.7% | NumPy | `CDOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.5% | NumPy | `DOUBLE_subtract` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.0% | Maths library | `pow` | `libsystem_m.dylib` |
| 1.9% | NumPy | `_aligned_contig_to_strided_size8` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.8% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.7% | NumPy | `CDOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |

The largest after `install()` (4,007 samples):

| Share | Class | Function | In |
|---|---|---|---|
| 17.5% | NumPy | `introselect_<npy::floating_point_type<double, (NPY_TYPES)12>, false, d` | `_multiarray_umath.cpython-312-darwin.so` |
| 15.6% | Other compiled code | `nanmedian_all_float64` | `reduce.cpython-312-darwin.so` |
| 9.6% | NumPy | `pocketfft::detail::cfftp<double>::pass8<false, pocketfft::detail::cmpl` | `_pocketfft_umath.cpython-312-darwin.so` |
| 7.8% | NumPy | `pocketfft::detail::general_nd<pocketfft::detail::pocketfft_c<double>, ` | `_pocketfft_umath.cpython-312-darwin.so` |
| 6.2% | Other compiled code | `nanstd_all_float64` | `reduce.cpython-312-darwin.so` |
| 3.5% | NumPy | `DOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 3.0% | NumPy | `_aligned_contig_cast_double_to_cdouble` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.9% | NumPy | `_contig_to_strided_size16` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.8% | NumPy | `DOUBLE_subtract` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.6% | NumPy | `CDOUBLE_multiply` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.3% | NumPy | `_aligned_contig_to_strided_size8` | `_multiarray_umath.cpython-312-darwin.so` |
| 2.3% | Maths library | `pow` | `libsystem_m.dylib` |
| 2.3% | Python interpreter | `_PyEval_EvalFrameDefault` | `libpython3.12.dylib` |
| 2.2% | NumPy | `DOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |
| 1.7% | NumPy | `CDOUBLE_add` | `_multiarray_umath.cpython-312-darwin.so` |

The Python functions with the most time of their own, in one plain run under `cProfile`:

| Share | Calls | Function |
|---|---|---|
| 15.5% | 30 | `numpy/fft/_pocketfft.py:58(_raw_fft)` |
| 12.9% | 44 | `<method 'partition' of 'numpy.ndarray' objects>` |
| 11.6% | 55 | `<built-in method bottleneck.reduce.nanmedian>` |
| 9.9% | 30 | `astropy/timeseries/periodograms/lombscargle/implementations/utils.py:327(nufft1_lra)` |
| 5.6% | 60 | `astropy/timeseries/periodograms/lombscargle/implementations/utils.py:217(chebyshev_polynomials)` |
| 4.8% | 55 | `<built-in method bottleneck.reduce.nanstd>` |
| 3.5% | 20,006 | `python3.12/inspect.py:2393(_signature_from_function)` |
| 3.0% | 40,012 | `python3.12/inspect.py:2497(_signature_from_callable)` |
| 2.6% | 4 | `astropy/stats/biweight.py:339(biweight_midvariance)` |
| 2.4% | 312 | `<method 'reduce' of 'numpy.ufunc' objects>` |
