# Copied benchmark code

Files under this directory are copied unchanged from published benchmark suites, so that a workload runs code that was not written for hotkernels. Do not edit them. A workload may convert their inputs and outputs, but not change their NumPy calls. See [docs/workloads-design.md](../../../docs/workloads-design.md).

| Directory | Source | Commit | Licence |
|---|---|---|---|
| `npbench/` | [NPBench](https://github.com/spcl/npbench) | `f2d7f27` | BSD-3-Clause, in `npbench/LICENSE`. `azimint_naive_numpy.py` and `azimint_hist_numpy.py` carry pyFAI's CC-BY 3.0 notice in their headers, and `mandelbrot1_numpy.py` and `mandelbrot2_numpy.py` Nicolas Rougier's BSD notice |
| `numpy_benchmarks/` | [numpy-benchmarks](https://github.com/serge-sans-paille/numpy-benchmarks) | `deef91c` | BSD-3-Clause, in `numpy_benchmarks/LICENSE` |
