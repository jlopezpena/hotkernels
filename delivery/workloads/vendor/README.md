# Copied benchmark code

Files under this directory are copied unchanged from published benchmark suites, so that a workload runs code that was not written for hotkernels. Do not edit them. A workload may convert their inputs and outputs, but not change their NumPy calls. See [docs/workloads-design.md](../../../docs/workloads-design.md).

| Directory | Source | Commit | Licence |
|---|---|---|---|
| `npbench/` | [NPBench](https://github.com/spcl/npbench) | `f2d7f27` | BSD-3-Clause, in `npbench/LICENSE`. `azimint_naive_numpy.py` carries pyFAI's CC-BY 3.0 notice in its header |
