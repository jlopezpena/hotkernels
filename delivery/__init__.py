"""Development tool that tests and times hotkernels as a user meets it.

The judge proves a kernel right as a native function. This package tests the
code between NumPy and the kernels: it runs NumPy calls with and without
hotkernels installed and compares everything a caller could observe.

Run with `uv run delivery <command>`.
"""
