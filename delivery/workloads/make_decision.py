"""numpy-benchmarks' make_decision: nearest constellation symbol for each received sample.

The NumPy code is in vendor/numpy_benchmarks, copied unchanged. For each
sample it takes `np.argmin` over the squared distances to 32 symbols: one
float64 `np.argmin` per sample, which reaches the argmin kernel.
"""
from __future__ import annotations

import numpy as np

from . import Workload
from .vendor.numpy_benchmarks.make_decision import make_decision

# Seed of NumPy's global generator, as the benchmark's own setup line sets it.
SEED = 0
# Number of constellation symbols, drawn from the samples, as in the benchmark.
SYMBOLS = 32
# Number of received samples. The benchmark's own setup uses 4,096. These are
# set for the times in SIZE_NAMES.
SIZES = {
    "S": {"samples": 8_192},
    "M": {"samples": 81_920},
    "L": {"samples": 819_200},
}


def make_data(samples: int) -> tuple:
    """Build the inputs as the benchmark's setup line does, at another length.

    Args:
        samples: Number of received complex samples.

    Returns:
        (samples, symbols), both complex128.
    """
    np.random.seed(SEED)
    received = np.random.randn(samples) + np.random.randn(samples) * 1.j
    symbols = np.random.choice(received, SYMBOLS)
    return received, symbols


def run(received, symbols) -> tuple:
    """Decide every sample once.

    Args:
        received: The received samples, complex128.
        symbols: The constellation, complex128.

    Returns:
        A one-element tuple: the symbol chosen for each sample.
    """
    return (make_decision(received, symbols),)


WORKLOAD = Workload(
    name="make_decision",
    title="numpy-benchmarks make_decision: nearest of 32 symbols for each sample, by np.argmin",
    source="https://github.com/serge-sans-paille/numpy-benchmarks, commit deef91c, "
           "numpy_benchmarks/benchmarks/make_decision.py",
    licence="BSD-3-Clause (numpy-benchmarks)",
    ours=False,
    sizes=SIZES,
    make_data=make_data,
    run=run,
)
