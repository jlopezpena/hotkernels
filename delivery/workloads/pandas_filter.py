"""A pandas boolean filter, `s[s > 0]`, as pandas users write it.

pandas makes the NumPy calls. The result is converted to NumPy arrays, values
and index, so that the two runs can be compared bit for bit.
"""
from __future__ import annotations

import numpy as np

from . import Workload

# Seed of the input data.
SEED = 0
# Element count of the Series, per size. No L: it would need about 2.4 GB per copy.
SIZES = {
    "S": {"elements": 3_000_000},
    "M": {"elements": 30_000_000},
}


def make_data(elements: int) -> tuple:
    """Build a Series of standard normal values with a default index.

    Args:
        elements: Length of the Series.

    Returns:
        A one-element tuple: the Series.
    """
    import pandas as pd

    return (pd.Series(np.random.default_rng(SEED).standard_normal(elements)),)


def run(series) -> tuple:
    """Keep the positive values.

    Args:
        series: A pandas Series of float64.

    Returns:
        (values kept, their index labels), as NumPy arrays.
    """
    kept = series[series > 0]
    return kept.to_numpy(), kept.index.to_numpy()


WORKLOAD = Workload(
    name="pandas_filter",
    title="pandas: s[s > 0] on a float64 Series",
    source="pandas.Series.__getitem__ with a boolean Series",
    licence=None,
    ours=False,
    sizes=SIZES,
    make_data=make_data,
    run=run,
    requires=("pandas",),
)
