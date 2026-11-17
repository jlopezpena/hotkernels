"""A small NumPy program, written with no knowledge of hotkernels.

It cleans a signal, summarises an 8-bit image and looks up some ids, with
the spellings people use: `x[mask]`, `x[mask] = v`, methods and functions.
Nothing here imports hotkernels. `uv run delivery demo` runs it as it is and
again after `hotkernels.install()`, and compares the two.
"""
import numpy as np

# Element count of each of the three arrays.
ELEMENTS = 2_000_000
# Readings beyond this many standard deviations are treated as faults.
FAULT_LIMIT = 3.0
# Range the cleaned signal is clamped to, and the bins of its histogram.
LOW, HIGH = -2.0, 2.0
BINS = 64
# Number of distinct ids, and the ones looked up.
ID_COUNT = 1000
WANTED_IDS = [3, 14, 15, 92, 65, 35, 89, 79]


def make_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the program's inputs.

    Args:
        seed: Seed of the random generator.

    Returns:
        Sensor readings as float64, an image as uint8 and ids as int64, each
        of ELEMENTS elements.
    """
    rng = np.random.default_rng(seed)
    readings = rng.standard_normal(ELEMENTS)
    pixels = rng.integers(0, 256, ELEMENTS, dtype=np.uint8)
    ids = rng.integers(0, ID_COUNT, ELEMENTS)
    return readings, pixels, ids


def analyse(readings: np.ndarray, pixels: np.ndarray, ids: np.ndarray) -> dict:
    """Run the analysis.

    Args:
        readings: Sensor readings. Not modified.
        pixels: An 8-bit image.
        ids: One id per reading.

    Returns:
        Every result, by name.
    """
    signal = readings.copy()
    faults = np.abs(signal) > FAULT_LIMIT
    signal[faults] = 0.0
    signal = np.clip(signal, LOW, HIGH)
    rising = signal[signal > 0]
    counts, edges = np.histogram(signal, BINS, (LOW, HIGH))

    wanted = np.isin(ids, WANTED_IDS)
    return {
        "peak": np.argmax(signal),
        "trough": np.argmin(signal),
        "rising": rising,
        "histogram": counts,
        "edges": edges,
        "image": pixels.astype(np.float32),
        "brightness": np.sum(pixels),
        "levels": np.bincount(pixels),
        "wanted_at": np.nonzero(wanted)[0],
        "id_running_total": np.cumsum(ids),
        "ids_in_use": np.count_nonzero(ids),
        "signal": signal,
    }


if __name__ == "__main__":
    analyse(*make_data())
