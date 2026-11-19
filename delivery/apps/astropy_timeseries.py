"""Reduces the light curves and images of a small survey with Astropy.

The routine statistics of an astronomer's reduction script, on simulated data:

- for each image: the sigma-clipped mean, median and standard deviation of
  the background, and its robust biweight location and scale;
- for each light curve, a star observed at irregular times with noise and a
  periodic signal: sigma clipping of outliers, a Lomb–Scargle periodogram by
  the fast method and its best period, and the median absolute deviation;
- for some of the light curves, their Bayesian blocks: the optimal piecewise
  constant segmentation of the measurements.

Astropy installs bottleneck where it finds it, as it does here: its sigma
clipping then calls bottleneck's NaN-aware reductions.
"""
from __future__ import annotations

import numpy as np

TITLE = "Survey statistics: sigma clipping, biweight, Lomb–Scargle periodograms, Bayesian blocks"
SOURCE = "Astropy (astropy.stats, astropy.timeseries), called as a library"
LICENCE = None
REQUIRES = ("astropy",)

# By size: (points in each light curve, side of each square image in pixels).
SHAPES = {"small": (1_000, 256), "large": (5_000, 1_024)}
# Light curves in the survey.
CURVES = 10
# Images in the survey.
IMAGES = 4
# Light curves segmented by Bayesian blocks, whose cost grows as the square of their length.
BLOCK_CURVES = 2
# Points of each of those curves segmented, at most: the first ones.
BLOCK_POINTS = {"small": 500, "large": 2_000}
# Days the observations span.
SPAN_DAYS = 100.0
# Fraction of the points of a light curve that are outliers.
OUTLIER_FRACTION = 0.01
# Fraction of the pixels of an image that are stars, far above the background.
STAR_FRACTION = 0.02
# Clipping threshold, in standard deviations.
CLIP_SIGMA = 3.0
# Seed of the data.
SEED = 0
# One line per size saying how large its data is.
SIZES = {size: f"{CURVES} light curves of {points:,} points, {IMAGES} images of {side:,} × {side:,}"
         for size, (points, side) in SHAPES.items()}


def make_data(size: str) -> tuple:
    """Simulate light curves and images.

    Args:
        size: One of SIZES.

    Returns:
        (observation times as a (CURVES × points) array in days, magnitudes
        of the same shape, their uncertainties, images as a list of float64
        arrays, the number of points segmented by Bayesian blocks).
    """
    points, side = SHAPES[size]
    rng = np.random.default_rng(SEED)
    times = np.sort(rng.uniform(0.0, SPAN_DAYS, (CURVES, points)), axis=1)
    periods = rng.uniform(0.2, 10.0, (CURVES, 1))
    errors = rng.uniform(0.01, 0.05, (CURVES, points))
    magnitudes = 15.0 + 0.3 * np.sin(2 * np.pi * times / periods) + errors * rng.standard_normal((CURVES, points))
    outliers = rng.random((CURVES, points)) < OUTLIER_FRACTION
    magnitudes[outliers] += rng.normal(0.0, 2.0, outliers.sum())
    images = []
    for _ in range(IMAGES):
        image = rng.normal(100.0, 5.0, (side, side))
        stars = rng.random((side, side)) < STAR_FRACTION
        image[stars] += rng.exponential(500.0, stars.sum())
        images.append(image)
    return times, magnitudes, errors, images, BLOCK_POINTS[size]


def run(times: np.ndarray, magnitudes: np.ndarray, errors: np.ndarray, images: list, block_points: int) -> tuple:
    """Reduce the survey.

    Args:
        times: Observation times of each light curve, in days, one curve per row.
        magnitudes: Measured magnitudes, the same shape.
        errors: Their uncertainties, the same shape.
        images: Images of the sky.
        block_points: Points of a light curve given to Bayesian blocks.

    Returns:
        Every statistic computed, as floats and arrays.
    """
    from astropy.stats import (bayesian_blocks, biweight_location, biweight_scale, mad_std, sigma_clip,
                               sigma_clipped_stats)
    from astropy.timeseries import LombScargle

    image_stats = tuple(
        (*sigma_clipped_stats(image, sigma=CLIP_SIGMA), biweight_location(image), biweight_scale(image))
        for image in images)
    curve_stats = []
    for t, m, e in zip(times, magnitudes, errors):
        clipped = sigma_clip(m, sigma=CLIP_SIGMA)
        keep = ~clipped.mask
        frequency, power = LombScargle(t[keep], m[keep], e[keep]).autopower(method="fast")
        curve_stats.append((clipped.mask, frequency, power, 1.0 / frequency[np.argmax(power)], mad_std(m)))
    blocks = tuple(
        bayesian_blocks(t[:block_points], m[:block_points], e[:block_points], fitness="measures")
        for t, m, e in zip(times[:BLOCK_CURVES], magnitudes[:BLOCK_CURVES], errors[:BLOCK_CURVES]))
    return image_stats, tuple(curve_stats), blocks
