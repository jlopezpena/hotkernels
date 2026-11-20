"""Tracks a moving target with a bootstrap particle filter, written for this study.

A target moves in the plane at a nearly constant velocity. A sensor at the
origin measures its range and bearing with noise. The filter keeps a cloud of
particles, each a guess of the state (x, y, vx, vy), as one (particles × 4)
array, which is how such filters are usually written:

- **predict:** move every particle by its velocity and add process noise, then
  slow down the particles that exceed the largest plausible speed;
- **weigh:** compare each particle's predicted range and bearing with the
  measurement, in log-likelihoods, and normalise the weights;
- **estimate:** the weighted mean and covariance of the cloud, and the
  particle of largest weight;
- **resample** when the effective sample size drops below half the cloud:
  systematic resampling by a cumulative sum of the weights and a search of
  evenly spaced positions in it.

The code is written the way a scientist would write it from a textbook, not
for hotkernels.
"""
from __future__ import annotations

import numpy as np

TITLE = "A bootstrap particle filter tracking a target from range and bearing"
SOURCE = "Written for this study"
LICENCE = None
REQUIRES = ()

# By size: particles in the cloud.
PARTICLES = {"small": 2_000, "large": 100_000}
# Time steps filtered in one run.
STEPS = 100
# Seconds between measurements.
DT = 1.0
# Standard deviation of the process noise on position (m) and velocity (m/s), per step.
PROCESS_NOISE = np.array([0.5, 0.5, 0.2, 0.2])
# Standard deviation of a range measurement, in metres.
RANGE_NOISE = 2.0
# Standard deviation of a bearing measurement, in radians.
BEARING_NOISE = 0.02
# The largest speed a particle may keep, in m/s; faster ones are scaled down to it.
MAX_SPEED = 15.0
# The target's true initial state: x, y (m), vx, vy (m/s).
TRUE_START = np.array([200.0, 100.0, 4.0, 6.0])
# The filter's initial guess of the state, the centre of its first cloud.
GUESS = np.array([180.0, 120.0, 0.0, 0.0])
# The standard deviation of the first cloud around GUESS, per component.
GUESS_SPREAD = np.array([30.0, 30.0, 5.0, 5.0])
# The cloud is resampled when the effective sample size falls below this fraction of it.
RESAMPLE_BELOW = 0.5
# Seed of the simulated target and its measurements.
TRUTH_SEED = 1
# Seed of the filter's own random draws, reset at the start of every run.
FILTER_SEED = 2
# One line per size saying how large its data is.
SIZES = {size: f"{particles:,} particles, {STEPS} steps" for size, particles in PARTICLES.items()}


def make_data(size: str) -> tuple:
    """Simulate a target's path and the sensor's measurements of it.

    Args:
        size: One of SIZES.

    Returns:
        (number of particles, measurements as a (STEPS × 2) float64 array of
        range in metres and bearing in radians).
    """
    rng = np.random.default_rng(TRUTH_SEED)
    state = TRUE_START.copy()
    measurements = np.empty((STEPS, 2))
    for step in range(STEPS):
        state[:2] += state[2:] * DT
        state[2:] += rng.normal(0.0, 0.1, 2)
        measurements[step] = (np.hypot(state[0], state[1]) + rng.normal(0.0, RANGE_NOISE),
                              np.arctan2(state[1], state[0]) + rng.normal(0.0, BEARING_NOISE))
    return PARTICLES[size], measurements


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Choose which particles survive, each in proportion to its weight.

    Args:
        weights: Normalised weights, summing to one.
        rng: The filter's random generator.

    Returns:
        The index of the particle each new particle copies, as int64.
    """
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0  # rounding must not leave the last position past the end
    return np.searchsorted(cumulative, positions)


def run(n: int, measurements: np.ndarray) -> tuple:
    """Filter every measurement.

    Args:
        n: Particles in the cloud.
        measurements: Range and bearing at each step.

    Returns:
        (weighted mean state at each step, covariance at each step, the most
        likely particle at each step, effective sample size at each step,
        number of times the cloud was resampled).
    """
    rng = np.random.default_rng(FILTER_SEED)
    particles = GUESS + GUESS_SPREAD * rng.standard_normal((n, 4))
    log_weights = np.zeros(n)
    means = np.empty((len(measurements), 4))
    covariances = np.empty((len(measurements), 4, 4))
    best = np.empty((len(measurements), 4))
    ess = np.empty(len(measurements))
    resampled = 0
    for step, (measured_range, measured_bearing) in enumerate(measurements):
        # Predict.
        particles[:, :2] += particles[:, 2:] * DT
        particles += PROCESS_NOISE * rng.standard_normal((n, 4))
        speed = np.hypot(particles[:, 2], particles[:, 3])
        too_fast = speed > MAX_SPEED
        particles[too_fast, 2:] *= (MAX_SPEED / speed[too_fast])[:, None]

        # Weigh.
        predicted_range = np.sqrt(particles[:, 0] ** 2 + particles[:, 1] ** 2)
        predicted_bearing = np.arctan2(particles[:, 1], particles[:, 0])
        range_error = (predicted_range - measured_range) / RANGE_NOISE
        bearing_error = (predicted_bearing - measured_bearing + np.pi) % (2 * np.pi) - np.pi
        bearing_error /= BEARING_NOISE
        log_weights += -0.5 * (range_error ** 2 + bearing_error ** 2)
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= np.sum(weights)

        # Estimate.
        means[step] = weights @ particles
        centred = particles - means[step]
        covariances[step] = (centred * weights[:, None]).T @ centred
        best[step] = particles[np.argmax(weights)]
        ess[step] = 1.0 / np.sum(weights ** 2)

        # Resample.
        if ess[step] < RESAMPLE_BELOW * n:
            particles = particles[systematic_resample(weights, rng)]
            log_weights = np.zeros(n)
            resampled += 1
        else:
            log_weights = np.log(weights)
    return means, covariances, best, ess, resampled
