"""A fractal flame renderer in plain NumPy, written by this project.

The algorithm is Draves and Reckase's, "The Fractal Flame Algorithm" (2003),
as Apophysis and flam3 render it. All points play the chaos game together:
on each step every point picks a transform by weight, goes through the
transform's affine map and its weighted variations, and moves its colour
towards the transform's. After the first steps the points are summed into a
density and a colour histogram, which are tone-mapped with the log of the
density and a gamma.

No benchmark suite has a NumPy flame renderer, so this one was written here,
in the spelling the algorithm suggests, not to reach hooks: one mask per
transform, `np.bincount` for the histograms, `np.interp` for the palette. It
is reported with the workloads this project wrote.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np

from . import Workload

# Seed of the generator that draws the starting points and every random choice.
SEED = 0
# The part of the plane drawn: (xmin, xmax, ymin, ymax). Its 4:3 aspect matches the images.
VIEW = (-1.6, 1.6, -1.2, 1.2)
# Gamma of the tone mapping, as flam3's default.
GAMMA = 2.2
# Entries in the palette gradient.
PALETTE_ENTRIES = 256
# Phase of each colour channel's cosine in the palette, in turns.
PALETTE_PHASES = (0.0, 0.33, 0.67)
# Points; the image's width and height in pixels; chaos-game steps; and how
# many of the first steps are run without plotting, so that the points settle
# onto the attractor (flam3 calls these the fuse). The image grows with the
# points so that each pixel gets about as many hits at M and L. S takes fewer
# steps, since each step costs about a millisecond in Python whatever the size.
SIZES = {
    "S": {"points": 10_000, "width": 256, "height": 192, "steps": 10, "fuse": 4},
    "M": {"points": 25_000, "width": 512, "height": 384, "steps": 40, "fuse": 15},
    "L": {"points": 300_000, "width": 1024, "height": 768, "steps": 40, "fuse": 15},
}


def linear(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Variation 0, linear: the point unchanged.

    Args:
        x: x coordinates after the affine map.
        y: y coordinates after the affine map.
        rng: The generator; unused, as in every variation but julia.

    Returns:
        The new (x, y).
    """
    return x, y


def sinusoidal(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Variation 1, sinusoidal: (sin x, sin y).

    Args:
        x: x coordinates after the affine map.
        y: y coordinates after the affine map.
        rng: The generator; unused.

    Returns:
        The new (x, y).
    """
    return np.sin(x), np.sin(y)


def spherical(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Variation 2, spherical: inversion in the unit circle, (x, y) / r².

    Args:
        x: x coordinates after the affine map.
        y: y coordinates after the affine map.
        rng: The generator; unused.

    Returns:
        The new (x, y). A point at the origin goes to infinity or NaN.
    """
    r2 = x * x + y * y
    return x / r2, y / r2


def swirl(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Variation 3, swirl: a rotation by r².

    Args:
        x: x coordinates after the affine map.
        y: y coordinates after the affine map.
        rng: The generator; unused.

    Returns:
        The new (x, y).
    """
    r2 = x * x + y * y
    s, c = np.sin(r2), np.cos(r2)
    return x * s - y * c, x * c + y * s


def horseshoe(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Variation 4, horseshoe: ((x - y)(x + y), 2xy) / r.

    Args:
        x: x coordinates after the affine map.
        y: y coordinates after the affine map.
        rng: The generator; unused.

    Returns:
        The new (x, y). A point at the origin goes to NaN.
    """
    r = np.sqrt(x * x + y * y)
    return (x - y) * (x + y) / r, 2 * x * y / r


def polar(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Variation 5, polar: (θ / π, r - 1), with θ measured from the y axis as in the paper.

    Args:
        x: x coordinates after the affine map.
        y: y coordinates after the affine map.
        rng: The generator; unused.

    Returns:
        The new (x, y).
    """
    return np.arctan2(x, y) / np.pi, np.sqrt(x * x + y * y) - 1


def julia(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Variation 13, julia: a square root of the point, with a random choice of the two roots.

    Args:
        x: x coordinates after the affine map.
        y: y coordinates after the affine map.
        rng: The generator, which picks the root for each point.

    Returns:
        The new (x, y).
    """
    sqrt_r = np.sqrt(np.sqrt(x * x + y * y))
    theta = np.arctan2(x, y) / 2 + np.pi * rng.integers(0, 2, x.shape)
    return sqrt_r * np.cos(theta), sqrt_r * np.sin(theta)


# A variation: (x, y, generator) -> new (x, y).
Variation = Callable[[np.ndarray, np.ndarray, np.random.Generator], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class Transform:
    """One function of the flame's iterated function system."""

    # Probability of picking this transform on a step, relative to the others.
    weight: float
    # Affine map (a, b, c, d, e, f): x' = a x + b y + c, y' = d x + e y + f.
    affine: tuple[float, float, float, float, float, float]
    # Variations applied after the affine map, each with its weight. Their
    # weighted results are summed.
    variations: tuple[tuple[Variation, float], ...]
    # Colour index in [0, 1] that the points it moves are drawn towards.
    colour: float


# The flame drawn. Chosen by eye, for an image with structure at every size.
TRANSFORMS = (
    Transform(0.45, (0.56, -0.43, 0.22, 0.43, 0.56, -0.10), ((linear, 0.4), (spherical, 0.6)), 0.0),
    Transform(0.30, (-0.35, 0.52, -0.60, 0.48, 0.30, 0.35), ((swirl, 0.7), (sinusoidal, 0.3)), 0.5),
    Transform(0.15, (0.42, 0.10, 0.55, -0.12, 0.40, 0.30), ((julia, 1.0),), 0.8),
    Transform(0.10, (0.30, -0.25, -0.15, 0.25, 0.30, -0.55), ((horseshoe, 0.5), (polar, 0.5)), 1.0),
)


def palette(index: np.ndarray) -> np.ndarray:
    """Look up colour indices in a gradient of cosines.

    Args:
        index: Colour indices in [0, 1], any shape.

    Returns:
        RGB in [0, 1], with one more axis of length 3.
    """
    stops = np.linspace(0, 1, PALETTE_ENTRIES)
    channels = [0.5 + 0.5 * np.cos(2 * np.pi * (stops + phase)) for phase in PALETTE_PHASES]
    return np.stack([np.interp(index, stops, channel) for channel in channels], axis=-1)


def step(x: np.ndarray, y: np.ndarray, colour: np.ndarray, rng: np.random.Generator) -> None:
    """Move every point once through a transform picked by weight, in place.

    Args:
        x: x coordinate of each point, float64.
        y: y coordinate of each point, float64.
        colour: Colour index of each point, float64 in [0, 1].
        rng: The generator.
    """
    weights = np.array([t.weight for t in TRANSFORMS])
    pick = rng.choice(len(TRANSFORMS), size=x.size, p=weights / weights.sum())
    for k, transform in enumerate(TRANSFORMS):
        chosen = pick == k
        a, b, c, d, e, f = transform.affine
        px, py = x[chosen], y[chosen]
        ax, ay = a * px + b * py + c, d * px + e * py + f
        nx = np.zeros_like(ax)
        ny = np.zeros_like(ay)
        for variation, weight in transform.variations:
            vx, vy = variation(ax, ay, rng)
            nx += weight * vx
            ny += weight * vy
        # A point sent to infinity or NaN starts again at the origin, as in flam3.
        lost = ~(np.isfinite(nx) & np.isfinite(ny))
        nx[lost] = 0.0
        ny[lost] = 0.0
        x[chosen], y[chosen] = nx, ny
        colour[chosen] = (colour[chosen] + transform.colour) / 2


def render(points: int, width: int, height: int, steps: int, fuse: int) -> np.ndarray:
    """Render the flame.

    Args:
        points: Number of points that play the chaos game together.
        width: Width of the image in pixels.
        height: Height of the image in pixels.
        steps: Chaos-game steps.
        fuse: Steps at the start that are not plotted; fewer than `steps`.

    Returns:
        The image, float64 RGB in [0, 1], of shape (height, width, 3).
    """
    rng = np.random.default_rng(SEED)
    x = rng.uniform(-1, 1, points)
    y = rng.uniform(-1, 1, points)
    colour = rng.uniform(0, 1, points)
    pixels = width * height
    density = np.zeros(pixels)
    colour_sum = np.zeros(pixels)
    xmin, xmax, ymin, ymax = VIEW

    with np.errstate(all="ignore"):
        for n in range(steps):
            step(x, y, colour, rng)
            if n < fuse:
                continue
            column = ((x - xmin) / (xmax - xmin) * width).astype(np.int64)
            row = ((y - ymin) / (ymax - ymin) * height).astype(np.int64)
            inside = (column >= 0) & (column < width) & (row >= 0) & (row < height)
            pixel = row[inside] * width + column[inside]
            density += np.bincount(pixel, minlength=pixels)
            colour_sum += np.bincount(pixel, weights=colour[inside], minlength=pixels)

    hit = density > 0
    mean_colour = np.zeros(pixels)
    mean_colour[hit] = colour_sum[hit] / density[hit]
    alpha = np.clip(np.log1p(density) / np.log1p(density.max()), 0.0, 1.0) ** (1 / GAMMA)
    image = np.clip(palette(mean_colour) * alpha[:, None], 0.0, 1.0)
    return image.reshape(height, width, 3)


def make_data(points: int, width: int, height: int, steps: int, fuse: int) -> tuple:
    """Pass the size through: the renderer draws its own points from SEED.

    Args:
        points: Number of points.
        width: Width of the image in pixels.
        height: Height of the image in pixels.
        steps: Chaos-game steps.
        fuse: Steps at the start that are not plotted.

    Returns:
        The arguments of `render`, in order.
    """
    return points, width, height, steps, fuse


def run(*arguments) -> tuple:
    """Render once.

    Args:
        arguments: The arguments make_data returned.

    Returns:
        A one-element tuple: the image.
    """
    return (render(*arguments),)


WORKLOAD = Workload(
    name="flame",
    title="Fractal flame: chaos game over four transforms, log-density tone mapping",
    source="delivery/workloads/flame.py in this repository",
    licence=None,
    ours=True,
    sizes=SIZES,
    make_data=make_data,
    run=run,
)
