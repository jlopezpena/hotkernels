"""Segments and measures the objects in noisy images with scikit-image.

The pipeline of a typical microscopy or inspection script, applied to each of
a few images of bright blobs on a noisy, unevenly lit background:

- equalise the contrast with adaptive histogram equalisation (CLAHE);
- smooth with a Gaussian, threshold at Otsu's level, remove small objects;
- label the objects and measure their area, centroid, bounding box,
  eccentricity and mean intensity;
- find edges with Canny, and their strength with Sobel;
- describe the image by its histogram of oriented gradients (HOG).
"""
from __future__ import annotations

import numpy as np

TITLE = "Segmenting and measuring objects in images: CLAHE, Otsu, labels, region properties, Canny, HOG"
SOURCE = "scikit-image (skimage.exposure, filters, morphology, measure, feature), called as a library"
LICENCE = None
REQUIRES = ("skimage",)

# Images processed in one run.
IMAGES = 4
# By size: the side of each square image, in pixels.
SIDE = {"small": 256, "large": 1_024}
# Fraction of each image covered by blobs, for skimage.data.binary_blobs.
BLOB_FRACTION = 0.3
# Standard deviation of the Gaussian noise added to each image, on a scale of 0 to 1.
NOISE = 0.15
# Standard deviation of the smoothing Gaussian, in pixels.
SMOOTHING_SIGMA = 2.0
# Standard deviation of the Gaussian inside Canny, in pixels.
CANNY_SIGMA = 1.5
# Objects smaller than this many pixels are removed before labelling.
SMALLEST_OBJECT = 64
# Properties measured of every object.
PROPERTIES = ("label", "area", "centroid", "bbox", "eccentricity", "intensity_mean")
# Seed of the images.
SEED = 0
# One line per size saying how large its data is.
SIZES = {size: f"{IMAGES} images of {side:,} × {side:,}" for size, side in SIDE.items()}


def make_data(size: str) -> tuple:
    """Build noisy, unevenly lit images of blobs.

    Args:
        size: One of SIZES.

    Returns:
        (the images, a list of float64 arrays with values from 0 to 1).
    """
    from skimage import data

    side = SIDE[size]
    rng = np.random.default_rng(SEED)
    # Lighting that rises from one corner to the other, which CLAHE corrects.
    ramp = np.linspace(0.0, 0.4, side)
    lighting = ramp[:, None] + ramp[None, :]
    images = []
    for index in range(IMAGES):
        blobs = data.binary_blobs(side, blob_size_fraction=0.05, volume_fraction=BLOB_FRACTION, rng=SEED + index)
        image = 0.5 * blobs + lighting + NOISE * rng.standard_normal((side, side))
        images.append(np.clip(image / image.max(), 0.0, 1.0))
    return (images,)


def run(images: list) -> tuple:
    """Run the pipeline on every image.

    Args:
        images: Grey images, float64 from 0 to 1.

    Returns:
        For each image: the equalised image, the object labels, every
        measured property as an array, the Canny edges, the Sobel magnitude
        and the HOG descriptor.
    """
    from skimage import exposure, feature, filters, measure, morphology

    results = []
    for image in images:
        equalised = exposure.equalize_adapthist(image)
        smooth = filters.gaussian(equalised, sigma=SMOOTHING_SIGMA)
        mask = smooth > filters.threshold_otsu(smooth)
        mask = morphology.remove_small_objects(mask, max_size=SMALLEST_OBJECT)
        labels = measure.label(mask)
        table = measure.regionprops_table(labels, intensity_image=image, properties=PROPERTIES)
        edges = feature.canny(image, sigma=CANNY_SIGMA)
        strength = filters.sobel(smooth)
        descriptor = feature.hog(equalised)
        results.append((equalised, labels, tuple(table[key] for key in sorted(table)), edges, strength, descriptor))
    return tuple(results)
