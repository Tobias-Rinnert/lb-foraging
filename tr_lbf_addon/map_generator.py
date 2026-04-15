"""Terrain generator for the survival simulation.

Generates a binary terrain map of stone (0) and grass (1) cells.
Uses continuous noise smoothed by a box filter, then thresholded at
``grass_ratio``, so the output grass fraction stays close to the requested
value regardless of the number of smoothing passes.
"""

import numpy as np


def generate_ca_map(
    field_size: int,
    grass_ratio: float = 0.70,
    smooth_iterations: int = 5,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a terrain map with spatially-smoothed noise.

    Initialises each cell with a uniform random value in [0, 1), then
    repeatedly blurs the noise with a 3x3 box filter (edge-padded).
    Finally, cells whose smoothed value falls below ``grass_ratio`` become
    grass (1); the rest become stone (0).

    Because the threshold is applied *after* smoothing a continuous signal,
    the output grass fraction tracks ``grass_ratio`` closely regardless of
    how many smoothing passes are applied — unlike a binary majority-vote CA,
    which amplifies the dominant state and converges to 0% or 100%.

    Args:
        field_size: width and height of the square grid
        grass_ratio: fraction of cells that should be grass (default 0.70)
        smooth_iterations: number of box-filter smoothing passes (default 5)
        seed: random seed for reproducibility (default None)

    Returns:
        np.ndarray of shape (field_size, field_size) with dtype int8.
        Values: 0 = stone, 1 = grass.
    """
    rng = np.random.default_rng(seed)
    noise = rng.random((field_size, field_size))

    for _ in range(smooth_iterations):
        padded = np.pad(noise, 1, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
        noise = windows.mean(axis=(-2, -1))

    # Use percentile thresholding so the output grass fraction equals
    # grass_ratio exactly, regardless of how much smoothing compressed the
    # noise distribution toward its mean.
    threshold = np.percentile(noise, grass_ratio * 100)
    return (noise <= threshold).astype(np.int8)
