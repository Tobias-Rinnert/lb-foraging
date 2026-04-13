"""Cellular automata terrain generator for the survival simulation.

Generates a binary terrain map of stone (0) and grass (1) cells using
cellular automata smoothing to produce natural-looking cave/island shapes.
"""

import numpy as np


def generate_ca_map(
    field_size: int,
    grass_ratio: float = 0.70,
    smooth_iterations: int = 5,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a terrain map using cellular automata smoothing.

    Initialises each cell randomly as grass with probability `grass_ratio`,
    then repeatedly smooths: a cell becomes grass if at least 50% of its
    3x3 neighbourhood is grass, otherwise stone.

    Args:
        field_size: width and height of the square grid
        grass_ratio: probability of a cell starting as grass (default 0.70)
        smooth_iterations: number of CA smoothing passes (default 5)
        seed: random seed for reproducibility (default None)

    Returns:
        np.ndarray of shape (field_size, field_size) with dtype int8.
        Values: 0 = stone, 1 = grass.
    """
    rng = np.random.default_rng(seed)
    grid = (rng.random((field_size, field_size)) < grass_ratio).astype(np.int8)

    for _ in range(smooth_iterations):
        new_grid = np.empty_like(grid)
        for row in range(field_size):
            for col in range(field_size):
                r0, r1 = max(0, row - 1), min(field_size, row + 2)
                c0, c1 = max(0, col - 1), min(field_size, col + 2)
                neighbourhood = grid[r0:r1, c0:c1]
                grass_count = neighbourhood.sum()
                total_count = neighbourhood.size
                new_grid[row, col] = np.int8(1) if grass_count * 2 >= total_count else np.int8(0)
        grid = new_grid

    return grid
