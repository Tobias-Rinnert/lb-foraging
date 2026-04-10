import numpy as np


def generate_ca_map(
    field_size: int,
    grass_ratio: float = 0.70,
    smooth_iterations: int = 5,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate a cellular automata terrain map.

    Returns a 2D numpy array of shape (field_size, field_size):
        0 = stone (no food growth, blocks movement)
        1 = grass (food grows here)

    CA rule: a cell becomes grass if >= 50% of its 3x3 neighbourhood is grass.
    """
    rng = np.random.default_rng(seed)
    grid = (rng.random((field_size, field_size)) < grass_ratio).astype(np.int8)

    for _ in range(smooth_iterations):
        new = grid.copy()
        for r in range(field_size):
            for c in range(field_size):
                r0, r1 = max(0, r - 1), min(field_size, r + 2)
                c0, c1 = max(0, c - 1), min(field_size, c + 2)
                new[r, c] = 1 if grid[r0:r1, c0:c1].mean() >= 0.5 else 0
        grid = new

    return grid
