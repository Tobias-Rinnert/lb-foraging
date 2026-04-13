"""Tests for tr_lbf_addon.map_generator — cellular automata terrain generator."""

import numpy as np
from tr_lbf_addon.map_generator import generate_ca_map


class TestGenerateCaMap:
    """Tests for generate_ca_map."""

    def test_output_shape(self):
        """Output shape matches field_size x field_size."""
        grid = generate_ca_map(field_size=10, seed=0)
        assert grid.shape == (10, 10)

    def test_output_dtype(self):
        """Output dtype is int8."""
        grid = generate_ca_map(field_size=10, seed=0)
        assert grid.dtype == np.int8

    def test_only_zero_and_one(self):
        """All values are exactly 0 (stone) or 1 (grass)."""
        grid = generate_ca_map(field_size=20, seed=42)
        assert set(np.unique(grid)).issubset({0, 1})

    def test_grass_ratio_roughly_correct(self):
        """Initial grass fraction matches target before smoothing (smooth_iterations=0)."""
        target = 0.50
        grid = generate_ca_map(field_size=100, grass_ratio=target, smooth_iterations=0, seed=7)
        actual = grid.mean()
        assert abs(actual - target) < 0.05

    def test_deterministic_with_seed(self):
        """Same seed and parameters produce the same map."""
        grid_a = generate_ca_map(field_size=15, seed=123)
        grid_b = generate_ca_map(field_size=15, seed=123)
        assert np.array_equal(grid_a, grid_b)

    def test_different_seeds_differ(self):
        """Different seeds produce different maps (no smoothing to prevent convergence)."""
        grid_a = generate_ca_map(field_size=20, smooth_iterations=0, seed=1)
        grid_b = generate_ca_map(field_size=20, smooth_iterations=0, seed=2)
        assert not np.array_equal(grid_a, grid_b)
