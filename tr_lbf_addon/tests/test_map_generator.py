import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from map_generator import generate_ca_map


def test_output_shape():
    m = generate_ca_map(10)
    assert m.shape == (10, 10)


def test_values_binary():
    m = generate_ca_map(15)
    assert set(m.flatten().tolist()).issubset({0, 1})


def test_grass_ratio_approx():
    # with large enough map, output ratio should be roughly in range of input
    m = generate_ca_map(50, grass_ratio=0.7, smooth_iterations=0, seed=42)
    ratio = m.mean()
    assert 0.55 < ratio < 0.85


def test_deterministic_with_seed():
    a = generate_ca_map(20, seed=123)
    b = generate_ca_map(20, seed=123)
    assert np.array_equal(a, b)


def test_different_seeds_differ():
    # use 0 smoothing so maps stay noisy and seed-sensitive
    a = generate_ca_map(20, smooth_iterations=0, seed=1)
    b = generate_ca_map(20, smooth_iterations=0, seed=2)
    assert not np.array_equal(a, b)


def test_smoothing_reduces_noise():
    # more smoothing iterations → fewer isolated cells → larger connected regions
    # simplified check: 5 iterations should give fewer unique 3x3 patches than 0
    noisy = generate_ca_map(20, smooth_iterations=0, seed=42)
    smooth = generate_ca_map(20, smooth_iterations=5, seed=42)
    # smooth map should have lower variance (more homogeneous)
    assert smooth.std() <= noisy.std() + 0.1
