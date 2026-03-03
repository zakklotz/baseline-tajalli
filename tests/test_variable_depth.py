"""Tests for variable depth curriculum."""

import random

from tajalli.training.trainer import _get_curriculum_depth_range


def test_variable_depth_curriculum_ranges():
    """Curriculum stages return (depth_min, depth_max) within specified ranges."""
    curriculum = [
        {"end_step": 10000, "depth_min": 2, "depth_max": 6},
        {"end_step": 50000, "depth_min": 6, "depth_max": 12},
        {"end_step": 300000, "depth_min": 8, "depth_max": 16},
    ]
    config = {"depth_curriculum": curriculum}

    r = _get_curriculum_depth_range(0, config)
    assert r == (2, 6)
    r = _get_curriculum_depth_range(5000, config)
    assert r == (2, 6)
    r = _get_curriculum_depth_range(10000, config)
    assert r == (2, 6)
    r = _get_curriculum_depth_range(10001, config)
    assert r == (6, 12)
    r = _get_curriculum_depth_range(50000, config)
    assert r == (6, 12)
    r = _get_curriculum_depth_range(50001, config)
    assert r == (8, 16)
    r = _get_curriculum_depth_range(300000, config)
    assert r == (8, 16)


def test_variable_depth_sampled_within_range():
    """Sampled depth should be within depth_min and depth_max."""
    curriculum = [
        {"end_step": 10000, "depth_min": 2, "depth_max": 6},
    ]
    config = {"depth_curriculum": curriculum}
    for _ in range(20):
        step = random.randint(0, 9999)
        r = _get_curriculum_depth_range(step, config)
        assert r is not None
        d_min, d_max = r
        depth = random.randint(d_min, d_max)
        assert d_min <= depth <= d_max
