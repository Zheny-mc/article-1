import numpy as np
from acfsa.filtering import adaptive_threshold


def test_adaptive_threshold_removes_low_scores():
    result = adaptive_threshold([0.92, 0.90, 0.88, 0.45, 0.32], beta=1.0, min_threshold=0.60)
    assert result.kept_indices == [0, 1, 2]
    assert 0.8 < result.threshold < 0.9


def test_equal_scores_use_median_and_floor():
    result = adaptive_threshold([0.8, 0.8, 0.8], min_threshold=0.35)
    assert result.mad == 0
    assert result.kept_indices == [0, 1, 2]


def test_all_below_floor_returns_empty():
    result = adaptive_threshold([0.2, 0.1, 0.05], min_threshold=0.35)
    assert result.kept_indices == []


def test_empty_input():
    result = adaptive_threshold([])
    assert result.kept_indices == []
    assert np.isinf(result.threshold)


def test_median_mad_are_robust_to_outlier():
    result = adaptive_threshold([0.91, 0.89, 0.87, 0.20, 0.10], beta=1.0, min_threshold=0.5)
    assert result.kept_indices == [0, 1, 2]
