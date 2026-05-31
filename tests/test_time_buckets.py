"""时间桶模块测试：边界正确性、桶分配、期望时间、窗口裁剪、概率对齐。"""

from __future__ import annotations

import numpy as np
import pytest

from src.time_buckets import (
    align_bucket_probabilities,
    assign_time_bucket,
    assign_time_buckets_batch,
    bucket_centers,
    clamp_time_to_window,
    expected_time_from_bucket_probs,
    get_time_buckets,
)


class TestBucketDefinitions:
    """验证桶区间定义正确。"""

    def test_t1_buckets(self):
        buckets = get_time_buckets("T1")
        assert len(buckets) == 4
        assert buckets == [(0.0, 3.0), (3.0, 6.0), (6.0, 12.0), (12.0, 24.0)]

    def test_t2_buckets(self):
        buckets = get_time_buckets("T2")
        assert len(buckets) == 4
        assert buckets == [(24.0, 36.0), (36.0, 48.0), (48.0, 60.0), (60.0, 72.0)]

    def test_t3_buckets(self):
        buckets = get_time_buckets("T3")
        assert len(buckets) == 4
        assert buckets == [(72.0, 96.0), (96.0, 120.0), (120.0, 144.0), (144.0, 168.0)]

    def test_unknown_window_raises(self):
        with pytest.raises(KeyError, match="T4"):
            get_time_buckets("T4")


class TestBucketCenters:
    """验证桶中心点计算。"""

    def test_t1_centers(self):
        centers = bucket_centers("T1")
        assert centers == [1.5, 4.5, 9.0, 18.0]

    def test_t2_centers(self):
        centers = bucket_centers("T2")
        assert centers == [30.0, 42.0, 54.0, 66.0]

    def test_t3_centers(self):
        centers = bucket_centers("T3")
        assert centers == [84.0, 108.0, 132.0, 156.0]


class TestAssignTimeBucket:
    """验证单时间值和批量桶分配逻辑。"""

    def test_t1_assign(self):
        assert assign_time_bucket("T1", 1.5) == 0
        assert assign_time_bucket("T1", 3.0) == 0
        assert assign_time_bucket("T1", 3.01) == 1
        assert assign_time_bucket("T1", 10.0) == 2
        assert assign_time_bucket("T1", 20.0) == 3

    def test_t1_boundary_clamps(self):
        assert assign_time_bucket("T1", 0.0) == 0
        assert assign_time_bucket("T1", -1.0) == 0
        assert assign_time_bucket("T1", 24.01) == 3

    def test_t2_assign(self):
        assert assign_time_bucket("T2", 24.0) == 0
        assert assign_time_bucket("T2", 42.0) == 1
        assert assign_time_bucket("T2", 72.0) == 3

    def test_t3_assign(self):
        assert assign_time_bucket("T3", 80.0) == 0
        assert assign_time_bucket("T3", 100.0) == 1
        assert assign_time_bucket("T3", 160.0) == 3

    def test_batch_assignment(self):
        times = np.array([1.5, 10.0, 5.0, 20.0, 0.0, 30.0])
        got = assign_time_buckets_batch("T1", times)
        expected = np.array([0, 2, 1, 3, 0, 3])
        np.testing.assert_array_equal(got, expected)


class TestExpectedTimeFromBucketProbs:
    """验证概率加权期望时间计算。"""

    def test_uniform(self):
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        t = expected_time_from_bucket_probs("T1", probs)
        assert t == pytest.approx(8.25)

    def test_peaked_first(self):
        probs = np.array([1.0, 0.0, 0.0, 0.0])
        t = expected_time_from_bucket_probs("T1", probs)
        assert t == pytest.approx(1.5)

    def test_peaked_last(self):
        probs = np.array([0.0, 0.0, 0.0, 1.0])
        t = expected_time_from_bucket_probs("T1", probs)
        assert t == pytest.approx(18.0)

    def test_all_zeros(self):
        probs = np.array([0.0, 0.0, 0.0, 0.0])
        t = expected_time_from_bucket_probs("T1", probs)
        assert t == pytest.approx(8.25)

    def test_batch_probs(self):
        probs = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        got = expected_time_from_bucket_probs("T1", probs)
        expected = np.array([1.5, 4.5, 9.0, 18.0])
        np.testing.assert_allclose(got, expected)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            expected_time_from_bucket_probs("T1", np.array([1.0, 2.0, 3.0]))


class TestClampTimeToWindow:
    """验证窗口内时间裁剪。"""

    def test_t1_in_bounds(self):
        assert clamp_time_to_window("T1", 5.0) == pytest.approx(5.0)

    def test_t1_below_lower(self):
        assert clamp_time_to_window("T1", -1.0) == pytest.approx(1e-6)

    def test_t1_above_upper(self):
        assert clamp_time_to_window("T1", 30.0) == pytest.approx(24.0)

    def test_nan_falls_back_to_midpoint(self):
        t = clamp_time_to_window("T1", float("nan"))
        assert t == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# align_bucket_probabilities 测试
# ---------------------------------------------------------------------------

class MockModel:
    def __init__(self, classes):
        self.classes_ = np.array(classes)


class TestAlignBucketProbabilities:
    """测试概率对齐各类场景。"""

    def test_already_4_cols_passthrough(self):
        raw = np.array([[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]])
        model = MockModel([0, 1, 2, 3])
        result = align_bucket_probabilities(model, raw)
        assert result.shape == (2, 4)
        np.testing.assert_allclose(result.sum(axis=1), [1.0, 1.0])

    def test_3_cols_missing_middle(self):
        raw = np.array([[0.2, 0.3, 0.5]])
        model = MockModel([0, 2, 3])
        result = align_bucket_probabilities(model, raw)
        assert result.shape == (1, 4)
        assert result[0, 0] == pytest.approx(0.2)
        assert result[0, 1] == pytest.approx(0.0)
        assert result[0, 2] == pytest.approx(0.3)
        assert result[0, 3] == pytest.approx(0.5)
        assert result.sum() == pytest.approx(1.0)

    def test_all_zeros_fallback(self):
        raw = np.zeros((2, 3))
        model = MockModel([0, 1, 2])
        result = align_bucket_probabilities(model, raw)
        assert result.shape == (2, 4)
        np.testing.assert_allclose(result, 0.25)

    def test_single_sample_1d(self):
        raw = np.array([0.1, 0.2, 0.3, 0.4])
        model = MockModel([0, 1, 2, 3])
        result = align_bucket_probabilities(model, raw)
        assert result.shape == (1, 4)
        np.testing.assert_allclose(result.sum(axis=1), [1.0])


# ---------------------------------------------------------------------------
# safe_extreme_probability 测试
# ---------------------------------------------------------------------------

class TestSafeExtremeProbability:
    def test_dummy_constant_0_returns_zeros(self):
        from src.time_buckets import safe_extreme_probability
        from sklearn.dummy import DummyClassifier
        m = DummyClassifier(strategy='constant', constant=0)
        m.fit([[0], [0], [0]], [0, 0, 0])
        probs = safe_extreme_probability(m, [[1], [2], [3]])
        assert probs.shape == (3,)
        assert np.allclose(probs, 0.0)

    def test_dummy_constant_1_returns_ones(self):
        from src.time_buckets import safe_extreme_probability
        from sklearn.dummy import DummyClassifier
        m = DummyClassifier(strategy='constant', constant=1)
        m.fit([[0], [0], [0]], [1, 1, 1])
        probs = safe_extreme_probability(m, [[1], [2]])
        assert np.allclose(probs, 1.0)

    def test_normal_binary_classifier(self):
        from src.time_buckets import safe_extreme_probability
        from sklearn.dummy import DummyClassifier
        m = DummyClassifier(strategy='uniform')
        m.fit([[0], [1], [2], [3]], [0, 0, 1, 1])
        probs = safe_extreme_probability(m, [[1], [2]])
        assert probs.shape == (2,)
        # uniform should give ~0.5
        assert np.all((probs >= 0.4) & (probs <= 0.6))
