"""时间桶模块。

将每个资格赛时间窗口（T1/T2/T3）划分为 4 个均匀/对数时间桶，
支持：桶分配、桶中心点计算、概率加权期望时间、窗口裁剪。
"""

from __future__ import annotations

import numpy as np

from src.qualification import WINDOW_BY_NAME

# ---------------------------------------------------------------------------
# 时间桶定义：每个窗口 4 个桶，均匀划分
# ---------------------------------------------------------------------------
_BUCKET_BINS: dict[str, list[tuple[float, float]]] = {
    "T1": [
        (0.0, 3.0),
        (3.0, 6.0),
        (6.0, 12.0),
        (12.0, 24.0),
    ],
    "T2": [
        (24.0, 36.0),
        (36.0, 48.0),
        (48.0, 60.0),
        (60.0, 72.0),
    ],
    "T3": [
        (72.0, 96.0),
        (96.0, 120.0),
        (120.0, 144.0),
        (144.0, 168.0),
    ],
    # ── H168: 4 桶覆盖全窗口 0-168h ──
    "H168": [
        (0.0, 12.0),
        (12.0, 48.0),
        (48.0, 96.0),
        (96.0, 168.0),
    ],
}

# 桶中心取区间中点
_BUCKET_CENTERS: dict[str, list[float]] = {
    window_name: [(lo + hi) / 2.0 for lo, hi in bins]
    for window_name, bins in _BUCKET_BINS.items()
}

# 每个窗口的合法边界 (left-exclusive, right-inclusive]
_WINDOW_BOUNDS: dict[str, tuple[float, float]] = {
    "T1": (0.0, 24.0),
    "T2": (24.0, 72.0),
    "T3": (72.0, 168.0),
    "H168": (0.0, 168.0),
}


def get_time_buckets(window_name: str) -> list[tuple[float, float]]:
    """返回窗口的时间桶区间列表。

    Args:
        window_name: "T1" | "T2" | "T3"

    Returns:
        list[(lo, hi), ...]，每个元素为 (下限, 上限] 小时。
    """
    if window_name not in _BUCKET_BINS:
        raise KeyError(f"Unknown window: {window_name!r}. Expected T1/T2/T3.")
    return list(_BUCKET_BINS[window_name])


def bucket_centers(window_name: str) -> list[float]:
    """返回窗口各时间桶的中心点小时数。

    Args:
        window_name: "T1" | "T2" | "T3"
    """
    if window_name not in _BUCKET_CENTERS:
        raise KeyError(f"Unknown window: {window_name!r}. Expected T1/T2/T3.")
    return list(_BUCKET_CENTERS[window_name])


def assign_time_bucket(window_name: str, time_hours: float) -> int:
    """将真实时间值分配到对应的时间桶索引 (0-3)。

    边界规则：左开右闭，即 time_hours ∈ (lo, hi]。
    若 time_hours <= 窗口下限，落入第 0 桶。
    若 time_hours > 窗口上限，落入最后一桶。

    Args:
        window_name: "T1" | "T2" | "T3"
        time_hours: 真实最大余震距离主震的小时数

    Returns:
        桶索引, 0-3
    """
    buckets = get_time_buckets(window_name)
    for idx, (lo, hi) in enumerate(buckets):
        if lo < time_hours <= hi:
            return idx
    if time_hours <= buckets[0][0]:
        return 0
    return len(buckets) - 1


def expected_time_from_bucket_probs(
    window_name: str,
    probs: np.ndarray,
) -> float:
    """根据 4 桶分类概率计算期望时间。

    使用桶中心加权平均：E[t] = Σ p_k * center_k。
    probs 会被归一化以确保和为 1。

    Args:
        window_name: "T1" | "T2" | "T3"
        probs: shape (4,) 或 (n, 4) 的分类概率数组

    Returns:
        期望时间小时数（标量或 shape (n,)）
    """
    centers = np.asarray(bucket_centers(window_name), dtype=float)
    probs = np.asarray(probs, dtype=float)

    if probs.ndim == 1:
        if probs.shape[0] != 4:
            raise ValueError(f"Expected 4 bucket probs, got shape {probs.shape}")
        total = probs.sum()
        if total <= 0:
            return float(centers.mean())
        return float(np.dot(probs / total, centers))

    if probs.ndim == 2:
        if probs.shape[1] != 4:
            raise ValueError(f"Expected 4 bucket columns, got {probs.shape[1]}")
        totals = probs.sum(axis=1, keepdims=True)
        totals = np.where(totals <= 0, 1.0, totals)
        return np.dot(probs / totals, centers)

    raise ValueError(f"probs must be 1D or 2D, got ndim={probs.ndim}")


def clamp_time_to_window(window_name: str, time_hours: float) -> float:
    """将时间值裁剪到窗口合法范围。

    使用窗口定义 (lower, upper]，时间被裁剪到 [lower + epsilon, upper]。
    非有限值回退到窗口中点。

    Args:
        window_name: "T1" | "T2" | "T3"
        time_hours: 待裁剪的时间值（小时）

    Returns:
        裁剪后的时间值
    """
    window = WINDOW_BY_NAME[window_name]
    if not np.isfinite(time_hours):
        return window.midpoint_hours
    lo, hi = window.lower_hours, window.upper_hours
    epsilon = 1e-6
    return float(np.clip(time_hours, lo + epsilon, hi))


# ---------------------------------------------------------------------------
# 批量接口
# ---------------------------------------------------------------------------

def assign_time_buckets_batch(
    window_name: str,
    time_hours: np.ndarray,
) -> np.ndarray:
    """向量化版本的 assign_time_bucket。

    Args:
        window_name: "T1" | "T2" | "T3"
        time_hours: shape (n,) 真实时间数组

    Returns:
        shape (n,) int 桶索引
    """
    buckets = get_time_buckets(window_name)
    result = np.full(len(time_hours), len(buckets) - 1, dtype=int)
    for idx, (lo, hi) in enumerate(buckets):
        result[(lo < time_hours) & (time_hours <= hi)] = idx
    result[time_hours <= buckets[0][0]] = 0
    return result


def bucket_centers_array(window_name: str) -> np.ndarray:
    """返回桶中心 NumPy 数组。"""
    return np.asarray(bucket_centers(window_name), dtype=float)


# ---------------------------------------------------------------------------
# 概率对齐：处理模型返回的 predict_proba 列数不足 4 的情况
# ---------------------------------------------------------------------------

def align_bucket_probabilities(
    model,
    raw_probs: np.ndarray,
    n_classes: int = 4,
) -> np.ndarray:
    """将模型返回的原始概率对齐到 n_classes 列。

    处理场景：
    - 训练集缺少某些桶类别时，LightGBM predict_proba 可能只返回 3 列。
    - 本函数通过 model.classes_ 做列映射，确保输出 shape=(n, n_classes)。

    Args:
        model: 已训练的 sklearn-风格分类器（需有 classes_ 属性）。
        raw_probs: shape (n, k) 原始概率，k 可能 < n_classes。
        n_classes: 期望的类别数，默认 4。

    Returns:
        shape (n, n_classes) 的对齐概率，每行归一化求和=1。
    """
    raw = np.asarray(raw_probs, dtype=float)

    # 如果已经是正确形状，直接返回
    if raw.ndim == 2 and raw.shape[1] == n_classes:
        probs = raw.copy()
    elif raw.ndim == 1 and n_classes == 4:
        # 单样本
        probs = np.zeros((1, n_classes), dtype=float)
        probs[0, : len(raw)] = raw
    else:
        # 需要根据 classes_ 映射
        clz = getattr(model, "classes_", None)
        if clz is not None:
            probs = np.zeros((len(raw), n_classes), dtype=float)
            for col_idx, class_label in enumerate(clz):
                if col_idx < raw.shape[1] and 0 <= int(class_label) < n_classes:
                    probs[:, int(class_label)] = raw[:, col_idx]
        elif raw.ndim == 2 and raw.shape[1] < n_classes:
            probs = np.zeros((len(raw), n_classes), dtype=float)
            probs[:, : raw.shape[1]] = raw
        elif raw.ndim == 1:
            probs = np.zeros((len(raw), n_classes), dtype=float)
            # 对类别预测做 one-hot fallback
            classes = raw.astype(int)
            probs[np.arange(len(raw)), np.clip(classes, 0, n_classes - 1)] = 1.0
        else:
            probs = raw[:, :n_classes] if raw.shape[1] >= n_classes else raw

    # 归一化每行；全零行回退为均匀概率
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = np.where(row_sums > 0, probs / row_sums, 1.0 / n_classes)
    return probs


# ---------------------------------------------------------------------------
# 安全极端概率提取（train / tune / package 共用）
# ---------------------------------------------------------------------------

def safe_extreme_probability(model, X) -> np.ndarray:
    """从二分类模型安全提取类别=1的概率。

    处理 DummyClassifier 单类别：
    - constant=0 → classes_=[0] → 返回全 0
    - constant=1 → classes_=[1] → 返回全 1
    - 正常二分类 → 返回第 2 列概率

    Returns:
        shape (n,) float array
    """
    try:
        if hasattr(model, "predict_proba"):
            raw = np.asarray(model.predict_proba(X), dtype=float)
            clz = getattr(model, "classes_", None)
            if clz is not None:
                if len(clz) == 1:
                    return np.full(len(X), 1.0 if int(clz[0]) == 1 else 0.0, dtype=float)
                if len(clz) >= 2:
                    idx_1 = int(np.where(clz == 1)[0][0]) if 1 in clz else 1
                    return raw[:, idx_1] if raw.ndim == 2 else raw
            if raw.ndim == 2 and raw.shape[1] >= 2:
                return raw[:, 1]
            return raw.ravel().astype(float)
        p = np.asarray(model.predict(X), dtype=int).ravel()
        return p.astype(float)
    except Exception:
        return np.zeros(len(X), dtype=float)
