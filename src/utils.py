from __future__ import annotations

import os
import random
from collections.abc import Iterable

import numpy as np


def set_random_seed(seed: int = 42) -> None:
    """固定常见随机源，保证实验尽可能可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: np.ndarray,
    lon2: np.ndarray,
    earth_radius_km: float = 6371.0,
) -> np.ndarray:
    """向量化计算经纬度球面距离，单位为公里。"""
    lat1_rad, lon1_rad = np.radians([lat1, lon1])
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    return earth_radius_km * 2.0 * np.arcsin(np.sqrt(a))


def seismic_moment_from_mw(magnitudes: Iterable[float] | np.ndarray) -> np.ndarray:
    """
    将矩震级 Mw 转为标量地震矩近似量。

    这里沿用当前项目已有经验式 10 ** (1.5 * Mw + 4.8)，用于相对能量/矩释放特征。
    """
    mags = np.asarray(magnitudes, dtype=float)
    return 10 ** (1.5 * mags + 4.8)
