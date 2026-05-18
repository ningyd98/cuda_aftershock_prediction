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


def get_torch_device(device_str: str = "auto") -> "torch.device":
    """
    按优先级自动选择 PyTorch 设备: CUDA → MPS → CPU。

    可通过 device_str 显式指定 ('cuda', 'mps', 'cpu', 'auto')。

    Usage:
        device = get_torch_device()         # auto-detect
        device = get_torch_device("mps")    # force MPS
        device = get_torch_device("cpu")    # force CPU
    """
    import torch

    normalized = device_str.strip().lower()
    if normalized in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA 不可用，无法使用 --device cuda")
    if normalized == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS 不可用，无法使用 --device mps")
    if normalized == "cpu":
        return torch.device("cpu")
    # auto: CUDA → MPS → CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def gardner_knopoff_windows() -> dict[tuple[float, float], tuple[float, float]]:
    """
    Gardner & Knopoff (1974) 余震时空窗参数。

    返回 dict: (Mw_min, Mw_max) → (time_window_days, distance_window_km)

    参考: Gardner, J. K., & Knopoff, L. (1974).
    "Is the sequence of earthquakes in Southern California, with aftershocks removed,
    Poissonian?" BSSA, 64(5), 1363-1367.
    """
    return {
        (2.0, 2.5): (6.0, 30.0),
        (2.5, 3.0): (11.5, 30.0),
        (3.0, 3.5): (22.0, 40.0),
        (3.5, 4.0): (42.0, 50.0),
        (4.0, 4.5): (83.0, 60.0),
        (4.5, 5.0): (155.0, 70.0),
        (5.0, 5.5): (290.0, 80.0),
        (5.5, 6.0): (510.0, 90.0),
        (6.0, 6.5): (790.0, 100.0),
        (6.5, 7.0): (915.0, 110.0),
        (7.0, 7.5): (960.0, 120.0),
        (7.5, 8.0): (985.0, 130.0),
        (8.0, 99.0): (1000.0, 140.0),
    }


def get_gk_window(magnitude: float) -> tuple[float, float]:
    """获取给定震级的 Gardner-Knopoff 时间窗口 (天) 和距离窗口 (km)。"""
    windows = gardner_knopoff_windows()
    for (lo, hi), (t, d) in windows.items():
        if lo <= magnitude < hi:
            return t, d
    return 1000.0, 140.0  # M8+ default


def gardner_knopoff_decluster(
    events: "pd.DataFrame",
    time_col: str = "time",
    mag_col: str = "mag",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    earth_radius_km: float = 6371.0,
) -> "pd.DataFrame":
    """
    Gardner-Knopoff 去聚类算法。

    按震级降序处理事件，每个事件去除其后发时-空窗内的所有较小事件
    （视为其"余震"），剩余事件构成去聚类目录。

    Args:
        events: 地震事件 DataFrame
        time_col: 时间列名
        mag_col: 震级列名
        lat_col: 纬度列名
        lon_col: 经度列名
        earth_radius_km: 地球半径

    Returns:
        去聚类后的事件表，新增 is_background 标记列
    """
    import pandas as pd

    df = events.copy()
    df["_idx"] = np.arange(len(df))
    df["_is_bg"] = True

    # 按震级降序排序
    sorted_df = df.sort_values(mag_col, ascending=False)

    for _, main_row in sorted_df.iterrows():
        if not main_row["_is_bg"]:
            continue

        ms_time = main_row[time_col]
        ms_mag = main_row[mag_col]
        ms_lat = main_row[lat_col]
        ms_lon = main_row[lon_col]

        time_win_days, dist_win_km = get_gk_window(ms_mag)
        time_end = ms_time + pd.Timedelta(days=time_win_days)

        # 选出时-空窗内的较小事件
        candidates = df[
            (df[time_col] > ms_time)
            & (df[time_col] <= time_end)
            & (df[mag_col] < ms_mag)
            & (df["_is_bg"])
        ]
        if candidates.empty:
            continue

        # 计算距离
        dists = haversine_km(
            ms_lat, ms_lon,
            candidates[lat_col].to_numpy(),
            candidates[lon_col].to_numpy(),
            earth_radius_km=earth_radius_km,
        )
        in_window = candidates.index[dists <= dist_win_km]
        df.loc[in_window, "_is_bg"] = False

    df["is_background"] = df["_is_bg"]
    df = df.drop(columns=["_idx", "_is_bg"])
    return df
