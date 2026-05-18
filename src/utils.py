from __future__ import annotations

import os
import random
from collections.abc import Iterable

import numpy as np


def set_random_seed(seed: int = 42, use_cuda_deterministic: bool = False) -> None:
    """固定常见随机源，保证实验尽可能可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if use_cuda_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            else:
                # 默认开启 benchmark 以自动选择最优卷积算法
                torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


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


def _is_cuda_available() -> bool:
    """检查 CUDA 是否可用（含真实设备访问测试）。

    仅 torch.cuda.is_available() 可能返回 True 但实际无法创建 CUDA context
    （如无 GPU、驱动不匹配等情况），因此增加一次实际设备访问验证。
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        # 实际尝试访问设备，确保 CUDA context 可正常创建
        _ = torch.cuda.get_device_properties(0)
        return True
    except Exception:
        return False


def get_cuda_device_name() -> str:
    """获取 CUDA 设备名称，不可用时返回空字符串。"""
    if not _is_cuda_available():
        return ""
    import torch

    return torch.cuda.get_device_name(0)


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
        if _is_cuda_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA 不可用，无法使用 --device cuda")
    if normalized == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS 不可用，无法使用 --device mps")
    if normalized == "cpu":
        return torch.device("cpu")
    # auto: CUDA → MPS → CPU
    if _is_cuda_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_cuda(
    device_str: str = "auto",
    deterministic: bool = False,
    allow_tf32: bool = True,
    matmul_precision: str = "high",
    benchmark: bool | None = None,
) -> "torch.device":
    """
    一站式 CUDA 环境配置，应在训练/推理入口调用。

    Args:
        device_str: 设备选择策略 ('auto', 'cuda', 'mps', 'cpu')
        deterministic: 是否开启 cudnn deterministic 模式
        allow_tf32: 是否允许 TF32 加速（Ampere+ GPU）
        matmul_precision: float32 矩阵乘法精度 ('highest', 'high', 'medium')
        benchmark: 是否开启 cudnn benchmark；None 时自动（deterministic=False → True）

    Returns:
        torch.device

    Usage:
        device = setup_cuda()                          # 全自动
        device = setup_cuda(deterministic=True)         # 可复现模式
        device = setup_cuda(matmul_precision="medium")  # 更激进的速度优化
    """
    import torch

    device = get_torch_device(device_str)

    if device.type == "cuda":
        # TF32 加速 (Ampere+ GPU)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        # CuDNN benchmark
        if benchmark is None:
            benchmark = not deterministic
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic

        # float32 matmul precision (PyTorch 2.0+)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(matmul_precision)

        # 打印设备信息（包装 try/except 防止 CUDA context 初始化失败）
        try:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            # PyTorch 2.x 用 total_memory，1.x 用 total_mem，兼容二者
            gpu_mem = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1024**3
            print(f"CUDA 已启用: {gpu_name} ({gpu_mem:.1f} GiB)")
            print(f"  benchmark={benchmark}, deterministic={deterministic}, "
                  f"tf32={allow_tf32}, matmul_precision={matmul_precision}")
        except Exception as exc:
            print(f"⚠ CUDA 设备信息获取失败: {exc}")
            print("  CUDA 驱动/库可能不完整，回退到 CPU 模式")
            return torch.device("cpu")

    elif device.type == "mps":
        print("MPS (Apple Silicon GPU) 已启用")

    return device


def try_torch_compile(model: "torch.nn.Module", warn_on_fail: bool = True) -> "torch.nn.Module":
    """
    尝试对模型使用 torch.compile (PyTorch 2.0+)，失败时回退到原模型。

    Args:
        model: PyTorch 模型
        warn_on_fail: 编译失败时是否打印警告

    Returns:
        编译后的模型（成功）或原模型（失败）
    """
    import torch

    if not hasattr(torch, "compile"):
        if warn_on_fail:
            print("⚠ torch.compile 需要 PyTorch >= 2.0，当前版本不支持，跳过编译")
        return model

    try:
        compiled = torch.compile(model, dynamic=False)
        print("✓ torch.compile 编译成功")
        return compiled
    except Exception as exc:
        if warn_on_fail:
            print(f"⚠ torch.compile 编译失败: {exc}，回退到 eager 模式")
        return model


# 缓存 LightGBM CUDA 检测结果，避免每次调用都做昂贵测试
_lightgbm_cuda_cache: bool | None = None


def _is_lightgbm_cuda_available() -> bool:
    """检测 LightGBM 是否编译了 CUDA 支持。

    不同于 _is_cuda_available()（检测 PyTorch/torch 的 CUDA），
    本函数会实际尝试用 LightGBM 创建 CUDA Booster 来确认 LightGBM 本身支持 CUDA。
    结果会被缓存，后续调用直接返回缓存值。
    """
    global _lightgbm_cuda_cache
    if _lightgbm_cuda_cache is not None:
        return _lightgbm_cuda_cache

    # 先确认系统有 GPU，否则直接跳过昂贵的 LightGBM 测试
    if not _is_cuda_available():
        _lightgbm_cuda_cache = False
        return False

    try:
        import numpy as np
        import lightgbm as lgb

        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        y = np.array([1.0, 2.0], dtype=np.float32)
        ds = lgb.Dataset(X, label=y, params={"verbose": -1})
        lgb.train(
            {"device": "cuda", "num_leaves": 2, "verbose": -1, "num_threads": 1},
            ds,
            num_boost_round=1,
        )
        _lightgbm_cuda_cache = True
        return True
    except Exception:
        _lightgbm_cuda_cache = False
        return False


def get_lightgbm_device(device_str: str = "auto") -> str:
    """
    获取 LightGBM 可用的设备参数。

    与 _is_cuda_available() 不同，本函数会实际测试 LightGBM 是否编译了 CUDA 支持，
    因为 PyTorch 有 CUDA ≠ LightGBM 编译了 CUDA。

    Returns:
        'cuda', 'cpu'（MPS 不支持 LightGBM）
    """
    normalized = device_str.strip().lower()
    if normalized in ("cuda", "gpu"):
        if _is_lightgbm_cuda_available():
            return "cuda"
        print("⚠ LightGBM CUDA 不可用，回退到 CPU")
        return "cpu"
    if normalized in ("auto",):
        if _is_lightgbm_cuda_available():
            return "cuda"
        return "cpu"
    return "cpu"


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
