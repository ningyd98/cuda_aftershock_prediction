from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.utils import haversine_km, seismic_moment_from_mw


def _round_to_bin(values: np.ndarray, bin_width: float) -> np.ndarray:
    """将震级归入指定精度的 bin，减少浮点误差对统计结果的影响。"""
    decimals = max(0, int(np.ceil(-np.log10(bin_width))))
    return np.round(np.round(values / bin_width) * bin_width, decimals)


def _empty_gr_result(n_events: int, mc: float = np.nan) -> dict:
    """返回统一格式的 Gutenberg-Richter 空结果。"""
    return {
        "gr_b_value": np.nan,
        "gr_a_value": np.nan,
        "gr_mc": mc,
        "gr_n": int(n_events),
        "gr_valid": False,
    }


def _empty_omori_result(n_events: int, nll: float = np.nan) -> dict:
    """返回统一格式的大森-宇津空结果。"""
    return {
        "omori_p": np.nan,
        "omori_c": np.nan,
        "omori_k": np.nan,
        "omori_nll": nll,
        "omori_n": int(n_events),
        "omori_p_boundary_hit": False,
        "omori_valid": False,
    }


def _empty_anisotropy_result(n_events: int) -> dict:
    """返回统一格式的空间各向异性空结果。"""
    return {
        "anisotropy_major_axis_km": np.nan,
        "anisotropy_minor_axis_km": np.nan,
        "anisotropy_axis_ratio": np.nan,
        "anisotropy_azimuth_deg": np.nan,
        "anisotropy_n": int(n_events),
        "anisotropy_valid": False,
    }


def calculate_bath_law_features(
    mainshock_mag: float,
    early_max_mag: float | None,
) -> dict:
    """
    计算 Båth's Law 相关特征。

    Båth's Law 经验上认为最大余震震级通常比主震低约 1.2 级。
    这里用主震震级与观测窗口内最大早期余震震级的差值，刻画序列
    已经释放出的最大余震强度缺口。
    """
    main_mag = float(mainshock_mag)
    early_mag = np.nan if early_max_mag is None else float(early_max_mag)
    valid = bool(np.isfinite(main_mag) and np.isfinite(early_mag) and early_mag > 0)

    if valid:
        bath_deficit = main_mag - early_mag
        bath_early_max_mag = early_mag
    else:
        bath_deficit = main_mag if np.isfinite(main_mag) else np.nan
        bath_early_max_mag = np.nan

    return {
        "bath_deficit": float(bath_deficit),
        "bath_early_max_mag": float(bath_early_max_mag)
        if np.isfinite(bath_early_max_mag)
        else np.nan,
        "bath_valid": valid,
    }


def estimate_mc_maxc(magnitudes: np.ndarray, bin_width: float = 0.1) -> float:
    """
    使用最大曲率法 MAXC 估计完整性震级 Mc。

    做法：按 bin_width 对震级做直方图，频次最高的震级 bin 即为 Mc；
    若多个 bin 频次并列最高，则取最小的震级 bin。
    """
    mags = magnitudes[np.isfinite(magnitudes)]
    if len(mags) == 0:
        return np.nan

    binned_mags = _round_to_bin(mags, bin_width)
    unique_bins, counts = np.unique(binned_mags, return_counts=True)
    mode_bins = unique_bins[counts == counts.max()]
    return float(mode_bins.min())


def estimate_gr_b_value(
    events: pd.DataFrame,
    mag_col: str = "mag",
    mc: float | None = None,
    bin_width: float = 0.1,
    min_events: int = 5,
) -> dict:
    """
    使用 Aki-Utsu 极大似然估计 Gutenberg-Richter 定律 b 值。

    若未显式传入 mc，则使用 MAXC 自动估计完整性震级。
    返回 a 值、b 值、Mc、参与拟合事件数和有效性标记。
    """
    if mag_col not in events.columns:
        return _empty_gr_result(0)

    mags = events[mag_col].dropna().astype(float).to_numpy()
    if len(mags) < min_events:
        return _empty_gr_result(len(mags))

    mc_hat = estimate_mc_maxc(mags, bin_width) if mc is None else float(mc)
    if not np.isfinite(mc_hat):
        return _empty_gr_result(len(mags), mc=mc_hat)

    complete_mags = mags[mags >= mc_hat]
    if len(complete_mags) < min_events:
        return _empty_gr_result(len(complete_mags), mc=mc_hat)

    denominator = complete_mags.mean() - (mc_hat - bin_width / 2.0)
    if denominator <= 0 or not np.isfinite(denominator):
        return _empty_gr_result(len(complete_mags), mc=mc_hat)

    b_value = np.log10(np.e) / denominator
    a_value = np.log10(len(complete_mags)) + b_value * mc_hat

    return {
        "gr_b_value": float(b_value),
        "gr_a_value": float(a_value),
        "gr_mc": float(mc_hat),
        "gr_n": int(len(complete_mags)),
        "gr_valid": bool(np.isfinite(b_value) and np.isfinite(a_value)),
    }


def calculate_productivity_index(
    mainshock_mag: float,
    gr_features: dict,
) -> dict:
    """计算地震生产率指数: a - b * M_main。"""
    a_value = float(gr_features.get("gr_a_value", np.nan))
    b_value = float(gr_features.get("gr_b_value", np.nan))
    main_mag = float(mainshock_mag)
    if not np.isfinite([a_value, b_value, main_mag]).all():
        return {"productivity_index": np.nan}
    return {"productivity_index": float(a_value - b_value * main_mag)}


def fit_omori_utsu(
    events: pd.DataFrame,
    mainshock_time,
    time_col: str = "time",
    obs_days: float = 3.0,
    min_events: int = 8,
    p_bounds: Sequence[float] = (0.2, 2.5),
    c_bounds: Sequence[float] = (1e-4, 10.0),
    initial_p: float = 1.0,
    initial_c: float = 0.05,
    boundary_tol: float = 1e-4,
    fit_start_day: float = 0.0,
) -> dict:
    """
    使用非齐次泊松过程 MLE 拟合大森-宇津定律。

    强度函数为 lambda(t) = K / (t + c)^p。若 p_hat 非常贴近优化边界，
    通常说明优化器撞到边界而非找到稳定极值，因此标记 omori_valid=False。
    """
    if time_col not in events.columns:
        return _empty_omori_result(0)

    if events.empty or len(events) < min_events:
        return _empty_omori_result(len(events))

    p_min, p_max = map(float, p_bounds)
    c_min, c_max = map(float, c_bounds)

    if p_min >= p_max or c_min <= 0 or c_min >= c_max:
        raise ValueError("大森定律拟合边界非法，请检查 p_bounds 与 c_bounds。")

    times = pd.to_datetime(events[time_col], utc=True, errors="coerce")
    mainshock_time = pd.to_datetime(mainshock_time, utc=True)
    elapsed_days = (times - mainshock_time).dt.total_seconds().to_numpy() / 86400.0
    elapsed_days = elapsed_days[np.isfinite(elapsed_days)]
    elapsed_days = elapsed_days[
        (elapsed_days > fit_start_day) & (elapsed_days <= obs_days)
    ]

    if len(elapsed_days) < min_events:
        return _empty_omori_result(len(elapsed_days))

    n_events = len(elapsed_days)

    def integral_term(p_value: float, c_value: float) -> float:
        """计算拟合窗口内强度函数积分项。"""
        start = fit_start_day + c_value
        end = obs_days + c_value
        if start <= 0 or end <= 0:
            return np.nan
        if abs(p_value - 1.0) < boundary_tol:
            return float(np.log(end / start))
        return float(
            (end ** (1.0 - p_value) - start ** (1.0 - p_value))
            / (1.0 - p_value)
        )

    def negative_log_likelihood(params: np.ndarray) -> float:
        """大森-宇津模型的负对数似然。"""
        p_value, log_c_value = params
        c_value = float(np.exp(log_c_value))

        integ = integral_term(float(p_value), c_value)
        if integ <= 0 or not np.isfinite(integ):
            return np.inf

        # 固定 p 与 c 后，K 的极大似然估计有解析解。
        k_value = n_events / integ
        if k_value <= 0 or not np.isfinite(k_value):
            return np.inf

        log_lambda = np.log(k_value) - p_value * np.log(elapsed_days + c_value)
        nll = -(np.sum(log_lambda) - k_value * integ)
        return float(nll) if np.isfinite(nll) else np.inf

    result = minimize(
        negative_log_likelihood,
        x0=np.array([initial_p, np.log(initial_c)], dtype=float),
        bounds=[(p_min, p_max), (np.log(c_min), np.log(c_max))],
        method="L-BFGS-B",
    )

    if not result.success:
        nll = float(result.fun) if np.isfinite(result.fun) else np.nan
        return _empty_omori_result(n_events, nll=nll)

    p_hat, log_c_hat = result.x
    c_hat = float(np.exp(log_c_hat))
    integ_hat = integral_term(float(p_hat), c_hat)
    k_hat = float(n_events / integ_hat) if integ_hat > 0 else np.nan

    p_boundary_hit = (
        abs(float(p_hat) - p_min) < boundary_tol
        or abs(float(p_hat) - p_max) < boundary_tol
    )

    return {
        "omori_p": float(p_hat),
        "omori_c": c_hat,
        "omori_k": k_hat,
        "omori_nll": float(result.fun),
        "omori_n": int(n_events),
        "omori_p_boundary_hit": bool(p_boundary_hit),
        "omori_valid": bool(
            result.success
            and not p_boundary_hit
            and np.isfinite(k_hat)
            and np.isfinite(result.fun)
        ),
    }


def calculate_spatial_anisotropy(
    events: pd.DataFrame,
    mainshock_lat: float,
    mainshock_lon: float,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    min_events: int = 3,
    ratio_epsilon: float = 1e-9,
    earth_radius_km: float = 6371.0,
) -> dict:
    """
    计算早期余震相对主震的空间各向异性特征。

    使用主震位置作为局部切平面原点，将经纬度近似投影到 km 坐标：
    x = R * cos(lat0) * dlon, y = R * dlat。随后对二维坐标做协方差
    分解，输出主轴长度、短轴长度、长短轴比和主轴方位角。
    """
    if lat_col not in events.columns or lon_col not in events.columns:
        return _empty_anisotropy_result(0)

    coords = events[[lat_col, lon_col]].dropna().astype(float)
    if len(coords) < min_events:
        return _empty_anisotropy_result(len(coords))

    lat0 = np.radians(float(mainshock_lat))
    lon0 = np.radians(float(mainshock_lon))
    lat = np.radians(coords[lat_col].to_numpy())
    lon = np.radians(coords[lon_col].to_numpy())

    x_km = earth_radius_km * np.cos(lat0) * (lon - lon0)
    y_km = earth_radius_km * (lat - lat0)
    xy = np.column_stack([x_km, y_km])

    if not np.isfinite(xy).all():
        xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < min_events:
        return _empty_anisotropy_result(len(xy))

    cov_matrix = np.cov(xy, rowvar=False)
    if cov_matrix.shape != (2, 2) or not np.isfinite(cov_matrix).all():
        return _empty_anisotropy_result(len(xy))

    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    eigvals = np.maximum(eigvals, 0.0)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    major_axis_km = float(np.sqrt(eigvals[0]))
    minor_axis_km = float(np.sqrt(eigvals[1]))
    axis_ratio = major_axis_km / max(minor_axis_km, ratio_epsilon)

    # 方位角定义为从正北顺时针到主轴方向，范围 [0, 180)。
    major_vec = eigvecs[:, 0]
    azimuth_deg = float((np.degrees(np.arctan2(major_vec[0], major_vec[1])) + 180.0) % 180.0)

    return {
        "anisotropy_major_axis_km": major_axis_km,
        "anisotropy_minor_axis_km": minor_axis_km,
        "anisotropy_axis_ratio": float(axis_ratio),
        "anisotropy_azimuth_deg": azimuth_deg,
        "anisotropy_n": int(len(xy)),
        "anisotropy_valid": bool(np.isfinite(axis_ratio)),
    }


def calculate_temporal_binned_features(
    events: pd.DataFrame,
    mainshock_time,
    time_col: str = "time",
    mag_col: str = "mag",
    bins_hours: Sequence[float] = (1.0, 6.0, 12.0, 24.0, 72.0),
    min_events: int = 1,
) -> dict:
    """
    计算主震后多个时间窗口内的余震频次与累积能量分布。

    对应 project_plan 第 3.2 节"时空演化特征"：
    震后 1h/6h/12h/24h/72h 内的余震发生频次、累积释放能量。

    返回 dict，每个时间窗产出 count_{h}h 和 energy_{h}h。
    """
    result: dict = {}
    if events.empty or time_col not in events.columns:
        for h in bins_hours:
            result[f"count_{h:.0f}h"] = 0
            result[f"energy_{h:.0f}h"] = 0.0
        return result

    times = pd.to_datetime(events[time_col], utc=True, errors="coerce")
    mainshock_time = pd.to_datetime(mainshock_time, utc=True)
    hours_elapsed = (times - mainshock_time).dt.total_seconds() / 3600.0

    mags = events[mag_col].astype(float).to_numpy() if mag_col in events.columns else np.zeros(len(events))

    for h in sorted(bins_hours):
        in_window = (hours_elapsed > 0) & (hours_elapsed <= h)
        count = int(in_window.sum())
        energy = float(seismic_moment_from_mw(mags[in_window]).sum()) if count > 0 else 0.0
        result[f"count_{h:.0f}h"] = count
        result[f"energy_{h:.0f}h"] = energy
    return result


def _empty_etas_result(n_events: int) -> dict:
    """返回统一格式的 ETAS 空结果。"""
    return {
        "etas_mu": np.nan,
        "etas_K0": np.nan,
        "etas_alpha": np.nan,
        "etas_c": np.nan,
        "etas_p": np.nan,
        "etas_nll": np.nan,
        "etas_n": int(n_events),
        "etas_valid": False,
    }


def estimate_etas_parameters(
    events: pd.DataFrame,
    mainshock_time,
    mainshock_mag: float | None = None,
    time_col: str = "time",
    mag_col: str = "mag",
    obs_days: float = 3.0,
    min_events: int = 10,
    max_events: int = 500,
    mc: float = 2.5,
    bin_width: float = 0.1,
) -> dict:
    """
    简化 ETAS (Epidemic Type Aftershock Sequence) 模型参数估计 (优化版)。

    对于超过 max_events 的序列做随机下采样以控制拟合时间。
    mainshock_mag 参数接收真实主震震级；若未传入则回退到早期余震
    最大震级（保持向后兼容）。
    """
    if events.empty or time_col not in events.columns:
        return _empty_etas_result(0)

    times = pd.to_datetime(events[time_col], utc=True, errors="coerce")
    mainshock_time = pd.to_datetime(mainshock_time, utc=True)
    elapsed_days = (times - mainshock_time).dt.total_seconds().to_numpy() / 86400.0
    elapsed_days = elapsed_days[np.isfinite(elapsed_days)]
    elapsed_days = elapsed_days[(elapsed_days > 0) & (elapsed_days <= obs_days)]

    if len(elapsed_days) < min_events:
        return _empty_etas_result(len(elapsed_days))

    n_events = len(elapsed_days)

    # 下采样：大序列截断到 max_events
    if n_events > max_events:
        rng = np.random.RandomState(42 + n_events)
        keep_idx = rng.choice(n_events, size=max_events, replace=False)
        elapsed_days = elapsed_days[keep_idx]
        n_events = max_events

    mags = events[mag_col].dropna().astype(float).to_numpy()
    mags_in_window = mags[np.isfinite(mags)]
    mags_in_window = mags_in_window[
        (pd.to_datetime(events[time_col], utc=True, errors="coerce") > mainshock_time)
        & (pd.to_datetime(events[time_col], utc=True, errors="coerce") <= mainshock_time + pd.Timedelta(days=obs_days))
    ]

    if len(mags_in_window) < min_events:
        return _empty_etas_result(n_events)

    # Step 1: 估计 b 值
    gr_result = estimate_gr_b_value(
        events[events[time_col].notna()].assign(mag=mags),
        mag_col="mag",
        mc=mc,
        bin_width=bin_width,
        min_events=min_events,
    )
    if not gr_result["gr_valid"]:
        return _empty_etas_result(n_events)

    b_value = gr_result["gr_b_value"]
    beta = b_value * np.log(10)

    # Step 2: 简化 ETAS — 固定 Omori 参数，仅估计 μ, K0, α
    # 强度函数: λ(t,m) = μ + Σ K0 * exp(α*(Mi-Mc)) / (t-ti + c)^p
    # 简化为仅用主震贡献: λ(t) ≈ μ + K0 * exp(α*(M0-Mc)) / (t + c)^p

    # 优先使用显式传入的真实主震震级，回退到早期余震最大震级
    if mainshock_mag is not None and np.isfinite(mainshock_mag):
        _mainshock_mag = float(mainshock_mag)
    else:
        _mainshock_mag = float(mags_in_window.max()) if len(mags_in_window) > 0 else mc + 1.0
    mag_factor = np.exp(beta * (_mainshock_mag - mc))

    # 先拟合大森定律获取 p, c
    omori_result = fit_omori_utsu(
        events,
        mainshock_time=mainshock_time,
        time_col=time_col,
        obs_days=obs_days,
        min_events=min_events,
    )
    p_hat = omori_result.get("omori_p", 1.0) if omori_result["omori_valid"] else 1.0
    c_hat = omori_result.get("omori_c", 0.05) if omori_result["omori_valid"] else 0.05

    if not np.isfinite(p_hat) or p_hat <= 0:
        p_hat = 1.0
    if not np.isfinite(c_hat) or c_hat <= 0:
        c_hat = 0.05

    # Step 3: 用 simplified MLE 估计 μ 和 K0
    # 对于给定 Omori 参数，λ(t) = μ + A / (t+c)^p, A = K0 * exp(α*(M0-Mc))
    # 使用 profile likelihood: 固定 α，优化 μ 和 K0

    def omori_integral(start_day: float, end_day: float, p_val: float, c_val: float) -> float:
        """∫ 1/(t+c)^p dt from start_day to end_day"""
        s = max(start_day + c_val, 1e-9)
        e = max(end_day + c_val, 1e-9)
        if abs(p_val - 1.0) < 1e-4:
            return float(np.log(e / s))
        return float((e ** (1.0 - p_val) - s ** (1.0 - p_val)) / (1.0 - p_val))

    def etas_neg_log_lik(params: np.ndarray) -> float:
        """简化 ETAS 负对数似然（固定 p, c）。"""
        log_mu, log_K0, alpha = params
        mu = float(np.exp(log_mu))
        K0 = float(np.exp(log_K0))
        alpha = float(alpha)

        if mu <= 0 or K0 <= 0:
            return np.inf

        A = K0 * np.exp(alpha * (_mainshock_mag - mc))
        integ_bg = mu * obs_days
        integ_triggered = A * omori_integral(0.0, obs_days, p_hat, c_hat)
        total_integ = integ_bg + integ_triggered

        if total_integ <= 0 or not np.isfinite(total_integ):
            return np.inf

        # Log-likelihood for each event
        triggered_contrib = A / (elapsed_days + c_hat) ** p_hat
        lambda_vals = mu + triggered_contrib
        if np.any(lambda_vals <= 0):
            return np.inf

        log_lik = np.sum(np.log(lambda_vals)) - total_integ
        nll = -float(log_lik)
        return nll if np.isfinite(nll) else np.inf

    try:
        result = minimize(
            etas_neg_log_lik,
            x0=np.array([np.log(0.01), np.log(0.1), 0.5], dtype=float),
            bounds=[
                (np.log(1e-6), np.log(10.0)),   # log_mu
                (np.log(1e-6), np.log(100.0)),  # log_K0
                (0.0, 3.0),                      # alpha
            ],
            method="L-BFGS-B",
            options={"maxiter": 30},  # 快速近似
        )

        if result.success:
            log_mu, log_K0, alpha_hat = result.x
            return {
                "etas_mu": float(np.exp(log_mu)),
                "etas_K0": float(np.exp(log_K0)),
                "etas_alpha": float(alpha_hat),
                "etas_c": float(c_hat),
                "etas_p": float(p_hat),
                "etas_nll": float(result.fun),
                "etas_n": int(n_events),
                "etas_valid": True,
            }
        else:
            return _empty_etas_result(n_events)
    except Exception:
        return _empty_etas_result(n_events)


def _require_geospatial_dependencies():
    """按需导入地理空间依赖，并给出清晰的安装提示。"""
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError as exc:
        raise ImportError(
            "地质构造特征需要 geopandas、shapely 和 pyproj。"
            "请先运行: pip install -r requirements.txt"
        ) from exc
    return gpd, Point


def _normalize_plate_boundary_type(
    raw_value,
    unknown_type: str = "UNK",
    subduction_label: str = "SUB",
) -> str:
    """将 PB2002 边界类型标准化为稳定类别。"""
    if raw_value is None or pd.isna(raw_value):
        return unknown_type

    value = str(raw_value).strip()
    if not value:
        return unknown_type

    upper_value = value.upper()
    if value.lower() == "subduction":
        return subduction_label
    return upper_value


def load_plate_boundaries(
    geojson_path: str | Path,
    type_field: str = "STEP_CLASS",
    fallback_type_field: str = "Type",
    unknown_type: str = "UNK",
    subduction_label: str = "SUB",
):
    """
    读取 Peter Bird 板块边界 GeoJSON，并标准化边界类型字段。

    类型字段优先使用 STEP_CLASS；若当前文件不存在该字段，则回退到 Type。
    """
    gpd, _ = _require_geospatial_dependencies()
    boundaries_gdf = gpd.read_file(geojson_path)

    if boundaries_gdf.empty:
        raise ValueError(f"板块边界文件为空: {geojson_path}")

    if boundaries_gdf.crs is None:
        boundaries_gdf = boundaries_gdf.set_crs("EPSG:4326")
    else:
        boundaries_gdf = boundaries_gdf.to_crs("EPSG:4326")

    if type_field in boundaries_gdf.columns:
        raw_type = boundaries_gdf[type_field]
    elif fallback_type_field in boundaries_gdf.columns:
        raw_type = boundaries_gdf[fallback_type_field]
    else:
        raw_type = pd.Series([unknown_type] * len(boundaries_gdf), index=boundaries_gdf.index)

    boundaries_gdf = boundaries_gdf.copy()
    boundaries_gdf["plate_boundary_type"] = raw_type.map(
        lambda value: _normalize_plate_boundary_type(
            value,
            unknown_type=unknown_type,
            subduction_label=subduction_label,
        )
    )
    return boundaries_gdf


def calculate_geological_features(
    sequence_df: pd.DataFrame,
    boundaries_gdf,
    lat_col: str = "mainshock_lat",
    lon_col: str = "mainshock_lon",
    distance_crs: str = "EPSG:3857",
    one_hot_types: Sequence[str] = ("SUB", "OTF", "OSR", "UNK"),
    unknown_type: str = "UNK",
) -> pd.DataFrame:
    """
    批量计算主震到最近板块边界的距离和边界类型特征。

    输出与 sequence_df 等长的 DataFrame，包含 mainshock_id、最近边界距离、
    最近边界类型，以及稳定 One-Hot 类型列。
    """
    gpd, Point = _require_geospatial_dependencies()
    required_cols = ["mainshock_id", lat_col, lon_col]
    missing_cols = [col for col in required_cols if col not in sequence_df.columns]
    if missing_cols:
        raise ValueError(f"基础样本表缺少地质特征所需字段: {missing_cols}")

    points_df = sequence_df[required_cols].copy()
    valid_mask = points_df[[lat_col, lon_col]].notna().all(axis=1)

    result_df = pd.DataFrame({"mainshock_id": sequence_df["mainshock_id"]})
    result_df["plate_boundary_distance_km"] = np.nan
    result_df["nearest_plate_boundary_type"] = unknown_type

    for plate_type in one_hot_types:
        result_df[f"plate_type_{plate_type}"] = 0

    if not valid_mask.any():
        result_df[f"plate_type_{unknown_type}"] = 1
        return result_df

    point_geometries = [
        Point(lon, lat)
        for lat, lon in points_df.loc[valid_mask, [lat_col, lon_col]].itertuples(index=False)
    ]
    points_gdf = gpd.GeoDataFrame(
        points_df.loc[valid_mask, ["mainshock_id"]].copy(),
        geometry=point_geometries,
        crs="EPSG:4326",
    )

    projected_points = points_gdf.to_crs(distance_crs)
    projected_boundaries = boundaries_gdf.to_crs(distance_crs)
    projected_boundaries = projected_boundaries[["plate_boundary_type", "geometry"]].copy()

    nearest = gpd.sjoin_nearest(
        projected_points,
        projected_boundaries,
        how="left",
        distance_col="plate_boundary_distance_m",
    )
    nearest = nearest.drop_duplicates(subset=["mainshock_id"], keep="first")
    nearest["plate_boundary_distance_km"] = (
        nearest["plate_boundary_distance_m"].astype(float) / 1000.0
    )
    nearest["nearest_plate_boundary_type"] = (
        nearest["plate_boundary_type"].fillna(unknown_type).astype(str)
    )
    nearest.loc[
        ~nearest["nearest_plate_boundary_type"].isin(one_hot_types),
        "nearest_plate_boundary_type",
    ] = unknown_type

    result_df = result_df.drop(columns=["plate_boundary_distance_km", "nearest_plate_boundary_type"])
    result_df = result_df.merge(
        nearest[
            [
                "mainshock_id",
                "plate_boundary_distance_km",
                "nearest_plate_boundary_type",
            ]
        ],
        on="mainshock_id",
        how="left",
    )
    result_df["nearest_plate_boundary_type"] = result_df[
        "nearest_plate_boundary_type"
    ].fillna(unknown_type)

    for plate_type in one_hot_types:
        result_df[f"plate_type_{plate_type}"] = (
            result_df["nearest_plate_boundary_type"] == plate_type
        ).astype(int)

    return result_df


# ============================================================
#  震源机制解特征 (Global CMT)
#  对应 project_plan 第 3.3 节
# ============================================================

def _load_gcmt_catalog(gcmt_path: Path) -> pd.DataFrame | None:
    """安全加载 GCMT 目录。"""
    if not gcmt_path.exists():
        return None
    try:
        gcmt = pd.read_csv(gcmt_path)
        gcmt["time"] = pd.to_datetime(
            gcmt["time"],
            utc=True,
            errors="coerce",
            format="mixed",
        )
        return gcmt.dropna(subset=["time", "latitude", "longitude"])
    except Exception:
        return None


GCMT_FEATURE_COLUMNS = [
    "strike1", "dip1", "rake1",
    "strike2", "dip2", "rake2",
    "fault_type",
    "fault_type_Thrust", "fault_type_Normal",
    "fault_type_Strike_Slip", "fault_type_Unknown",
    "fault_type_NF", "fault_type_SS", "fault_type_TF", "fault_type_UNK",
    "plunge_P", "trend_P", "plunge_T", "trend_T",
    "f_clvd",
    "gcmt_time_diff_seconds", "gcmt_distance_km",
    "focal_mechanism_valid",
]


def classify_fault_type_from_rake(rake: float) -> str:
    """
    根据 rake 角粗分类震源机制。

    Thrust 对应逆冲/逆断层，Normal 对应正断层，Strike-Slip 对应走滑。
    斜滑样本在三分类中按 rake 落入的主导象限归入最接近类别。
    """
    if not np.isfinite(rake):
        return "Unknown"
    rake_norm = ((float(rake) + 180.0) % 360.0) - 180.0
    if 45.0 <= rake_norm <= 135.0:
        return "Thrust"
    if -135.0 <= rake_norm <= -45.0:
        return "Normal"
    if abs(rake_norm) <= 45.0 or abs(rake_norm) >= 135.0:
        return "Strike-Slip"
    return "Unknown"


def _legacy_fault_type_code(fault_type: str) -> str:
    """把新 fault_type 标签映射到旧版 TF/NF/SS/UNK 编码。"""
    mapping = {
        "Thrust": "TF",
        "Normal": "NF",
        "Strike-Slip": "SS",
        "Unknown": "UNK",
    }
    return mapping.get(fault_type, "UNK")


def _empty_gcmt_features() -> dict:
    """返回未匹配 GCMT 时的稳定空特征。"""
    return {
        "strike1": np.nan, "dip1": np.nan, "rake1": np.nan,
        "strike2": np.nan, "dip2": np.nan, "rake2": np.nan,
        "fault_type": "Unknown",
        "fault_type_Thrust": 0, "fault_type_Normal": 0,
        "fault_type_Strike_Slip": 0, "fault_type_Unknown": 1,
        "fault_type_NF": 0, "fault_type_SS": 0,
        "fault_type_TF": 0, "fault_type_UNK": 1,
        "plunge_P": np.nan, "trend_P": np.nan,
        "plunge_T": np.nan, "trend_T": np.nan,
        "f_clvd": np.nan,
        "gcmt_time_diff_seconds": np.nan,
        "gcmt_distance_km": np.nan,
        "focal_mechanism_valid": False,
    }


def _row_to_gcmt_features(row: pd.Series, time_diff_seconds: float, distance_km: float) -> dict:
    """把一条 GCMT 匹配记录转为模型特征。"""
    rake1 = float(row.get("rake1", np.nan))
    fault_type = classify_fault_type_from_rake(rake1)
    legacy_code = _legacy_fault_type_code(fault_type)
    features = _empty_gcmt_features()
    features.update(
        {
            "strike1": float(row.get("strike1", np.nan)),
            "dip1": float(row.get("dip1", np.nan)),
            "rake1": rake1,
            "strike2": float(row.get("strike2", np.nan)),
            "dip2": float(row.get("dip2", np.nan)),
            "rake2": float(row.get("rake2", np.nan)),
            "fault_type": fault_type,
            "fault_type_Thrust": 1 if fault_type == "Thrust" else 0,
            "fault_type_Normal": 1 if fault_type == "Normal" else 0,
            "fault_type_Strike_Slip": 1 if fault_type == "Strike-Slip" else 0,
            "fault_type_Unknown": 1 if fault_type == "Unknown" else 0,
            "fault_type_NF": 1 if legacy_code == "NF" else 0,
            "fault_type_SS": 1 if legacy_code == "SS" else 0,
            "fault_type_TF": 1 if legacy_code == "TF" else 0,
            "fault_type_UNK": 1 if legacy_code == "UNK" else 0,
            "plunge_P": float(row.get("plunge_P", np.nan)),
            "trend_P": float(row.get("trend_P", np.nan)),
            "plunge_T": float(row.get("plunge_T", np.nan)),
            "trend_T": float(row.get("trend_T", np.nan)),
            "f_clvd": float(row.get("f_clvd", np.nan)),
            "gcmt_time_diff_seconds": float(time_diff_seconds),
            "gcmt_distance_km": float(distance_km),
            "focal_mechanism_valid": bool(np.isfinite(row.get("strike1", np.nan))),
        }
    )
    return features


def merge_gcmt_features(
    sequence_df: pd.DataFrame,
    gcmt_csv_path: str | Path,
    time_tolerance_seconds: float = 60.0,
    spatial_radius_km: float = 50.0,
    earth_radius_km: float = 6371.0,
    time_col: str = "mainshock_time",
    lat_col: str = "mainshock_lat",
    lon_col: str = "mainshock_lon",
    id_col: str = "mainshock_id",
) -> pd.DataFrame:
    """
    将主震序列表与 Global CMT 震源机制目录近似匹配。

    匹配条件：事件时间差小于 time_tolerance_seconds，且震中距离小于
    spatial_radius_km。若候选多条，优先选择时间差最小，其次距离最近。
    """
    result_df = sequence_df.copy()
    required_cols = [id_col, time_col, lat_col, lon_col]
    missing_cols = [col for col in required_cols if col not in result_df.columns]
    if missing_cols:
        raise ValueError(f"GCMT 匹配缺少必要字段: {missing_cols}")

    for col in GCMT_FEATURE_COLUMNS:
        if col in result_df.columns:
            result_df = result_df.drop(columns=col)

    gcmt_path = Path(gcmt_csv_path)
    gcmt_df = _load_gcmt_catalog(gcmt_path)
    empty_template = _empty_gcmt_features()
    if gcmt_df is None or gcmt_df.empty:
        feature_rows = [
            {id_col: row[id_col], **empty_template}
            for _, row in result_df.iterrows()
        ]
        return result_df.merge(pd.DataFrame(feature_rows), on=id_col, how="left")

    seq_times = pd.to_datetime(result_df[time_col], utc=True, errors="coerce")
    gcmt_df = gcmt_df.sort_values("time").reset_index(drop=True)
    feature_rows: list[dict] = []

    for idx, row in result_df.iterrows():
        features = _empty_gcmt_features()
        ms_time = seq_times.iloc[idx]
        if pd.notna(ms_time):
            time_diff_seconds = (
                (gcmt_df["time"] - ms_time).dt.total_seconds().abs()
            )
            time_mask = time_diff_seconds <= float(time_tolerance_seconds)
            candidates = gcmt_df.loc[time_mask].copy()

            if not candidates.empty:
                candidate_time_diff = time_diff_seconds.loc[candidates.index].to_numpy()
                distances = haversine_km(
                    row[lat_col],
                    row[lon_col],
                    candidates["latitude"].to_numpy(),
                    candidates["longitude"].to_numpy(),
                    earth_radius_km=earth_radius_km,
                )
                spatial_mask = distances <= float(spatial_radius_km)
                if spatial_mask.any():
                    valid_candidates = candidates.loc[spatial_mask].copy()
                    valid_time_diff = candidate_time_diff[spatial_mask]
                    valid_distances = distances[spatial_mask]
                    order = np.lexsort((valid_distances, valid_time_diff))
                    best_pos = int(order[0])
                    features = _row_to_gcmt_features(
                        valid_candidates.iloc[best_pos],
                        time_diff_seconds=float(valid_time_diff[best_pos]),
                        distance_km=float(valid_distances[best_pos]),
                    )

        feature_rows.append({id_col: row[id_col], **features})

    focal_df = pd.DataFrame(feature_rows)
    return result_df.merge(focal_df, on=id_col, how="left")


def match_focal_mechanism(
    mainshock_time,
    mainshock_lat: float,
    mainshock_lon: float,
    gcmt_df: pd.DataFrame,
    time_window_days: float = 1.0,
    spatial_radius_km: float = 200.0,
    earth_radius_km: float = 6371.0,
) -> dict:
    """
    匹配主震到 Global CMT 目录中最近的震源机制解。

    返回 strike/dip/rake、断层类型 One-Hot 和 P/T 轴信息。
    """
    from src.utils import haversine_km as _hav

    empty = {
        "strike1": np.nan, "dip1": np.nan, "rake1": np.nan,
        "strike2": np.nan, "dip2": np.nan, "rake2": np.nan,
        "fault_type_NF": 0, "fault_type_SS": 0,
        "fault_type_TF": 0, "fault_type_UNK": 1,
        "plunge_P": np.nan, "trend_P": np.nan,
        "plunge_T": np.nan, "trend_T": np.nan,
        "f_clvd": np.nan,
        "focal_mechanism_valid": False,
    }

    if gcmt_df is None or gcmt_df.empty:
        return empty

    ms_time = pd.to_datetime(mainshock_time, utc=True)
    t_mask = (
        (gcmt_df["time"] >= ms_time - pd.Timedelta(days=time_window_days))
        & (gcmt_df["time"] <= ms_time + pd.Timedelta(days=time_window_days))
    )
    candidates = gcmt_df.loc[t_mask].copy()
    if candidates.empty:
        return empty

    dists = _hav(
        mainshock_lat, mainshock_lon,
        candidates["latitude"].to_numpy(),
        candidates["longitude"].to_numpy(),
        earth_radius_km=earth_radius_km,
    )
    in_range = dists <= spatial_radius_km
    if not in_range.any():
        return empty

    # 仅在空间范围内选择距离最近的候选
    in_range_indices = np.where(in_range)[0]
    best_local_idx = in_range_indices[np.argmin(dists[in_range])]
    row = candidates.iloc[best_local_idx]

    ft = str(row.get("fault_type", "UNK"))
    fault_types = {"NF", "SS", "TF", "UNK"}
    ft_clean = ft if ft in fault_types else "UNK"

    return {
        "strike1": float(row.get("strike1", np.nan)),
        "dip1": float(row.get("dip1", np.nan)),
        "rake1": float(row.get("rake1", np.nan)),
        "strike2": float(row.get("strike2", np.nan)),
        "dip2": float(row.get("dip2", np.nan)),
        "rake2": float(row.get("rake2", np.nan)),
        "fault_type_NF": 1 if ft_clean == "NF" else 0,
        "fault_type_SS": 1 if ft_clean == "SS" else 0,
        "fault_type_TF": 1 if ft_clean == "TF" else 0,
        "fault_type_UNK": 1 if ft_clean == "UNK" else 0,
        "plunge_P": float(row.get("plunge_P", np.nan)),
        "trend_P": float(row.get("trend_P", np.nan)),
        "plunge_T": float(row.get("plunge_T", np.nan)),
        "trend_T": float(row.get("trend_T", np.nan)),
        "f_clvd": float(row.get("f_clvd", np.nan)),
        "focal_mechanism_valid": bool(np.isfinite(row.get("strike1", np.nan))),
    }
