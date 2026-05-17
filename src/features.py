from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


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
