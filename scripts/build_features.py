from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.features import (
    calculate_bath_law_features,
    calculate_geological_features,
    calculate_productivity_index,
    calculate_spatial_anisotropy,
    calculate_temporal_binned_features,
    estimate_etas_parameters,
    estimate_gr_b_value,
    fit_omori_utsu,
    load_plate_boundaries,
    merge_gcmt_features,
)
from src.utils import haversine_km


def load_config(config_path: Path) -> dict:
    """读取 YAML 配置文件。"""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_project_path(path_value: str | Path) -> Path:
    """将配置中的相对路径解析为项目根目录下的绝对路径。"""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_gcmt_catalog(gcmt_path: Path, gcmt_cfg: dict) -> None:
    """GCMT 启用但本地目录缺失时，自动下载官方 NDK 目录并解析为 CSV。"""
    if gcmt_path.exists():
        return
    if not bool(gcmt_cfg.get("auto_download", True)):
        return

    print(f"GCMT 目录不存在，开始自动下载: {gcmt_path}")
    try:
        from scripts.download_gcmt import download_gcmt_catalog

        download_gcmt_catalog(
            start_year=int(gcmt_cfg.get("start_year", 1976)),
            end_year=int(gcmt_cfg.get("end_year", 2024)),
            output_dir=gcmt_path.parent,
            request_sleep=float(gcmt_cfg.get("request_sleep", 0.2)),
        )
    except Exception as exc:
        print(f"⚠ GCMT 自动下载失败: {exc}")


def normalize_event_catalog(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    统一原始事件目录字段。

    支持 USGS 原始字段，也兼容 test_eq_data 中 Date + Time 的字段形式。
    """
    df = raw_df.copy()
    df = df.rename(
        columns={
            "Lat": "latitude",
            "Lon": "longitude",
            "Mag": "mag",
            "Depth": "depth",
        }
    )

    if "time" not in df.columns and {"Date", "Time"}.issubset(df.columns):
        df["time"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            utc=True,
            errors="coerce",
        )
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    else:
        raise ValueError("事件目录缺少 time 字段，且无法从 Date + Time 合成。")

    required_cols = ["time", "latitude", "longitude", "mag"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"事件目录缺少必要字段: {missing_cols}")

    df = df.dropna(subset=required_cols)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def extract_early_aftershocks(
    sequence_row,
    event_df: pd.DataFrame,
    obs_days: float,
    spatial_radius_km: float,
    earth_radius_km: float,
) -> pd.DataFrame:
    """根据基础样本表中的主震信息，重建观测窗口内的早期余震明细。"""
    mainshock_time = pd.to_datetime(sequence_row.mainshock_time, utc=True)
    obs_end_time = mainshock_time + pd.Timedelta(days=obs_days)

    time_mask = (event_df["time"] > mainshock_time) & (event_df["time"] <= obs_end_time)
    candidates = event_df.loc[time_mask].copy()
    if candidates.empty:
        return candidates

    candidates["distance_km"] = haversine_km(
        sequence_row.mainshock_lat,
        sequence_row.mainshock_lon,
        candidates["latitude"].to_numpy(),
        candidates["longitude"].to_numpy(),
        earth_radius_km=earth_radius_km,
    )
    return candidates.loc[candidates["distance_km"] <= spatial_radius_km].copy()


def build_one_sequence_features(
    sequence_row,
    event_df: pd.DataFrame,
    phase1_cfg: dict,
    gr_kwargs: dict,
    omori_kwargs: dict,
    anisotropy_kwargs: dict,
) -> dict:
    """单个主震序列的高级特征计算单元，供 joblib 并行调用。"""
    early_events = extract_early_aftershocks(
        sequence_row=sequence_row,
        event_df=event_df,
        obs_days=phase1_cfg["obs_days"],
        spatial_radius_km=phase1_cfg["spatial_radius_km"],
        earth_radius_km=phase1_cfg["earth_radius_km"],
    )

    gr_features = estimate_gr_b_value(early_events, **gr_kwargs)
    productivity_features = calculate_productivity_index(
        mainshock_mag=sequence_row.mainshock_mag,
        gr_features=gr_features,
    )
    omori_features = fit_omori_utsu(
        early_events,
        mainshock_time=sequence_row.mainshock_time,
        obs_days=phase1_cfg["obs_days"],
        **omori_kwargs,
    )
    anisotropy_features = calculate_spatial_anisotropy(
        early_events,
        mainshock_lat=sequence_row.mainshock_lat,
        mainshock_lon=sequence_row.mainshock_lon,
        earth_radius_km=phase1_cfg["earth_radius_km"],
        **anisotropy_kwargs,
    )
    temporal_features = calculate_temporal_binned_features(
        early_events,
        mainshock_time=sequence_row.mainshock_time,
    )
    etas_features = {}
    try:
        etas_features = estimate_etas_parameters(
            early_events,
            mainshock_time=sequence_row.mainshock_time,
            mainshock_mag=sequence_row.mainshock_mag,
            obs_days=phase1_cfg["obs_days"],
        )
    except Exception:
        etas_features = {
            "etas_mu": np.nan, "etas_K0": np.nan, "etas_alpha": np.nan,
            "etas_c": np.nan, "etas_p": np.nan, "etas_nll": np.nan,
            "etas_n": int(len(early_events)), "etas_valid": False,
        }

    early_max_mag = (
        float(early_events["mag"].max())
        if "mag" in early_events.columns and not early_events.empty
        else np.nan
    )
    bath_features = calculate_bath_law_features(
        mainshock_mag=sequence_row.mainshock_mag,
        early_max_mag=early_max_mag,
    )

    return {
        "mainshock_id": sequence_row.mainshock_id,
        "advanced_early_event_count": int(len(early_events)),
        **bath_features,
        **gr_features,
        **productivity_features,
        **omori_features,
        **anisotropy_features,
        **temporal_features,
        **etas_features,
    }


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="并行生成阶段一高级地震学特征")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理前 N 条主震样本，用于快速冒烟测试",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="覆盖配置中的输出路径",
    )
    return parser.parse_args()


def main() -> None:
    """读取基础序列表与事件目录，并行生成高级特征。"""
    args = parse_args()
    cfg = load_config(args.config)

    sequences_path = resolve_project_path(cfg["paths"]["base_sequences_csv"])
    event_catalog_path = resolve_project_path(cfg["paths"]["event_catalog_csv"])
    plate_boundaries_path = resolve_project_path(cfg["paths"]["plate_boundaries_geojson"])
    output_path = (
        resolve_project_path(args.output)
        if args.output is not None
        else resolve_project_path(cfg["paths"]["advanced_features_csv"])
    )

    # 若完整事件目录不存在，回退到主震目录
    if not event_catalog_path.exists():
        fallback_path = resolve_project_path(
            cfg["paths"].get("mainshock_catalog_csv", "data/raw/USGS_Mw6.0_Depth70_1970-2023.csv")
        )
        print(f"⚠ 完整事件目录不存在: {event_catalog_path}")
        print(f"  回退到主震目录: {fallback_path}")
        print(f"  提示: 运行 python main.py download-full-catalog 下载完整目录以获得更好的特征质量")
        event_catalog_path = fallback_path

    sequence_df = pd.read_csv(sequences_path)
    if args.limit is not None:
        sequence_df = sequence_df.head(args.limit).copy()

    event_df = normalize_event_catalog(pd.read_csv(event_catalog_path))

    phase1_cfg = cfg["phase1"]
    gr_kwargs = dict(phase1_cfg["gr"])
    omori_kwargs = dict(phase1_cfg["omori"])
    anisotropy_kwargs = dict(phase1_cfg["anisotropy"])
    geology_kwargs = dict(phase1_cfg["geology"])
    parallel_cfg = cfg["parallel"]
    rows = list(sequence_df.itertuples(index=False))

    geology_enabled = bool(geology_kwargs.pop("enabled", True))
    if geology_enabled:
        boundaries_gdf = load_plate_boundaries(
            plate_boundaries_path,
            type_field=geology_kwargs.pop("type_field"),
            fallback_type_field=geology_kwargs.pop("fallback_type_field"),
            unknown_type=geology_kwargs["unknown_type"],
            subduction_label=geology_kwargs.pop("subduction_label"),
        )
        geology_df = calculate_geological_features(
            sequence_df,
            boundaries_gdf,
            **geology_kwargs,
        )
    else:
        geology_df = pd.DataFrame({"mainshock_id": sequence_df["mainshock_id"]})

    results = Parallel(
        n_jobs=parallel_cfg["n_jobs"],
        backend=parallel_cfg["backend"],
        verbose=parallel_cfg.get("verbose", 0),
        max_nbytes=parallel_cfg.get("max_nbytes", "64M"),
    )(
        delayed(build_one_sequence_features)(
            sequence_row=row,
            event_df=event_df,
            phase1_cfg=phase1_cfg,
            gr_kwargs=gr_kwargs,
            omori_kwargs=omori_kwargs,
            anisotropy_kwargs=anisotropy_kwargs,
        )
        for row in tqdm(rows, desc="提取高级特征")
    )

    feature_df = pd.DataFrame(results)
    merged_df = sequence_df.merge(feature_df, on="mainshock_id", how="left")
    merged_df = merged_df.merge(geology_df, on="mainshock_id", how="left")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- GCMT 震源机制解匹配 ----
    gcmt_path = cfg["paths"].get("gcmt_catalog_csv")
    gcmt_cfg = cfg.get("phase1", {}).get("gcmt", {})
    gcmt_enabled = bool(gcmt_cfg.get("enabled", True))
    if gcmt_enabled and gcmt_path:
        gcmt_path = resolve_project_path(gcmt_path)
        ensure_gcmt_catalog(gcmt_path, gcmt_cfg)
        if gcmt_path.exists():
            merged_df = merge_gcmt_features(
                merged_df,
                gcmt_csv_path=gcmt_path,
                time_tolerance_seconds=float(gcmt_cfg.get("time_tolerance_seconds", 60.0)),
                spatial_radius_km=float(gcmt_cfg.get("spatial_radius_km", 50.0)),
                earth_radius_km=float(phase1_cfg.get("earth_radius_km", 6371.0)),
            )
            print(f"GCMT 匹配完成: {int(merged_df['focal_mechanism_valid'].sum())} 条有效")
        else:
            print("⚠ GCMT 目录不可用，跳过震源机制解特征")
    else:
        print("⚠ 跳过震源机制解特征（GCMT 目录未找到或被禁用）")

    merged_df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"高级特征已保存: {output_path}")
    print(f"样本数: {len(merged_df)}")
    print(f"总特征列数: {len(merged_df.columns)}")
    print(f"有效 b 值样本数: {int(merged_df['gr_valid'].sum())}")
    print(f"有效大森参数样本数: {int(merged_df['omori_valid'].sum())}")
    print(f"有效各向异性样本数: {int(merged_df['anisotropy_valid'].sum())}")
    print(f"有效震源机制解样本数: {int(merged_df.get('focal_mechanism_valid', pd.Series([0]*len(merged_df))).sum())}")
    if geology_enabled:
        print("最近板块边界类型分布:")
        print(merged_df["nearest_plate_boundary_type"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
