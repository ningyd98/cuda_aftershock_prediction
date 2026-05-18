from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import haversine_km, seismic_moment_from_mw


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_event_catalog(raw_csv_path: str | Path) -> pd.DataFrame:
    """读取并标准化 USGS 地震目录。"""
    df = pd.read_csv(raw_csv_path)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce", format="mixed")
    df = df.dropna(subset=["time", "latitude", "longitude", "mag", "depth"])
    df = df.sort_values("time").reset_index(drop=True)

    # 若目录缺少 id，则用稳定的行号补一个事件标识，保证后续去重逻辑可用。
    if "id" not in df.columns:
        df["id"] = [f"event_{idx}" for idx in range(len(df))]

    return df


def build_earthquake_sequences(
    raw_csv_path: str | Path,
    obs_days: float = 3.0,
    target_days: float = 30.0,
    spatial_radius_km: float = 100.0,
    min_mainshock_mag: float = 6.0,
    max_depth_km: float = 70.0,
    earth_radius_km: float = 6371.0,
) -> pd.DataFrame:
    """
    从原始地震目录构建以主震为行的标准化样本表。

    每行包含主震信息、观测窗口内早期余震统计，以及目标窗口内最大余震标签。
    """
    print("1. 加载并清洗基础数据...")
    df = load_event_catalog(raw_csv_path)

    potential_mainshocks = df[
        (df["mag"] >= min_mainshock_mag) & (df["depth"] <= max_depth_km)
    ].copy()

    sequence_data: list[dict] = []
    processed_mainshock_ids: set[str] = set()

    print(f"2. 开始划分序列... 发现潜在主震 {len(potential_mainshocks)} 次")

    for _, mainshock in potential_mainshocks.iterrows():
        if mainshock["id"] in processed_mainshock_ids:
            continue

        ms_time = mainshock["time"]
        ms_lat = float(mainshock["latitude"])
        ms_lon = float(mainshock["longitude"])
        obs_end_time = ms_time + timedelta(days=obs_days)
        target_end_time = ms_time + timedelta(days=target_days)

        mask_time = (df["time"] > ms_time) & (df["time"] <= target_end_time)
        candidates = df.loc[mask_time].copy()
        if candidates.empty:
            continue

        candidates["distance_km"] = haversine_km(
            ms_lat,
            ms_lon,
            candidates["latitude"].to_numpy(),
            candidates["longitude"].to_numpy(),
            earth_radius_km=earth_radius_km,
        )
        aftershocks = candidates.loc[candidates["distance_km"] <= spatial_radius_km]

        # 若后续窗口内出现更大地震，当前事件更像前震，暂不作为主震样本。
        if not aftershocks.empty and aftershocks["mag"].max() > mainshock["mag"]:
            continue

        processed_mainshock_ids.update(aftershocks["id"].tolist())

        early_aftershocks = aftershocks.loc[aftershocks["time"] <= obs_end_time]
        future_aftershocks = aftershocks.loc[aftershocks["time"] > obs_end_time]

        early_mags = early_aftershocks["mag"].astype(float)
        early_max_mag = float(early_mags.max()) if len(early_mags) else np.nan
        early_mean_mag = float(early_mags.mean()) if len(early_mags) else np.nan
        early_energy = (
            float(seismic_moment_from_mw(early_mags).sum())
            if len(early_mags)
            else 0.0
        )

        if future_aftershocks.empty:
            record = {
                "mainshock_id": mainshock["id"],
                "mainshock_time": ms_time,
                "mainshock_lat": ms_lat,
                "mainshock_lon": ms_lon,
                "mainshock_mag": float(mainshock["mag"]),
                "mainshock_depth": float(mainshock["depth"]),
                "early_aftershock_count": int(len(early_aftershocks)),
                "early_max_mag": early_max_mag if np.isfinite(early_max_mag) else 0.0,
                "early_mean_mag": early_mean_mag if np.isfinite(early_mean_mag) else 0.0,
                "early_energy_sum": early_energy,
                "has_target_aftershock": False,
                "target_max_mag": np.nan,
                "target_time_to_max_days": np.nan,
            }
        else:
            max_idx = future_aftershocks["mag"].idxmax()
            max_aftershock = future_aftershocks.loc[max_idx]
            max_aftershock_mag = float(max_aftershock["mag"])
            time_to_max = (
                max_aftershock["time"] - obs_end_time
            ).total_seconds() / 86400.0
            record = {
                "mainshock_id": mainshock["id"],
                "mainshock_time": ms_time,
                "mainshock_lat": ms_lat,
                "mainshock_lon": ms_lon,
                "mainshock_mag": float(mainshock["mag"]),
                "mainshock_depth": float(mainshock["depth"]),
                "early_aftershock_count": int(len(early_aftershocks)),
                "early_max_mag": early_max_mag if np.isfinite(early_max_mag) else 0.0,
                "early_mean_mag": early_mean_mag if np.isfinite(early_mean_mag) else 0.0,
                "early_energy_sum": early_energy,
                "has_target_aftershock": True,
                "target_max_mag": max_aftershock_mag,
                "target_time_to_max_days": float(time_to_max),
            }
        sequence_data.append(record)

    print("3. 序列化完成！")
    return pd.DataFrame(sequence_data)


def parse_args() -> argparse.Namespace:
    """解析基础样本构建脚本参数。"""
    parser = argparse.ArgumentParser(description="构建主震-余震基础样本表")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "USGS_Mw6.0_Depth70_1970-2023.csv",
        help="原始 USGS 地震目录路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "ML_Ready_Sequences.csv",
        help="基础样本表输出路径",
    )
    parser.add_argument("--obs-days", type=float, default=3.0, help="观测窗口天数")
    parser.add_argument("--target-days", type=float, default=30.0, help="目标窗口天数")
    parser.add_argument("--radius-km", type=float, default=100.0, help="空间半径")
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    sequence_df = build_earthquake_sequences(
        raw_csv_path=args.input,
        obs_days=args.obs_days,
        target_days=args.target_days,
        spatial_radius_km=args.radius_km,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sequence_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"基础样本表已保存: {args.output}")
    print(f"样本数: {len(sequence_df)}")


if __name__ == "__main__":
    main()
