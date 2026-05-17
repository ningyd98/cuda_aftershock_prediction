from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.features import estimate_gr_b_value, fit_omori_utsu
from src.utils import haversine_km


def load_config(config_path: Path) -> dict:
    """读取 YAML 配置文件。"""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_project_path(path_value: str | Path) -> Path:
    """将配置中的相对路径解析为项目根目录下的绝对路径。"""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


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
    omori_features = fit_omori_utsu(
        early_events,
        mainshock_time=sequence_row.mainshock_time,
        obs_days=phase1_cfg["obs_days"],
        **omori_kwargs,
    )

    return {
        "mainshock_id": sequence_row.mainshock_id,
        "advanced_early_event_count": int(len(early_events)),
        **gr_features,
        **omori_features,
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
    output_path = (
        resolve_project_path(args.output)
        if args.output is not None
        else resolve_project_path(cfg["paths"]["advanced_features_csv"])
    )

    sequence_df = pd.read_csv(sequences_path)
    if args.limit is not None:
        sequence_df = sequence_df.head(args.limit).copy()

    event_df = normalize_event_catalog(pd.read_csv(event_catalog_path))

    phase1_cfg = cfg["phase1"]
    gr_kwargs = dict(phase1_cfg["gr"])
    omori_kwargs = dict(phase1_cfg["omori"])
    parallel_cfg = cfg["parallel"]
    rows = list(sequence_df.itertuples(index=False))

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
        )
        for row in tqdm(rows, desc="提取高级特征")
    )

    feature_df = pd.DataFrame(results)
    merged_df = sequence_df.merge(feature_df, on="mainshock_id", how="left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"高级特征已保存: {output_path}")
    print(f"样本数: {len(merged_df)}")
    print(f"有效 b 值样本数: {int(merged_df['gr_valid'].sum())}")
    print(f"有效大森参数样本数: {int(merged_df['omori_valid'].sum())}")


if __name__ == "__main__":
    main()
