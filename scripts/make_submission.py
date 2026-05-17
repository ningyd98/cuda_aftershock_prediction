from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.features import (
    calculate_geological_features,
    calculate_spatial_anisotropy,
    calculate_temporal_binned_features,
    estimate_etas_parameters,
    estimate_gr_b_value,
    fit_omori_utsu,
    load_plate_boundaries,
)
from src.utils import haversine_km, seismic_moment_from_mw


DEFAULT_TARGET_COLS = ["target_max_mag", "target_time_to_max_days"]


def resolve_project_path(path_value: str | Path) -> Path:
    """将相对路径解析为项目根目录下的绝对路径。"""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_event_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """兼容 USGS 原始表、Date/Time 事件表和 Year/Month/Day 主震目录表。"""
    df = raw_df.copy()
    df = df.rename(
        columns={
            "Lat": "latitude",
            "Lon": "longitude",
            "Mag": "mag",
            "Depth": "depth",
        }
    )

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce", format="mixed")
    elif {"Date", "Time"}.issubset(df.columns):
        df["time"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            utc=True,
            errors="coerce",
            format="mixed",
        )
    elif {"Year", "Month", "Day", "Hour", "Minute", "Second"}.issubset(df.columns):
        df["time"] = pd.to_datetime(
            dict(
                year=df["Year"],
                month=df["Month"],
                day=df["Day"],
                hour=df["Hour"],
                minute=df["Minute"],
                second=df["Second"],
            ),
            utc=True,
            errors="coerce",
        )
    else:
        raise ValueError("输入 CSV 缺少 time，且无法从 Date/Time 或 Year/Month/Day 合成时间。")

    required_cols = ["time", "latitude", "longitude", "mag", "depth"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"输入 CSV 缺少必要字段: {missing_cols}")

    df = df.dropna(subset=required_cols).sort_values("time").reset_index(drop=True)
    if "id" not in df.columns:
        df["id"] = df["time"].dt.strftime("%Y%m%d%H%M%S").astype(str) + "_eq"
    return df


def pick_mainshock(event_df: pd.DataFrame) -> pd.Series:
    """选择输入事件表中震级最大的事件作为待预测主震。"""
    if event_df.empty:
        raise ValueError("输入事件表为空，无法识别主震。")
    max_mag = event_df["mag"].max()
    mainshock = event_df.loc[event_df["mag"] == max_mag].sort_values("time").iloc[0]
    return mainshock


def extract_early_events(
    event_df: pd.DataFrame,
    mainshock: pd.Series,
    obs_days: float,
    spatial_radius_km: float,
    earth_radius_km: float,
) -> pd.DataFrame:
    """截取主震后观测窗口内、空间半径内的早期余震。"""
    obs_end = mainshock["time"] + pd.Timedelta(days=obs_days)
    candidates = event_df.loc[
        (event_df["time"] > mainshock["time"]) & (event_df["time"] <= obs_end)
    ].copy()
    if candidates.empty:
        return candidates

    candidates["distance_km"] = haversine_km(
        float(mainshock["latitude"]),
        float(mainshock["longitude"]),
        candidates["latitude"].to_numpy(),
        candidates["longitude"].to_numpy(),
        earth_radius_km=earth_radius_km,
    )
    return candidates.loc[candidates["distance_km"] <= spatial_radius_km].copy()


def build_single_sequence_features(
    event_df: pd.DataFrame,
    plate_boundaries_path: Path,
    obs_days: float = 3.0,
    spatial_radius_km: float = 100.0,
    earth_radius_km: float = 6371.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """为单个待预测主震实时构建阶段一特征。"""
    mainshock = pick_mainshock(event_df)
    early_events = extract_early_events(
        event_df,
        mainshock,
        obs_days=obs_days,
        spatial_radius_km=spatial_radius_km,
        earth_radius_km=earth_radius_km,
    )

    early_mags = early_events["mag"].astype(float)
    mainshock_id = str(mainshock.get("id", mainshock["time"].strftime("%Y%m%d%H%M%S")))

    base_features = {
        "mainshock_id": mainshock_id,
        "mainshock_time": mainshock["time"],
        "mainshock_lat": float(mainshock["latitude"]),
        "mainshock_lon": float(mainshock["longitude"]),
        "mainshock_mag": float(mainshock["mag"]),
        "mainshock_depth": float(mainshock["depth"]),
        "early_aftershock_count": int(len(early_events)),
        "early_max_mag": float(early_mags.max()) if len(early_mags) else 0.0,
        "early_mean_mag": float(early_mags.mean()) if len(early_mags) else 0.0,
        "early_energy_sum": float(seismic_moment_from_mw(early_mags).sum())
        if len(early_mags)
        else 0.0,
        # 占位目标列不进入 submission，仅用于复用训练特征选择逻辑。
        "target_max_mag": np.nan,
        "target_time_to_max_days": np.nan,
        "advanced_early_event_count": int(len(early_events)),
    }
    base_features.update(estimate_gr_b_value(early_events))
    base_features.update(
        fit_omori_utsu(
            early_events,
            mainshock_time=mainshock["time"],
            obs_days=obs_days,
        )
    )
    base_features.update(
        calculate_spatial_anisotropy(
            early_events,
            mainshock_lat=float(mainshock["latitude"]),
            mainshock_lon=float(mainshock["longitude"]),
            earth_radius_km=earth_radius_km,
        )
    )
    base_features.update(
        calculate_temporal_binned_features(
            early_events,
            mainshock_time=mainshock["time"],
        )
    )
    base_features.update(
        estimate_etas_parameters(
            early_events,
            mainshock_time=mainshock["time"],
            obs_days=obs_days,
        )
    )

    feature_df = pd.DataFrame([base_features])
    boundaries_gdf = load_plate_boundaries(plate_boundaries_path)
    geology_df = calculate_geological_features(feature_df, boundaries_gdf)
    feature_df = feature_df.merge(geology_df, on="mainshock_id", how="left")
    return feature_df, early_events


def load_feature_cols(path: Path) -> list[str]:
    """读取训练阶段保存的特征列。"""
    with path.open("r", encoding="utf-8") as file:
        feature_cols = json.load(file)
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError(f"特征列文件格式非法: {path}")
    return feature_cols


def make_model_matrix(feature_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """按训练特征列构建推理矩阵，缺失列补 NaN，布尔列转 0/1。"""
    model_df = feature_df.copy()
    for col in feature_cols:
        if col not in model_df.columns:
            model_df[col] = np.nan
        if pd.api.types.is_bool_dtype(model_df[col]):
            model_df[col] = model_df[col].astype(int)
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    return model_df[feature_cols]


def load_ensemble_weights(path: Path | None) -> dict:
    """读取融合权重，缺省使用 baseline=1。"""
    if path is None or not path.exists():
        return {"baseline": 1.0, "dl": 0.0}
    with path.open("r", encoding="utf-8") as file:
        weights = json.load(file)
    return {
        "baseline": float(weights.get("baseline", 1.0)),
        "dl": float(weights.get("dl", 0.0)),
    }


def predict_with_baseline(model_path: Path, X: pd.DataFrame) -> np.ndarray:
    """加载 baseline 模型并预测。"""
    model = joblib.load(model_path)
    return np.asarray(model.predict(X), dtype=float).reshape(1, 2)


def predict_with_dl(
    dl_model_path: Path,
    dl_meta_path: Path,
    event_df: pd.DataFrame,
    global_feature_df: pd.DataFrame,
    device: str = "cpu",
) -> np.ndarray | None:
    """加载深度学习 Transformer 模型并预测。

    若模型文件不存在或加载失败，返回 None。
    """
    if not dl_model_path.exists() or not dl_meta_path.exists():
        return None

    try:
        import torch

        with open(dl_meta_path, "r") as f:
            dl_meta = json.load(f)

        from src.models_dl import Seq2SeqAftershockPredictor
        from src.dataset import EarthquakeSequenceDataset, SequenceBuildConfig

        model = Seq2SeqAftershockPredictor(
            event_feature_dim=dl_meta["event_feature_dim"],
            global_feature_dim=dl_meta["global_feature_dim"],
            d_model=dl_meta.get("d_model", 128),
            nhead=dl_meta.get("nhead", 4),
            num_layers=dl_meta.get("num_layers", 3),
        )
        model.load_state_dict(torch.load(dl_model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()

        # 构建 Dataset 获取单样本
        seq_config = SequenceBuildConfig(obs_days=3.0, spatial_radius_km=100.0, max_seq_len=256)
        dataset = EarthquakeSequenceDataset(
            sequence_df=global_feature_df,
            event_catalog_df=event_df,
            global_feature_cols=dl_meta["global_feature_cols"],
            config=seq_config,
        )
        from src.dataset import earthquake_collate_fn

        sample = dataset[0]
        batch = earthquake_collate_fn([sample])

        with torch.no_grad():
            seq_x = batch["seq_x"].to(device)
            global_x = batch["global_x"].to(device)
            mask = batch["seq_padding_mask"].to(device)
            preds = model(seq_x, global_x, mask)
            return preds.cpu().numpy().reshape(1, 2)
    except Exception as exc:
        print(f"   DL 模型预测失败: {exc}")
        return None


def rule_fallback_prediction(mainshock_mag: float, early_count: int) -> np.ndarray:
    """模型缺失或异常时的保底预测。"""
    fallback_mag = max(0.0, min(mainshock_mag - 1.2, mainshock_mag + 0.5))
    fallback_time = 1.0 if early_count > 0 else 0.0
    return np.array([[fallback_mag, fallback_time]], dtype=float)


def postprocess_prediction(
    pred: np.ndarray,
    mainshock_mag: float,
    early_count: int,
) -> tuple[float, float]:
    """按比赛语义裁剪异常预测。"""
    if pred.shape != (1, 2) or not np.isfinite(pred).all():
        pred = rule_fallback_prediction(mainshock_mag, early_count)

    predicted_mag = float(pred[0, 0])
    predicted_time = float(pred[0, 1])

    predicted_time = max(predicted_time, 0.0)
    predicted_mag = max(predicted_mag, max(0.0, mainshock_mag - 3.0))
    predicted_mag = min(predicted_mag, mainshock_mag + 0.5)
    return predicted_mag, predicted_time


def parse_args() -> argparse.Namespace:
    """解析推理参数。"""
    parser = argparse.ArgumentParser(description="生成余震预测比赛 submission.csv")
    parser.add_argument("--input", type=Path, required=True, help="待预测单序列 CSV")
    parser.add_argument("--output", type=Path, required=True, help="submission 输出路径")
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "baseline_model.joblib",
        help="baseline 模型路径",
    )
    parser.add_argument(
        "--feature-cols",
        type=Path,
        default=PROJECT_ROOT / "models" / "feature_cols.json",
        help="训练阶段保存的特征列路径",
    )
    parser.add_argument(
        "--ensemble-weights",
        type=Path,
        default=PROJECT_ROOT / "models" / "ensemble_weights.json",
        help="融合权重 JSON 路径",
    )
    parser.add_argument(
        "--plate-boundaries",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "PB2002_boundaries.json",
        help="PB2002 板块边界 GeoJSON",
    )
    parser.add_argument("--obs-days", type=float, default=3.0)
    parser.add_argument("--spatial-radius-km", type=float, default=100.0)
    parser.add_argument("--earth-radius-km", type=float, default=6371.0)
    parser.add_argument(
        "--allow-rule-fallback",
        action="store_true",
        help="模型产物缺失或预测失败时允许规则兜底输出",
    )
    return parser.parse_args()


def main() -> None:
    """端到端生成 submission.csv。"""
    args = parse_args()
    input_path = resolve_project_path(args.input)
    output_path = resolve_project_path(args.output)
    baseline_model_path = resolve_project_path(args.baseline_model)
    feature_cols_path = resolve_project_path(args.feature_cols)
    ensemble_weights_path = resolve_project_path(args.ensemble_weights)
    plate_boundaries_path = resolve_project_path(args.plate_boundaries)

    event_df = normalize_event_table(pd.read_csv(input_path))
    feature_df, early_events = build_single_sequence_features(
        event_df,
        plate_boundaries_path=plate_boundaries_path,
        obs_days=args.obs_days,
        spatial_radius_km=args.spatial_radius_km,
        earth_radius_km=args.earth_radius_km,
    )

    mainshock_id = str(feature_df.loc[0, "mainshock_id"])
    mainshock_mag = float(feature_df.loc[0, "mainshock_mag"])
    early_count = int(len(early_events))

    try:
        feature_cols = load_feature_cols(feature_cols_path)
        X = make_model_matrix(feature_df, feature_cols)
        weights = load_ensemble_weights(ensemble_weights_path)

        # Baseline 预测
        baseline_pred = predict_with_baseline(baseline_model_path, X)
        baseline_weight = max(float(weights.get("baseline", 1.0)), 0.0)

        # XGBoost 预测（如可用）
        xgb_pred = None
        xgb_weight = float(weights.get("xgboost", 0.0))
        if xgb_weight > 0:
            xgb_model_path = baseline_model_path.parent / "xgboost_model.joblib"
            if xgb_model_path.exists():
                xgb_pred = predict_with_baseline(xgb_model_path, X)
                print(f"   XGBoost 模型已加载，权重: {xgb_weight}")

        # DL 预测（如可用）
        dl_pred = None
        dl_weight = float(weights.get("dl", 0.0))
        if dl_weight > 0:
            dl_model_path = baseline_model_path.parent / "dl_model.pt"
            dl_meta_path = baseline_model_path.parent / "dl_meta.json"
            dl_pred = predict_with_dl(
                dl_model_path, dl_meta_path,
                event_df, feature_df,
            )
            if dl_pred is not None:
                print(f"   DL 模型已加载，权重: {dl_weight}")
            else:
                print("   DL 模型不可用，回退")

        # 加权融合
        total_weight = baseline_weight
        fused_pred = baseline_pred * baseline_weight
        if xgb_pred is not None and xgb_weight > 0:
            fused_pred = fused_pred + xgb_pred * xgb_weight
            total_weight += xgb_weight
        if dl_pred is not None and dl_weight > 0:
            fused_pred = fused_pred + dl_pred * dl_weight
            total_weight += dl_weight
        pred = fused_pred / total_weight if total_weight > 0 else baseline_pred
    except Exception as exc:
        if not args.allow_rule_fallback:
            raise RuntimeError(
                "模型推理失败。请确认 baseline_model.joblib、feature_cols.json "
                "和 ensemble_weights.json 已生成；或加 --allow-rule-fallback。"
            ) from exc
        print(f"警告：模型推理失败，使用规则兜底。原因: {exc}")
        pred = rule_fallback_prediction(mainshock_mag, early_count)

    predicted_mag, predicted_time = postprocess_prediction(
        pred,
        mainshock_mag=mainshock_mag,
        early_count=early_count,
    )

    submission_df = pd.DataFrame(
        [
            {
                "mainshock_id": mainshock_id,
                "predicted_max_mag": predicted_mag,
                "predicted_time_to_max": predicted_time,
            }
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"submission 已保存: {output_path}")
    print(submission_df.to_string(index=False))


if __name__ == "__main__":
    main()
