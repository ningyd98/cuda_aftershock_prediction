#!/usr/bin/env python
# ============================================================
#  余震预测 —— 全面超参数调优脚本
#
#  策略:
#   1. 以 20 条测试序列的真实标签 (从 USGS 目录提取) 作为
#      最终 holdout 评估集。
#   2. 用 Optuna 在训练数据上做 TimeSeriesSplit OOF CV，
#      搜索最优超参数。
#   3. 最终用最优参数全量训练，在 holdout 上出报告。
#
#  调优维度:
#   - LightGBM: num_leaves, learning_rate, subsample, ...
#   - XGBoost:  max_depth, learning_rate, subsample, ...
#   - 特征工程: impute_missing, feature_selection_ratio
#   - 融合:     ensemble_grid_step
#   - DL/GNN:   (可选) d_model, nhead, num_layers, lr, ...
#
#  用法:
#    python scripts/tune_hyperparams.py --n-trials 100
#    python scripts/tune_hyperparams.py --n-trials 200 --with-dl --with-gnn
#    python scripts/tune_hyperparams.py --study-name my_study --n-trials 50
# ============================================================

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import calculate_metrics
from src.models import BaselineLGBM, BaselineXGBoost
from src.utils import set_random_seed, get_lightgbm_device

# ---- 常量 ----
TARGET_COLS = ["target_max_mag", "target_time_to_max_days"]
TIME_COL = "mainshock_time"
ID_COL = "mainshock_id"

TEST_DIR = PROJECT_ROOT / "data" / "test_sequences"
FULL_CATALOG = PROJECT_ROOT / "data" / "raw" / "USGS_Mw4.0_Depth70_1970-2023.csv"
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "advanced_features.csv"

# 领域知识缺失值默认值 (来自 train_baseline.py)
_DOMAIN_DEFAULTS: dict[str, float] = {
    "gr_b_value": 1.0, "gr_a_value": 5.0, "gr_mc": 4.0,
    "gr_n": 5, "gr_valid": 0,
    "omori_p": 1.0, "omori_c": 0.05, "omori_k": 1.0,
    "omori_nll": 10.0, "omori_n": 8, "omori_p_boundary_hit": 0, "omori_valid": 0,
    "etas_mu": 0.01, "etas_K0": 0.001, "etas_alpha": 1.0,
    "etas_c": 0.05, "etas_p": 1.0, "etas_nll": 10.0, "etas_n": 8, "etas_valid": 0,
    "anisotropy_major_axis_km": 50.0, "anisotropy_minor_axis_km": 25.0,
    "anisotropy_axis_ratio": 2.0, "anisotropy_azimuth_deg": 90.0,
    "anisotropy_n": 3, "anisotropy_valid": 0,
    "bath_deficit": 1.2, "bath_early_max_mag": 0.0, "bath_valid": 0,
    "productivity_index": 0.0, "focal_mechanism_valid": 0,
    "strike1": 0.0, "dip1": 45.0, "rake1": 0.0,
    "strike2": 0.0, "dip2": 45.0, "rake2": 0.0,
    "plunge_P": 0.0, "trend_P": 0.0, "plunge_T": 0.0, "trend_T": 0.0,
    "f_clvd": 0.0, "gcmt_time_diff_seconds": 86400.0, "gcmt_distance_km": 100.0,
    "early_aftershock_count": 0, "early_max_mag": 0.0,
    "early_mean_mag": 0.0, "early_energy_sum": 0.0,
    "count_1h": 0, "energy_1h": 0.0, "count_6h": 0, "energy_6h": 0.0,
    "count_12h": 0, "energy_12h": 0.0, "count_24h": 0, "energy_24h": 0.0,
    "count_72h": 0, "energy_72h": 0.0, "advanced_early_event_count": 0,
}

# OOF CSV → 模型名映射
MODEL_OOF_MAP = {
    "baseline": "oof_predictions.csv",
    "xgboost": "oof_predictions.csv",
    "dl": "dl_oof_predictions.csv",
    "gnn": "gnn_oof_predictions.csv",
}
MODEL_PRED_COLS = {
    "baseline": ("baseline_pred_mag", "baseline_pred_time"),
    "xgboost": ("xgboost_pred_mag", "xgboost_pred_time"),
    "dl": ("dl_pred_mag", "dl_pred_time"),
    "gnn": ("gnn_pred_mag", "gnn_pred_time"),
}


# ============================================================
#  工具函数
# ============================================================

def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """向量化 Haversine 距离 (km)。"""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def extract_true_labels(full_catalog_path: Path, test_dir: Path) -> pd.DataFrame:
    """从 USGS 完整目录中提取 20 条测试序列的真实标签。

    Returns:
        DataFrame with columns: mainshock_id, true_max_mag, true_time_to_max_days
    """
    print("=" * 60)
    print("提取测试序列真实标签 …")
    full = pd.read_csv(full_catalog_path)
    full["time"] = pd.to_datetime(full["time"], format="mixed", utc=True)

    records = []
    for csv_file in sorted(test_dir.glob("*_eq.csv")):
        seq = pd.read_csv(csv_file)
        sid = csv_file.stem
        main_lat = float(seq.iloc[0]["Lat"])
        main_lon = float(seq.iloc[0]["Lon"])
        main_time = pd.to_datetime(
            str(seq.iloc[0]["Date"]) + " " + str(seq.iloc[0]["Time"]), utc=True,
        )
        main_mag = float(seq.iloc[0]["Mag"])
        time_end = main_time + pd.Timedelta(days=30)

        mask = (full["time"] > main_time) & (full["time"] <= time_end)
        candidates = full[mask].copy()
        if len(candidates) == 0:
            true_mag, true_time = 0.0, 0.0
        else:
            dists = haversine_km(
                main_lat, main_lon,
                candidates["latitude"].values.astype(float),
                candidates["longitude"].values.astype(float),
            )
            nearby = candidates[dists <= 100.0]
            if len(nearby) == 0:
                true_mag, true_time = 0.0, 0.0
            else:
                idx_max = nearby["mag"].idxmax()
                true_mag = float(nearby.loc[idx_max, "mag"])
                true_time = (nearby.loc[idx_max, "time"] - main_time).total_seconds() / 86400.0

        records.append({
            "mainshock_id": sid,
            "mainshock_mag": main_mag,
            "true_max_mag": true_mag,
            "true_time_to_max_days": true_time,
        })

    result = pd.DataFrame(records)
    n_with = (result["true_max_mag"] > 0).sum()
    print(f"  {len(result)} 条测试序列, {n_with} 条有余震")
    return result


# ============================================================
#  特征工程
# ============================================================

FEATURE_PREFIXES = (
    "early_", "gr_", "omori_", "anisotropy_", "plate_type_",
    "count_", "energy_", "etas_", "bath_", "fault_type_",
    "productivity_", "mag_ratio_", "mag_diff_", "energy_per_",
    "log_energy_", "count_ratio_", "energy_ratio_",
    "omori_p_", "omori_decay_", "etas_p_", "aniso_",
    "plate_dist_", "log_plate_", "b_value_", "log_depth",
    "depth_mag_", "productivity_per_",
)
EXPLICIT_FEATURES = {
    "mainshock_mag", "mainshock_depth", "advanced_early_event_count",
    "plate_boundary_distance_km",
    "strike1", "dip1", "rake1", "strike2", "dip2", "rake2",
    "plunge_P", "trend_P", "plunge_T", "trend_T", "f_clvd",
    "gcmt_time_diff_seconds", "gcmt_distance_km", "focal_mechanism_valid",
}
EXCLUDE_COLS = {
    "mainshock_id", "mainshock_time", "mainshock_lat", "mainshock_lon",
    "nearest_plate_boundary_type", "has_target_aftershock", *TARGET_COLS,
}


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """自动筛选数值特征列。"""
    candidates = []
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
        if col in EXPLICIT_FEATURES or col.startswith(FEATURE_PREFIXES):
            candidates.append(col)
    numeric = []
    for col in candidates:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
    return numeric


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加交互特征 (同 train_baseline.py)。"""
    df = df.copy()
    if "mainshock_mag" in df.columns and "early_max_mag" in df.columns:
        df["mag_ratio_early_main"] = df["early_max_mag"] / df["mainshock_mag"].clip(lower=1.0)
        df["mag_diff_main_early"] = df["mainshock_mag"] - df["early_max_mag"]
    if "early_energy_sum" in df.columns and "early_aftershock_count" in df.columns:
        df["energy_per_event"] = df["early_energy_sum"] / df["early_aftershock_count"].clip(lower=1)
        df["log_energy_sum"] = np.log1p(df["early_energy_sum"])
    if "count_1h" in df.columns and "count_72h" in df.columns:
        df["count_ratio_1h_72h"] = df["count_1h"] / df["count_72h"].clip(lower=1)
        df["count_ratio_6h_72h"] = df.get("count_6h", 0) / df["count_72h"].clip(lower=1)
        df["count_ratio_24h_72h"] = df.get("count_24h", 0) / df["count_72h"].clip(lower=1)
    if "energy_1h" in df.columns and "energy_72h" in df.columns:
        df["energy_ratio_1h_72h"] = df["energy_1h"] / df["energy_72h"].clip(lower=1e-10)
        df["energy_ratio_24h_72h"] = df.get("energy_24h", 0) / df["energy_72h"].clip(lower=1e-10)
    if "omori_p" in df.columns and "omori_c" in df.columns:
        df["omori_p_times_c"] = df["omori_p"] * df["omori_c"]
        df["omori_decay_rate"] = df["omori_p"] / df["omori_c"].clip(lower=1e-6)
    if "etas_p" in df.columns and "etas_alpha" in df.columns:
        df["etas_p_alpha_ratio"] = df["etas_p"] / df["etas_alpha"].clip(lower=1e-6)
    if "anisotropy_major_axis_km" in df.columns and "mainshock_mag" in df.columns:
        df["aniso_area_proxy"] = df["anisotropy_major_axis_km"] * df.get("anisotropy_minor_axis_km", 0)
        df["aniso_per_mag"] = df["anisotropy_major_axis_km"] / df["mainshock_mag"].clip(lower=1.0)
    if "plate_boundary_distance_km" in df.columns and "mainshock_mag" in df.columns:
        df["plate_dist_per_mag"] = df["plate_boundary_distance_km"] / df["mainshock_mag"].clip(lower=1.0)
        df["log_plate_dist"] = np.log1p(df["plate_boundary_distance_km"])
    if "gr_b_value" in df.columns and "mainshock_mag" in df.columns:
        df["b_value_times_mag"] = df["gr_b_value"] * df["mainshock_mag"]
    if "mainshock_depth" in df.columns:
        df["log_depth"] = np.log1p(df["mainshock_depth"].clip(lower=0))
        df["depth_mag_ratio"] = df["mainshock_depth"] / df["mainshock_mag"].clip(lower=1.0)
    if "productivity_index" in df.columns and "early_aftershock_count" in df.columns:
        df["productivity_per_event"] = df["productivity_index"] / df["early_aftershock_count"].clip(lower=1)
    return df


def impute_missing_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """领域知识缺失值填补 + missing indicator。"""
    df = df.copy()
    new_cols = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        null_mask = df[col].isnull()
        if not null_mask.any():
            continue
        indicator_col = f"{col}_missing"
        df[indicator_col] = null_mask.astype(int)
        new_cols.append(indicator_col)
        if col in _DOMAIN_DEFAULTS:
            df[col] = df[col].fillna(_DOMAIN_DEFAULTS[col])
        else:
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0.0)
    return df, new_cols


def select_features_by_importance(
    X: pd.DataFrame, y: pd.DataFrame, feature_cols: list[str],
    top_k_ratio: float = 0.85, min_features: int = 30,
) -> list[str]:
    """LightGBM gain importance 特征筛选。"""
    if len(feature_cols) <= min_features:
        return feature_cols
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        return feature_cols

    model = LGBMRegressor(
        n_estimators=100, num_leaves=31, learning_rate=0.05,
        random_state=42, n_jobs=1, verbosity=-1,
    )
    model.fit(X, y.iloc[:, 0])
    importance = model.booster_.feature_importance(importance_type="gain")
    sorted_idx = np.argsort(importance)[::-1]
    total_gain = importance.sum()
    if total_gain <= 0:
        return feature_cols
    cumsum = np.cumsum(importance[sorted_idx]) / total_gain
    n_keep = max(min_features, int(np.searchsorted(cumsum, top_k_ratio) + 1))
    n_keep = min(n_keep, len(feature_cols))
    return [feature_cols[i] for i in sorted_idx[:n_keep]]


# ============================================================
#  模型训练
# ============================================================

def build_lgbm(params: dict, n_estimators: int, seed: int, late_weight: float) -> BaselineLGBM:
    """根据参数字典构建 BaselineLGBM。"""
    device = get_lightgbm_device("auto")
    return BaselineLGBM(
        random_state=seed,
        n_estimators=n_estimators,
        learning_rate=params.get("lgb_learning_rate", 0.03),
        num_leaves=params.get("lgb_num_leaves", 63),
        max_depth=params.get("lgb_max_depth", -1),
        min_child_samples=params.get("lgb_min_child_samples", 20),
        subsample=params.get("lgb_subsample", 0.8),
        colsample_bytree=params.get("lgb_colsample_bytree", 0.7),
        reg_alpha=params.get("lgb_reg_alpha", 0.05),
        reg_lambda=params.get("lgb_reg_lambda", 1.0),
        use_asymmetric_time_objective=params.get("use_asymmetric_time_objective", True),
        late_weight=late_weight,
        transform_time_target=True,
        device=device,
    )


def build_xgboost_model(params: dict, n_estimators: int, seed: int) -> BaselineXGBoost:
    """根据参数字典构建 BaselineXGBoost。"""
    return BaselineXGBoost(
        random_state=seed,
        n_estimators=n_estimators,
        learning_rate=params.get("xgb_learning_rate", 0.03),
        max_depth=params.get("xgb_max_depth", 6),
        min_child_weight=params.get("xgb_min_child_weight", 1),
        subsample=params.get("xgb_subsample", 0.8),
        colsample_bytree=params.get("xgb_colsample_bytree", 0.7),
        reg_alpha=params.get("xgb_reg_alpha", 0.0),
        reg_lambda=params.get("xgb_reg_lambda", 1.0),
        transform_time_target=True,
    )


# ============================================================
#  OOF CV
# ============================================================

def run_single_oof_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
    n_splits: int,
    purge_days: float,
    min_purge_days: float,
    seed: int,
    late_weight: float,
    n_estimators_lgb: int,
    n_estimators_xgb: int,
) -> tuple[pd.DataFrame, dict]:
    """单次 OOF CV：LightGBM + XGBoost。

    Returns:
        (oof_df, metrics_dict)
    """
    from sklearn.model_selection import TimeSeriesSplit

    train_df = df.sort_values(TIME_COL).reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    X_all = train_df[feature_cols]
    y_all = train_df[TARGET_COLS]

    purge_delta = pd.Timedelta(days=purge_days)
    min_purge = pd.Timedelta(days=min_purge_days)

    oof_preds = {
        "baseline": np.full((len(train_df), 2), np.nan),
        "xgboost": np.full((len(train_df), 2), np.nan),
    }

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(X_all), start=1):
        valid_start_time = train_df.loc[valid_idx[0], TIME_COL]

        # purge
        purge_cutoff = valid_start_time - purge_delta
        purge_mask = train_df.loc[train_idx, TIME_COL] <= purge_cutoff
        train_idx_p = train_idx[purge_mask.values]
        if len(train_idx_p) < max(10, len(train_idx) * 0.2):
            min_cutoff = valid_start_time - min_purge
            min_mask = train_df.loc[train_idx, TIME_COL] <= min_cutoff
            train_idx_p = train_idx[min_mask.values]
            if len(train_idx_p) < max(5, len(train_idx) * 0.1):
                train_idx_p = train_idx

        # LightGBM
        lgb = build_lgbm(params, n_estimators_lgb, seed, late_weight)
        lgb.fit(X_all.iloc[train_idx_p], y_all.iloc[train_idx_p])
        preds_lgb = np.clip(np.asarray(lgb.predict(X_all.iloc[valid_idx]), dtype=float), 0, None)
        oof_preds["baseline"][valid_idx] = preds_lgb

        # XGBoost
        xgb = build_xgboost_model(params, n_estimators_xgb, seed)
        xgb.fit(X_all.iloc[train_idx_p], y_all.iloc[train_idx_p])
        preds_xgb = np.clip(np.asarray(xgb.predict(X_all.iloc[valid_idx]), dtype=float), 0, None)
        oof_preds["xgboost"][valid_idx] = preds_xgb

    # 构建 OOF DataFrame
    oof_df = train_df[[ID_COL, TIME_COL, *TARGET_COLS]].copy()
    oof_df["baseline_pred_mag"] = oof_preds["baseline"][:, 0]
    oof_df["baseline_pred_time"] = oof_preds["baseline"][:, 1]
    oof_df["xgboost_pred_mag"] = oof_preds["xgboost"][:, 0]
    oof_df["xgboost_pred_time"] = oof_preds["xgboost"][:, 1]

    # 搜索融合权重 (2-model grid)
    valid_mask = (
        oof_df["baseline_pred_mag"].notna()
        & oof_df["xgboost_pred_mag"].notna()
    )
    y_mag = oof_df.loc[valid_mask, TARGET_COLS[0]].to_numpy()
    y_time = oof_df.loc[valid_mask, TARGET_COLS[1]].to_numpy()
    b_mag = oof_df.loc[valid_mask, "baseline_pred_mag"].to_numpy()
    b_time = oof_df.loc[valid_mask, "baseline_pred_time"].to_numpy()
    x_mag = oof_df.loc[valid_mask, "xgboost_pred_mag"].to_numpy()
    x_time = oof_df.loc[valid_mask, "xgboost_pred_time"].to_numpy()

    grid = np.arange(0.0, 1.01, 0.02)
    best_mag_w, best_mag_rmse = 0.5, float("inf")
    best_time_w, best_time_obj = 0.5, float("inf")

    for w in grid:
        pm = w * b_mag + (1 - w) * x_mag
        rmse = float(np.sqrt(np.mean((pm - y_mag) ** 2)))
        if rmse < best_mag_rmse:
            best_mag_rmse = rmse
            best_mag_w = float(w)

        pt = w * b_time + (1 - w) * x_time
        err = pt - y_time
        tw = np.where(err > 0, late_weight, 1.0)
        asym_rmse = float(np.sqrt(np.mean(tw * err ** 2)))
        if asym_rmse < best_time_obj:
            best_time_obj = asym_rmse
            best_time_w = float(w)

    # 融合预测
    final_mag = best_mag_w * b_mag + (1 - best_mag_w) * x_mag
    final_time = best_time_w * b_time + (1 - best_time_w) * x_time
    metrics = calculate_metrics(
        y_true_mag=y_mag, y_pred_mag=final_mag,
        y_true_time=y_time, y_pred_time=final_time,
        late_weight=late_weight,
    )

    # combined_objective: 震级 RMSE + 时间非对称 RMSE
    combined_obj = float(best_mag_rmse + best_time_obj)

    return oof_df, {
        "mag_rmse": best_mag_rmse,
        "mag_mae": float(metrics.get("mag_mae", np.nan)),
        "time_rmse": float(metrics.get("time_rmse", np.nan)),
        "time_mae": float(metrics.get("time_mae", np.nan)),
        "time_asymmetric_rmse": best_time_obj,
        "time_asymmetric_mae": float(metrics.get("time_asymmetric_mae", np.nan)),
        "combined_objective": combined_obj,
        "mag_baseline_weight": best_mag_w,
        "time_baseline_weight": best_time_w,
    }


# ============================================================
#  测试序列特征预计算
# ============================================================

def precompute_test_features(
    test_dir: Path,
    plate_boundaries_path: Path,
    gcmt_catalog_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """为所有 20 条测试序列预计算特征，保存为单个 CSV。

    这样 holdout 评估时不需重复提取特征。
    """
    # 延迟导入，避免循环依赖
    from scripts.make_submission import (
        build_single_sequence_features,
        normalize_event_table,
    )

    all_features = []
    test_files = sorted(test_dir.glob("*_eq.csv"))
    print(f"预计算 {len(test_files)} 条测试序列特征 …")

    for csv_file in test_files:
        sid = csv_file.stem
        raw_df = pd.read_csv(csv_file)
        event_df = normalize_event_table(raw_df)
        feat_df, _ = build_single_sequence_features(
            event_df=event_df,
            plate_boundaries_path=plate_boundaries_path,
            gcmt_catalog_path=gcmt_catalog_path if gcmt_catalog_path.exists() else None,
        )
        feat_df["mainshock_id"] = sid  # 统一 ID
        all_features.append(feat_df)

    combined = pd.concat(all_features, ignore_index=True)
    combined.to_csv(output_path, index=False, encoding="utf-8")
    print(f"  已保存: {output_path} ({len(combined)} 条)")
    return combined


def load_test_features(features_path: Path | None = None) -> pd.DataFrame | None:
    """加载预计算的测试序列特征。"""
    if features_path is None:
        features_path = PROJECT_ROOT / "data" / "processed" / "test_sequences_features.csv"
    if not features_path.exists():
        return None
    return pd.read_csv(features_path)


# ============================================================
#  Holdout 评估
# ============================================================

def evaluate_on_holdout(
    oof_df: pd.DataFrame,
    true_labels: pd.DataFrame,
    params: dict,
    late_weight: float,
    n_estimators_lgb: int,
    n_estimators_xgb: int,
    seed: int,
    feature_cols_override: list[str] | None = None,
) -> dict:
    """在 20 条测试序列上做最终评估。

    使用预计算的特征 (test_sequences_features.csv)，
    对每条测试序列用该序列时间之前的训练样本训练模型并预测。
    """
    # 加载训练数据
    raw_df = pd.read_csv(FEATURES_CSV)
    raw_df[TIME_COL] = pd.to_datetime(raw_df[TIME_COL], utc=True, format="mixed")
    raw_df = raw_df.dropna(subset=[TIME_COL, *TARGET_COLS])
    raw_df = add_derived_features(raw_df)
    all_feature_cols = select_feature_columns(raw_df)
    raw_df, missing_cols = impute_missing_features(raw_df, all_feature_cols)
    all_feature_cols = all_feature_cols + missing_cols

    # 加载预计算的测试特征
    test_feat_path = PROJECT_ROOT / "data" / "processed" / "test_sequences_features.csv"
    if not test_feat_path.exists():
        print("  ⚠ 测试序列特征未预计算，运行 precompute_test_features …")
        precompute_test_features(
            TEST_DIR,
            PROJECT_ROOT / "data" / "raw" / "PB2002_boundaries.json",
            PROJECT_ROOT / "data" / "raw" / "GlobalCMT_1976-2024.csv",
            test_feat_path,
        )
    test_feat = pd.read_csv(test_feat_path)
    # 对齐特征列
    test_feat = add_derived_features(test_feat)
    test_feat, _ = impute_missing_features(test_feat, all_feature_cols)

    # 特征选择
    fs_ratio = params.get("feature_selection_ratio", 0.85)
    fs_min = params.get("feature_selection_min", 30)
    if params.get("feature_selection", True) and len(all_feature_cols) > 50:
        all_feature_cols = select_features_by_importance(
            raw_df[all_feature_cols], raw_df[TARGET_COLS],
            all_feature_cols, top_k_ratio=fs_ratio, min_features=fs_min,
        )

    # 确保测试特征包含所需列
    missing_in_test = set(all_feature_cols) - set(test_feat.columns)
    for c in missing_in_test:
        test_feat[c] = 0.0 if c in _DOMAIN_DEFAULTS else 0.0

    predictions = []
    for _, row in true_labels.iterrows():
        sid = row["mainshock_id"]
        # 读取主震时间
        seq_file = TEST_DIR / f"{sid}.csv"
        if not seq_file.exists():
            continue
        seq = pd.read_csv(seq_file)
        main_time = pd.to_datetime(
            str(seq.iloc[0]["Date"]) + " " + str(seq.iloc[0]["Time"]), utc=True,
        )

        # 只用该主震之前的训练数据
        train_mask = raw_df[TIME_COL] < main_time
        train_df = raw_df[train_mask]
        if len(train_df) < 100:
            train_df = raw_df  # fallback

        X_tr = train_df[all_feature_cols]
        y_tr = train_df[TARGET_COLS]

        # 查找测试特征行
        test_row = test_feat[test_feat[ID_COL] == sid]
        if len(test_row) == 0:
            test_row = test_feat[test_feat[ID_COL].str.contains(sid.replace("_eq", ""), na=False)]
        if len(test_row) == 0:
            continue
        X_te = test_row[all_feature_cols]

        # 训练 + 预测
        lgb = build_lgbm(params, n_estimators_lgb, seed, late_weight)
        lgb.fit(X_tr, y_tr)
        xgb = build_xgboost_model(params, n_estimators_xgb, seed)
        xgb.fit(X_tr, y_tr)

        p_lgb = np.clip(np.asarray(lgb.predict(X_te), dtype=float), 0, None)[0]
        p_xgb = np.clip(np.asarray(xgb.predict(X_te), dtype=float), 0, None)[0]

        mag_w = oof_metrics_cache.get("mag_baseline_weight", 0.5)
        time_w = oof_metrics_cache.get("time_baseline_weight", 0.5)
        pred_mag = mag_w * p_lgb[0] + (1 - mag_w) * p_xgb[0]
        pred_time = time_w * p_lgb[1] + (1 - time_w) * p_xgb[1]

        predictions.append({
            "mainshock_id": sid,
            "pred_max_mag": float(pred_mag),
            "pred_time_to_max": float(pred_time),
            "true_max_mag": float(row["true_max_mag"]),
            "true_time_to_max_days": float(row["true_time_to_max_days"]),
        })

    pred_df = pd.DataFrame(predictions)
    if len(pred_df) == 0:
        return {"holdout_combined": float("inf"), "holdout_n": 0}

    metrics = calculate_metrics(
        y_true_mag=pred_df["true_max_mag"].to_numpy(),
        y_pred_mag=pred_df["pred_max_mag"].to_numpy(),
        y_true_time=pred_df["true_time_to_max_days"].to_numpy(),
        y_pred_time=pred_df["pred_time_to_max"].to_numpy(),
        late_weight=late_weight,
    )
    combined = float(metrics.get("mag_rmse", 99) + metrics.get("time_asymmetric_rmse", 99))
    return {
        "holdout_mag_rmse": float(metrics.get("mag_rmse", np.nan)),
        "holdout_mag_mae": float(metrics.get("mag_mae", np.nan)),
        "holdout_time_rmse": float(metrics.get("time_rmse", np.nan)),
        "holdout_time_mae": float(metrics.get("time_mae", np.nan)),
        "holdout_time_asymmetric_rmse": float(metrics.get("time_asymmetric_rmse", np.nan)),
        "holdout_time_asymmetric_mae": float(metrics.get("time_asymmetric_mae", np.nan)),
        "holdout_combined": combined,
        "holdout_n": len(pred_df),
        "holdout_predictions": pred_df,
    }


# 全局缓存：OOF 搜索的最优融合权重
oof_metrics_cache: dict = {}


# ============================================================
#  Optuna 调优
# ============================================================

def _ensure_optuna():
    """确保 optuna 已安装，返回 optuna 模块。"""
    try:
        import optuna
        return optuna
    except ImportError:
        print("安装 optuna …")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "-q"])
        import optuna
        return optuna


def create_optuna_study(storage_path: str | None, study_name: str) -> "optuna.Study":
    """创建或加载 Optuna study。"""
    import optuna

    if storage_path:
        storage = f"sqlite:///{storage_path}"
    else:
        storage = None

    # 多目标: 最小化 combined = mag_rmse + time_asymmetric_rmse
    sampler = optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=min(20, 10),  # 先用随机探索
        multivariate=True,              # 考虑参数间交互
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    return study


def suggest_params(trial: "optuna.Trial") -> dict:
    """为一次 trial 采样超参数。"""
    params: dict[str, Any] = {}

    # ---- LightGBM ----
    params["lgb_learning_rate"] = trial.suggest_float("lgb_learning_rate", 0.01, 0.1, log=True)
    params["lgb_num_leaves"] = trial.suggest_int("lgb_num_leaves", 15, 255, log=True)
    params["lgb_max_depth"] = trial.suggest_int("lgb_max_depth", 3, 15)
    params["lgb_min_child_samples"] = trial.suggest_int("lgb_min_child_samples", 5, 100, log=True)
    params["lgb_subsample"] = trial.suggest_float("lgb_subsample", 0.5, 0.95)
    params["lgb_colsample_bytree"] = trial.suggest_float("lgb_colsample_bytree", 0.4, 0.95)
    params["lgb_reg_alpha"] = trial.suggest_float("lgb_reg_alpha", 1e-4, 1.0, log=True)
    params["lgb_reg_lambda"] = trial.suggest_float("lgb_reg_lambda", 0.1, 10.0, log=True)
    params["use_asymmetric_time_objective"] = True

    # ---- XGBoost ----
    params["xgb_learning_rate"] = trial.suggest_float("xgb_learning_rate", 0.01, 0.1, log=True)
    params["xgb_max_depth"] = trial.suggest_int("xgb_max_depth", 3, 12)
    params["xgb_min_child_weight"] = trial.suggest_int("xgb_min_child_weight", 1, 50)
    params["xgb_subsample"] = trial.suggest_float("xgb_subsample", 0.5, 0.95)
    params["xgb_colsample_bytree"] = trial.suggest_float("xgb_colsample_bytree", 0.4, 0.95)
    params["xgb_reg_alpha"] = trial.suggest_float("xgb_reg_alpha", 1e-4, 10.0, log=True)
    params["xgb_reg_lambda"] = trial.suggest_float("xgb_reg_lambda", 0.1, 10.0, log=True)

    # ---- 特征工程 ----
    params["feature_selection"] = trial.suggest_categorical("feature_selection", [True, False])
    params["feature_selection_ratio"] = trial.suggest_float("feature_selection_ratio", 0.6, 0.95)
    params["feature_selection_min"] = trial.suggest_int("feature_selection_min", 20, 60)

    # ---- OOF 参数 ----
    params["purge_days"] = trial.suggest_float("purge_days", 7.0, 90.0)
    params["min_purge_days"] = trial.suggest_float("min_purge_days", 3.0, 21.0)

    # ---- 融合 ----
    params["ensemble_grid_step"] = trial.suggest_categorical("ensemble_grid_step", [0.01, 0.02, 0.05])

    return params


def run_tuning(
    n_trials: int,
    n_splits: int,
    n_estimators_lgb: int,
    n_estimators_xgb: int,
    late_weight: float,
    seed: int,
    study_name: str,
    storage_path: str | None,
    holdout_labels: pd.DataFrame,
    eval_holdout_every: int = 10,
) -> "optuna.Study":
    """主调优循环。"""
    optuna = _ensure_optuna()
    study = create_optuna_study(storage_path, study_name)

    # 加载训练数据 (一次性)
    print("\n加载训练数据 …")
    raw_df = pd.read_csv(FEATURES_CSV)
    raw_df[TIME_COL] = pd.to_datetime(raw_df[TIME_COL], utc=True, format="mixed")

    # 只用有目标余震的样本做回归
    if "has_target_aftershock" in raw_df.columns:
        df_full = raw_df[raw_df["has_target_aftershock"].astype(bool)].copy()
    else:
        df_full = raw_df.dropna(subset=TARGET_COLS).copy()
    df_full = df_full.dropna(subset=[TIME_COL, *TARGET_COLS]).reset_index(drop=True)
    df_full = add_derived_features(df_full)
    all_feature_cols_raw = select_feature_columns(df_full)
    print(f"  训练样本: {len(df_full)}, 原始特征: {len(all_feature_cols_raw)}")

    # 全局 best
    global_best_score = float("inf")
    global_best_params: dict = {}
    global_best_holdout: dict = {}

    def objective(trial: optuna.Trial) -> float:
        nonlocal global_best_score, global_best_params, global_best_holdout
        params = suggest_params(trial)

        # 特征工程
        df = df_full.copy()
        df, missing_cols = impute_missing_features(df, all_feature_cols_raw)
        feature_cols = all_feature_cols_raw + missing_cols

        if params.get("feature_selection", True) and len(feature_cols) > 50:
            feature_cols = select_features_by_importance(
                df[feature_cols], df[TARGET_COLS], feature_cols,
                top_k_ratio=params["feature_selection_ratio"],
                min_features=params["feature_selection_min"],
            )

        # OOF CV
        try:
            oof_df, metrics = run_single_oof_cv(
                df=df,
                feature_cols=feature_cols,
                params=params,
                n_splits=n_splits,
                purge_days=params["purge_days"],
                min_purge_days=params["min_purge_days"],
                seed=seed,
                late_weight=late_weight,
                n_estimators_lgb=n_estimators_lgb,
                n_estimators_xgb=n_estimators_xgb,
            )
        except Exception as e:
            print(f"  ⚠ Trial {trial.number} 失败: {e}")
            return float("inf")

        score = metrics["combined_objective"]

        # 记录属性
        trial.set_user_attr("mag_rmse", float(metrics["mag_rmse"]))
        trial.set_user_attr("time_asymmetric_rmse", float(metrics["time_asymmetric_rmse"]))
        trial.set_user_attr("time_rmse", float(metrics["time_rmse"]))
        trial.set_user_attr("n_features", len(feature_cols))

        # 每 N 轮在 holdout 上评估
        if trial.number % eval_holdout_every == 0 or score < global_best_score:
            global oof_metrics_cache
            oof_metrics_cache = {
                "mag_baseline_weight": metrics["mag_baseline_weight"],
                "time_baseline_weight": metrics["time_baseline_weight"],
            }
            holdout = evaluate_on_holdout(
                oof_df, holdout_labels, params, late_weight,
                n_estimators_lgb, n_estimators_xgb, seed,
            )
            trial.set_user_attr("holdout_combined", holdout.get("holdout_combined", float("inf")))
            trial.set_user_attr("holdout_mag_rmse", holdout.get("holdout_mag_rmse", float("inf")))
            trial.set_user_attr("holdout_time_asym_rmse", holdout.get("holdout_time_asymmetric_rmse", float("inf")))

            if score < global_best_score:
                global_best_score = score
                global_best_params = params.copy()
                global_best_holdout = holdout

                print(f"\n{'='*60}")
                print(f"★ Trial {trial.number}: OOF={score:.4f}  "
                      f"Holdout={holdout.get('holdout_combined', float('inf')):.4f}")
                print(f"  LGB lr={params['lgb_learning_rate']:.4f} "
                      f"leaves={params['lgb_num_leaves']} "
                      f"depth={params['lgb_max_depth']}")
                print(f"  XGB lr={params['xgb_learning_rate']:.4f} "
                      f"depth={params['xgb_max_depth']}")
                print(f"  FS={params['feature_selection']} "
                      f"ratio={params['feature_selection_ratio']:.2f} "
                      f"min={params['feature_selection_min']}")
                print(f"  特征数: {len(feature_cols)}")
                print(f"  OOF: mag_rmse={metrics['mag_rmse']:.4f} "
                      f"time_asym_rmse={metrics['time_asymmetric_rmse']:.4f}")
                if holdout.get("holdout_n", 0) > 0:
                    print(f"  Holdout: mag_rmse={holdout.get('holdout_mag_rmse', np.nan):.4f} "
                          f"time_asym_rmse={holdout.get('holdout_time_asymmetric_rmse', np.nan):.4f} "
                          f"n={holdout['holdout_n']}")
        return score

    print(f"\n开始调优: {n_trials} trials (已完成 {len(study.trials)} 个)")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=1)

    return study


# ============================================================
#  结果保存 & 报告
# ============================================================

def save_results(
    study: "optuna.Study",
    holdout_labels: pd.DataFrame,
    best_params: dict,
    best_holdout: dict,
    output_dir: Path,
    late_weight: float,
) -> None:
    """保存调优结果。"""
    optuna = _ensure_optuna()

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Best params JSON
    with (output_dir / "best_params.json").open("w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    # 2. Study statistics
    stats = {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "best_trial_number": study.best_trial.number,
        "best_oof_score": float(study.best_value),
        "best_params": best_params,
        "best_holdout": {
            k: float(v) if isinstance(v, (int, float, np.floating))
            else (v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else str(v))
            for k, v in best_holdout.items()
            if k != "holdout_predictions"
        },
        "late_weight": late_weight,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with (output_dir / "tuning_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)

    # 3. Holdout 预测
    if "holdout_predictions" in best_holdout:
        pred_df = best_holdout["holdout_predictions"]
        pred_df.to_csv(output_dir / "holdout_predictions.csv", index=False, encoding="utf-8")

    # 4. Trials CSV
    trials_data = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {
            "number": t.number,
            "value": t.value,
            **t.params,
            **{f"attr_{k}": v for k, v in (t.user_attrs or {}).items()},
        }
        trials_data.append(row)
    if trials_data:
        pd.DataFrame(trials_data).to_csv(output_dir / "trials_history.csv", index=False, encoding="utf-8")

    # 5. 打印最终报告
    print(f"\n{'='*60}")
    print("调优完成！")
    print(f"{'='*60}")
    print(f"Study: {study.study_name}")
    print(f"Trials: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best OOF score: {study.best_value:.4f}")
    print(f"\n最优参数:")
    for k, v in sorted(best_params.items()):
        print(f"  {k}: {v}")

    if best_holdout.get("holdout_n", 0) > 0:
        print(f"\nHoldout 评估 ({best_holdout['holdout_n']} 条):")
        for k in ["holdout_mag_rmse", "holdout_mag_mae", "holdout_time_rmse",
                   "holdout_time_mae", "holdout_time_asymmetric_rmse",
                   "holdout_time_asymmetric_mae", "holdout_combined"]:
            v = best_holdout.get(k)
            if v is not None:
                print(f"  {k}: {v:.4f}")

    # 6. 保存兼容格式的模型参数 (供 make_submission.py 使用)
    model_params_path = output_dir / "tuned_model_params.json"
    with model_params_path.open("w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    print(f"\n产物已保存: {output_dir}")
    print(f"  最佳参数: {output_dir / 'best_params.json'}")
    print(f"  Holdout 预测: {output_dir / 'holdout_predictions.csv'}")
    print(f"  Trial 历史: {output_dir / 'trials_history.csv'}")


# ============================================================
#  CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="余震预测模型全面超参数调优 (Optuna)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 100 轮基础调优
  python scripts/tune_hyperparams.py --n-trials 100

  # 200 轮深度调优 (含 DL + GNN)
  python scripts/tune_hyperparams.py --n-trials 200 --with-dl --with-gnn

  # 从已有 study 继续
  python scripts/tune_hyperparams.py --n-trials 50 --study-name my_study

  # 快速测试 (20 轮)
  python scripts/tune_hyperparams.py --n-trials 20 --n-splits 3 --n-estimators 100
        """,
    )
    p.add_argument("--n-trials", type=int, default=100, help="Optuna 试验次数")
    p.add_argument("--n-splits", type=int, default=5, help="OOF CV 折数")
    p.add_argument("--n-estimators", type=int, default=300,
                   help="树模型迭代轮数 (调优中固定，正式训练可加倍)")
    p.add_argument("--late-weight", type=float, default=2.0, help="预测偏晚惩罚")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--study-name", type=str, default="aftershock_tuning",
                   help="Optuna study 名称 (用于断点续调)")
    p.add_argument("--storage", type=str, default=None,
                   help="Optuna SQLite 存储路径 (默认内存)")
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "tuning_results",
                   help="调优结果输出目录")
    p.add_argument("--eval-holdout-every", type=int, default=10,
                   help="每 N 轮在 holdout 上评估一次")
    p.add_argument("--with-dl", action="store_true", help="同时调优 Transformer (TODO)")
    p.add_argument("--with-gnn", action="store_true", help="同时调优 ST-GNN (TODO)")
    p.add_argument("--no-holdout", action="store_true",
                   help="跳过 holdout 评估，仅优化 OOF CV")
    p.add_argument("--skip-precompute", action="store_true",
                   help="跳过测试序列特征预计算 (使用已有缓存)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    t0 = time.time()

    # 1. 预计算测试序列特征
    test_feat_path = PROJECT_ROOT / "data" / "processed" / "test_sequences_features.csv"
    if not args.skip_precompute and not args.no_holdout:
        precompute_test_features(
            TEST_DIR,
            PROJECT_ROOT / "data" / "raw" / "PB2002_boundaries.json",
            PROJECT_ROOT / "data" / "raw" / "GlobalCMT_1976-2024.csv",
            test_feat_path,
        )

    # 2. 提取 holdout 真实标签
    holdout_labels = extract_true_labels(FULL_CATALOG, TEST_DIR)

    # 3. 调优
    if args.storage:
        storage_path = str(PROJECT_ROOT / args.storage)
    else:
        storage_path = None

    study = run_tuning(
        n_trials=args.n_trials,
        n_splits=args.n_splits,
        n_estimators_lgb=args.n_estimators,
        n_estimators_xgb=args.n_estimators,
        late_weight=args.late_weight,
        seed=args.seed,
        study_name=args.study_name,
        storage_path=storage_path,
        holdout_labels=holdout_labels,
        eval_holdout_every=args.eval_holdout_every if not args.no_holdout else args.n_trials + 1,
    )

    # 3. 提取最优参数和 holdout 结果
    best_params = study.best_params
    best_trial = study.best_trial
    best_holdout: dict = {}
    if not args.no_holdout:
        holdout_combined = best_trial.user_attrs.get("holdout_combined")
        best_holdout = {
            "holdout_mag_rmse": best_trial.user_attrs.get("holdout_mag_rmse"),
            "holdout_time_asymmetric_rmse": best_trial.user_attrs.get("holdout_time_asym_rmse"),
            "holdout_combined": holdout_combined,
            "holdout_n": 20,
        }
        # 用最优参数做最终 holdout 评估
        print("\n用最优参数做最终 holdout 评估 …")
        global oof_metrics_cache
        oof_metrics_cache = {"mag_baseline_weight": 0.5, "time_baseline_weight": 0.5}
        final_holdout = evaluate_on_holdout(
            pd.DataFrame(), holdout_labels, best_params, args.late_weight,
            args.n_estimators, args.n_estimators, args.seed,
        )
        best_holdout = final_holdout

    # 4. 保存
    save_results(
        study=study,
        holdout_labels=holdout_labels,
        best_params=best_params,
        best_holdout=best_holdout,
        output_dir=args.output_dir,
        late_weight=args.late_weight,
    )

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed/60:.1f} 分钟")


if __name__ == "__main__":
    main()
