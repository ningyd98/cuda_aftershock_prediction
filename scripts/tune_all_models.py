#!/usr/bin/env python
# ============================================================
#  余震预测 —— 全模型联合超参数调优
#
#  策略:
#   每个 Optuna trial 同时采样:
#   - LightGBM 超参数 (8 维)
#   - XGBoost 超参数 (7 维)
#   - Transformer/DL 超参数 (7 维)
#   - ST-GNN 超参数 (5 维)
#   - 特征工程参数 (3 维)
#   - OOF 参数 (2 维)
#
#  每个 trial:
#   1. 领域知识缺失值填补 + missing indicator
#   2. 可选特征选择
#   3. 树模型 OOF CV (5-fold)
#   4. DL Transformer OOF CV (3-fold, 减少调优耗时)
#   5. GNN ST-GNN OOF CV (3-fold)
#   6. 4 模型 simplex 网格搜索融合权重
#   7. 计算 combined objective = mag_rmse + time_asymmetric_rmse
#   8. 每 10 轮在 20 条 holdout 测试序列上评估
#
#  用法:
#   python scripts/tune_all_models.py --n-trials 100
#   python scripts/tune_all_models.py --n-trials 200 --storage tuning.db
#   python scripts/tune_all_models.py --n-trials 50 --fast  # 快速模式
# ============================================================

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

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
EVENT_CATALOG = PROJECT_ROOT / "data" / "raw" / "USGS_Mw4.0_Depth70_1970-2023.csv"
PLATE_PATH = PROJECT_ROOT / "data" / "raw" / "PB2002_boundaries.json"
GCMT_PATH = PROJECT_ROOT / "data" / "raw" / "GlobalCMT_1976-2024.csv"

MODEL_PRED_COLS = {
    "baseline": ("baseline_pred_mag", "baseline_pred_time"),
    "xgboost": ("xgboost_pred_mag", "xgboost_pred_time"),
    "dl": ("dl_pred_mag", "dl_pred_time"),
    "gnn": ("gnn_pred_mag", "gnn_pred_time"),
}

# ---- 领域默认值 ----
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

# ============================================================
#  工具函数
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def extract_true_labels(full_catalog_path: Path, test_dir: Path) -> pd.DataFrame:
    """从 USGS 完整目录提取 20 条测试序列的真实标签。"""
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
        main_time = pd.to_datetime(str(seq.iloc[0]["Date"]) + " " + str(seq.iloc[0]["Time"]), utc=True)
        main_mag = float(seq.iloc[0]["Mag"])
        time_end = main_time + pd.Timedelta(days=30)
        mask = (full["time"] > main_time) & (full["time"] <= time_end)
        candidates = full[mask].copy()
        if len(candidates) == 0:
            true_mag, true_time = 0.0, 0.0
        else:
            dists = haversine_km(main_lat, main_lon,
                                 candidates["latitude"].values.astype(float),
                                 candidates["longitude"].values.astype(float))
            nearby = candidates[dists <= 100.0]
            if len(nearby) == 0:
                true_mag, true_time = 0.0, 0.0
            else:
                idx_max = nearby["mag"].idxmax()
                true_mag = float(nearby.loc[idx_max, "mag"])
                true_time = (nearby.loc[idx_max, "time"] - main_time).total_seconds() / 86400.0
        records.append({"mainshock_id": sid, "mainshock_mag": main_mag,
                        "true_max_mag": true_mag, "true_time_to_max_days": true_time})
    result = pd.DataFrame(records)
    n_with = (result["true_max_mag"] > 0).sum()
    print(f"  {len(result)} 条测试序列, {n_with} 条有余震")
    return result


# ============================================================
#  特征工程 (与 train_baseline.py 一致)
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
    candidates = []
    for col in df.columns:
        if col in EXCLUDE_COLS: continue
        if col in EXPLICIT_FEATURES or col.startswith(FEATURE_PREFIXES):
            candidates.append(col)
    numeric = []
    for col in candidates:
        if pd.api.types.is_bool_dtype(df[col]): df[col] = df[col].astype(int)
        if pd.api.types.is_numeric_dtype(df[col]): numeric.append(col)
    return numeric


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
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


def impute_missing_features(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """领域知识缺失值填补 + missing indicator。"""
    df = df.copy()
    new_cols = []
    for col in feature_cols:
        if col not in df.columns: continue
        null_mask = df[col].isnull()
        if not null_mask.any(): continue
        indicator_col = f"{col}_missing"
        df[indicator_col] = null_mask.astype(int)
        new_cols.append(indicator_col)
        if col in _DOMAIN_DEFAULTS:
            df[col] = df[col].fillna(_DOMAIN_DEFAULTS[col])
        else:
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0.0)
    return df, new_cols


def select_features_by_importance(X, y, feature_cols, top_k_ratio=0.85, min_features=30):
    if len(feature_cols) <= min_features: return feature_cols
    from lightgbm import LGBMRegressor
    model = LGBMRegressor(n_estimators=100, num_leaves=31, learning_rate=0.05,
                          random_state=42, n_jobs=-1, verbosity=-1)
    model.fit(X, y.iloc[:, 0])
    importance = model.booster_.feature_importance(importance_type="gain")
    sorted_idx = np.argsort(importance)[::-1]
    total_gain = importance.sum()
    if total_gain <= 0: return feature_cols
    cumsum = np.cumsum(importance[sorted_idx]) / total_gain
    n_keep = max(min_features, int(np.searchsorted(cumsum, top_k_ratio) + 1))
    n_keep = min(n_keep, len(feature_cols))
    return [feature_cols[i] for i in sorted_idx[:n_keep]]


# ============================================================
#  模型构建
# ============================================================

def build_lgbm(params, n_estimators, seed, late_weight):
    device = get_lightgbm_device("cuda")
    return BaselineLGBM(
        random_state=seed, n_estimators=n_estimators,
        learning_rate=params.get("lgb_lr", 0.03),
        num_leaves=params.get("lgb_num_leaves", 63),
        max_depth=params.get("lgb_max_depth", -1),
        min_child_samples=params.get("lgb_min_child_samples", 20),
        subsample=params.get("lgb_subsample", 0.8),
        colsample_bytree=params.get("lgb_colsample_bytree", 0.7),
        reg_alpha=params.get("lgb_reg_alpha", 0.05),
        reg_lambda=params.get("lgb_reg_lambda", 1.0),
        use_asymmetric_time_objective=True,
        late_weight=late_weight, transform_time_target=True, device=device,
    )


def build_xgb(params, n_estimators, seed):
    return BaselineXGBoost(
        random_state=seed, n_estimators=n_estimators,
        learning_rate=params.get("xgb_lr", 0.03),
        max_depth=params.get("xgb_max_depth", 6),
        min_child_weight=params.get("xgb_min_child_weight", 1),
        subsample=params.get("xgb_subsample", 0.8),
        colsample_bytree=params.get("xgb_colsample_bytree", 0.7),
        reg_alpha=params.get("xgb_reg_alpha", 0.0),
        reg_lambda=params.get("xgb_reg_lambda", 1.0),
        transform_time_target=True,
    )


def build_dl_model(event_feat_dim, global_feat_dim, params, device):
    """构建 Transformer 模型。"""
    from src.models_dl import Seq2SeqAftershockPredictor
    model = Seq2SeqAftershockPredictor(
        event_feature_dim=event_feat_dim,
        global_feature_dim=global_feat_dim,
        d_model=params.get("dl_d_model", 128),
        nhead=params.get("dl_nhead", 4),
        num_layers=params.get("dl_num_layers", 3),
        dim_feedforward=params.get("dl_dim_ff", 256),
        dropout=params.get("dl_dropout", 0.1),
        global_hidden_dim=params.get("dl_global_hidden", 128),
        fusion_hidden_dim=params.get("dl_fusion_hidden", 128),
        output_dim=2, max_seq_len=256,
    ).to(device)
    return model


def build_gnn_model(event_feat_dim, global_feat_dim, params, device):
    """构建 ST-GNN 模型。"""
    from src.models_gnn import STGNNPredictor
    model = STGNNPredictor(
        event_feature_dim=event_feat_dim,
        global_feature_dim=global_feat_dim,
        node_hidden_dim=params.get("gnn_node_hidden", 64),
        num_gnn_layers=params.get("gnn_layers", 3),
        gnn_sigma=params.get("gnn_sigma", 50.0),
        gnn_radius_km=params.get("gnn_radius_km", 100.0),
        gru_hidden_dim=params.get("gnn_gru_hidden", 64),
        gru_layers=params.get("gnn_gru_layers", 2),
        global_hidden_dim=params.get("gnn_global_hidden", 128),
        fusion_hidden_dim=params.get("gnn_fusion_hidden", 128),
        output_dim=2,
        dropout=params.get("gnn_dropout", 0.1),
    ).to(device)
    return model


# ============================================================
#  OOF CV
# ============================================================

def run_tree_oof_cv(df_train, feature_cols, params, n_splits, purge_days,
                    min_purge_days, seed, late_weight, n_est_lgb, n_est_xgb):
    """树模型 (LightGBM + XGBoost) OOF CV。"""
    from sklearn.model_selection import TimeSeriesSplit

    df = df_train.sort_values(TIME_COL).reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    X_all = df[feature_cols]
    y_all = df[TARGET_COLS]
    purge_delta = pd.Timedelta(days=purge_days)
    min_purge = pd.Timedelta(days=min_purge_days)

    oof_b, oof_x = (np.full((len(df), 2), np.nan) for _ in range(2))

    fold_iter = tqdm(
        enumerate(splitter.split(X_all), start=1),
        total=n_splits,
        desc="🌲 Tree OOF CV",
        unit="fold",
        leave=False,
    )
    for fold_idx, (tr_idx, v_idx) in fold_iter:
        v_start = df.loc[v_idx[0], TIME_COL]
        # purge
        cutoff = v_start - purge_delta
        pm = df.loc[tr_idx, TIME_COL] <= cutoff
        tr_p = tr_idx[pm.values]
        if len(tr_p) < max(10, len(tr_idx) * 0.2):
            mc = v_start - min_purge
            mm = df.loc[tr_idx, TIME_COL] <= mc
            tr_p = tr_idx[mm.values]
            if len(tr_p) < max(5, len(tr_idx) * 0.1): tr_p = tr_idx

        fold_iter.set_postfix(train=len(tr_p), valid=len(v_idx))

        lgb = build_lgbm(params, n_est_lgb, seed, late_weight)
        lgb.fit(X_all.iloc[tr_p], y_all.iloc[tr_p])
        oof_b[v_idx] = np.clip(np.asarray(lgb.predict(X_all.iloc[v_idx]), float), 0, None)

        xgb = build_xgb(params, n_est_xgb, seed)
        xgb.fit(X_all.iloc[tr_p], y_all.iloc[tr_p])
        oof_x[v_idx] = np.clip(np.asarray(xgb.predict(X_all.iloc[v_idx]), float), 0, None)

    oof_df = df[[ID_COL, TIME_COL, *TARGET_COLS]].copy()
    oof_df["baseline_pred_mag"] = oof_b[:, 0]
    oof_df["baseline_pred_time"] = oof_b[:, 1]
    oof_df["xgboost_pred_mag"] = oof_x[:, 0]
    oof_df["xgboost_pred_time"] = oof_x[:, 1]
    return oof_df


def run_dl_oof_cv(df_train, event_df_path, feature_cols, params, n_splits, purge_days,
                  seed, late_weight, device_str):
    """Transformer OOF CV。返回 OOF DataFrame。"""
    import torch
    from src.dataset import (EarthquakeSequenceDataset, SequenceBuildConfig,
                             earthquake_collate_fn, fit_dataset_preprocessors)
    from src.models_dl import asymmetric_time_mse_loss
    from src.utils import setup_cuda
    from sklearn.model_selection import TimeSeriesSplit

    device = setup_cuda(
        device_str,
        deterministic=False,
        allow_tf32=True,
        matmul_precision="medium",
    )
    df = df_train.sort_values(TIME_COL).reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    purge_delta = pd.Timedelta(days=purge_days)

    event_df = pd.read_csv(event_df_path)
    event_df["time"] = pd.to_datetime(event_df["time"], utc=True, errors="coerce")

    seq_config = SequenceBuildConfig(obs_days=3.0, spatial_radius_km=100.0, max_seq_len=256)
    oof_preds = np.full((len(df), 2), np.nan, dtype=float)

    epochs = params.get("dl_epochs", 30)
    batch_size = params.get("dl_batch_size", 32)
    lr = params.get("dl_lr", 1e-3)
    use_amp = True  # Use bfloat16 to avoid NaN risk while getting speedup
    amp_dtype = torch.bfloat16

    dl_fold_iter = tqdm(
        enumerate(splitter.split(df), start=1),
        total=n_splits,
        desc="🧠 DL OOF CV",
        unit="fold",
        leave=False,
    )
    for fold_idx, (tr_idx, v_idx) in dl_fold_iter:
        v_start = df.loc[v_idx[0], TIME_COL]
        cutoff = v_start - purge_delta
        pm = df.loc[tr_idx, TIME_COL] <= cutoff
        tr_p = tr_idx[pm.values]
        if len(tr_p) < max(10, len(tr_idx) * 0.2): tr_p = tr_idx

        dl_fold_iter.set_postfix(train=len(tr_p), valid=len(v_idx))

        fold_tr = df.iloc[tr_p].copy()
        fold_v = df.iloc[v_idx].copy()

        preps = fit_dataset_preprocessors(
            sequence_df=fold_tr, event_catalog_df=event_df,
            global_feature_cols=feature_cols, target_cols=TARGET_COLS,
            config=seq_config, scaler_type="robust", add_missing_indicators=True,
        )
        tr_ds = EarthquakeSequenceDataset(
            sequence_df=fold_tr, event_catalog_df=event_df,
            global_feature_cols=feature_cols, target_cols=TARGET_COLS,
            config=seq_config, preprocessors=preps, fit_preprocessors=False,
        )
        v_ds = EarthquakeSequenceDataset(
            sequence_df=fold_v, event_catalog_df=event_df,
            global_feature_cols=feature_cols, target_cols=TARGET_COLS,
            config=seq_config, preprocessors=preps, fit_preprocessors=False,
        )
        num_workers = min(8, len(tr_ds) // batch_size + 1)
        pin_memory = device.type == "cuda"
        tr_ldr = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                                             collate_fn=earthquake_collate_fn,
                                             num_workers=num_workers,
                                             pin_memory=pin_memory)
        v_ldr = torch.utils.data.DataLoader(v_ds, batch_size=batch_size, shuffle=False,
                                            collate_fn=earthquake_collate_fn,
                                            num_workers=num_workers,
                                            pin_memory=pin_memory)

        model = build_dl_model(len(tr_ds.event_feature_cols), tr_ds.global_feature_dim, params, device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best_loss, best_state = float("inf"), None
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        ep_iter = tqdm(range(epochs), desc=f"  DL Fold {fold_idx} epochs", unit="ep", leave=False)
        for ep in ep_iter:
            model.train()
            for batch in tr_ldr:
                sx = batch["seq_x"].to(device)
                gx = batch["global_x"].to(device)
                yb = batch["y"].to(device)
                mk = batch["seq_padding_mask"].to(device)
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    loss = asymmetric_time_mse_loss(model(sx, gx, mk), yb, late_weight=late_weight)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            # val
            model.eval()
            v_loss, v_cnt = 0.0, 0
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                for batch in v_ldr:
                    sx = batch["seq_x"].to(device)
                    gx = batch["global_x"].to(device)
                    yb = batch["y"].to(device)
                    mk = batch["seq_padding_mask"].to(device)
                    pp = model(sx, gx, mk)
                    v_loss += asymmetric_time_mse_loss(pp, yb, late_weight=late_weight).item() * len(yb)
                    v_cnt += len(yb)
            v_loss /= max(v_cnt, 1)
            ep_iter.set_postfix(val_loss=f"{v_loss:.4f}", best=f"{best_loss:.4f}")
            if v_loss < best_loss:
                best_loss = v_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            sched.step()

        if best_state is None: raise RuntimeError("DL fold best_state is None")
        model.load_state_dict(best_state)
        model.eval()

        all_p, all_t = [], []
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            pred_iter = tqdm(v_ldr, desc=f"  DL Fold {fold_idx} 预测", unit="batch", leave=False)
            for batch in pred_iter:
                sx = batch["seq_x"].to(device)
                gx = batch["global_x"].to(device)
                yb = batch["y"].to(device)
                mk = batch["seq_padding_mask"].to(device)
                pp = model(sx, gx, mk)
                all_p.append(pp.cpu().numpy())
                all_t.append(yb.cpu().numpy())
        preds = np.clip(np.concatenate(all_p, axis=0), 0, None)
        # time from log1p back to days
        preds[:, 1] = np.expm1(np.clip(preds[:, 1], 0.0, 50.0))
        oof_preds[v_idx] = preds

    oof_df = df[[ID_COL, TIME_COL, *TARGET_COLS]].copy()
    oof_df["dl_pred_mag"] = oof_preds[:, 0]
    oof_df["dl_pred_time"] = oof_preds[:, 1]
    return oof_df


def run_gnn_oof_cv(df_train, event_df_path, feature_cols, params, n_splits, purge_days,
                   seed, late_weight, device_str):
    """ST-GNN OOF CV。返回 OOF DataFrame。"""
    import torch
    from src.dataset import (EarthquakeSequenceDataset, SequenceBuildConfig,
                             earthquake_collate_fn, fit_dataset_preprocessors)
    from src.models_gnn import stgnn_asymmetric_loss
    from src.utils import setup_cuda
    from sklearn.model_selection import TimeSeriesSplit

    device = setup_cuda(
        device_str,
        deterministic=False,
        allow_tf32=True,
        matmul_precision="medium",
    )
    df = df_train.sort_values(TIME_COL).reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    purge_delta = pd.Timedelta(days=purge_days)

    event_df = pd.read_csv(event_df_path)
    event_df["time"] = pd.to_datetime(event_df["time"], utc=True, errors="coerce")

    seq_config = SequenceBuildConfig(obs_days=3.0, spatial_radius_km=100.0, max_seq_len=256)
    oof_preds = np.full((len(df), 2), np.nan, dtype=float)

    epochs = params.get("gnn_epochs", 30)
    batch_size = params.get("gnn_batch_size", 16)
    lr = params.get("gnn_lr", 1e-3)
    use_amp = True  # Use bfloat16 to avoid NaN risk
    amp_dtype = torch.bfloat16

    gnn_fold_iter = tqdm(
        enumerate(splitter.split(df), start=1),
        total=n_splits,
        desc="🔗 GNN OOF CV",
        unit="fold",
        leave=False,
    )
    for fold_idx, (tr_idx, v_idx) in gnn_fold_iter:
        v_start = df.loc[v_idx[0], TIME_COL]
        cutoff = v_start - purge_delta
        pm = df.loc[tr_idx, TIME_COL] <= cutoff
        tr_p = tr_idx[pm.values]
        if len(tr_p) < max(10, len(tr_idx) * 0.2): tr_p = tr_idx

        gnn_fold_iter.set_postfix(train=len(tr_p), valid=len(v_idx))

        fold_tr = df.iloc[tr_p].copy()
        fold_v = df.iloc[v_idx].copy()

        preps = fit_dataset_preprocessors(
            sequence_df=fold_tr, event_catalog_df=event_df,
            global_feature_cols=feature_cols, target_cols=TARGET_COLS,
            config=seq_config, scaler_type="robust", add_missing_indicators=True,
        )
        tr_ds = EarthquakeSequenceDataset(
            sequence_df=fold_tr, event_catalog_df=event_df,
            global_feature_cols=feature_cols, target_cols=TARGET_COLS,
            config=seq_config, preprocessors=preps, fit_preprocessors=False,
        )
        v_ds = EarthquakeSequenceDataset(
            sequence_df=fold_v, event_catalog_df=event_df,
            global_feature_cols=feature_cols, target_cols=TARGET_COLS,
            config=seq_config, preprocessors=preps, fit_preprocessors=False,
        )
        num_workers = min(8, len(tr_ds) // batch_size + 1)
        pin_memory = device.type == "cuda"
        tr_ldr = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                                             collate_fn=earthquake_collate_fn,
                                             num_workers=num_workers,
                                             pin_memory=pin_memory)
        v_ldr = torch.utils.data.DataLoader(v_ds, batch_size=batch_size, shuffle=False,
                                            collate_fn=earthquake_collate_fn,
                                            num_workers=num_workers,
                                            pin_memory=pin_memory)

        model = build_gnn_model(len(tr_ds.event_feature_cols), tr_ds.global_feature_dim, params, device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best_loss, best_state = float("inf"), None
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        gn_ep_iter = tqdm(range(epochs), desc=f"  GNN Fold {fold_idx} epochs", unit="ep", leave=False)
        for ep in gn_ep_iter:
            model.train()
            for batch in tr_ldr:
                sx = batch["seq_x"].to(device)
                gx = batch["global_x"].to(device)
                coords = batch["graph_coords_km"].to(device)
                gtd = batch["graph_time_days"].to(device)
                yb = batch["y"].to(device)
                mk = batch["seq_padding_mask"].to(device)
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    loss = stgnn_asymmetric_loss(model(sx, gx, mk, graph_coords_km=coords, graph_time_days=gtd, graph_strike_rad=batch.get("mainshock_strike_rad")), yb, late_weight=late_weight)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            model.eval()
            v_loss, v_cnt = 0.0, 0
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                for batch in v_ldr:
                    sx = batch["seq_x"].to(device)
                    gx = batch["global_x"].to(device)
                    coords = batch["graph_coords_km"].to(device)
                    gtd = batch["graph_time_days"].to(device)
                    yb = batch["y"].to(device)
                    mk = batch["seq_padding_mask"].to(device)
                    pp = model(sx, gx, mk, graph_coords_km=coords, graph_time_days=gtd, graph_strike_rad=batch.get("mainshock_strike_rad"))
                    v_loss += stgnn_asymmetric_loss(pp, yb, late_weight=late_weight).item() * len(yb)
                    v_cnt += len(yb)
            v_loss /= max(v_cnt, 1)
            gn_ep_iter.set_postfix(val_loss=f"{v_loss:.4f}", best=f"{best_loss:.4f}")
            if v_loss < best_loss:
                best_loss = v_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            sched.step()

        if best_state is None: raise RuntimeError("GNN fold best_state is None")
        model.load_state_dict(best_state)
        model.eval()

        all_p, all_t = [], []
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            gn_pred_iter = tqdm(v_ldr, desc=f"  GNN Fold {fold_idx} 预测", unit="batch", leave=False)
            for batch in gn_pred_iter:
                sx = batch["seq_x"].to(device)
                gx = batch["global_x"].to(device)
                coords = batch["graph_coords_km"].to(device)
                gtd = batch["graph_time_days"].to(device)
                yb = batch["y"].to(device)
                mk = batch["seq_padding_mask"].to(device)
                pp = model(sx, gx, mk, graph_coords_km=coords, graph_time_days=gtd, graph_strike_rad=batch.get("mainshock_strike_rad"))
                all_p.append(pp.cpu().numpy())
                all_t.append(yb.cpu().numpy())
        preds = np.clip(np.concatenate(all_p, axis=0), 0, None)
        preds[:, 1] = np.expm1(np.clip(preds[:, 1], 0.0, 50.0))
        oof_preds[v_idx] = preds

    oof_df = df[[ID_COL, TIME_COL, *TARGET_COLS]].copy()
    oof_df["gnn_pred_mag"] = oof_preds[:, 0]
    oof_df["gnn_pred_time"] = oof_preds[:, 1]
    return oof_df


def search_ensemble_weights(merged, available_models, target_idx, late_weight, grid_step=0.02):
    """Simplex 网格搜索最优融合权重。支持 1-4 个模型。"""
    pred_cols = [MODEL_PRED_COLS[m][target_idx] for m in available_models]
    pred_mat = merged[pred_cols].to_numpy(float)
    y_true = merged[TARGET_COLS[target_idx]].to_numpy(float)

    # 过滤 NaN 行
    valid = np.isfinite(pred_mat).all(axis=1) & np.isfinite(y_true)
    if not valid.any():
        return {available_models[0]: 1.0} if available_models else {}, float("inf")
    pred_mat = pred_mat[valid]
    y_true = y_true[valid]
    n = len(available_models)

    def _obj(w):
        p = pred_mat @ w
        if target_idx == 0:
            return float(np.sqrt(np.mean((p - y_true) ** 2)))
        err = p - y_true
        tw = np.where(err > 0, late_weight, 1.0)
        return float(np.sqrt(np.mean(tw * err ** 2)))

    if n == 1:
        return {available_models[0]: 1.0}, _obj(np.array([1.0]))

    # Simplex grid
    n_pts = int(1.0 / grid_step)
    if n_pts < 1:
        n_pts = 25  # default for 0.04 step
    best_w, best_obj = None, float("inf")

    def gen_simplex(dim):
        if dim == 2:
            for i in range(n_pts + 1):
                yield np.array([i, n_pts - i], float) / n_pts
        elif dim == 3:
            for i in range(n_pts + 1):
                for j in range(n_pts + 1 - i):
                    yield np.array([i, j, n_pts - i - j], float) / n_pts
        elif dim == 4:
            for i in range(n_pts + 1):
                for j in range(n_pts + 1 - i):
                    for k in range(n_pts + 1 - i - j):
                        yield np.array([i, j, k, n_pts - i - j - k], float) / n_pts

    # 预计算 simplex 组合总数用于进度条
    _simplex_total = {2: n_pts + 1, 3: (n_pts + 1) * (n_pts + 2) // 2,
                      4: (n_pts + 1) * (n_pts + 2) * (n_pts + 3) // 6}
    target_label = "mag" if target_idx == 0 else "time"
    simplex_iter = tqdm(
        gen_simplex(n),
        total=_simplex_total.get(n, n_pts + 1),
        desc=f"  ⚖️  融合权重搜索 ({target_label})",
        unit="comb",
        leave=False,
        disable=(n <= 2),  # 1-2 模型时组合数少，不显示进度条
    )
    for wvec in simplex_iter:
        obj = _obj(wvec)
        if obj < best_obj:
            best_obj = obj
            best_w = {available_models[i]: round(float(wvec[i]), 4) for i in range(n)}

    if best_w is None:
        # fallback to uniform
        best_w = {m: round(1.0 / n, 4) for m in available_models}
        best_obj = _obj(np.array(list(best_w.values())))

    total = sum(best_w.values())
    best_w = {k: round(v / total, 4) for k, v in best_w.items()}
    return best_w, best_obj


def compute_final_combined(oof_df, available_models, mag_w, time_w, late_weight):
    """计算融合后的 combined objective。"""
    valid = np.ones(len(oof_df), dtype=bool)
    for m in available_models:
        mc, tc = MODEL_PRED_COLS[m]
        valid &= oof_df[mc].notna().to_numpy() & oof_df[tc].notna().to_numpy()
    vm = oof_df.loc[valid]
    if len(vm) == 0: return float("inf")

    pm, pt = np.zeros(len(vm)), np.zeros(len(vm))
    for m in available_models:
        mc, tc = MODEL_PRED_COLS[m]
        pm += vm[mc].to_numpy() * mag_w.get(m, 0)
        pt += vm[tc].to_numpy() * time_w.get(m, 0)

    m_rmse = np.sqrt(np.mean((pm - vm[TARGET_COLS[0]].to_numpy()) ** 2))
    err = pt - vm[TARGET_COLS[1]].to_numpy()
    tw = np.where(err > 0, late_weight, 1.0)
    ta_rmse = np.sqrt(np.mean(tw * err ** 2))
    return float(m_rmse + ta_rmse)


def precompute_test_features():
    """预计算测试序列特征（如尚未存在）。"""
    out_path = PROJECT_ROOT / "data" / "processed" / "test_sequences_features.csv"
    if out_path.exists(): return True
    print("预计算测试序列特征 …")
    try:
        from scripts.make_submission import build_single_sequence_features, normalize_event_table
    except ImportError:
        print("  ⚠ 无法导入 make_submission，跳过预计算")
        return False
    all_feats = []
    for csv_file in sorted(TEST_DIR.glob("*_eq.csv")):
        raw_df = pd.read_csv(csv_file)
        event_df = normalize_event_table(raw_df)
        feat_df, _ = build_single_sequence_features(
            event_df=event_df, plate_boundaries_path=PLATE_PATH,
            gcmt_catalog_path=GCMT_PATH if GCMT_PATH.exists() else None,
        )
        feat_df["mainshock_id"] = csv_file.stem
        all_feats.append(feat_df)
    pd.concat(all_feats, ignore_index=True).to_csv(out_path, index=False, encoding="utf-8")
    print(f"  已保存: {out_path}")
    return True


def evaluate_on_holdout(true_labels, params, late_weight, n_est_lgb, n_est_xgb, seed):
    """在 20 条测试序列上做最终评估。"""
    test_feat_path = PROJECT_ROOT / "data" / "processed" / "test_sequences_features.csv"
    if not test_feat_path.exists():
        precompute_test_features()
    test_feat = pd.read_csv(test_feat_path)

    raw_df = pd.read_csv(FEATURES_CSV)
    raw_df[TIME_COL] = pd.to_datetime(raw_df[TIME_COL], utc=True, format="mixed")
    if "has_target_aftershock" in raw_df.columns:
        raw_df = raw_df[raw_df["has_target_aftershock"].astype(bool)]
    raw_df = raw_df.dropna(subset=[TIME_COL, *TARGET_COLS])
    raw_df = add_derived_features(raw_df)
    all_cols = select_feature_columns(raw_df)
    raw_df, missing = impute_missing_features(raw_df, all_cols)
    all_cols = all_cols + missing
    test_feat = add_derived_features(test_feat)
    test_feat, _ = impute_missing_features(test_feat, all_cols)

    if params.get("feature_selection", True) and len(all_cols) > 50:
        all_cols = select_features_by_importance(
            raw_df[all_cols], raw_df[TARGET_COLS], all_cols,
            top_k_ratio=params.get("feature_selection_ratio", 0.85),
            min_features=params.get("feature_selection_min", 30),
        )

    for c in set(all_cols) - set(test_feat.columns):
        test_feat[c] = 0.0

    predictions = []
    holdout_iter = tqdm(
        true_labels.iterrows(),
        total=len(true_labels),
        desc="🎯 Holdout 评估",
        unit="seq",
        leave=False,
    )
    for _, row in holdout_iter:
        sid = row["mainshock_id"]
        seq_file = TEST_DIR / f"{sid}.csv"
        if not seq_file.exists(): continue
        seq = pd.read_csv(seq_file)
        main_time = pd.to_datetime(str(seq.iloc[0]["Date"]) + " " + str(seq.iloc[0]["Time"]), utc=True)

        tr_mask = raw_df[TIME_COL] < main_time
        tr_df = raw_df[tr_mask] if tr_mask.sum() >= 100 else raw_df

        # Tree models
        lgb = build_lgbm(params, n_est_lgb, seed, late_weight)
        lgb.fit(tr_df[all_cols], tr_df[TARGET_COLS])
        xgb = build_xgb(params, n_est_xgb, seed)
        xgb.fit(tr_df[all_cols], tr_df[TARGET_COLS])

        te_row = test_feat[test_feat[ID_COL] == sid]
        if len(te_row) == 0: continue
        X_te = te_row[all_cols]

        p_lgb = np.clip(np.asarray(lgb.predict(X_te), float), 0, None)[0]
        p_xgb = np.clip(np.asarray(xgb.predict(X_te), float), 0, None)[0]

        # Use OOF-searched weights
        mw = _last_ensemble_cache.get("mag", {}).get("baseline", 0.5)
        tw = _last_ensemble_cache.get("time", {}).get("baseline", 0.5)
        pm = mw * p_lgb[0] + _last_ensemble_cache.get("mag", {}).get("xgboost", 0) * p_xgb[0]
        pt = tw * p_lgb[1] + _last_ensemble_cache.get("time", {}).get("xgboost", 0) * p_xgb[1]
        # add dl/gnn if present
        for mm in ["dl", "gnn"]:
            if mm in _last_ensemble_cache.get("mag", {}):
                pm += _last_ensemble_cache["mag"].get(mm, 0) * p_lgb[0]  # fallback to lgb
            if mm in _last_ensemble_cache.get("time", {}):
                pt += _last_ensemble_cache["time"].get(mm, 0) * p_lgb[1]

        predictions.append({
            "mainshock_id": sid,
            "pred_max_mag": float(pm),
            "pred_time_to_max": float(pt),
            "true_max_mag": float(row["true_max_mag"]),
            "true_time_to_max_days": float(row["true_time_to_max_days"]),
        })

    pred_df = pd.DataFrame(predictions)
    if len(pred_df) == 0: return {"holdout_combined": float("inf"), "holdout_n": 0}
    # 传递主震震级用于 Båth 检验
    ms_mag = None
    if len(merged_labels := true_labels[true_labels["mainshock_id"].isin(pred_df["mainshock_id"])]):
        ms_mag = merged_labels.set_index("mainshock_id").reindex(pred_df["mainshock_id"])["mainshock_mag"].to_numpy()
    m = calculate_metrics(
        y_true_mag=pred_df["true_max_mag"].to_numpy(),
        y_pred_mag=pred_df["pred_max_mag"].to_numpy(),
        y_true_time=pred_df["true_time_to_max_days"].to_numpy(),
        y_pred_time=pred_df["pred_time_to_max"].to_numpy(),
        late_weight=late_weight,
        mainshock_mag=ms_mag,
    )
    return {
        "holdout_mag_rmse": float(m.get("mag_rmse", np.nan)),
        "holdout_mag_mae": float(m.get("mag_mae", np.nan)),
        "holdout_mag_medae": float(m.get("mag_medae", np.nan)),
        "holdout_time_rmse": float(m.get("time_rmse", np.nan)),
        "holdout_time_mae": float(m.get("time_mae", np.nan)),
        "holdout_time_medae": float(m.get("time_medae", np.nan)),
        "holdout_time_asymmetric_rmse": float(m.get("time_asymmetric_rmse", np.nan)),
        "holdout_time_asymmetric_mae": float(m.get("time_asymmetric_mae", np.nan)),
        "holdout_energy_ratio_median": float(m.get("mag_energy_ratio_median", np.nan)),
        "holdout_bath_deviation": float(m.get("bath_deviation", np.nan)) if "bath_deviation" in m else None,
        "holdout_combined": float(m.get("mag_rmse", 99) + m.get("time_asymmetric_rmse", 99)),
        "holdout_n": len(pred_df),
        "holdout_predictions": pred_df,
    }


# Cache for ensemble weights found in OOF
_last_ensemble_cache: dict = {}


# ============================================================
#  Optuna
# ============================================================

def _ensure_optuna():
    try:
        import optuna; return optuna
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "-q"])
        import optuna; return optuna


def suggest_all_params(trial, include_dl: bool, include_gnn: bool, fast: bool) -> dict:
    """一次采样所有超参数。"""
    p: dict[str, Any] = {}

    # === LightGBM (8 params) ===
    p["lgb_lr"] = trial.suggest_float("lgb_lr", 0.01, 0.1, log=True)
    p["lgb_num_leaves"] = trial.suggest_int("lgb_num_leaves", 15, 255, log=True)
    p["lgb_max_depth"] = trial.suggest_int("lgb_max_depth", 3, 15)
    p["lgb_min_child_samples"] = trial.suggest_int("lgb_min_child_samples", 5, 100, log=True)
    p["lgb_subsample"] = trial.suggest_float("lgb_subsample", 0.5, 0.95)
    p["lgb_colsample_bytree"] = trial.suggest_float("lgb_colsample_bytree", 0.4, 0.95)
    p["lgb_reg_alpha"] = trial.suggest_float("lgb_reg_alpha", 1e-4, 1.0, log=True)
    p["lgb_reg_lambda"] = trial.suggest_float("lgb_reg_lambda", 0.1, 10.0, log=True)

    # === XGBoost (7 params) ===
    p["xgb_lr"] = trial.suggest_float("xgb_lr", 0.01, 0.1, log=True)
    p["xgb_max_depth"] = trial.suggest_int("xgb_max_depth", 3, 12)
    p["xgb_min_child_weight"] = trial.suggest_int("xgb_min_child_weight", 1, 50)
    p["xgb_subsample"] = trial.suggest_float("xgb_subsample", 0.5, 0.95)
    p["xgb_colsample_bytree"] = trial.suggest_float("xgb_colsample_bytree", 0.4, 0.95)
    p["xgb_reg_alpha"] = trial.suggest_float("xgb_reg_alpha", 1e-4, 10.0, log=True)
    p["xgb_reg_lambda"] = trial.suggest_float("xgb_reg_lambda", 0.1, 10.0, log=True)

    # === Features (3 params) ===
    p["feature_selection"] = trial.suggest_categorical("feature_selection", [True, False])
    p["feature_selection_ratio"] = trial.suggest_float("feature_selection_ratio", 0.6, 0.95)
    p["feature_selection_min"] = trial.suggest_int("feature_selection_min", 20, 60)

    # === OOF params (2) ===
    p["purge_days"] = trial.suggest_float("purge_days", 7.0, 90.0)
    p["min_purge_days"] = trial.suggest_float("min_purge_days", 3.0, 21.0)

    # === DL/Transformer (if enabled) ===
    if include_dl:
        p["dl_d_model"] = trial.suggest_categorical("dl_d_model", [64, 96, 128, 192, 256])
        p["dl_nhead"] = trial.suggest_categorical("dl_nhead", [2, 4, 8])
        p["dl_num_layers"] = trial.suggest_int("dl_num_layers", 1, 6)
        p["dl_dim_ff"] = trial.suggest_categorical("dl_dim_ff", [128, 256, 512])
        p["dl_dropout"] = trial.suggest_float("dl_dropout", 0.05, 0.4)
        p["dl_lr"] = trial.suggest_float("dl_lr", 5e-4, 5e-3, log=True)
        p["dl_batch_size"] = trial.suggest_categorical("dl_batch_size", [16, 32, 64])
        p["dl_epochs"] = trial.suggest_int("dl_epochs", 10, 15) if fast else trial.suggest_int("dl_epochs", 15, 40)
        p["dl_global_hidden"] = trial.suggest_categorical("dl_global_hidden", [64, 128, 256])
        p["dl_fusion_hidden"] = trial.suggest_categorical("dl_fusion_hidden", [64, 128, 256])

    # === GNN (if enabled) ===
    if include_gnn:
        p["gnn_node_hidden"] = trial.suggest_categorical("gnn_node_hidden", [32, 64, 128])
        p["gnn_layers"] = trial.suggest_int("gnn_layers", 1, 5)
        p["gnn_sigma"] = trial.suggest_float("gnn_sigma", 20.0, 100.0)
        p["gnn_radius_km"] = trial.suggest_float("gnn_radius_km", 50.0, 150.0)
        p["gnn_gru_hidden"] = trial.suggest_categorical("gnn_gru_hidden", [32, 64, 128])
        p["gnn_gru_layers"] = trial.suggest_int("gnn_gru_layers", 1, 3)
        p["gnn_dropout"] = trial.suggest_float("gnn_dropout", 0.05, 0.4)
        p["gnn_lr"] = trial.suggest_float("gnn_lr", 5e-4, 5e-3, log=True)
        p["gnn_batch_size"] = trial.suggest_categorical("gnn_batch_size", [8, 16, 32])
        p["gnn_epochs"] = trial.suggest_int("gnn_epochs", 8, 12) if fast else trial.suggest_int("gnn_epochs", 12, 30)
        p["gnn_global_hidden"] = trial.suggest_categorical("gnn_global_hidden", [64, 128, 256])
        p["gnn_fusion_hidden"] = trial.suggest_categorical("gnn_fusion_hidden", [64, 128, 256])

    # Ensemble grid step
    p["ensemble_grid_step"] = trial.suggest_categorical("ensemble_grid_step", [0.02, 0.04])

    return p


def run_tuning(
    n_trials, n_splits_tree, n_splits_dl, n_splits_gnn,
    n_est_lgb, n_est_xgb, late_weight, seed,
    study_name, storage_path,
    holdout_labels, eval_holdout_every,
    include_dl, include_gnn, fast, device_str,
    mag_weight=3.0,
    optuna_n_jobs=1,
):
    optuna = _ensure_optuna()
    if storage_path:
        storage = f"sqlite:///{storage_path}"
    else:
        storage = None

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10, multivariate=True)
    study = optuna.create_study(
        study_name=study_name, storage=storage,
        direction="minimize", sampler=sampler,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )

    # 一次性加载数据
    print("\n加载训练数据 …")
    raw_df = pd.read_csv(FEATURES_CSV)
    raw_df[TIME_COL] = pd.to_datetime(raw_df[TIME_COL], utc=True, format="mixed")
    if "has_target_aftershock" in raw_df.columns:
        df_full = raw_df[raw_df["has_target_aftershock"].astype(bool)].copy()
    else:
        df_full = raw_df.dropna(subset=TARGET_COLS).copy()
    df_full = df_full.dropna(subset=[TIME_COL, *TARGET_COLS]).reset_index(drop=True)
    df_full = add_derived_features(df_full)
    all_raw_cols = select_feature_columns(df_full)
    print(f"  训练样本: {len(df_full)}, 原始特征: {len(all_raw_cols)}")
    print(f"  DL={include_dl}, GNN={include_gnn}, Fast={fast}")
    print(f"  DL splits={n_splits_dl}, GNN splits={n_splits_gnn}, Tree splits={n_splits_tree}")
    print(f"  Device: {device_str}")

    # 提前验证 CUDA 可用性
    from src.utils import get_torch_device, get_lightgbm_device
    _dev = get_torch_device(device_str)
    _lgb_dev = get_lightgbm_device(device_str)
    print(f"  PyTorch 设备: {_dev}  |  LightGBM 设备: {_lgb_dev}")

    global_best_score = float("inf")
    global_best_params: dict = {}
    global_best_holdout: dict = {}

    def objective(trial):
        nonlocal global_best_score, global_best_params, global_best_holdout
        tqdm.write(f"\n{'─'*50}\n🔬 Trial {trial.number}: 采样超参数并开始评估 …")
        params = suggest_all_params(trial, include_dl, include_gnn, fast)
        params["mag_weight"] = mag_weight  # CLI 指定的震级权重，注入到每 trial

        # 特征工程
        tqdm.write(f"  📐 特征工程 (特征选择={params.get('feature_selection', True)})")
        df = df_full.copy()
        df, missing_cols = impute_missing_features(df, all_raw_cols)
        feature_cols = all_raw_cols + missing_cols
        if params.get("feature_selection", True) and len(feature_cols) > 50:
            feature_cols = select_features_by_importance(
                df[feature_cols], df[TARGET_COLS], feature_cols,
                top_k_ratio=params["feature_selection_ratio"],
                min_features=params["feature_selection_min"],
            )
        trial.set_user_attr("n_features", len(feature_cols))
        trial.set_user_attr("feature_selection", params["feature_selection"])

        purge_days = params["purge_days"]
        min_purge = params["min_purge_days"]

        available_models = ["baseline", "xgboost"]

        # 1. Tree OOF
        tqdm.write(f"  🌲 Step 1/4: 树模型 OOF CV ({n_splits_tree}-fold, purge={purge_days:.0f}d)")
        try:
            oof_tree = run_tree_oof_cv(
                df, feature_cols, params, n_splits_tree, purge_days, min_purge,
                seed, late_weight, n_est_lgb, n_est_xgb,
            )
        except Exception as e:
            tqdm.write(f"  ⚠ Trial {trial.number} tree OOF 失败: {e}")
            return float("inf")

        # Start with tree OOF as base
        oof_all = oof_tree.copy()

        # 2. DL OOF
        if include_dl:
            tqdm.write(f"  🧠 Step 2/4: Transformer OOF CV ({n_splits_dl}-fold, {params.get('dl_epochs',30)}ep)")
            try:
                oof_dl = run_dl_oof_cv(
                    df, EVENT_CATALOG, feature_cols, params,
                    n_splits_dl, purge_days, seed, late_weight, device_str,
                )
                oof_all["dl_pred_mag"] = oof_dl["dl_pred_mag"]
                oof_all["dl_pred_time"] = oof_dl["dl_pred_time"]
                available_models.append("dl")
            except Exception as e:
                tqdm.write(f"  ⚠ Trial {trial.number} DL OOF 失败: {e}，跳过 DL")
                trial.set_user_attr("dl_error", str(e)[:200])

        # 3. GNN OOF
        if include_gnn:
            tqdm.write(f"  🔗 Step 3/4: ST-GNN OOF CV ({n_splits_gnn}-fold, {params.get('gnn_epochs',30)}ep)")
            try:
                oof_gnn = run_gnn_oof_cv(
                    df, EVENT_CATALOG, feature_cols, params,
                    n_splits_gnn, purge_days, seed, late_weight, device_str,
                )
                oof_all["gnn_pred_mag"] = oof_gnn["gnn_pred_mag"]
                oof_all["gnn_pred_time"] = oof_gnn["gnn_pred_time"]
                available_models.append("gnn")
            except Exception as e:
                tqdm.write(f"  ⚠ Trial {trial.number} GNN OOF 失败: {e}，跳过 GNN")
                trial.set_user_attr("gnn_error", str(e)[:200])

        # 4. Ensemble
        tqdm.write(f"  ⚖️  Step 4/4: 融合权重搜索 (模型: {available_models}, grid_step={params.get('ensemble_grid_step',0.04)})")
        gs = params.get("ensemble_grid_step", 0.04)
        mag_w, mag_obj = search_ensemble_weights(oof_all, available_models, 0, late_weight, gs)
        time_w, time_obj = search_ensemble_weights(oof_all, available_models, 1, late_weight, gs)
        combined = float(params.get("mag_weight", 3.0) * mag_obj + time_obj)

        trial.set_user_attr("mag_rmse", mag_obj)
        trial.set_user_attr("time_asym_rmse", time_obj)
        trial.set_user_attr("time_rmse", float(np.sqrt(np.mean(
            (oof_all[available_models[0] + "_pred_time"].fillna(0).values - 
             oof_all[TARGET_COLS[1]].values) ** 2))))
        trial.set_user_attr("n_models", len(available_models))
        trial.set_user_attr("available_models", ",".join(available_models))

        # 5. Holdout
        if trial.number % eval_holdout_every == 0 or combined < global_best_score:
            tqdm.write(f"  🎯 运行 Holdout 评估 …")
            global _last_ensemble_cache
            _last_ensemble_cache = {"mag": mag_w, "time": time_w}
            holdout = evaluate_on_holdout(holdout_labels, params, late_weight, n_est_lgb, n_est_xgb, seed)
            trial.set_user_attr("holdout_combined", holdout.get("holdout_combined", float("inf")))
            trial.set_user_attr("holdout_mag_rmse", holdout.get("holdout_mag_rmse", float("inf")))
            trial.set_user_attr("holdout_time_asym_rmse", holdout.get("holdout_time_asymmetric_rmse", float("inf")))

            if combined < global_best_score:
                global_best_score = combined
                global_best_params = deepcopy(params)
                global_best_holdout = deepcopy(holdout)
                global_best_holdout["_available_models"] = available_models
                global_best_holdout["_mag_weights"] = mag_w
                global_best_holdout["_time_weights"] = time_w
                tqdm.write(f"\n{'='*60}")
                tqdm.write(f"★ Trial {trial.number}: OOF={combined:.4f}  "
                      f"(= {mag_weight:.1f}×{mag_obj:.4f} + {time_obj:.4f})")
                tqdm.write(f"  Holdout={holdout.get('holdout_combined', float('inf')):.4f}")
                tqdm.write(f"  模型: {available_models}")
                tqdm.write(f"  LGB lr={params['lgb_lr']:.4f} leaves={params['lgb_num_leaves']} depth={params['lgb_max_depth']}")
                tqdm.write(f"  XGB lr={params['xgb_lr']:.4f} depth={params['xgb_max_depth']}")
                if include_dl:
                    tqdm.write(f"  DL dm={params.get('dl_d_model')} head={params.get('dl_nhead')} layers={params.get('dl_num_layers')} lr={params.get('dl_lr',0):.4f}")
                if include_gnn:
                    tqdm.write(f"  GNN hidden={params.get('gnn_node_hidden')} layers={params.get('gnn_layers')} radius={params.get('gnn_radius_km',0):.0f}km")
                tqdm.write(f"  特征: {len(feature_cols)} (FS={params['feature_selection']})")
                tqdm.write(f"  Mag 权重: {mag_w}")
                tqdm.write(f"  Time 权重: {time_w}")
                tqdm.write(f"  📊 震级: RMSE={mag_obj:.4f}  时间: AsymRMSE={time_obj:.4f}")
                if holdout.get("holdout_n", 0) > 0:
                    tqdm.write(f"  Holdout: mag_rmse={holdout.get('holdout_mag_rmse',np.nan):.4f} "
                          f"time_asym_rmse={holdout.get('holdout_time_asymmetric_rmse',np.nan):.4f}")

        return combined

    print(f"\n开始全模型调优: {n_trials} trials (已完成 {len(study.trials)})")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=optuna_n_jobs)

    # Save all results into best params
    global_best_params.update({
        "_available_models": global_best_holdout.pop("_available_models", []),
        "_mag_weights": global_best_holdout.pop("_mag_weights", {}),
        "_time_weights": global_best_holdout.pop("_time_weights", {}),
    })
    return study, global_best_params, global_best_holdout


# ============================================================
#  CLI & 保存
# ============================================================

def save_results(study, best_params, best_holdout, output_dir, late_weight, include_dl, include_gnn, mag_weight=3.0):
    optuna = _ensure_optuna()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 提取纯模型参数和元数据
    model_params = {k: v for k, v in best_params.items() if not k.startswith("_")}
    meta = {k: v for k, v in best_params.items() if k.startswith("_")}

    with (output_dir / "best_params.json").open("w", encoding="utf-8") as f:
        json.dump(model_params, f, ensure_ascii=False, indent=2)

    with (output_dir / "ensemble_weights.json").open("w", encoding="utf-8") as f:
        json.dump({"mag": meta.get("_mag_weights", {}), "time": meta.get("_time_weights", {})},
                  f, ensure_ascii=False, indent=2)

    stats = {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "best_trial_number": study.best_trial.number,
        "best_oof_score": float(study.best_value),
        "best_params": model_params,
        "best_available_models": meta.get("_available_models", []),
        "best_mag_weights": meta.get("_mag_weights", {}),
        "best_time_weights": meta.get("_time_weights", {}),
        "best_holdout": {k: float(v) if isinstance(v, (int, float, np.floating)) else (
            v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else str(v)
        ) for k, v in best_holdout.items() if k != "holdout_predictions"},
        "late_weight": late_weight,
        "include_dl": include_dl,
        "include_gnn": include_gnn,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with (output_dir / "tuning_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)

    if "holdout_predictions" in best_holdout:
        best_holdout["holdout_predictions"].to_csv(output_dir / "holdout_predictions.csv", index=False, encoding="utf-8")

    # Trials history
    trials_data = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE: continue
        row = {"number": t.number, "value": t.value, **t.params,
               **{f"attr_{k}": v for k, v in (t.user_attrs or {}).items()}}
        trials_data.append(row)
    if trials_data:
        pd.DataFrame(trials_data).to_csv(output_dir / "trials_history.csv", index=False, encoding="utf-8")

    # ─── 生成可读调优报告 ───
    _write_tuning_report(output_dir, study, model_params, meta, best_holdout,
                         late_weight, include_dl, include_gnn, mag_weight)

    # Final console report
    print(f"\n{'='*60}")
    print("全模型联合调优完成！")
    print(f"{'='*60}")
    print(f"Study: {study.study_name}")
    print(f"Trials: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best OOF score: {study.best_value:.4f}")
    print(f"\n可用模型: {meta.get('_available_models', [])}")
    print(f"\n最优 Mag 融合权重: {meta.get('_mag_weights', {})}")
    print(f"最优 Time 融合权重: {meta.get('_time_weights', {})}")
    print(f"\n最优模型参数:")
    for k, v in sorted(model_params.items()):
        print(f"  {k}: {v}")

    if best_holdout.get("holdout_n", 0) > 0:
        print(f"\nHoldout 评估 ({best_holdout['holdout_n']} 条):")
        print(f"  📊 震级: RMSE={best_holdout.get('holdout_mag_rmse',np.nan):.4f}  "
              f"MAE={best_holdout.get('holdout_mag_mae',np.nan):.4f}  "
              f"MedAE={best_holdout.get('holdout_mag_medae',np.nan):.4f}")
        er = best_holdout.get("holdout_energy_ratio_median")
        if er is not None: print(f"        EnergyRatio(median): {er:.2f}×")
        print(f"  ⏱️  时间: RMSE={best_holdout.get('holdout_time_rmse',np.nan):.2f}d  "
              f"MAE={best_holdout.get('holdout_time_mae',np.nan):.2f}d  "
              f"MedAE={best_holdout.get('holdout_time_medae',np.nan):.2f}d")
        print(f"  🎯 综合: {best_holdout.get('holdout_combined',np.nan):.4f}  "
              f"AsymRMSE={best_holdout.get('holdout_time_asymmetric_rmse',np.nan):.2f}")
        bd = best_holdout.get("holdout_bath_deviation")
        if bd is not None: print(f"  🔬 Båth ΔM偏差: {bd:.4f}")

    print(f"\n产物已保存: {output_dir}")
    print(f"📄 可读报告: {output_dir / 'tuning_report.md'}")


# ============================================================
#  生成可读调优报告 (Markdown)
# ============================================================

def _write_tuning_report(
    output_dir: Path,
    study,
    model_params: dict,
    meta: dict,
    best_holdout: dict,
    late_weight: float,
    include_dl: bool,
    include_gnn: bool,
    mag_weight: float = 3.0,
) -> None:
    """生成调优报告 Markdown 文件。"""
    available_models = meta.get("_available_models", [])
    mag_weights = meta.get("_mag_weights", {})
    time_weights = meta.get("_time_weights", {})

    best_oof = float(study.best_value)
    best_trial = study.best_trial.number

    lines: list[str] = []
    lines.append(f"# 余震预测全模型超参数调优报告")
    lines.append(f"")
    lines.append(f"> 生成时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"> 调优算法: Optuna TPESampler (多变量联合采样)")
    lines.append(f"> 试验次数: {len(study.trials)}")
    lines.append(f"")

    # ── 1. 总体概述 ──
    lines.append(f"## 1. 总体概述")
    lines.append(f"")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|:---|---:|")
    lines.append(f"| 最优 Trial 编号 | **#{best_trial}** |")
    lines.append(f"| 最优 OOF 综合分 (越低越好) | **{best_oof:.4f}** |")
    lines.append(f"| 参与融合的模型 | {', '.join(available_models) if available_models else '无'} |")
    lines.append(f"| 调优 DL 模型 | {'✅' if include_dl else '❌'} |")
    lines.append(f"| 调优 GNN 模型 | {'✅' if include_gnn else '❌'} |")
    lines.append(f"| 时间非对称惩罚权重 | {late_weight:.1f}× |")
    lines.append(f"")

    # ── 2. OOF 交叉验证指标 ──
    lines.append(f"## 2. OOF 交叉验证指标")
    lines.append(f"")
    lines.append(f"OOF (Out-of-Fold) 指标反映模型对**未见过的未来数据**的预测能力。")
    lines.append(f"")
    mag_rmse = study.best_trial.user_attrs.get("mag_rmse", float("nan"))
    time_asym = study.best_trial.user_attrs.get("time_asym_rmse", float("nan"))
    lines.append(f"| 指标 | 最优 Trial 值 | 说明 |")
    lines.append(f"|:---|---:|:---|")
    lines.append(f"| 震级 RMSE | {mag_rmse:.4f} | 预测余震震级的均方根误差 |")
    lines.append(f"| 时间非对称 RMSE | {time_asym:.4f} | 预测偏晚惩罚 {late_weight:.0f}× 的时间误差 |")
    lines.append(f"| **综合分 (mag × {mag_weight:.0f} + time)** | **{best_oof:.4f}** | Optuna 优化目标 |")
    lines.append(f"")

    # ── 3. Holdout 测试集评估 ──
    holdout_n = best_holdout.get("holdout_n", 0)
    if holdout_n > 0:
        lines.append(f"## 3. Holdout 测试集评估")
        lines.append(f"")
        lines.append(f"在 **{holdout_n} 条独立测试序列**上的最终评估（未参与训练/验证）：")
        lines.append(f"")
        lines.append(f"### 3.1 震级预测")
        lines.append(f"| 指标 | 值 | 说明 |")
        lines.append(f"|:---|---:|:---|")
        for key, label, note in [
            ("holdout_mag_rmse", "RMSE", "震级均方根误差"),
            ("holdout_mag_mae", "MAE", "平均绝对误差"),
            ("holdout_mag_medae", "MedAE", "中位数绝对误差（鲁棒）"),
        ]:
            v = best_holdout.get(key)
            if v is not None: lines.append(f"| {label} | {v:.4f} | {note} |")
        er = best_holdout.get("holdout_energy_ratio_median")
        if er is not None: lines.append(f"| Energy Ratio (median) | {er:.2f}× | 典型能量偏差倍数 |")
        lines.append(f"")
        lines.append(f"### 3.2 时间预测")
        lines.append(f"| 指标 | 值 | 说明 |")
        lines.append(f"|:---|---:|:---|")
        for key, label, note in [
            ("holdout_time_rmse", "RMSE", "时间均方根误差（天）"),
            ("holdout_time_mae", "MAE", "平均绝对误差（天）"),
            ("holdout_time_medae", "MedAE", "中位数绝对误差（天）"),
            ("holdout_time_asymmetric_rmse", f"非对称 RMSE", f"预测偏晚 {late_weight:.0f}× 惩罚"),
            ("holdout_time_asymmetric_mae", f"非对称 MAE", f"非对称平均绝对误差"),
        ]:
            v = best_holdout.get(key)
            if v is not None: lines.append(f"| {label} | {v:.4f} | {note} |")
        lines.append(f"")
        lines.append(f"### 3.3 物理一致性")
        bd = best_holdout.get("holdout_bath_deviation")
        lines.append(f"| 指标 | 值 | 说明 |")
        lines.append(f"|:---|---:|:---|")
        if bd is not None: lines.append(f"| Båth ΔM Deviation | {bd:.4f} | ΔM 预测偏差，越低越好 |")
        comb = best_holdout.get("holdout_combined")
        if comb is not None: lines.append(f"| **Holdout 综合分** | **{comb:.4f}** | mag_rmse + time_asymmetric_rmse |")
        lines.append(f"")

    # ── 4. 最优融合权重 ──
    lines.append(f"## 4. 最优融合权重")
    lines.append(f"")
    lines.append(f"震级 (mag) 和时间 (time) 目标独立搜索最优权重。")
    lines.append(f"")
    lines.append(f"### 震级预测 (Mag) 融合权重")
    lines.append(f"")
    lines.append(f"| 模型 | 权重 | 占比 |")
    lines.append(f"|:---|---:|---:|")
    mag_total = sum(mag_weights.values()) or 1.0
    for model in ["baseline", "xgboost", "dl", "gnn"]:
        w = mag_weights.get(model, 0.0)
        lines.append(f"| {_model_label(model)} | {w:.4f} | {w / mag_total * 100:.1f}% |")
    lines.append(f"")
    lines.append(f"### 时间预测 (Time) 融合权重")
    lines.append(f"")
    lines.append(f"| 模型 | 权重 | 占比 |")
    lines.append(f"|:---|---:|---:|")
    time_total = sum(time_weights.values()) or 1.0
    for model in ["baseline", "xgboost", "dl", "gnn"]:
        w = time_weights.get(model, 0.0)
        lines.append(f"| {_model_label(model)} | {w:.4f} | {w / time_total * 100:.1f}% |")
    lines.append(f"")
    lines.append(f"> 权重为 0 的模型不参与最终推理。")
    lines.append(f"")

    # ── 5. 最优超参数 ──
    lines.append(f"## 5. 最优超参数")
    lines.append(f"")

    # 分组显示
    groups = [
        ("LightGBM", [k for k in sorted(model_params) if k.startswith("lgb_")]),
        ("XGBoost", [k for k in sorted(model_params) if k.startswith("xgb_")]),
        ("Transformer (DL)", [k for k in sorted(model_params) if k.startswith("dl_")]),
        ("ST-GNN", [k for k in sorted(model_params) if k.startswith("gnn_")]),
        ("特征工程 & OOF", [k for k in sorted(model_params)
                          if not any(k.startswith(p) for p in ("lgb_", "xgb_", "dl_", "gnn_"))]),
    ]

    for group_name, keys in groups:
        if not keys:
            continue
        lines.append(f"### {group_name}")
        lines.append(f"")
        lines.append(f"| 参数 | 最优值 |")
        lines.append(f"|:---|---:|")
        for k in keys:
            v = model_params[k]
            if isinstance(v, float):
                lines.append(f"| `{k}` | {v:.6f} |")
            else:
                lines.append(f"| `{k}` | {v} |")
        lines.append(f"")

    # ── 6. 使用建议 ──
    lines.append(f"## 6. 使用建议")
    lines.append(f"")
    lines.append(f"### 使用最优参数重新训练")
    lines.append(f"")
    lines.append(f"```bash")
    lines.append(f"# 查看最优参数")
    lines.append(f"cat {output_dir}/best_params.json")
    lines.append(f"")
    lines.append(f"# 查看融合权重")
    lines.append(f"cat {output_dir}/ensemble_weights.json")
    lines.append(f"")
    lines.append(f"# 使用最优融合权重运行 OOF 全流程")
    lines.append(f"cp {output_dir}/ensemble_weights.json data/models/ensemble_weights.json")
    lines.append(f"./run.sh --skip-download --train-oof-ensemble")
    lines.append(f"```")
    lines.append(f"")
    lines.append(f"### 产物清单")
    lines.append(f"")
    lines.append(f"| 文件 | 说明 |")
    lines.append(f"|:---|:---|")
    lines.append(f"| `{output_dir}/best_params.json` | 最优超参数 (JSON) |")
    lines.append(f"| `{output_dir}/ensemble_weights.json` | 双目标最优融合权重 (JSON) |")
    lines.append(f"| `{output_dir}/tuning_stats.json` | 调优统计汇总 (JSON) |")
    lines.append(f"| `{output_dir}/holdout_predictions.csv` | Holdout 预测 vs 真实值 |")
    lines.append(f"| `{output_dir}/trials_history.csv` | 所有 trial 记录 |")
    lines.append(f"| `{output_dir}/tuning_report.md` | **本报告** |")
    lines.append(f"")

    # 写入文件
    report_path = output_dir / "tuning_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _model_label(model: str) -> str:
    """模型代号 → 可读名称。"""
    return {
        "baseline": "LightGBM (基线)",
        "xgboost": "XGBoost",
        "dl": "Transformer",
        "gnn": "ST-GNN",
    }.get(model, model)


def parse_args():
    p = argparse.ArgumentParser(description="余震预测全模型联合超参数调优 (Optuna)")
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--optuna-n-jobs", type=int, default=1, help="并行 trial 数量")
    p.add_argument("--n-splits-tree", type=int, default=5, help="树模型 OOF CV 折数")
    p.add_argument("--n-splits-dl", type=int, default=3, help="DL OOF CV 折数 (少折加速)")
    p.add_argument("--n-splits-gnn", type=int, default=3, help="GNN OOF CV 折数")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--late-weight", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--study-name", type=str, default="aftershock_full_tune")
    p.add_argument("--storage", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "tuning_results")
    p.add_argument("--eval-holdout-every", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--no-dl", action="store_true", help="不调优 Transformer")
    p.add_argument("--no-gnn", action="store_true", help="不调优 ST-GNN")
    p.add_argument("--fast", action="store_true",
                   help="快速模式: 减少 DL/GNN epochs")
    p.add_argument("--no-holdout", action="store_true")
    p.add_argument("--mag-weight", type=float, default=3.0,
                   help="综合目标中震级权重 (默认 3.0, 即震级重要度=时间3倍)")
    return p.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)

    # 确保 optuna 已安装
    _ensure_optuna()

    t0 = time.time()

    internal_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"tune_{internal_timestamp}"

    include_dl = not args.no_dl
    include_gnn = not args.no_gnn

    print("=" * 60)
    print("全模型联合超参数调优")
    print("=" * 60)
    print(f"Trials: {args.n_trials}")
    print(f"Tree splits: {args.n_splits_tree}  DL splits: {args.n_splits_dl}  GNN splits: {args.n_splits_gnn}")
    print(f"DL: {include_dl}  GNN: {include_gnn}  Fast: {args.fast}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")

    # 预计算测试特征
    if not args.no_holdout:
        precompute_test_features()

    # 提取真实标签
    holdout_labels = extract_true_labels(FULL_CATALOG, TEST_DIR)
    eval_every = args.eval_holdout_every if not args.no_holdout else args.n_trials + 1

    storage = str(PROJECT_ROOT / args.storage) if args.storage else None

    study, best_params, best_holdout = run_tuning(
        n_trials=args.n_trials,
        n_splits_tree=args.n_splits_tree,
        n_splits_dl=args.n_splits_dl,
        n_splits_gnn=args.n_splits_gnn,
        n_est_lgb=args.n_estimators,
        n_est_xgb=args.n_estimators,
        late_weight=args.late_weight,
        seed=args.seed,
        study_name=args.study_name,
        storage_path=storage,
        holdout_labels=holdout_labels,
        eval_holdout_every=eval_every,
        include_dl=include_dl,
        include_gnn=include_gnn,
        fast=args.fast,
        device_str=args.device,
        mag_weight=args.mag_weight,
        optuna_n_jobs=args.optuna_n_jobs,
    )

    save_results(study, best_params, best_holdout, output_dir, args.late_weight, include_dl, include_gnn, args.mag_weight)

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed/60:.1f} 分钟 ({elapsed/3600:.1f} 小时)")


if __name__ == "__main__":
    main()
