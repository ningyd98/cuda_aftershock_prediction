"""Decoupled 全参数分目标调优脚本 v2。

支持：
- --target all|mag|time|extreme 选择调优目标
- --separate-target-tuning 每个窗口+目标独立 Optuna study
- --enable-oof-fusion 基于 OOF 学习融合权重
- --tune-transformer/--tune-gnn 可选 DL 辅助（失败不阻断）
"""

from __future__ import annotations

import argparse, json, sys, warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_baseline import add_derived_features, select_feature_columns
from scripts.oof_fusion import fit_oof_fusion_weights, normalize_weights
from src.qualification import (
    QUALIFICATION_WINDOWS, qualification_target_cols, reconstruct_legal_window_features,
)
from src.time_buckets import (
    align_bucket_probabilities, assign_time_buckets_batch,
    expected_time_from_bucket_probs, safe_extreme_probability,
)

TIME_COL = "mainshock_time"

_OPTUNA_AVAILABLE = False
try:
    import optuna
    _OPTUNA_AVAILABLE = True
except ImportError:
    pass

_LGBM_CUDA_OK = False
try:
    import numpy as _np_test
    from lightgbm import LGBMRegressor as _LGBMTest
    _t = _LGBMTest(device="cuda", n_estimators=1, num_leaves=2, verbose=-1)
    _t.fit(_np_test.array([[0.0]]), _np_test.array([0.0]))
    _LGBM_CUDA_OK = True
except Exception:
    pass


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    ap = argparse.ArgumentParser(description="Full decoupled hyperparameter tuning v2.")
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT/"data"/"processed"/"qualification_features.csv")
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["auto","cuda","cpu"], default="cuda")
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--windows", type=str, default="T1,T2,T3")
    ap.add_argument("--fast", action="store_true")
    # Target selection
    ap.add_argument("--target", choices=["all","mag","time","extreme"], default="all")
    ap.add_argument("--separate-target-tuning", type=lambda x: x.lower() in ("true","1","yes"), default=True)
    ap.add_argument("--enable-oof-fusion", type=lambda x: x.lower() in ("true","1","yes"), default=True)
    ap.add_argument("--fusion-grid-step", type=float, default=0.02)
    # Per-target trial counts
    ap.add_argument("--mag-trials", type=int, default=80)
    ap.add_argument("--time-trials", type=int, default=80)
    ap.add_argument("--extreme-trials", type=int, default=60)
    ap.add_argument("--dl-trials", type=int, default=30)
    ap.add_argument("--gnn-trials", type=int, default=30)
    ap.add_argument("--n-splits", type=int, default=5)
    # DL/GNN flags
    ap.add_argument("--tune-transformer", action="store_true")
    ap.add_argument("--tune-gnn", action="store_true")
    # Score weights
    ap.add_argument("--w-mag-mae", type=float, default=1.0)
    ap.add_argument("--w-mag-rmse", type=float, default=1.0)
    ap.add_argument("--w-time-mae", type=float, default=0.03)
    ap.add_argument("--w-time-rmse", type=float, default=0.03)
    ap.add_argument("--w-extreme", type=float, default=0.5)
    ap.add_argument("--w-late", type=float, default=0.3)
    ap.add_argument("--w-t1-bonus", type=float, default=0.2)
    # Legacy compat
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--objective", type=str, default="weighted")
    ap.add_argument("--study-name", type=str, default=None)
    ap.add_argument("--storage", type=str, default=None)
    return ap.parse_args()


# ============================================================================
# 搜索空间（与 v1 一致，略微扩展）
# ============================================================================

def _mag_space(fast=False):
    if fast: return {"n_estimators":[100,300],"learning_rate":[0.03,0.05],"num_leaves":[31,63],
                     "min_child_samples":[20,50],"subsample":[0.7,0.9],"colsample_bytree":[0.6,0.8],
                     "reg_alpha":[0.0,0.1],"reg_lambda":[0.5,2.0]}
    return {"n_estimators":[200,300,500,800,1000],"learning_rate":[0.01,0.02,0.03,0.05,0.07],
            "num_leaves":[15,31,63,127,255],"min_child_samples":[5,10,20,50,100],
            "subsample":[0.5,0.6,0.7,0.8,0.9,1.0],"colsample_bytree":[0.4,0.5,0.6,0.7,0.8,0.9,1.0],
            "reg_alpha":[0.0,0.01,0.05,0.1,0.5,1.0],"reg_lambda":[0.0,0.5,1.0,2.0,5.0]}

def _bucket_space(fast=False):
    if fast: return {"n_estimators":[100,300],"learning_rate":[0.03,0.05],"num_leaves":[15,31],
                     "min_child_samples":[15,30],"subsample":[0.7,0.9],"colsample_bytree":[0.6,0.8],
                     "reg_alpha":[0.0,0.1],"reg_lambda":[1.0,2.0]}
    return {"n_estimators":[200,300,500,800],"learning_rate":[0.01,0.02,0.03,0.05,0.07],
            "num_leaves":[15,31,63,127],"min_child_samples":[5,10,15,20,30,50],
            "subsample":[0.6,0.7,0.8,0.9,1.0],"colsample_bytree":[0.5,0.6,0.7,0.8,0.9,1.0],
            "reg_alpha":[0.0,0.05,0.1,0.3,0.5],"reg_lambda":[0.5,1.0,2.0,5.0]}

def _extreme_space(fast=False):
    if fast: return {"extreme_margin":[1.0,1.5],"n_estimators":[100,300],"learning_rate":[0.03,0.05],
                     "num_leaves":[31],"subsample":[0.8],"colsample_bytree":[0.7]}
    return {"extreme_margin":[0.5,0.8,1.0,1.2,1.5,2.0],"n_estimators":[100,200,300,500],
            "learning_rate":[0.01,0.02,0.03,0.05,0.07],"num_leaves":[15,31,63],
            "subsample":[0.7,0.8,0.9,1.0],"colsample_bytree":[0.6,0.7,0.8,0.9]}

def _postproc_space(fast=False):
    if fast: return {"mag_bias_T1":[-0.2,0.0,0.2],"mag_bias_T2":[-0.2,0.0,0.2],"mag_bias_T3":[-0.2,0.0,0.2],
                     "time_bias_T1":[-1.0,0.0],"time_bias_T2":[-2.0,0.0],"time_bias_T3":[0.0,5.0],
                     "extreme_prob_threshold":[0.5],"high_risk_mag_quantile_weight":[0.5],
                     "early_time_shift_strength":[0.1],"t1_early_delta_bonus":[0.0,0.2]}
    return {"mag_bias_T1":[-0.3,-0.2,-0.1,0.0,0.1],"mag_bias_T2":[-0.1,0.0,0.1,0.2],
            "mag_bias_T3":[-0.1,0.0,0.1,0.2,0.3],"time_bias_T1":[-3.0,-2.0,-1.0,0.0],
            "time_bias_T2":[-6.0,-4.0,-2.0,0.0,2.0],"time_bias_T3":[0.0,3.0,6.0,9.0,12.0],
            "extreme_prob_threshold":[0.3,0.4,0.5,0.6,0.7],
            "high_risk_mag_quantile_weight":[0.3,0.5,0.7,0.9],
            "early_time_shift_strength":[0.0,0.1,0.2,0.35],
            "t1_early_delta_bonus":[0.0,0.1,0.2,0.3,0.5]}


# ============================================================================
# 帮助函数
# ============================================================================

def _sample(space, trial_or_rng, prefix):
    clean, prefixed = {}, {}
    if _OPTUNA_AVAILABLE and hasattr(trial_or_rng, "suggest_categorical"):
        for k, v in space.items():
            val = trial_or_rng.suggest_categorical(f"{prefix}_{k}", list(v))
            prefixed[f"{prefix}_{k}"] = val; clean[k] = val
    else:
        rng = trial_or_rng
        for k, v in space.items():
            val = rng.choice(list(v)); clean[k] = val; prefixed[f"{prefix}_{k}"] = val
    return clean, prefixed

def _rmse(e): return float(np.sqrt(np.mean(np.square(e))))
def _mae(e): return float(np.mean(np.abs(e)))

def _build_reg(params, seed, device):
    try:
        from lightgbm import LGBMRegressor
        p = {"n_estimators":int(params.get("n_estimators",500)),"learning_rate":float(params.get("learning_rate",0.03)),
             "num_leaves":int(params.get("num_leaves",63)),"min_child_samples":int(params.get("min_child_samples",20)),
             "subsample":float(params.get("subsample",0.8)),"colsample_bytree":float(params.get("colsample_bytree",0.7)),
             "reg_alpha":float(params.get("reg_alpha",0.05)),"reg_lambda":float(params.get("reg_lambda",1.0)),
             "random_state":seed,"n_jobs":-1,"verbosity":-1,"objective":"regression"}
        if device=="cuda": p["device"]="cuda"
        return LGBMRegressor(**p)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=int(p["n_estimators"]),learning_rate=float(p["learning_rate"]),
                                             max_leaf_nodes=int(p["num_leaves"]),random_state=seed)

def _build_clf(params, seed, device, n_class=4):
    try:
        from lightgbm import LGBMClassifier
        p = {"n_estimators":int(params.get("n_estimators",300)),"learning_rate":float(params.get("learning_rate",0.03)),
             "num_leaves":int(params.get("num_leaves",31)),"min_child_samples":int(params.get("min_child_samples",15)),
             "subsample":float(params.get("subsample",0.85)),"colsample_bytree":float(params.get("colsample_bytree",0.8)),
             "reg_alpha":float(params.get("reg_alpha",0.1)),"reg_lambda":float(params.get("reg_lambda",2.0)),
             "random_state":seed,"n_jobs":-1,"verbosity":-1}
        if device=="cuda": p["device"]="cuda"
        if n_class>2: p["objective"]="multiclass"; p["num_class"]=n_class
        return LGBMClassifier(**p)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=int(p["n_estimators"]),learning_rate=float(p["learning_rate"]),random_state=seed)

def _build_extreme(params, seed, device):
    try:
        from lightgbm import LGBMClassifier
        p = {"n_estimators":int(params.get("n_estimators",300)),"learning_rate":float(params.get("learning_rate",0.03)),
             "num_leaves":int(params.get("num_leaves",31)),"subsample":float(params.get("subsample",0.85)),
             "colsample_bytree":float(params.get("colsample_bytree",0.8)),"reg_lambda":2.0,
             "random_state":seed,"n_jobs":-1,"verbosity":-1,"class_weight":"balanced"}
        if device=="cuda": p["device"]="cuda"
        return LGBMClassifier(**p)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=int(p["n_estimators"]),learning_rate=float(p["learning_rate"]),random_state=seed)

# --------------- XGBoost & 独立时间回归构造器 (v2) ---------------

def _build_reg_xgb(params, seed):
    """XGBoost 震级回归器（decoupled v2 新增候选）。"""
    try:
        from xgboost import XGBRegressor
        p = {"n_estimators":int(params.get("n_estimators",500)),"learning_rate":float(params.get("learning_rate",0.03)),
             "max_depth":int(params.get("max_depth",7)),"subsample":float(params.get("subsample",0.8)),
             "colsample_bytree":float(params.get("colsample_bytree",0.7)),"reg_alpha":float(params.get("reg_alpha",0.05)),
             "reg_lambda":float(params.get("reg_lambda",1.5)),"min_child_weight":int(params.get("min_child_weight",5)),
             "random_state":seed,"n_jobs":-1,"verbosity":0,"tree_method":"hist"}
        return XGBRegressor(objective="reg:squarederror",**p)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=int(p.get("n_estimators",500)),
                                             learning_rate=float(p.get("learning_rate",0.03)),random_state=seed)

def _build_clf_xgb(params, seed, n_class=4):
    """XGBoost 时间桶分类器（decoupled v2 新增候选）。"""
    try:
        from xgboost import XGBClassifier
        p = {"n_estimators":int(params.get("n_estimators",300)),"learning_rate":float(params.get("learning_rate",0.03)),
             "max_depth":int(params.get("max_depth",6)),"subsample":float(params.get("subsample",0.85)),
             "colsample_bytree":float(params.get("colsample_bytree",0.8)),"reg_alpha":float(params.get("reg_alpha",0.1)),
             "reg_lambda":float(params.get("reg_lambda",2.0)),"random_state":seed,"n_jobs":-1,"verbosity":0}
        if n_class>2: p["objective"]="multi:softprob"; p["num_class"]=n_class
        return XGBClassifier(**p)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=int(p.get("n_estimators",300)),
                                              learning_rate=float(p.get("learning_rate",0.03)),random_state=seed)

def _build_time_reg_lgbm(params, seed, device):
    """LightGBM 独立时间回归器（decoupled v2: 直接回归 log-hours，与 bucket 分类形成异构候选）。"""
    try:
        from lightgbm import LGBMRegressor
        p = {"n_estimators":int(params.get("n_estimators",500)),"learning_rate":float(params.get("learning_rate",0.03)),
             "num_leaves":int(params.get("num_leaves",63)),"min_child_samples":int(params.get("min_child_samples",20)),
             "subsample":float(params.get("subsample",0.8)),"colsample_bytree":float(params.get("colsample_bytree",0.7)),
             "reg_alpha":float(params.get("reg_alpha",0.1)),"reg_lambda":float(params.get("reg_lambda",2.0)),
             "random_state":seed,"n_jobs":-1,"verbosity":-1,"objective":"regression"}
        if device=="cuda": p["device"]="cuda"
        return LGBMRegressor(**p)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=int(p["n_estimators"]),learning_rate=float(p["learning_rate"]),random_state=seed)

def _build_time_reg_xgb(params, seed):
    """XGBoost 独立时间回归器（decoupled v2 新增候选）。"""
    try:
        from xgboost import XGBRegressor
        p = {"n_estimators":int(params.get("n_estimators",500)),"learning_rate":float(params.get("learning_rate",0.03)),
             "max_depth":int(params.get("max_depth",6)),"subsample":float(params.get("subsample",0.8)),
             "colsample_bytree":float(params.get("colsample_bytree",0.7)),"reg_alpha":float(params.get("reg_alpha",0.1)),
             "reg_lambda":float(params.get("reg_lambda",2.0)),"min_child_weight":int(params.get("min_child_weight",5)),
             "random_state":seed,"n_jobs":-1,"verbosity":0,"tree_method":"hist"}
        return XGBRegressor(objective="reg:squarederror",**p)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=int(p.get("n_estimators",500)),
                                             learning_rate=float(p.get("learning_rate",0.03)),random_state=seed)

def _postprocess(mag, time, ep, mm, window, pp):
    from src.qualification import WINDOW_BY_NAME
    w = WINDOW_BY_NAME[window.name if hasattr(window,'name') else str(window)]
    ma, ta = mag.copy(), time.copy()
    thr = float(pp.get("extreme_prob_threshold",0.5))
    high = ep >= thr
    if not high.any(): return ma, ta
    uplift = float(pp.get("high_risk_mag_quantile_weight",0.5))
    margin = float(pp.get("extreme_margin",1.2))
    shift = float(pp.get("early_time_shift_strength",0.1))
    floor = np.clip(mm[high]-margin,0.0,None)
    ma[high] = (1-uplift)*mag[high]+uplift*np.maximum(mag[high],floor)
    anchor = w.lower_hours+(w.upper_hours-w.lower_hours)*0.3
    ta[high] = (1-shift)*time[high]+shift*np.minimum(time[high],anchor)
    ta[high] = np.clip(ta[high],w.lower_hours+1e-6,w.upper_hours)
    if w.name=="T1":
        bonus = float(pp.get("t1_early_delta_bonus",0.0))
        if bonus>0: ma[high] = np.clip(ma[high]+bonus,0.0,None)
    return np.clip(ma,0.0,None), ta


# ============================================================================
# Per-target OOF evaluation
# ============================================================================

def _oof_mag_only(df, wn, params, n_splits, seed, device, pp):
    """震级 OOF (v2)：返回 {mag_lgbm_raw, mag_lgbm_pp, mag_xgb_raw, mag_xgb_pp, ym, ef}。
    
    同时训练 LGBM 和 XGBoost 两个候选，使 OOF 融合不再只有单列权重 1.0。
    """
    window = next(w for w in QUALIFICATION_WINDOWS if w.name==wn)
    ldf = reconstruct_legal_window_features(df, wn); ldf[TIME_COL]=df[TIME_COL]
    ldf = ldf.dropna(subset=[TIME_COL,window.mag_col,window.time_col]).reset_index(drop=True)
    fcs = select_feature_columns(ldf)
    if not fcs: return None
    ym = ldf[window.mag_col].to_numpy(dtype=float)
    mm = ldf["mainshock_mag"].to_numpy(dtype=float)
    em = float(params.get("extreme_margin",1.2))
    ef = (ym>0.0)&(ym>=mm-em)
    spl = TimeSeriesSplit(n_splits=n_splits); X=ldf[fcs]
    oof_lgbm = np.full(len(ldf),np.nan); oof_xgb = np.full(len(ldf),np.nan)
    with warnings.catch_warnings(): warnings.simplefilter("ignore")
    for ti,vi in spl.split(X):
        # LGBM
        m_lgbm = _build_reg(params, seed, device); m_lgbm.fit(X.iloc[ti],ym[ti])
        oof_lgbm[vi] = np.clip(np.asarray(m_lgbm.predict(X.iloc[vi]),dtype=float),0.0,None)
        # XGBoost
        m_xgb = _build_reg_xgb(params, seed); m_xgb.fit(X.iloc[ti],ym[ti])
        oof_xgb[vi] = np.clip(np.asarray(m_xgb.predict(X.iloc[vi]),dtype=float),0.0,None)
    mb = float(pp.get(f"mag_bias_{wn}",0.0))
    return {"mag_lgbm_raw": oof_lgbm, "mag_lgbm_pp": oof_lgbm+mb,
            "mag_xgb_raw": oof_xgb, "mag_xgb_pp": oof_xgb+mb,
            "ym": ym, "ef": ef}

def _oof_time_direct_reg(df, wn, params, n_splits, seed, device, pp):
    """独立时间回归 OOF（decoupled v2）：直接回归 log-hours，与 bucket 分类形成异构候选。
    
    返回 {time_direct_raw, time_direct_pp, yt} 或 None。
    """
    window = next(w for w in QUALIFICATION_WINDOWS if w.name==wn)
    ldf = reconstruct_legal_window_features(df, wn); ldf[TIME_COL]=df[TIME_COL]
    ldf = ldf.dropna(subset=[TIME_COL,window.mag_col,window.time_col]).reset_index(drop=True)
    fcs = select_feature_columns(ldf)
    if not fcs: return None
    yt = ldf[window.time_col].to_numpy(dtype=float)
    # 回归 log-time，避免大值主导 loss
    yt_log = np.log1p(np.clip(yt,1e-6,None))
    spl = TimeSeriesSplit(n_splits=n_splits); X=ldf[fcs]
    oof_lgbm = np.full(len(ldf),np.nan); oof_xgb = np.full(len(ldf),np.nan)
    with warnings.catch_warnings(): warnings.simplefilter("ignore")
    for ti,vi in spl.split(X):
        m_lgbm = _build_time_reg_lgbm(params, seed, device); m_lgbm.fit(X.iloc[ti],yt_log[ti])
        oof_lgbm[vi] = np.expm1(np.asarray(m_lgbm.predict(X.iloc[vi]),dtype=float))
        m_xgb = _build_time_reg_xgb(params, seed); m_xgb.fit(X.iloc[ti],yt_log[ti])
        oof_xgb[vi] = np.expm1(np.asarray(m_xgb.predict(X.iloc[vi]),dtype=float))
    oof_lgbm = np.clip(oof_lgbm, window.lower_hours+1e-6, window.upper_hours)
    oof_xgb  = np.clip(oof_xgb,  window.lower_hours+1e-6, window.upper_hours)
    tb = float(pp.get(f"time_bias_{wn}",0.0))
    return {"time_direct_lgbm_raw": oof_lgbm, "time_direct_lgbm_pp": np.clip(oof_lgbm+tb,window.lower_hours+1e-6,window.upper_hours),
            "time_direct_xgb_raw": oof_xgb, "time_direct_xgb_pp": np.clip(oof_xgb+tb,window.lower_hours+1e-6,window.upper_hours),
            "yt": yt}


def _oof_time_only(df, wn, params, n_splits, seed, device, pp):
    """时间 OOF (v2)：返回 {time_bucket_expected_raw, time_bucket_expected_pp, time_direct_lgbm_raw, ..., yt, yb, bprobs}。
    
    除原有的时间桶分类→期望时间外，新增独立时间回归候选，使 OOF 融合有 2+ 异构列。
    """
    window = next(w for w in QUALIFICATION_WINDOWS if w.name==wn)
    ldf = reconstruct_legal_window_features(df, wn); ldf[TIME_COL]=df[TIME_COL]
    ldf = ldf.dropna(subset=[TIME_COL,window.mag_col,window.time_col]).reset_index(drop=True)
    fcs = select_feature_columns(ldf)
    if not fcs: return None
    yt = ldf[window.time_col].to_numpy(dtype=float)
    yb = assign_time_buckets_batch(wn, yt)
    spl = TimeSeriesSplit(n_splits=n_splits); X=ldf[fcs]
    oof_t = np.full(len(ldf),np.nan); oof_bp = np.full((len(ldf),4),np.nan)
    with warnings.catch_warnings(): warnings.simplefilter("ignore")
    for ti,vi in spl.split(X):
        m = _build_clf(params, seed, device, 4); m.fit(X.iloc[ti],yb[ti])
        bp = align_bucket_probabilities(m, np.asarray(m.predict_proba(X.iloc[vi]),dtype=float))
        oof_t[vi] = expected_time_from_bucket_probs(wn, bp)
        oof_bp[vi] = bp
    tb = float(pp.get(f"time_bias_{wn}",0.0))
    result = {"time_bucket_expected_raw": oof_t,
              "time_bucket_expected_pp": np.clip(oof_t+tb,window.lower_hours+1e-6,window.upper_hours),
              "yt": yt, "yb": yb, "bprobs": oof_bp}

    # 附加直接时间回归候选
    direct = _oof_time_direct_reg(df, wn, params, n_splits, seed, device, pp)
    if direct is not None:
        for k in ("time_direct_lgbm_raw","time_direct_lgbm_pp","time_direct_xgb_raw","time_direct_xgb_pp"):
            if k in direct:
                result[k] = direct[k]

    return result

def _oof_extreme_only(df, wn, params, n_splits, seed, device, pp):
    """极端风险 OOF：返回 extreme_prob, true_extreme"""
    window = next(w for w in QUALIFICATION_WINDOWS if w.name==wn)
    ldf = reconstruct_legal_window_features(df, wn); ldf[TIME_COL]=df[TIME_COL]
    ldf = ldf.dropna(subset=[TIME_COL,window.mag_col,window.time_col]).reset_index(drop=True)
    fcs = select_feature_columns(ldf)
    if not fcs: return None
    ym = ldf[window.mag_col].to_numpy(dtype=float)
    mm = ldf["mainshock_mag"].to_numpy(dtype=float)
    em = float(params.get("extreme_margin",1.2))
    ef = (ym>0.0)&(ym>=mm-em); ye = ef.astype(int)
    spl = TimeSeriesSplit(n_splits=n_splits); X=ldf[fcs]
    oof = np.full(len(ldf),np.nan)
    with warnings.catch_warnings(): warnings.simplefilter("ignore")
    for ti,vi in spl.split(X):
        if ye[ti].sum()>0 and len(np.unique(ye[ti]))>1:
            m = _build_extreme(params, seed, device); m.fit(X.iloc[ti],ye[ti])
            oof[vi] = safe_extreme_probability(m, X.iloc[vi])
        else: oof[vi] = 0.0
    return oof, ef


# ============================================================================
# Per-target scoring
# ============================================================================

def _mag_score(ym, oof_pp, ef, args):
    e = oof_pp - ym
    ema = float(np.mean(np.abs(e[ef]))) if ef.any() else 0.0
    return args.w_mag_mae * _mae(e) + args.w_mag_rmse * _rmse(e) + args.w_extreme * ema

def _time_score(yt, oof_pp, args):
    e = oof_pp - yt
    lw = np.where(e>0,2.0,1.0)
    late_pen = float(np.sqrt(np.mean(lw*np.square(e)))) - _rmse(e)
    return (args.w_time_mae*_mae(e)/24.0 + args.w_time_rmse*_rmse(e)/24.0
            + args.w_late*max(0.0,late_pen))


# ============================================================================
# Single-target single-window tuning study
# ============================================================================

def _tune_one(study_name, space, target_type, wn, df, n_splits, seed, device, n_trials, pp, args):
    """返回 (best_params_dict, best_score)"""
    best_score = float("inf"); best_p = {}
    if not _OPTUNA_AVAILABLE:
        rng = np.random.RandomState(seed)
        for i in range(n_trials):
            p, _ = _sample(space, rng, target_type[:4])
            score = _eval_one(p, target_type, wn, df, n_splits, seed, device, pp, args)
            if score < best_score: best_score = score; best_p = p
        return best_p, best_score

    import optuna
    study = optuna.create_study(study_name=study_name, direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    def fn(trial):
        nonlocal best_score, best_p
        p, _ = _sample(space, trial, target_type[:4])
        s = _eval_one(p, target_type, wn, df, n_splits, seed, device, pp, args)
        if s < best_score: best_score = s; best_p = p
        return s
    study.optimize(fn, n_trials=n_trials, show_progress_bar=True)
    return best_p, best_score


def _eval_one(p, target_type, wn, df, n_splits, seed, device, pp, args):
    """评估一组参数，返回 score。v2 适配 dict 格式 OOF。"""
    if target_type == "mag":
        r = _oof_mag_only(df, wn, p, n_splits, seed, device, pp)
        if r is None: return 1e9
        # 使用 LGBM postprocessed 作为主评估（保持向后兼容）
        mag_pp = r.get("mag_lgbm_pp"); ym = r["ym"]; ef = r["ef"]
        v = np.isfinite(mag_pp)
        return _mag_score(ym[v], mag_pp[v], ef[v], args) if v.any() else 1e9
    elif target_type == "time":
        r = _oof_time_only(df, wn, p, n_splits, seed, device, pp)
        if r is None: return 1e9
        # 使用 bucket expected postprocessed 作为主评估
        time_pp = r.get("time_bucket_expected_pp"); yt = r["yt"]
        v = np.isfinite(time_pp)
        return _time_score(yt[v], time_pp[v], args) if v.any() else 1e9
    elif target_type == "extreme":
        r = _oof_extreme_only(df, wn, p, n_splits, seed, device, pp)
        if r is None: return 1e9
        ep, ef = r
        v = np.isfinite(ep) & np.isfinite(ef)
        if not v.any(): return 1e9
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(ef[v].astype(int), ep[v]) if len(np.unique(ef[v]))>1 else 0.5
        except: auc = 0.5
        recall = float(np.mean(ep[v][ef[v]]>=0.5)) if ef[v].any() else 1.0
        return 0.6*(1-recall)+0.4*(1-auc)
    return 1e9


# ============================================================================
# DL/GNN placeholder (非阻断)
# ============================================================================

def _try_dl_oof(df, wn, target_type):
    """尝试用 Transformer 生成 OOF。失败返回 None。"""
    try:
        print(f"  [DL] attempting {target_type} OOF for {wn}...")
        # 实际工程中调用 train_dl.py 的 OOF 入口，此处为轻量 placeholder
        return None
    except Exception as e:
        print(f"  [DL] failed (non-blocking): {e}")
        return None

def _try_gnn_oof(df, wn, target_type):
    """尝试用 GNN 生成 OOF。失败返回 None。"""
    try:
        print(f"  [GNN] attempting {target_type} OOF for {wn}...")
        return None
    except Exception as e:
        print(f"  [GNN] failed (non-blocking): {e}")
        return None


# ============================================================================
# 主流程
# ============================================================================

def main():
    args = parse_args()
    # 自动检测 LightGBM CUDA 是否可用，不可用则回退 cpu
    if args.device in ("cuda", "auto") and not _LGBM_CUDA_OK:
        print("[WARN] LightGBM CUDA not available; falling back to cpu device.")
        args.device = "cpu"
    dp = Path(args.data) if Path(args.data).is_absolute() else PROJECT_ROOT/args.data
    if args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT/"data"/"tuning_results"/f"decoupled_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    windows = [w.strip() for w in args.windows.split(",") if w.strip()]
    separate = args.separate_target_tuning
    target = args.target
    fast = args.fast

    df = pd.read_csv(dp)
    miss = [c for c in qualification_target_cols() if c not in df.columns]
    if miss: raise ValueError(f"Missing: {miss}")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce", format="mixed")
    df = add_derived_features(df)
    df = df.dropna(subset=[TIME_COL,*qualification_target_cols()]).reset_index(drop=True).sort_values(TIME_COL).reset_index(drop=True)

    # 默认后处理（不参与 target-specific study 时可以共用一组）
    pp_default = {}
    pp_space = _postproc_space(fast)
    rng = np.random.RandomState(args.seed)
    for k, vs in pp_space.items(): pp_default[k] = rng.choice(list(vs))
    
    # 确定 trial 数
    n_mag = args.mag_trials; n_time = args.time_trials; n_ext = args.extreme_trials
    if args.n_trials is not None: n_mag = n_time = n_ext = args.n_trials

    # 收集所有最佳参数 (v2: 多候选 OOF 融合，含 XGBoost + 独立时间回归)
    best_params = {"windows": {}, "global": {"seed": args.seed, "separate_target_tuning": separate,
                   "oof_fusion": args.enable_oof_fusion, "dl_enabled": args.tune_transformer,
                   "gnn_enabled": args.tune_gnn, "version": "decoupled_v2"}}
    fusion_weights = {}
    all_oof_rows = []

    for wn in windows:
        wp = {}
        fw_wn = {}
        print(f"\n{'='*60}\n  Window {wn}\n{'='*60}")

        # --- Mag tuning (LGBM + XGB 双候选) ---
        if target in ("all", "mag"):
            print(f"\n  [{wn}] Tuning MAG model ({n_mag} trials)...")
            mp, ms = _tune_one(f"{wn}_mag", _mag_space(fast), "mag", wn, df,
                               args.n_splits, args.seed, args.device, n_mag, pp_default, args)
            wp["mag_model"] = mp
            r = _oof_mag_only(df, wn, mp, args.n_splits, args.seed, args.device, pp_default)
            if r:
                mag_rows = pd.DataFrame({
                    "window": wn,
                    "oof_mag_lgbm": r["mag_lgbm_raw"], "oof_mag_xgb": r["mag_xgb_raw"],
                    "oof_mag_lgbm_postprocessed": r["mag_lgbm_pp"],
                    "oof_mag_xgb_postprocessed": r["mag_xgb_pp"],
                    "true_mag": r["ym"], "extreme_flag": r["ef"],
                    "mainshock_id": np.arange(len(r["ym"])),
                })
                all_oof_rows.append(mag_rows)
                if args.enable_oof_fusion:
                    valid = mag_rows["true_mag"].notna()
                    pred_cols = [c for c in mag_rows.columns if c.startswith("oof_mag_") and not c.endswith("_postprocessed")]
                    for c in pred_cols:
                        valid &= mag_rows[c].notna()
                    if valid.sum() > 5 and len(pred_cols) >= 2:
                        wts = fit_oof_fusion_weights(mag_rows[valid], "true_mag", pred_cols, "rmse", args.fusion_grid_step)
                        fw_wn["mag"] = wts
                        print(f"  [{wn}] MAG fusion weights: {wts}")
                for key in ("mag_lgbm","mag_xgb"):
                    col = f"oof_{key}"
                    if col in mag_rows.columns:
                        v = np.isfinite(mag_rows[col]) & np.isfinite(mag_rows["true_mag"])
                        if v.any():
                            e = mag_rows.loc[v,col].to_numpy()-mag_rows.loc[v,"true_mag"].to_numpy()
                            print(f"  [{wn}] {key}: mae={_mae(e):.4f} rmse={_rmse(e):.4f}")
            print(f"  [{wn}] MAG best score: {ms:.4f}")

        # --- Time tuning (bucket 分类 + 直接回归 异构双候选) ---
        if target in ("all", "time"):
            print(f"\n  [{wn}] Tuning TIME model ({n_time} trials)...")
            tp, ts = _tune_one(f"{wn}_time", _bucket_space(fast), "time", wn, df,
                               args.n_splits, args.seed, args.device, n_time, pp_default, args)
            wp["time_model"] = tp
            r = _oof_time_only(df, wn, tp, args.n_splits, args.seed, args.device, pp_default)
            if r:
                time_cols = {"window": wn, "true_time": r["yt"], "true_bucket": r["yb"],
                             "mainshock_id": np.arange(len(r["yt"]))}
                if "time_bucket_expected_raw" in r:
                    time_cols["oof_time_bucket_raw"] = r["time_bucket_expected_raw"]
                    time_cols["oof_time_bucket_postprocessed"] = r.get("time_bucket_expected_pp", r["time_bucket_expected_raw"])
                for dk in ("time_direct_lgbm_raw","time_direct_lgbm_pp","time_direct_xgb_raw","time_direct_xgb_pp"):
                    if dk in r and r[dk] is not None:
                        time_cols[f"oof_{dk}"] = r[dk]
                if "bprobs" in r and r["bprobs"] is not None:
                    for i in range(4):
                        time_cols[f"bucket_prob_{i}"] = r["bprobs"][:, i]
                time_rows = pd.DataFrame(time_cols)
                all_oof_rows.append(time_rows)
                if args.enable_oof_fusion:
                    valid = time_rows["true_time"].notna()
                    pred_cols = [c for c in time_rows.columns if c.startswith("oof_time_") and not c.endswith("_postprocessed")]
                    for c in pred_cols:
                        valid &= time_rows[c].notna()
                    if valid.sum() > 5 and len(pred_cols) >= 2:
                        wts = fit_oof_fusion_weights(time_rows[valid], "true_time", pred_cols, "rmse", args.fusion_grid_step)
                        fw_wn["time"] = wts
                        print(f"  [{wn}] TIME fusion weights: {wts}")
                for key in ("time_bucket","time_direct_lgbm","time_direct_xgb"):
                    raw_col = f"oof_{key}_raw"
                    if raw_col in time_rows.columns:
                        v = np.isfinite(time_rows[raw_col]) & np.isfinite(time_rows["true_time"])
                        if v.any():
                            e = time_rows.loc[v,raw_col].to_numpy()-time_rows.loc[v,"true_time"].to_numpy()
                            print(f"  [{wn}] {key}: mae={_mae(e):.2f}h rmse={_rmse(e):.2f}h")
            print(f"  [{wn}] TIME best score: {ts:.4f}")

        # --- Extreme tuning ---
        if target in ("all", "extreme"):
            print(f"\n  [{wn}] Tuning EXTREME model ({n_ext} trials)...")
            ep, es = _tune_one(f"{wn}_extreme", _extreme_space(fast), "extreme", wn, df,
                               args.n_splits, args.seed, args.device, n_ext, pp_default, args)
            wp["extreme_model"] = ep
            r = _oof_extreme_only(df, wn, ep, args.n_splits, args.seed, args.device, pp_default)
            if r:
                eprob, ef = r
                ext_rows = pd.DataFrame({"window": wn, "oof_extreme_prob_lgbm": eprob,
                                         "true_extreme": ef, "mainshock_id": np.arange(len(ef))})
                all_oof_rows.append(ext_rows)
                if args.enable_oof_fusion:
                    valid = np.isfinite(eprob)
                    if valid.sum()>5:
                        pred_cols = [c for c in ext_rows.columns if c.startswith("oof_extreme_prob_")]
                        if pred_cols:
                            wts = fit_oof_fusion_weights(ext_rows[valid], "true_extreme", pred_cols, "rmse", args.fusion_grid_step)
                            fw_wn["extreme"] = wts
            print(f"  [{wn}] EXTREME best score: {es:.4f}")

        # --- DL/GNN hooks (placeholder; 不声称真实参与) ---
        dl_enabled = args.tune_transformer; gnn_enabled = args.tune_gnn
        if dl_enabled and target in ("all", "mag"):
            _try_dl_oof(df, wn, "mag")
        if gnn_enabled and target in ("all", "mag"):
            _try_gnn_oof(df, wn, "mag")

        wp["postprocessing"] = pp_default
        if fw_wn: wp["fusion"] = fw_wn; fusion_weights[wn] = fw_wn
        best_params["windows"][wn] = wp

    # ========================================================================
    # 后处理调优 (v2: 加入 T2/T3 时间 MAE 约束，防止时间预测退化)
    # ========================================================================
    print(f"\n{'='*60}\n  Tuning postprocessing params (v2 with T2/T3 time-MAE guard)\n{'='*60}")
    pp_best = {}
    pp_best_score = float("inf")
    pp_best_t23_mae = float("inf")
    rng = np.random.RandomState(args.seed)
    n_pp_trials = min(40, max(n_mag, n_time, n_ext))
    # 计算 T2/T3 baseline 时间 MAE
    t23_baseline_mae = {}
    for wn in windows:
        if wn in ("T2", "T3"):
            rmt = _oof_time_only(df, wn, best_params["windows"].get(wn,{}).get("time_model",{}),
                                 args.n_splits, args.seed, args.device, pp_default)
            if rmt:
                tp_raw = rmt.get("time_bucket_expected_raw")
                if tp_raw is not None:
                    vv = np.isfinite(tp_raw) & np.isfinite(rmt["yt"])
                    if vv.any():
                        t23_baseline_mae[wn] = _mae(tp_raw[vv]-rmt["yt"][vv])
    for _ in range(n_pp_trials):
        cand = {}
        for k, vs in pp_space.items(): cand[k] = rng.choice(list(vs))
        score = 0; cur_t23_mae = 0
        for wn in windows:
            wp = best_params["windows"].get(wn, {})
            mp = wp.get("mag_model", {}); tp = wp.get("time_model", {})
            r_m = _oof_mag_only(df, wn, mp, args.n_splits, args.seed, args.device, cand)
            r_t = _oof_time_only(df, wn, tp, args.n_splits, args.seed, args.device, cand)
            if r_m and r_t:
                mag_pp = r_m.get("mag_lgbm_pp"); ym = r_m["ym"]; ef = r_m["ef"]
                time_pp = r_t.get("time_bucket_expected_pp"); yt = r_t["yt"]
                v = np.isfinite(mag_pp) & np.isfinite(time_pp)
                if v.any():
                    score += _mag_score(ym[v], mag_pp[v], ef[v], args) + _time_score(yt[v], time_pp[v], args)
                    if wn in ("T2", "T3"):
                        cur_t23_mae += _mae(time_pp[v]-yt[v])
        # T2/T3 时间 MAE 约束：不超过 baseline 的 1.3 倍
        violation = False
        for wn in ("T2", "T3"):
            if wn in t23_baseline_mae and wn in windows:
                bl = t23_baseline_mae[wn]
                if bl > 0 and cur_t23_mae > bl * 1.3:
                    violation = True; break
        if not violation and score < pp_best_score:
            pp_best_score = score; pp_best = cand; pp_best_t23_mae = cur_t23_mae

    best_params["postprocessing"] = pp_best
    best_params["global"]["pp_best_score"] = pp_best_score
    best_params["global"]["t23_time_mae_guard"] = {"enabled": True, "max_ratio": 1.3,
                                                     "baseline_mae": t23_baseline_mae,
                                                     "selected_mae": pp_best_t23_mae}

    # 保存
    (out/"best_params.json").write_text(json.dumps(best_params, indent=2, ensure_ascii=False), encoding="utf-8")
    if fusion_weights:
        (out/"fusion_weights.json").write_text(json.dumps(fusion_weights, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "windows": windows,
               "separate_target_tuning": separate, "oof_fusion": args.enable_oof_fusion,
               "dl_enabled": args.tune_transformer, "gnn_enabled": args.tune_gnn,
               "per_target_trials": {"mag": n_mag, "time": n_time, "extreme": n_ext},
               "pp_best_score": pp_best_score,
               "version": "decoupled_v2", "candidates": {"mag": ["lgbm","xgb"], "time": ["bucket","direct_lgbm","direct_xgb"]}}
    (out/"tuning_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if all_oof_rows:
        pd.concat(all_oof_rows, axis=1).loc[:, ~pd.concat(all_oof_rows, axis=1).columns.duplicated()].to_csv(
            out/"best_oof_predictions.csv", index=False)

    print(f"\nDone. All results: {out}")
    print("  best_params.json, fusion_weights.json, tuning_summary.json")


if __name__ == "__main__":
    main()
