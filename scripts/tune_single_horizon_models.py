"""Single-horizon H168 调优脚本 v1。

每个 target（mag / time / extreme）独立调优，支持：
- LightGBM + XGBoost 双候选
- OOF 融合权重搜索
- fast smoke 模式
- 默认不使用余震观测特征
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
    H168_WINDOW, H168_MAG_COL, H168_TIME_COL, H168_FLAG_COL,
    derive_h168_labels, reconstruct_h168_features,
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
    from lightgbm import LGBMRegressor as _LGBMTest
    _t = _LGBMTest(device="cuda", n_estimators=1, num_leaves=2, verbose=-1)
    _t.fit(np.array([[0.0]]), np.array([0.0]))
    _LGBM_CUDA_OK = True
except Exception:
    pass


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    ap = argparse.ArgumentParser(description="Single-horizon H168 hyperparameter tuning.")
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT/"data"/"processed"/"qualification_features.csv")
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["auto","cuda","cpu"], default="cuda")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--target", choices=["all","mag","time","extreme"], default="all")
    ap.add_argument("--enable-oof-fusion", type=lambda x: x.lower() in ("true","1","yes"), default=True)
    ap.add_argument("--fusion-grid-step", type=float, default=0.02)
    ap.add_argument("--mag-trials", type=int, default=80)
    ap.add_argument("--time-trials", type=int, default=80)
    ap.add_argument("--extreme-trials", type=int, default=60)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--tune-transformer", action="store_true")
    ap.add_argument("--tune-gnn", action="store_true")
    ap.add_argument("--w-mag-mae", type=float, default=1.0)
    ap.add_argument("--w-mag-rmse", type=float, default=1.0)
    ap.add_argument("--w-time-mae", type=float, default=0.03)
    ap.add_argument("--w-time-rmse", type=float, default=0.03)
    ap.add_argument("--w-extreme", type=float, default=0.5)
    ap.add_argument("--w-late", type=float, default=0.3)
    ap.add_argument("--w-t1-bonus", type=float, default=0.2)
    ap.add_argument("--n-trials", type=int, default=None)
    return ap.parse_args()


# ============================================================================
# 搜索空间
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

def _build_reg_xgb(params, seed):
    try:
        from xgboost import XGBRegressor
        p = {"n_estimators":int(params.get("n_estimators",500)),"learning_rate":float(params.get("learning_rate",0.03)),
             "max_depth":7,"subsample":float(params.get("subsample",0.8)),
             "colsample_bytree":float(params.get("colsample_bytree",0.7)),"reg_alpha":float(params.get("reg_alpha",0.05)),
             "reg_lambda":float(params.get("reg_lambda",1.5)),"min_child_weight":5,
             "random_state":seed,"n_jobs":-1,"verbosity":0,"tree_method":"hist"}
        return XGBRegressor(objective="reg:squarederror",**p)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=int(p.get("n_estimators",500)),
                                             learning_rate=float(p.get("learning_rate",0.03)),random_state=seed)

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

def _build_time_reg_lgbm(params, seed, device):
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
    try:
        from xgboost import XGBRegressor
        p = {"n_estimators":int(params.get("n_estimators",500)),"learning_rate":float(params.get("learning_rate",0.03)),
             "max_depth":6,"subsample":float(params.get("subsample",0.8)),
             "colsample_bytree":float(params.get("colsample_bytree",0.7)),"reg_alpha":float(params.get("reg_alpha",0.1)),
             "reg_lambda":float(params.get("reg_lambda",2.0)),"min_child_weight":5,
             "random_state":seed,"n_jobs":-1,"verbosity":0,"tree_method":"hist"}
        return XGBRegressor(objective="reg:squarederror",**p)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=int(p.get("n_estimators",500)),
                                             learning_rate=float(p.get("learning_rate",0.03)),random_state=seed)


# ============================================================================
# Per-target OOF
# ============================================================================

WNAME = "H168"
WIN = H168_WINDOW

def _prepare_data(df):
    """统一数据准备：推导 H168 标签 + 合法特征重建。"""
    df2 = derive_h168_labels(df)
    ldf = reconstruct_h168_features(df2)
    ldf[TIME_COL] = df[TIME_COL]
    ldf = ldf.dropna(subset=[TIME_COL, H168_MAG_COL, H168_TIME_COL]).reset_index(drop=True)
    return ldf


def _oof_mag(df, params, n_splits, seed, device):
    ldf = _prepare_data(df)
    fcs = select_feature_columns(ldf)
    if not fcs: return None
    ym = ldf[H168_MAG_COL].to_numpy(dtype=float)
    mm = ldf["mainshock_mag"].to_numpy(dtype=float)
    em = float(params.get("extreme_margin",1.2))
    ef = (ym>0.0)&(ym>=mm-em)
    spl = TimeSeriesSplit(n_splits=n_splits); X=ldf[fcs]
    oof_lgbm = np.full(len(ldf),np.nan); oof_xgb = np.full(len(ldf),np.nan)
    with warnings.catch_warnings(): warnings.simplefilter("ignore")
    for ti,vi in spl.split(X):
        m_lgbm = _build_reg(params, seed, device); m_lgbm.fit(X.iloc[ti],ym[ti])
        oof_lgbm[vi] = np.clip(np.asarray(m_lgbm.predict(X.iloc[vi]),dtype=float),0.0,None)
        m_xgb = _build_reg_xgb(params, seed); m_xgb.fit(X.iloc[ti],ym[ti])
        oof_xgb[vi] = np.clip(np.asarray(m_xgb.predict(X.iloc[vi]),dtype=float),0.0,None)
    return {"mag_lgbm_raw": oof_lgbm, "mag_xgb_raw": oof_xgb,
            "ym": ym, "ef": ef}


def _oof_time_direct_reg(df, params, n_splits, seed, device):
    ldf = _prepare_data(df)
    fcs = select_feature_columns(ldf)
    if not fcs: return None
    yt = ldf[H168_TIME_COL].to_numpy(dtype=float)
    yt_log = np.log1p(np.clip(yt,1e-6,None))
    spl = TimeSeriesSplit(n_splits=n_splits); X=ldf[fcs]
    oof_lgbm = np.full(len(ldf),np.nan); oof_xgb = np.full(len(ldf),np.nan)
    with warnings.catch_warnings(): warnings.simplefilter("ignore")
    for ti,vi in spl.split(X):
        m_lgbm = _build_time_reg_lgbm(params, seed, device); m_lgbm.fit(X.iloc[ti],yt_log[ti])
        oof_lgbm[vi] = np.clip(np.expm1(np.asarray(m_lgbm.predict(X.iloc[vi]),dtype=float)),
                               WIN.lower_hours+1e-6, WIN.upper_hours)
        m_xgb = _build_time_reg_xgb(params, seed); m_xgb.fit(X.iloc[ti],yt_log[ti])
        oof_xgb[vi] = np.clip(np.expm1(np.asarray(m_xgb.predict(X.iloc[vi]),dtype=float)),
                              WIN.lower_hours+1e-6, WIN.upper_hours)
    return {"time_direct_lgbm_raw": oof_lgbm, "time_direct_xgb_raw": oof_xgb, "yt": yt}


def _oof_time_bucket(df, params, n_splits, seed, device):
    ldf = _prepare_data(df)
    fcs = select_feature_columns(ldf)
    if not fcs: return None
    yt = ldf[H168_TIME_COL].to_numpy(dtype=float)
    yb = assign_time_buckets_batch(WNAME, yt)
    spl = TimeSeriesSplit(n_splits=n_splits); X=ldf[fcs]
    oof_t = np.full(len(ldf),np.nan); oof_bp = np.full((len(ldf),4),np.nan)
    with warnings.catch_warnings(): warnings.simplefilter("ignore")
    for ti,vi in spl.split(X):
        m = _build_clf(params, seed, device, 4); m.fit(X.iloc[ti],yb[ti])
        bp = align_bucket_probabilities(m, np.asarray(m.predict_proba(X.iloc[vi]),dtype=float))
        oof_t[vi] = expected_time_from_bucket_probs(WNAME, bp)
        oof_bp[vi] = bp
    return {"time_bucket_raw": oof_t, "yt": yt, "yb": yb, "bprobs": oof_bp}


def _oof_extreme(df, params, n_splits, seed, device):
    ldf = _prepare_data(df)
    fcs = select_feature_columns(ldf)
    if not fcs: return None
    ym = ldf[H168_MAG_COL].to_numpy(dtype=float)
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
# Scoring
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
# Tuning
# ============================================================================

def _tune_one(study_name, space, target_type, df, n_splits, seed, device, n_trials, args):
    best_score = float("inf"); best_p = {}
    if not _OPTUNA_AVAILABLE:
        rng = np.random.RandomState(seed)
        for i in range(n_trials):
            p, _ = _sample(space, rng, target_type[:4])
            score = _eval_one(p, target_type, df, n_splits, seed, device, args)
            if score < best_score: best_score = score; best_p = p
        return best_p, best_score

    import optuna
    study = optuna.create_study(study_name=study_name, direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    def fn(trial):
        nonlocal best_score, best_p
        p, _ = _sample(space, trial, target_type[:4])
        s = _eval_one(p, target_type, df, n_splits, seed, device, args)
        if s < best_score: best_score = s; best_p = p
        return s
    study.optimize(fn, n_trials=n_trials, show_progress_bar=True)
    return best_p, best_score


def _eval_one(p, target_type, df, n_splits, seed, device, args):
    if target_type == "mag":
        r = _oof_mag(df, p, n_splits, seed, device)
        if r is None: return 1e9
        mag_pp = r["mag_lgbm_raw"]; ym = r["ym"]; ef = r["ef"]
        v = np.isfinite(mag_pp)
        return _mag_score(ym[v], mag_pp[v], ef[v], args) if v.any() else 1e9
    elif target_type == "time":
        r = _oof_time_bucket(df, p, n_splits, seed, device)
        if r is None: return 1e9
        time_pp = r["time_bucket_raw"]; yt = r["yt"]
        v = np.isfinite(time_pp)
        return _time_score(yt[v], time_pp[v], args) if v.any() else 1e9
    elif target_type == "extreme":
        oof, ef = _oof_extreme(df, p, n_splits, seed, device)
        if oof is None: return 1e9
        v = np.isfinite(oof) & np.isfinite(ef)
        if not v.any(): return 1e9
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(ef[v].astype(int), oof[v]) if len(np.unique(ef[v]))>1 else 0.5
        except: auc = 0.5
        recall = float(np.mean(oof[v][ef[v]]>=0.5)) if ef[v].any() else 1.0
        return 0.6*(1-recall)+0.4*(1-auc)
    return 1e9


# ============================================================================
# DL/GNN placeholder (不阻断)
# ============================================================================

def _try_dl_oof(df, target_type):
    try:
        print(f"  [DL] attempting {target_type} OOF for H168...")
        return None
    except Exception as e:
        print(f"  [DL] failed (non-blocking): {e}")
        return None

def _try_gnn_oof(df, target_type):
    try:
        print(f"  [GNN] attempting {target_type} OOF for H168...")
        return None
    except Exception as e:
        print(f"  [GNN] failed (non-blocking): {e}")
        return None


# ============================================================================
# 主流程
# ============================================================================

def main():
    args = parse_args()
    if args.device in ("cuda", "auto") and not _LGBM_CUDA_OK:
        print("[WARN] LightGBM CUDA not available; falling back to cpu device.")
        args.device = "cpu"
    dp = Path(args.data) if Path(args.data).is_absolute() else PROJECT_ROOT/args.data
    if args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT/"data"/"tuning_results"/f"single_horizon_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    target = args.target; fast = args.fast

    df = pd.read_csv(dp)
    miss = [c for c in ["target_T1_max_mag","target_T2_max_mag","target_T3_max_mag"] if c not in df.columns]
    if miss: raise ValueError(f"Missing T1/T2/T3 label columns: {miss}")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce", format="mixed")
    df = add_derived_features(df)
    df = derive_h168_labels(df)
    df = df.dropna(subset=[TIME_COL, H168_MAG_COL, H168_TIME_COL]).reset_index(drop=True).sort_values(TIME_COL).reset_index(drop=True)

    n_mag = args.mag_trials; n_time = args.time_trials; n_ext = args.extreme_trials
    if args.n_trials is not None: n_mag = n_time = n_ext = args.n_trials

    best_params = {"global": {"seed": args.seed, "oof_fusion": args.enable_oof_fusion,
                   "dl_enabled": args.tune_transformer, "gnn_enabled": args.tune_gnn,
                   "version": "single_horizon_v1"}}
    fusion_weights = {}; all_rows = {}

    # ── Mag ──
    if target in ("all", "mag"):
        print(f"\n[H168] Tuning MAG model ({n_mag} trials)...")
        mp, ms = _tune_one("H168_mag", _mag_space(fast), "mag", df, args.n_splits, args.seed, args.device, n_mag, args)
        best_params["mag_model"] = mp
        r = _oof_mag(df, mp, args.n_splits, args.seed, args.device)
        if r:
            all_rows["true_mag"] = r["ym"]; all_rows["extreme_flag"] = r["ef"]
            all_rows["oof_mag_lgbm"] = r["mag_lgbm_raw"]; all_rows["oof_mag_xgb"] = r["mag_xgb_raw"]
            if args.enable_oof_fusion:
                mag_df = pd.DataFrame({k: v for k, v in all_rows.items() if k.startswith(("true_mag","oof_mag_","extreme_flag"))})
                valid = mag_df["true_mag"].notna()
                pred_cols = [c for c in mag_df.columns if c.startswith("oof_mag_")]
                for c in pred_cols: valid &= mag_df[c].notna()
                if valid.sum() > 5 and len(pred_cols) >= 2:
                    wts = fit_oof_fusion_weights(mag_df[valid], "true_mag", pred_cols, "rmse", args.fusion_grid_step)
                    fusion_weights["mag"] = wts
                    print(f"  MAG fusion: {wts}")
        print(f"  MAG best score: {ms:.4f}")

    # ── Time ──
    if target in ("all", "time"):
        print(f"\n[H168] Tuning TIME model ({n_time} trials)...")
        tp, ts = _tune_one("H168_time", _bucket_space(fast), "time", df, args.n_splits, args.seed, args.device, n_time, args)
        best_params["time_model"] = tp
        rb = _oof_time_bucket(df, tp, args.n_splits, args.seed, args.device)
        rd = _oof_time_direct_reg(df, tp, args.n_splits, args.seed, args.device)
        if rb:
            all_rows["true_time"] = rb["yt"]; all_rows["true_bucket"] = rb["yb"]
            all_rows["oof_time_bucket_raw"] = rb["time_bucket_raw"]
            for i in range(4): all_rows[f"bucket_prob_{i}"] = rb["bprobs"][:, i] if rb.get("bprobs") is not None else np.full(len(rb["yt"]), np.nan)
        if rd:
            all_rows["oof_time_direct_lgbm_raw"] = rd["time_direct_lgbm_raw"]
            all_rows["oof_time_direct_xgb_raw"] = rd["time_direct_xgb_raw"]
        if args.enable_oof_fusion and "true_time" in all_rows:
            time_df = pd.DataFrame({k: v for k, v in all_rows.items() if k.startswith(("true_time","oof_time_","bucket_prob_"))})
            valid = time_df["true_time"].notna()
            pred_cols = [c for c in time_df.columns if c.startswith("oof_time_")]
            for c in pred_cols: valid &= time_df[c].notna()
            if valid.sum() > 5 and len(pred_cols) >= 2:
                wts = fit_oof_fusion_weights(time_df[valid], "true_time", pred_cols, "rmse", args.fusion_grid_step)
                fusion_weights["time"] = wts
                print(f"  TIME fusion: {wts}")
        print(f"  TIME best score: {ts:.4f}")

    # ── Extreme ──
    if target in ("all", "extreme"):
        print(f"\n[H168] Tuning EXTREME model ({n_ext} trials)...")
        ep, es = _tune_one("H168_extreme", _extreme_space(fast), "extreme", df, args.n_splits, args.seed, args.device, n_ext, args)
        best_params["extreme_model"] = ep
        oof, ef = _oof_extreme(df, ep, args.n_splits, args.seed, args.device)
        if oof is not None:
            all_rows["oof_extreme_prob_lgbm"] = oof; all_rows["true_extreme"] = ef
        print(f"  EXTREME best score: {es:.4f}")

    # DL/GNN
    if args.tune_transformer and target in ("all", "mag"):
        _try_dl_oof(df, "mag")
    if args.tune_gnn and target in ("all", "mag"):
        _try_gnn_oof(df, "mag")

    best_params["fusion"] = fusion_weights

    # 保存
    (out/"best_params.json").write_text(json.dumps(best_params, indent=2, ensure_ascii=False), encoding="utf-8")
    if fusion_weights:
        (out/"fusion_weights.json").write_text(json.dumps(fusion_weights, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "window": "H168",
               "oof_fusion": args.enable_oof_fusion, "dl_enabled": args.tune_transformer, "gnn_enabled": args.tune_gnn,
               "per_target_trials": {"mag": n_mag, "time": n_time, "extreme": n_ext},
               "version": "single_horizon_v1"}
    (out/"tuning_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if all_rows:
        # 找出最短 numpy 数组长度，用于对齐各列
        arr_lens = [len(v) for v in all_rows.values() if isinstance(v, np.ndarray)]
        if arr_lens:
            min_len = min(arr_lens)
            row_data = {}
            for k, v in all_rows.items():
                if isinstance(v, np.ndarray):
                    row_data[k] = v[:min_len]
                else:
                    row_data[k] = v
            pd.DataFrame(row_data).to_csv(out/"best_oof_predictions.csv", index=False)

    print(f"\nDone. All results: {out}")


if __name__ == "__main__":
    main()
