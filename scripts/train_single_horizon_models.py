"""Single-horizon H168 模型训练脚本。

训练 H168 单窗口模型：
- MagModel (LGBM + XGBoost)
- TimeBucketModel (LGBM bucket classifier)
- Direct Time Regression (LGBM + XGB)
- ExtremeClassifier

输出 qualification_single_horizon_model.joblib + metrics + OOF predictions。
"""

from __future__ import annotations

import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

import joblib, numpy as np, pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_baseline import add_derived_features, select_feature_columns
from src.qualification import (
    H168_WINDOW, H168_MAG_COL, H168_TIME_COL, H168_FLAG_COL,
    derive_h168_labels, reconstruct_h168_features,
)
from scripts.oof_fusion import apply_fusion, normalize_weights
from src.time_buckets import (
    align_bucket_probabilities, assign_time_buckets_batch,
    expected_time_from_bucket_probs, safe_extreme_probability,
)

TIME_COL = "mainshock_time"
WNAME = "H168"
WIN = H168_WINDOW

_LGBM_CUDA_OK = False
try:
    from lightgbm import LGBMRegressor as _LGBMTest
    _t = _LGBMTest(device="cuda", n_estimators=1, num_leaves=2, verbose=-1)
    _t.fit(np.array([[0.0]]), np.array([0.0]))
    _LGBM_CUDA_OK = True
except Exception:
    pass


def resolve_project_path(p):
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_args():
    ap = argparse.ArgumentParser(description="Train single-horizon H168 model.")
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT/"data"/"processed"/"qualification_features.csv")
    ap.add_argument("--save-dir", type=Path, default=PROJECT_ROOT/"data"/"models"/"single_horizon")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["auto","cuda","cpu"], default="cuda")
    ap.add_argument("--model-type", choices=["lightgbm","xgboost","both"], default="both")
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--late-weight", type=float, default=2.0)
    ap.add_argument("--extreme-margin", type=float, default=1.2)
    ap.add_argument("--best-params", type=Path, default=None)
    return ap.parse_args()


# ── 模型构造器 ──

def build_mag_model(seed, device, overrides=None):
    p = {"n_estimators":500,"learning_rate":0.03,"num_leaves":63,"max_depth":-1,
         "min_child_samples":20,"subsample":0.8,"colsample_bytree":0.7,
         "reg_alpha":0.05,"reg_lambda":1.0}
    if overrides: p.update(overrides)
    try:
        from lightgbm import LGBMRegressor
        params = {**p,"random_state":seed,"n_jobs":-1,"verbosity":-1,"objective":"regression"}
        if device=="cuda": params["device"]="cuda"
        return LGBMRegressor(**params)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=int(p["n_estimators"]),learning_rate=float(p["learning_rate"]),
                                             max_leaf_nodes=int(p["num_leaves"]),random_state=seed)

def build_mag_model_xgb(seed, overrides=None):
    p = {"n_estimators":500,"learning_rate":0.03,"max_depth":7,"subsample":0.8,
         "colsample_bytree":0.7,"reg_alpha":0.05,"reg_lambda":1.5,"min_child_weight":5}
    if overrides: p.update(overrides)
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(objective="reg:squarederror",**p,random_state=seed,n_jobs=-1,verbosity=0,tree_method="hist")
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=int(p["n_estimators"]),learning_rate=float(p["learning_rate"]),random_state=seed)

def build_bucket_classifier(seed, device, overrides=None):
    p = {"n_estimators":500,"learning_rate":0.03,"num_leaves":31,"max_depth":-1,
         "min_child_samples":15,"subsample":0.85,"colsample_bytree":0.8,
         "reg_alpha":0.1,"reg_lambda":2.0}
    if overrides: p.update(overrides)
    try:
        from lightgbm import LGBMClassifier
        params = {"objective":"multiclass","num_class":4,**p,"random_state":seed,"n_jobs":-1,"verbosity":-1}
        if device=="cuda": params["device"]="cuda"
        return LGBMClassifier(**params)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=int(p["n_estimators"]),learning_rate=float(p["learning_rate"]),random_state=seed)

def build_extreme_classifier(seed, device, overrides=None, y_for_check=None):
    if y_for_check is not None:
        uniq = np.unique(y_for_check)
        if len(uniq) < 2:
            return DummyClassifier(strategy="constant", constant=int(uniq[0]) if len(uniq) else 0)
    p = {"n_estimators":300,"learning_rate":0.03,"num_leaves":31,
         "subsample":0.85,"colsample_bytree":0.8,"reg_lambda":2.0}
    if overrides: p.update(overrides)
    try:
        from lightgbm import LGBMClassifier
        params = {**p,"random_state":seed,"n_jobs":-1,"verbosity":-1,"class_weight":"balanced"}
        if device=="cuda": params["device"]="cuda"
        return LGBMClassifier(**params)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=int(p["n_estimators"]),learning_rate=float(p["learning_rate"]),random_state=seed)


# ── 度量 ──

def _rmse(e, w=None):
    if w is None: return float(np.sqrt(np.mean(np.square(e))))
    w = np.asarray(w, dtype=float)
    return float(np.sqrt(np.sum(w * np.square(e)) / np.sum(w)))

def calc_metrics(y_mag, y_time, p_mag, p_time, late=2.0):
    me, te = p_mag - y_mag, p_time - y_time
    lw = np.where(te > 0, late, 1.0)
    return {"mag_rmse": _rmse(me), "mag_mae": float(np.mean(np.abs(me))),
            "time_hour_rmse": _rmse(te), "time_hour_mae": float(np.mean(np.abs(te))),
            "time_hour_asym_rmse": float(np.sqrt(np.mean(lw * np.square(te)))),
            "time_hour_hit_rate": float(np.mean(np.abs(te) <= np.maximum(0.2 * y_time, 3.0)))}


def load_best_params(p):
    if p is None: return {}
    p = resolve_project_path(p)
    if not p.exists():
        print(f"[WARN] best_params not found: {p}")
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ── OOF ──

def _oof_mag(df, feats, y_mag, names, args, overrides=None):
    spl = TimeSeriesSplit(n_splits=args.n_splits)
    X = df[feats]; oof_map, models_map, recs = {}, {}, []
    for nm in names:
        oof = np.full(len(df), np.nan)
        for ti, vi in spl.split(X):
            m = build_mag_model_xgb(args.seed, overrides=overrides) if nm=="xgboost" else build_mag_model(args.seed, args.device, overrides=overrides)
            m.fit(X.iloc[ti], y_mag[ti])
            oof[vi] = np.clip(np.asarray(m.predict(X.iloc[vi]), dtype=float), 0.0, None)
        fin = build_mag_model_xgb(args.seed, overrides=overrides) if nm=="xgboost" else build_mag_model(args.seed, args.device, overrides=overrides)
        fin.fit(X, y_mag)
        oof_map[nm], models_map[nm] = oof, fin
        v = np.isfinite(oof)
        if v.any():
            recs.append({"model_type":"mag","model":nm,"mag_rmse":_rmse(oof[v]-y_mag[v]),
                         "mag_mae":float(np.mean(np.abs(oof[v]-y_mag[v])))})
    return models_map, oof_map, recs


def _oof_bucket(df, feats, yb, args, overrides=None):
    spl = TimeSeriesSplit(n_splits=args.n_splits)
    X = df[feats]
    oof_p = np.full((len(df), 4), np.nan); oof_c = np.full(len(df), np.nan)
    recs = []
    for fi, (ti, vi) in enumerate(spl.split(X), 1):
        m = build_bucket_classifier(args.seed, args.device, overrides=overrides)
        m.fit(X.iloc[ti], yb[ti])
        raw = np.asarray(m.predict_proba(X.iloc[vi]), dtype=float)
        p = align_bucket_probabilities(m, raw)
        oof_p[vi], pred_c = p, np.argmax(p, axis=1)
        oof_c[vi] = pred_c
        recs.append({"model_type":"time_bucket","fold":fi,"accuracy":float(np.mean(pred_c==yb[vi]))})
    fin = build_bucket_classifier(args.seed, args.device, overrides=overrides)
    fin.fit(X, yb)
    return fin, oof_p, oof_c, recs


def _oof_time_direct_reg(df, feats, yt, args):
    """直接回归 log-time 作为独立时间候选。"""
    spl = TimeSeriesSplit(n_splits=args.n_splits); X = df[feats]
    yt_log = np.log1p(np.clip(yt, 1e-6, None))
    oof_lgbm = np.full(len(df), np.nan); oof_xgb = np.full(len(df), np.nan)
    recs = []
    for fi, (ti, vi) in enumerate(spl.split(X), 1):
        try:
            from lightgbm import LGBMRegressor
            m_lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=63,
                                    random_state=args.seed, n_jobs=-1, verbosity=-1)
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor
            m_lgbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, random_state=args.seed)
        m_lgbm.fit(X.iloc[ti], yt_log[ti])
        oof_lgbm[vi] = np.clip(np.expm1(np.asarray(m_lgbm.predict(X.iloc[vi]), dtype=float)),
                               WIN.lower_hours+1e-6, WIN.upper_hours)
        try:
            from xgboost import XGBRegressor
            m_xgb = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=6,
                                 random_state=args.seed, n_jobs=-1, verbosity=0, tree_method="hist")
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor
            m_xgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, random_state=args.seed)
        m_xgb.fit(X.iloc[ti], yt_log[ti])
        oof_xgb[vi] = np.clip(np.expm1(np.asarray(m_xgb.predict(X.iloc[vi]), dtype=float)),
                              WIN.lower_hours+1e-6, WIN.upper_hours)
        recs.append({"model_type":"time_regression","fold":fi,
                     "lgbm_mae":float(np.mean(np.abs(oof_lgbm[vi]-yt[vi]))),
                     "xgb_mae":float(np.mean(np.abs(oof_xgb[vi]-yt[vi])))})
    return oof_lgbm, oof_xgb, recs


def _oof_extreme(df, feats, ye, args, overrides=None):
    spl = TimeSeriesSplit(n_splits=args.n_splits); X = df[feats]
    oof_p = np.full(len(df), np.nan); ys = pd.Series(ye); recs = []
    for fi, (ti, vi) in enumerate(spl.split(X), 1):
        ytf = ys.iloc[ti].to_numpy(dtype=int)
        m = build_extreme_classifier(args.seed, args.device, overrides=overrides, y_for_check=ytf)
        m.fit(X.iloc[ti], ytf)
        pr = safe_extreme_probability(m, X.iloc[vi]); oof_p[vi] = pr
        vy = ys.iloc[vi].to_numpy(dtype=int)
        try:
            auc = float(roc_auc_score(vy, pr)) if len(np.unique(vy))>1 else float("nan")
        except ValueError: auc = float("nan")
        recs.append({"model_type":"extreme","fold":fi,"positive_rate":float(np.mean(vy)),"roc_auc":auc})
    fin = build_extreme_classifier(args.seed, args.device, overrides=overrides, y_for_check=ye)
    fin.fit(X, ye)
    return fin, oof_p, recs


def fuse_mag(oof_map):
    return np.nanmean(list(oof_map.values()), axis=0)


# ── main ──

def main():
    args = parse_args()
    if args.device in ("cuda","auto") and not _LGBM_CUDA_OK:
        print("[WARN] LightGBM CUDA not available; falling back to cpu device.")
        args.device = "cpu"
    dp, sd = resolve_project_path(args.data), resolve_project_path(args.save_dir)
    sd.mkdir(parents=True, exist_ok=True)
    bp = load_best_params(args.best_params)

    df = pd.read_csv(dp)
    miss = [c for c in ["target_T1_max_mag","target_T2_max_mag","target_T3_max_mag"] if c not in df.columns]
    if miss: raise ValueError("Missing T1/T2/T3 labels: "+", ".join(miss))
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce", format="mixed")
    df = add_derived_features(df)
    df = derive_h168_labels(df)
    df = df.dropna(subset=[TIME_COL, H168_MAG_COL, H168_TIME_COL]).reset_index(drop=True)
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    pp = bp.get("postprocessing", {})
    fusion_w = bp.get("fusion", {})

    art = {"artifact_type": "qualification_single_horizon_v2",
           "created_at": datetime.now(timezone.utc).isoformat(),
           "data": str(dp), "target_unit": "hours_since_mainshock",
           "postprocessing": pp, "fusion_weights": fusion_w}
    all_recs, all_met = [], {}

    ldf = reconstruct_h168_features(df)
    ldf[TIME_COL] = df[TIME_COL]
    ldf = ldf.dropna(subset=[TIME_COL, H168_MAG_COL, H168_TIME_COL]).reset_index(drop=True)
    fcs = select_feature_columns(ldf)
    if not fcs: raise ValueError("No features for H168")

    ym, yt = ldf[H168_MAG_COL].to_numpy(dtype=float), ldf[H168_TIME_COL].to_numpy(dtype=float)
    mm = ldf["mainshock_mag"].to_numpy(dtype=float)
    em = float(pp.get("extreme_margin", float(args.extreme_margin)))
    ef = (ym>0.0)&(ym>=mm-em); ye = ef.astype(int)
    yb = assign_time_buckets_batch(WNAME, yt)

    mo = bp.get("mag_model", bp.get("mag", {}))
    bo = bp.get("time_model", bp.get("bucket_model", bp.get("time_bucket", {})))

    nms = []
    if args.model_type in ("lightgbm","both"): nms.append("baseline")
    if args.model_type in ("xgboost","both"): nms.append("xgboost")

    print(f"\n{'='*50}\n  Single-Horizon H168\n{'='*50}")

    mag_mods, mag_oom, mag_recs = _oof_mag(ldf, fcs, ym, nms, args, overrides=mo)
    all_recs.extend(mag_recs)
    fm_raw = fuse_mag(mag_oom)

    bm, bprobs, bcls, brecs = _oof_bucket(ldf, fcs, yb, args, overrides=bo)
    all_recs.extend(brecs)
    dt_raw = expected_time_from_bucket_probs(WNAME, bprobs)

    dt_reg_lgbm, dt_reg_xgb, trecs = _oof_time_direct_reg(ldf, fcs, yt, args)
    all_recs.extend(trecs)

    # v2: 在完整 H168 训练集上训练最终时间回归模型并存储为模型对象
    yt_log = np.log1p(np.clip(yt, 1e-6, None))
    try:
        from lightgbm import LGBMRegressor
        time_direct_model_lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=63,
                                                random_state=args.seed, n_jobs=-1, verbosity=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        time_direct_model_lgbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, random_state=args.seed)
    time_direct_model_lgbm.fit(ldf[fcs], yt_log)
    try:
        from xgboost import XGBRegressor
        time_direct_model_xgb = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=6,
                                              random_state=args.seed, n_jobs=-1, verbosity=0, tree_method="hist")
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        time_direct_model_xgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, random_state=args.seed)
    time_direct_model_xgb.fit(ldf[fcs], yt_log)
    print(f"  [H168] direct time reg LGBM fitted, XGB fitted")

    emod, eprobs, erecs = _oof_extreme(ldf, fcs, ye, args)
    all_recs.extend(erecs)

    fm_pp, dt_pp = fm_raw.copy(), dt_raw.copy()
    # extreme 高风险触发：提升震级，提前时间
    pp_thr = float(pp.get("extreme_prob_threshold",0.5))
    high = eprobs >= pp_thr
    if high.any():
        uplift = float(pp.get("high_risk_mag_quantile_weight",0.5))
        margin = float(pp.get("extreme_margin",1.2))
        floor = np.clip(mm[high]-margin,0.0,None)
        fm_pp[high] = (1-uplift)*fm_raw[high]+uplift*np.maximum(fm_raw[high],floor)
        shift = float(pp.get("early_time_shift_strength",0.1))
        anchor = WIN.lower_hours + (WIN.upper_hours-WIN.lower_hours)*0.3
        dt_pp[high] = (1-shift)*dt_raw[high]+shift*np.minimum(dt_raw[high],anchor)
        dt_pp[high] = np.clip(dt_pp[high], WIN.lower_hours+1e-6, WIN.upper_hours)
    fm_pp = np.clip(fm_pp,0.0,None)

    valid = np.isfinite(fm_raw) & np.isfinite(dt_raw)
    if valid.any():
        raw_m = calc_metrics(ym[valid], yt[valid], fm_raw[valid], dt_raw[valid], args.late_weight)
        pp_m = calc_metrics(ym[valid], yt[valid], fm_pp[valid], dt_pp[valid], args.late_weight)
        era = float(np.mean(np.abs(fm_raw[valid][ef[valid]]-ym[valid][ef[valid]]))) if ef[valid].any() else float("nan")
        epa = float(np.mean(np.abs(fm_pp[valid][ef[valid]]-ym[valid][ef[valid]]))) if ef[valid].any() else float("nan")
        vb = np.isfinite(bcls)&np.isfinite(np.array(yb,dtype=float))
        ba = float(np.mean(bcls[vb].astype(int)==yb[vb])) if vb.any() else float("nan")
        all_met = {"raw":raw_m,"postprocessed":pp_m,
                   "extreme":{"raw_mag_mae":era,"postprocessed_mag_mae":epa,
                              "extreme_count":int(np.sum(ef[valid])),
                              "extreme_sample_rate":float(np.mean(ef[valid]))},
                   "time_bucket_accuracy":ba}
        print(f"  raw:  mag_rmse={raw_m['mag_rmse']:.4f} time_mae={raw_m['time_hour_mae']:.2f}")
        print(f"  post: mag_rmse={pp_m['mag_rmse']:.4f} time_mae={pp_m['time_hour_mae']:.2f} bucket_acc={ba:.4f}")

    art["H168"] = {"observation_hours":0.0,"feature_cols":fcs,
                    "mag_models":mag_mods,"bucket_model":bm,
                    "extreme_model":emod,
                    "time_direct_model_lgbm": time_direct_model_lgbm,
                    "time_direct_model_xgb": time_direct_model_xgb,
                    "weights":{"mag":{n:1.0/len(nms) for n in nms}},
                    "fusion":fusion_w, "version":"single_horizon_v2"}

    mp = sd / "qualification_single_horizon_model.joblib"
    joblib.dump(art, mp)
    (sd/"single_horizon_metrics.json").write_text(
        json.dumps({"created_at":art["created_at"],"data":str(dp),"metrics":all_met},
                   indent=2,ensure_ascii=False),encoding="utf-8")
    pd.DataFrame(all_recs).to_csv(sd/"single_horizon_oof_metrics.csv",index=False)

    row = ldf[["mainshock_id",TIME_COL,"mainshock_mag",H168_MAG_COL,H168_TIME_COL]].copy()
    row["extreme_flag"]=ef; row["extreme_prob"]=eprobs
    row["mag_raw"]=fm_raw; row["time_raw"]=dt_raw
    row["mag_postprocessed"]=fm_pp; row["time_postprocessed"]=dt_pp
    for i in range(4): row[f"bucket_{i}_prob"]=bprobs[:,i]
    row.to_csv(sd/"single_horizon_oof_predictions.csv",index=False)

    if bp: (sd/"best_params.json").write_text(json.dumps(bp,indent=2,ensure_ascii=False),encoding="utf-8")

    print(f"\nDone. Artifact: {mp}")


if __name__ == "__main__":
    main()
