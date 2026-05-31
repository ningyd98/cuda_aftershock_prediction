"""Decoupled 资格赛模型训练脚本。

每个窗口独立训练：MagModel / TimeBucketModel / ExtremeClassifier。
metrics 输出 raw 和 postprocessed 两套指标（口径与最终预测一致）。
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
    QUALIFICATION_WINDOWS, observation_hours_for_window,
    qualification_target_cols, reconstruct_legal_window_features,
)
from scripts.oof_fusion import apply_fusion, normalize_weights
from src.time_buckets import (
    align_bucket_probabilities, assign_time_buckets_batch,
    expected_time_from_bucket_probs, safe_extreme_probability,
)

TIME_COL = "mainshock_time"

_LGBM_CUDA_OK = False
try:
    import numpy as _np_test
    from lightgbm import LGBMRegressor as _LGBMTest
    _t = _LGBMTest(device="cuda", n_estimators=1, num_leaves=2, verbose=-1)
    _t.fit(_np_test.array([[0.0]]), _np_test.array([0.0]))
    _LGBM_CUDA_OK = True
except Exception:
    pass


def resolve_project_path(p):
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_args():
    ap = argparse.ArgumentParser(description="Train decoupled models per window.")
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "processed" / "qualification_features.csv")
    ap.add_argument("--save-dir", type=Path, default=PROJECT_ROOT / "data" / "models" / "qualification_decoupled")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["auto","cuda","cpu"], default="cuda")
    ap.add_argument("--model-type", choices=["lightgbm","xgboost","both"], default="both")
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--late-weight", type=float, default=2.0)
    ap.add_argument("--extreme-margin", type=float, default=1.2)
    ap.add_argument("--quantiles", type=str, default="0.5,0.75,0.9")
    ap.add_argument("--tuning-config", type=Path, default=None)
    ap.add_argument("--best-params", type=Path, default=None)
    return ap.parse_args()


# --------------- 后处理（可复用） ---------------

def apply_extreme_postprocessing(mag_pred, time_pred, extreme_prob, mainshock_mag, window_name, pp):
    """extreme 校正 + 时间偏移。返回 (mag_adj, time_adj)。"""
    from src.qualification import WINDOW_BY_NAME
    window = WINDOW_BY_NAME[window_name]
    mag_adj, time_adj = mag_pred.copy(), time_pred.copy()
    threshold = float(pp.get("extreme_prob_threshold", 0.5))
    high = extreme_prob >= threshold
    if not high.any():
        return mag_adj, time_adj
    # mag floor correction
    uplift = float(pp.get("high_risk_mag_quantile_weight", 0.5))
    margin = float(pp.get("extreme_margin", 1.2))
    floor = np.clip(mainshock_mag[high] - margin, 0.0, None)
    mag_adj[high] = (1 - uplift) * mag_pred[high] + uplift * np.maximum(mag_pred[high], floor)
    # time early shift
    shift = float(pp.get("early_time_shift_strength", 0.1))
    anchor = window.lower_hours + (window.upper_hours - window.lower_hours) * 0.3
    time_adj[high] = (1 - shift) * time_pred[high] + shift * np.minimum(time_pred[high], anchor)
    time_adj[high] = np.clip(time_adj[high], window.lower_hours + 1e-6, window.upper_hours)
    if window_name == "T1":
        bonus = float(pp.get("t1_early_delta_bonus", 0.0))
        if bonus > 0:
            mag_adj[high] = np.clip(mag_adj[high] + bonus, 0.0, None)
    mag_adj = np.clip(mag_adj, 0.0, None)
    return mag_adj, time_adj


# --------------- 度量 ---------------

def _rmse(e, w=None):
    if w is None:
        return float(np.sqrt(np.mean(np.square(e))))
    w = np.asarray(w, dtype=float)
    return float(np.sqrt(np.sum(w * np.square(e)) / np.sum(w)))


def calc_metrics(y_mag, y_time, p_mag, p_time, late=2.0):
    me, te = p_mag - y_mag, p_time - y_time
    lw = np.where(te > 0, late, 1.0)
    return {"mag_rmse": _rmse(me), "mag_mae": float(np.mean(np.abs(me))),
            "time_hour_rmse": _rmse(te), "time_hour_mae": float(np.mean(np.abs(te))),
            "time_hour_asym_rmse": float(np.sqrt(np.mean(lw * np.square(te)))),
            "time_hour_hit_rate": float(np.mean(np.abs(te) <= np.maximum(0.2 * y_time, 3.0)))}


# --------------- 模型构造器 ---------------

def build_mag_model(seed, device, overrides=None):
    p = {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 63, "max_depth": -1,
         "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.7,
         "reg_alpha": 0.05, "reg_lambda": 1.0}
    if overrides:
        p.update(overrides)
    try:
        from lightgbm import LGBMRegressor
        params = {**p, "random_state": seed, "n_jobs": -1, "verbosity": -1, "objective": "regression"}
        if device == "cuda":
            params["device"] = "cuda"
        return LGBMRegressor(**params)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=int(p["n_estimators"]),
                                             learning_rate=float(p["learning_rate"]),
                                             max_leaf_nodes=int(p["num_leaves"]), random_state=seed)


def build_mag_model_xgb(seed, overrides=None):
    p = {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 7, "subsample": 0.8,
         "colsample_bytree": 0.7, "reg_alpha": 0.05, "reg_lambda": 1.5, "min_child_weight": 5}
    if overrides:
        p.update(overrides)
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(objective="reg:squarederror", **p,
                            random_state=seed, n_jobs=-1, verbosity=0, tree_method="hist")
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=int(p["n_estimators"]),
                                             learning_rate=float(p["learning_rate"]),
                                             max_depth=int(p.get("max_depth", 7)), random_state=seed)


def build_bucket_classifier(seed, device, overrides=None):
    p = {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 31, "max_depth": -1,
         "min_child_samples": 15, "subsample": 0.85, "colsample_bytree": 0.8,
         "reg_alpha": 0.1, "reg_lambda": 2.0}
    if overrides:
        p.update(overrides)
    try:
        from lightgbm import LGBMClassifier
        params = {"objective": "multiclass", "num_class": 4, **p,
                  "random_state": seed, "n_jobs": -1, "verbosity": -1}
        if device == "cuda":
            params["device"] = "cuda"
        return LGBMClassifier(**params)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=int(p["n_estimators"]),
                                              learning_rate=float(p["learning_rate"]), random_state=seed)


def build_bucket_classifier_xgb(seed, overrides=None):
    """XGBoost 时间桶分类器（decoupled v2 新增候选）。"""
    p = {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 6,
         "subsample": 0.85, "colsample_bytree": 0.8, "reg_alpha": 0.1,
         "reg_lambda": 2.0}
    if overrides:
        p.update(overrides)
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(objective="multi:softprob", num_class=4, **p,
                            random_state=seed, n_jobs=-1, verbosity=0)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=int(p["n_estimators"]),
                                              learning_rate=float(p["learning_rate"]), random_state=seed)


def build_extreme_classifier(seed, device, overrides=None, y_for_check=None):
    """构建极端大余震二分类器。单类别自动回退 DummyClassifier。"""
    if y_for_check is not None:
        uniq = np.unique(y_for_check)
        if len(uniq) < 2:
            return DummyClassifier(strategy="constant", constant=int(uniq[0]) if len(uniq) else 0)
    p = {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31,
         "subsample": 0.85, "colsample_bytree": 0.8, "reg_lambda": 2.0}
    if overrides:
        p.update(overrides)
    try:
        from lightgbm import LGBMClassifier
        params = {**p, "random_state": seed, "n_jobs": -1, "verbosity": -1, "class_weight": "balanced"}
        if device == "cuda":
            params["device"] = "cuda"
        return LGBMClassifier(**params)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=int(p["n_estimators"]),
                                              learning_rate=float(p["learning_rate"]), random_state=seed)


# --------------- OOF ---------------

def _oof_mag(df, feats, y_mag, names, args, wn, overrides=None):
    spl = TimeSeriesSplit(n_splits=args.n_splits)
    X, oof_map, models_map, recs = df[feats], {}, {}, []
    for nm in names:
        oof = np.full(len(df), np.nan)
        for ti, vi in spl.split(X):
            m = build_mag_model_xgb(args.seed, overrides=overrides) if nm == "xgboost" else build_mag_model(args.seed, args.device, overrides=overrides)
            m.fit(X.iloc[ti], y_mag[ti])
            oof[vi] = np.clip(np.asarray(m.predict(X.iloc[vi]), dtype=float), 0.0, None)
        fin = build_mag_model_xgb(args.seed, overrides=overrides) if nm == "xgboost" else build_mag_model(args.seed, args.device, overrides=overrides)
        fin.fit(X, y_mag)
        oof_map[nm], models_map[nm] = oof, fin
        v = np.isfinite(oof)
        if v.any():
            recs.append({"window": wn, "model_type": "mag", "model": nm, "fold": 0,
                         "mag_rmse": _rmse(oof[v] - y_mag[v]),
                         "mag_mae": float(np.mean(np.abs(oof[v] - y_mag[v])))})
    return models_map, oof_map, recs


def _oof_bucket(df, feats, yb, args, wn, overrides=None):
    spl = TimeSeriesSplit(n_splits=args.n_splits)
    X = df[feats]
    oof_p = np.full((len(df), 4), np.nan)
    oof_c = np.full(len(df), np.nan)
    oof_xgb_p = np.full((len(df), 4), np.nan)
    oof_xgb_c = np.full(len(df), np.nan)
    recs = []
    for fi, (ti, vi) in enumerate(spl.split(X), 1):
        # LGBM bucket
        m = build_bucket_classifier(args.seed, args.device, overrides=overrides)
        m.fit(X.iloc[ti], yb[ti])
        raw = np.asarray(m.predict_proba(X.iloc[vi]), dtype=float)
        p = align_bucket_probabilities(m, raw)
        oof_p[vi], pred_c = p, np.argmax(p, axis=1)
        oof_c[vi] = pred_c
        # XGBoost bucket
        m_xgb = build_bucket_classifier_xgb(args.seed, overrides=overrides)
        m_xgb.fit(X.iloc[ti], yb[ti])
        raw_xgb = np.asarray(m_xgb.predict_proba(X.iloc[vi]), dtype=float)
        p_xgb = align_bucket_probabilities(m_xgb, raw_xgb)
        oof_xgb_p[vi], pred_c_xgb = p_xgb, np.argmax(p_xgb, axis=1)
        oof_xgb_c[vi] = pred_c_xgb
        recs.append({"window": wn, "model_type": "time_bucket", "fold": fi,
                     "accuracy_lgbm": float(np.mean(pred_c == yb[vi])),
                     "accuracy_xgb": float(np.mean(pred_c_xgb == yb[vi]))})
    fin = build_bucket_classifier(args.seed, args.device, overrides=overrides)
    fin.fit(X, yb)
    fin_xgb = build_bucket_classifier_xgb(args.seed, overrides=overrides)
    fin_xgb.fit(X, yb)
    return fin, oof_p, oof_c, fin_xgb, oof_xgb_p, oof_xgb_c, recs


def _oof_time_direct_reg(df, feats, yt, args, wn):
    """独立时间回归 OOF（decoupled v2）：直接回归 log-hours，与 bucket 分类形成异构候选。"""
    spl = TimeSeriesSplit(n_splits=args.n_splits)
    X = df[feats]
    from src.qualification import WINDOW_BY_NAME
    window = WINDOW_BY_NAME[wn]
    yt_log = np.log1p(np.clip(yt, 1e-6, None))
    oof_lgbm = np.full(len(df), np.nan)
    oof_xgb = np.full(len(df), np.nan)
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
        pred_log = np.asarray(m_lgbm.predict(X.iloc[vi]), dtype=float)
        oof_lgbm[vi] = np.clip(np.expm1(pred_log), window.lower_hours + 1e-6, window.upper_hours)
        try:
            from xgboost import XGBRegressor
            m_xgb = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=6,
                                 random_state=args.seed, n_jobs=-1, verbosity=0, tree_method="hist")
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor
            m_xgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, random_state=args.seed)
        m_xgb.fit(X.iloc[ti], yt_log[ti])
        pred_log_xgb = np.asarray(m_xgb.predict(X.iloc[vi]), dtype=float)
        oof_xgb[vi] = np.clip(np.expm1(pred_log_xgb), window.lower_hours + 1e-6, window.upper_hours)
        recs.append({"window": wn, "model_type": "time_regression", "fold": fi,
                     "lgbm_mae": float(np.mean(np.abs(oof_lgbm[vi] - yt[vi]))),
                     "xgb_mae": float(np.mean(np.abs(oof_xgb[vi] - yt[vi])))})
    return oof_lgbm, oof_xgb, recs


def _oof_extreme(df, feats, ye, args, wn, overrides=None):
    spl = TimeSeriesSplit(n_splits=args.n_splits)
    X = df[feats]
    oof_p = np.full(len(df), np.nan)
    ys = pd.Series(ye)
    recs = []
    for fi, (ti, vi) in enumerate(spl.split(X), 1):
        ytf = ys.iloc[ti].to_numpy(dtype=int)
        m = build_extreme_classifier(args.seed, args.device, overrides=overrides, y_for_check=ytf)
        m.fit(X.iloc[ti], ytf)
        pr = safe_extreme_probability(m, X.iloc[vi])
        oof_p[vi] = pr
        vy = ys.iloc[vi].to_numpy(dtype=int)
        try:
            auc = float(roc_auc_score(vy, pr)) if len(np.unique(vy)) > 1 else float("nan")
        except ValueError:
            auc = float("nan")
        recs.append({"window": wn, "model_type": "extreme", "fold": fi,
                     "positive_rate": float(np.mean(vy)), "roc_auc": auc})
    fin = build_extreme_classifier(args.seed, args.device, overrides=overrides, y_for_check=ye)
    fin.fit(X, ye)
    return fin, oof_p, recs


def fuse_mag(oof_map):
    return np.nanmean(list(oof_map.values()), axis=0)


def load_best_params(p):
    if p is None:
        return {}
    p = resolve_project_path(p)
    if not p.exists():
        print(f"[WARN] best_params not found: {p}")
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# --------------- main ---------------

def main():
    args = parse_args()
    if args.device in ("cuda", "auto") and not _LGBM_CUDA_OK:
        print("[WARN] LightGBM CUDA not available; falling back to cpu device.")
        args.device = "cpu"
    dp, sd = resolve_project_path(args.data), resolve_project_path(args.save_dir)
    sd.mkdir(parents=True, exist_ok=True)
    bp = load_best_params(args.best_params)

    df = pd.read_csv(dp)
    miss = [c for c in qualification_target_cols() if c not in df.columns]
    if miss:
        raise ValueError("Missing targets: " + ", ".join(miss))
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce", format="mixed")
    df = add_derived_features(df)
    df = df.dropna(subset=[TIME_COL, *qualification_target_cols()]).reset_index(drop=True)
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    pp = bp.get("postprocessing", {})

    fusion_w = bp.get("fusion_weights", {}) if bp else {}
    art = {"artifact_type": "qualification_decoupled_v2",
           "created_at": datetime.now(timezone.utc).isoformat(),
           "data": str(dp), "target_unit": "hours_since_mainshock",
           "postprocessing": pp, "fusion_weights": fusion_w, "windows": {}}
    all_recs, all_rows, all_met = [], [], {}

    for win in QUALIFICATION_WINDOWS:
        wn = win.name
        ldf = reconstruct_legal_window_features(df, wn)
        ldf[TIME_COL] = df[TIME_COL]
        ldf = ldf.dropna(subset=[TIME_COL, win.mag_col, win.time_col]).reset_index(drop=True)
        fcs = select_feature_columns(ldf)
        if not fcs:
            raise ValueError(f"No features for {wn}")

        ym, yt = ldf[win.mag_col].to_numpy(dtype=float), ldf[win.time_col].to_numpy(dtype=float)
        mm = ldf["mainshock_mag"].to_numpy(dtype=float)
        em = float(pp.get("extreme_margin", float(args.extreme_margin)))
        ef = (ym > 0.0) & (ym >= mm - em)
        ye = ef.astype(int)
        yb = assign_time_buckets_batch(wn, yt)

        wp = bp.get("windows", {}).get(wn, {})
        mo, bo, eo = wp.get("mag_model", wp.get("mag", {})), wp.get("bucket_model", wp.get("time_bucket", {})), wp.get("extreme_model", wp.get("extreme", {}))

        nms = []
        if args.model_type in ("lightgbm", "both"):
            nms.append("baseline")
        if args.model_type in ("xgboost", "both"):
            nms.append("xgboost")

        print(f"\n{'='*50}\n  Window {wn}\n{'='*50}")

        mag_mods, mag_oom, mag_recs = _oof_mag(ldf, fcs, ym, nms, args, wn, overrides=mo)
        all_recs.extend(mag_recs)
        fm_raw = fuse_mag(mag_oom)

        # v2: bucket 返回 (lgbm_model, lgbm_probs, lgbm_classes, xgb_model, xgb_probs, xgb_classes, recs)
        bm, bprobs, bcls, bm_xgb, bprobs_xgb, bcls_xgb, brecs = _oof_bucket(ldf, fcs, yb, args, wn, overrides=bo)
        all_recs.extend(brecs)
        dt_raw = expected_time_from_bucket_probs(wn, bprobs)

        # v2: 独立时间回归 OOF
        dt_reg_lgbm, dt_reg_xgb, trecs = _oof_time_direct_reg(ldf, fcs, yt, args, wn)
        all_recs.extend(trecs)

        emod, eprobs, erecs = _oof_extreme(ldf, fcs, ye, args, wn, overrides=eo)
        all_recs.extend(erecs)

        # 后处理
        fm_pp, dt_pp = apply_extreme_postprocessing(fm_raw, dt_raw, eprobs, mm, wn, pp)
        mb = float(pp.get(f"mag_bias_{wn}", 0.0))
        tb = float(pp.get(f"time_bias_{wn}", 0.0))
        fm_pp += mb
        dt_pp = np.clip(dt_pp + tb, win.lower_hours + 1e-6, win.upper_hours)

        valid = np.isfinite(fm_raw) & np.isfinite(dt_raw)
        if valid.any():
            raw_m = calc_metrics(ym[valid], yt[valid], fm_raw[valid], dt_raw[valid], args.late_weight)
            pp_m = calc_metrics(ym[valid], yt[valid], fm_pp[valid], dt_pp[valid], args.late_weight)
            era = float(np.mean(np.abs(fm_raw[valid][ef[valid]] - ym[valid][ef[valid]]))) if ef[valid].any() else float("nan")
            epa = float(np.mean(np.abs(fm_pp[valid][ef[valid]] - ym[valid][ef[valid]]))) if ef[valid].any() else float("nan")
            vb = np.isfinite(bcls) & np.isfinite(np.array(yb, dtype=float))
            ba = float(np.mean(bcls[vb].astype(int) == yb[vb])) if vb.any() else float("nan")
            all_met[wn] = {
                "raw": raw_m, "postprocessed": pp_m,
                "extreme": {"raw_mag_mae": era, "postprocessed_mag_mae": epa,
                            "extreme_count": int(np.sum(ef[valid])),
                            "extreme_sample_rate": float(np.mean(ef[valid]))},
                "time_bucket_accuracy": ba,
            }
            print(f"  raw:  mag_rmse={raw_m['mag_rmse']:.4f} time_mae={raw_m['time_hour_mae']:.2f}")
            print(f"  post: mag_rmse={pp_m['mag_rmse']:.4f} time_mae={pp_m['time_hour_mae']:.2f} bucket_acc={ba:.4f}")

        art["windows"][wn] = {
            "observation_hours": observation_hours_for_window(wn),
            "feature_cols": fcs, "mag_models": mag_mods,
            "bucket_model": bm, "bucket_model_xgb": bm_xgb,
            "extreme_model": emod,
            "weights": {"mag": {n: 1.0 / len(nms) for n in nms}},
            "postprocessing": {"mag_bias": mb, "time_bias": tb},
            "fusion": fusion_w.get(wn, {}),
            # v2: 独立时间回归结果（无模型对象，仅有 OOF 数组用于评估）
            "time_direct_reg_lgbm_oof": dt_reg_lgbm.tolist() if len(dt_reg_lgbm) < 10000 else None,
            "time_direct_reg_xgb_oof": dt_reg_xgb.tolist() if len(dt_reg_xgb) < 10000 else None,
            "version": "decoupled_v2",
        }

        row = ldf[["mainshock_id", TIME_COL, "mainshock_mag", win.mag_col, win.time_col]].copy()
        row["window"] = wn
        row["extreme_flag"] = ef
        row["extreme_prob"] = eprobs
        row["decoupled_mag_raw"] = fm_raw
        row["decoupled_time_raw"] = dt_raw
        row["decoupled_mag_postprocessed"] = fm_pp
        row["decoupled_time_postprocessed"] = dt_pp
        for i in range(4):
            row[f"bucket_{i}_prob"] = bprobs[:, i]
        row["bucket_assigned"] = bcls
        all_rows.append(row)

    mp = sd / "qualification_decoupled_models.joblib"
    joblib.dump(art, mp)
    (sd / "decoupled_metrics.json").write_text(
        json.dumps({"created_at": art["created_at"], "data": str(dp), "window_metrics": all_met},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(all_recs).to_csv(sd / "decoupled_oof_metrics.csv", index=False)
    pd.concat(all_rows, ignore_index=True).to_csv(sd / "decoupled_oof_predictions.csv", index=False)
    if bp:
        (sd / "best_params.json").write_text(json.dumps(bp, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone. Artifact: {mp}")


if __name__ == "__main__":
    main()
