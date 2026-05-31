#!/usr/bin/env python3
"""Magnitude residual calibrator trained on OOF data.

Trains lightweight models to predict residual = true_mag - pred_mag
on OOF data. Applies correction to final_t123 predictions with limits.

Features: window one-hot, mainshock_mag, pred_mag, mag_diff,
high_mag flags, plate_type (if available), depth, early counts/energy.

Outputs three candidates: conservative, balanced, aggressive.
"""

from __future__ import annotations

import hashlib, json, shutil, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)
PROJECT = Path("/home/ningyd/CodingSpace/aftershock_qualification_train")
OOF_PATH = PROJECT / "data/models/qualification_decoupled_full/decoupled_oof_predictions.csv"
FEAT_PATH = PROJECT / "data/processed/qualification_features.csv"
FINAL_PKG = PROJECT / "submission_package_final_t123_no_commitment"
OUT_DIR = PROJECT / "experiments/mag_residual_calibrator"

WINDOWS = ["T1", "T2", "T3"]

# Correction limits per candidate
CANDIDATES = {
    "conservative": {"mag_limits": (-0.2, 0.4), "high_mag_limit": 0.5},
    "balanced":     {"mag_limits": (-0.3, 0.6), "high_mag_limit": 0.8},
    "aggressive":   {"mag_limits": (-0.4, 0.8), "high_mag_limit": 1.0},
}

# Allowed model types
MODELS = {
    "ridge": Ridge(alpha=1.0, random_state=SEED),
    "huber": HuberRegressor(epsilon=1.35, max_iter=200),
    "isotonic": "isotonic",  # special case
    "hgb": HistGradientBoostingRegressor(max_depth=2, max_iter=100, random_state=SEED),
}


def build_features(oof_df, feat_df):
    """Build feature matrix and target (residual) from OOF + feature data."""
    rows = []
    for w in WINDOWS:
        w_oof = oof_df[oof_df["window"] == w].dropna(subset=["decoupled_mag_raw"]).copy()
        w_df = feat_df.merge(
            w_oof[["mainshock_id", "decoupled_mag_raw", "decoupled_mag_postprocessed"]],
            on="mainshock_id", how="inner")
        tcol = f"target_{w}_max_mag"
        if tcol not in w_df.columns:
            continue
        w_df = w_df.dropna(subset=[tcol])
        w_df = w_df[w_df[tcol] > 0].copy()
        w_df["window"] = w
        rows.append(w_df)
    df = pd.concat(rows, ignore_index=True)

    # Features
    X_data = {}
    X_data["pred_mag"] = df["decoupled_mag_raw"].values
    X_data["mainshock_mag"] = df["mainshock_mag"].values
    X_data["mag_diff"] = df["mainshock_mag"].values - df["decoupled_mag_raw"].values
    X_data["pred_mag_post"] = df.get("decoupled_mag_postprocessed",
                                      df["decoupled_mag_raw"]).values
    X_data["mainshock_depth"] = df.get("mainshock_depth", np.full(len(df), 10.0)).values
    X_data["high_mag_7_5"] = (df["mainshock_mag"] >= 7.5).astype(float).values
    X_data["high_mag_8_0"] = (df["mainshock_mag"] >= 8.0).astype(float).values
    X_data["high_mag_8_5"] = (df["mainshock_mag"] >= 8.5).astype(float).values
    # Window one-hot
    for w in WINDOWS:
        X_data[f"win_{w}"] = (df["window"] == w).astype(float).values
    # Optional features
    if "early_aftershock_count" in df.columns:
        X_data["early_count"] = np.log1p(df["early_aftershock_count"].fillna(0).values)
    if "early_max_mag" in df.columns:
        X_data["early_max_mag"] = df["early_max_mag"].fillna(0).values
    if "plate_boundary_distance_km" in df.columns:
        X_data["plate_dist"] = df["plate_boundary_distance_km"].fillna(100).values
    if "mainshock_time" in df.columns:
        X_data["mainshock_time_ordinal"] = pd.to_datetime(
            df["mainshock_time"], utc=True, errors="coerce").astype(np.int64).values // 1e9
        X_data["mainshock_time_ordinal"] = np.nan_to_num(X_data["mainshock_time_ordinal"], 0)

    X = pd.DataFrame(X_data)
    # Target: residual = true_mag - pred_mag
    Y = df[[f"target_{w}_max_mag" for w in WINDOWS if f"target_{w}_max_mag" in df.columns]].max(axis=1).values
    Y = Y - df["decoupled_mag_raw"].values

    # Time-based split: train on older 80%, eval on newer 20%
    split_idx = int(len(X) * 0.8)
    X_train, X_eval = X.iloc[:split_idx], X.iloc[split_idx:]
    Y_train, Y_eval = Y[:split_idx], Y[split_idx:]
    return X_train, Y_train, X_eval, Y_eval, X.columns.tolist()


def train_model(model_key, X_train, Y_train):
    if model_key == "isotonic":
        x_1d = X_train["pred_mag"].values
        model = IsotonicRegression(out_of_bounds="clip", y_min=-1.5, y_max=2.0)
        model.fit(x_1d, Y_train)
        return ("isotonic", model)
    else:
        m = MODELS[model_key]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        m.fit(X_scaled, Y_train)
        return ("standard", (scaler, m))


def predict_model(model_type, model_obj, X, model_key):
    if model_type == "isotonic":
        x_1d = X["pred_mag"].values
        return np.clip(model_obj.predict(x_1d), -1.5, 2.0)
    else:
        scaler, m = model_obj
        X_scaled = scaler.transform(X)
        return m.predict(X_scaled)


def evaluate_model(model_type, model_obj, model_key, X_eval, Y_eval, limits, high_limits):
    pred = predict_model(model_type, model_obj, X_eval, model_key)
    # Apply limits
    pred_limited = apply_limits(pred, X_eval, limits, high_limits)
    mae = float(np.mean(np.abs(pred_limited - Y_eval)))
    rmse = float(np.sqrt(np.mean((pred_limited - Y_eval) ** 2)))
    return mae, rmse, pred_limited


def apply_limits(residuals, X, limits, high_limits):
    lo, hi = limits
    hi_boost = high_limits
    limited = residuals.copy()
    limited = np.clip(limited, lo, hi)
    # For mainshock_mag >= 8.0, allow larger positive correction
    if "high_mag_8_0" in X.columns:
        high_mask = X["high_mag_8_0"].values > 0.5
        limited[high_mask] = np.clip(residuals[high_mask], lo, hi_boost)
    return limited


def parse_line(line: str) -> dict | None:
    parts = line.strip().split()
    if len(parts) < 7:
        return None
    return {"token": parts[0], "lon": float(parts[1]), "lat": float(parts[2]),
            "main_mag": float(parts[3]), "pred_mag": float(parts[4]),
            "pred_time_str": parts[-1]}


def load_predictions():
    pred_dir = FINAL_PKG / "predictions"
    records = []
    for fp in sorted(pred_dir.glob("*-T1-T2.csv")):
        tok = fp.stem.replace("-T1-T2", "")
        for i, line in enumerate(fp.read_text().strip().splitlines()):
            info = parse_line(line)
            if info:
                info["window"] = "T1" if i == 0 else "T2"
                records.append(info)
    for fp in sorted(pred_dir.glob("*-T3.csv")):
        tok = fp.stem.replace("-T3", "")
        for line in fp.read_text().strip().splitlines():
            info = parse_line(line)
            if info:
                info["window"] = "T3"
                records.append(info)
    return pd.DataFrame(records)


def build_test_features(preds_df):
    """Build features for final_t123 predictions."""
    X = {}
    X["pred_mag"] = preds_df["pred_mag"].values
    X["mainshock_mag"] = preds_df["main_mag"].values
    X["mag_diff"] = preds_df["main_mag"].values - preds_df["pred_mag"].values
    X["pred_mag_post"] = preds_df["pred_mag"].values
    X["mainshock_depth"] = np.full(len(preds_df), 10.0)
    X["high_mag_7_5"] = (preds_df["main_mag"] >= 7.5).astype(float).values
    X["high_mag_8_0"] = (preds_df["main_mag"] >= 8.0).astype(float).values
    X["high_mag_8_5"] = (preds_df["main_mag"] >= 8.5).astype(float).values
    for w in WINDOWS:
        X[f"win_{w}"] = (preds_df["window"] == w).astype(float).values
    # Placeholder for optional features used in training
    X["early_count"] = np.zeros(len(preds_df))
    X["early_max_mag"] = np.zeros(len(preds_df))
    X["plate_dist"] = np.full(len(preds_df), 100.0)
    X["mainshock_time_ordinal"] = np.zeros(len(preds_df))
    return pd.DataFrame(X)


def write_package(preds, out_dir):
    pdir = out_dir / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)
    for tok in sorted(preds["token"].unique()):
        rows = preds[preds["token"] == tok]
        t1, t2, t3 = rows[rows["window"] == "T1"], rows[rows["window"] == "T2"], rows[rows["window"] == "T3"]
        if len(t1) and len(t2):
            r1, r2 = t1.iloc[0], t2.iloc[0]
            lines = [
                f"{tok} {r1['lon']:.2f} {r1['lat']:.2f} {r1['main_mag']:.1f} "
                f"{r1['pred_mag']:.1f} (Ms) {r1['pred_time_str']}",
                f"{tok} {r2['lon']:.2f} {r2['lat']:.2f} {r2['main_mag']:.1f} "
                f"{r2['pred_mag']:.1f} (Ms) {r2['pred_time_str']}",
            ]
            (pdir / f"{tok}-T1-T2.csv").write_text("\n".join(lines) + "\n")
        if len(t3):
            r3 = t3.iloc[0]
            (pdir / f"{tok}-T3.csv").write_text(
                f"{tok} {r3['lon']:.2f} {r3['lat']:.2f} {r3['main_mag']:.1f} "
                f"{r3['pred_mag']:.1f} (Ms) {r3['pred_time_str']}\n")

    src = FINAL_PKG / "technical_docs"
    dst = out_dir / "technical_docs"
    if src.exists() and not dst.exists():
        shutil.copytree(src, dst)
    (out_dir / "MANIFEST.json").write_text(json.dumps({
        "description": "Magnitude residual calibrator",
        "created": datetime.now(timezone.utc).isoformat(),
    }, indent=2))


def zip_hash(out_dir, zip_path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(out_dir.rglob("*")):
            if fp.is_file():
                zf.write(fp, fp.relative_to(out_dir))
    sha = hashlib.sha256()
    with open(zip_path, "rb") as f:
        while data := f.read(8192):
            sha.update(data)
    h = sha.hexdigest()
    (zip_path.parent / (zip_path.name + ".sha256")).write_text(h)
    return h


def evaluate_zip(zip_path, name, output_csv):
    import subprocess
    args = [sys.executable, str(PROJECT / "scripts/evaluate_qualification_package.py"),
            "--zip-path", str(zip_path),
            "--output", str(output_csv),
            "--package-name", name]
    if output_csv.exists():
        args.append("--append")
    result = subprocess.run(args, capture_output=True, text=True, cwd=str(PROJECT))
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"[WARN] {result.stderr}")


def main():
    print("=" * 60)
    print("Magnitude Residual Calibrator")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_csv = OUT_DIR / "summary.csv"
    if summary_csv.exists():
        summary_csv.unlink()

    print("\n[1/4] Loading OOF data & building features...")
    oof_df = pd.read_csv(OOF_PATH)
    feat_df = pd.read_csv(FEAT_PATH)
    X_train, Y_train, X_eval, Y_eval, feature_names = build_features(oof_df, feat_df)
    print(f"  Train: {len(X_train)}, Eval: {len(X_eval)}, Features: {len(feature_names)}")

    print("\n[2/4] Training residual models...")
    best_models = {}
    for model_key in ["ridge", "huber", "isotonic"]:  # skip hgb for speed
        mtype, mobj = train_model(model_key, X_train, Y_train)
        best_models[model_key] = (mtype, mobj)
        for cand_name, cand_cfg in CANDIDATES.items():
            mae, rmse, _ = evaluate_model(mtype, mobj, model_key, X_eval, Y_eval,
                                          cand_cfg["mag_limits"], cand_cfg["high_mag_limit"])
            print(f"  {model_key}/{cand_name}: eval_mae={mae:.4f} eval_rmse={rmse:.4f}")

    print("\n[3/4] Loading final_t123 predictions & applying corrections...")
    preds = load_predictions()
    test_X = build_test_features(preds)

    # Select best model per candidate based on eval
    model_selector = {}
    for cand_name in CANDIDATES:
        best_mk, best_rmse = None, 999
        for mk, (mt, mo) in best_models.items():
            _, rmse, _ = evaluate_model(mt, mo, mk, X_eval, Y_eval,
                                        CANDIDATES[cand_name]["mag_limits"],
                                        CANDIDATES[cand_name]["high_mag_limit"])
            if rmse < best_rmse:
                best_rmse = rmse
                best_mk = mk
        model_selector[cand_name] = best_mk

    for cand_name, cand_cfg in CANDIDATES.items():
        mk = model_selector[cand_name]
        mtype, mobj = best_models[mk]
        residuals = predict_model(mtype, mobj, test_X, mk)
        residuals = apply_limits(residuals, test_X, cand_cfg["mag_limits"],
                                 cand_cfg["high_mag_limit"])

        corrected = preds.copy()
        corrected["pred_mag"] = np.clip(
            corrected["pred_mag"].values + residuals, 0.0, None)
        corrected["pred_mag"] = corrected["pred_mag"].round(1)
        corrected["_residual"] = residuals
        changed = np.abs(residuals) > 0.05
        print(f"\n  [{cand_name}] model={mk}, changed={changed.sum()}/{len(preds)}")
        if changed.any():
            for i in np.where(changed)[0][:5]:
                print(f"    {preds.iloc[i]['token']} {preds.iloc[i]['window']}: "
                      f"{preds.iloc[i]['pred_mag']:.1f} + {residuals[i]:.2f} → {corrected.iloc[i]['pred_mag']:.1f}")

        pkg_dir = OUT_DIR / f"package_{cand_name}"
        zip_path = OUT_DIR / f"qualification_submission_mag_residual_{cand_name}.zip"
        if pkg_dir.exists():
            shutil.rmtree(pkg_dir)
        pkg_dir.mkdir(parents=True)
        write_package(corrected, pkg_dir)
        zip_hash(pkg_dir, zip_path)
        evaluate_zip(zip_path, f"mag_residual_{cand_name}", summary_csv)

    # Recommendation
    lines = ["# Magnitude Residual Calibrator 报告", "",
             f"- 生成时间: {datetime.now(timezone.utc).isoformat()}",
             f"- 训练样本: {len(X_train)} OOF rows, {len(feature_names)} features",
             f"- 评估样本: {len(X_eval)} OOF rows", "",
             "## 候选模型选择", ""]
    for cand_name, mk in model_selector.items():
        lines.append(f"- **{cand_name}**: model={mk}, limits={CANDIDATES[cand_name]['mag_limits']}, "
                     f"high_mag_limit={CANDIDATES[cand_name]['high_mag_limit']}")
    lines.append("")
    lines.append("## 指标汇总")
    if summary_csv.exists():
        lines.append("```csv")
        lines.append(summary_csv.read_text())
        lines.append("```")
    lines.extend(["", "## 推荐", "",
                  "推荐 `balanced` 候选用于正式提交。",
                  "`conservative` 适合对泛化要求极高的场景。",
                  "`aggressive` 风险较大，仅用于对比实验。"])
    (OUT_DIR / "recommendation.md").write_text("\n".join(lines))
    print(f"\nDone! Results: {OUT_DIR}")


if __name__ == "__main__":
    main()
