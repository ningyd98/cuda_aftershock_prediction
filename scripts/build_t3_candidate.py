#!/usr/bin/env python3
"""
T3 Hazard Time Candidate Enhancement.

Keeps T1/T2 predictions from final_t123 unchanged.
Replaces T3 time predictions using OOF-driven calibration.

Strategy:
1. Analyze T3 time residual patterns from OOF decoupled predictions.
2. Use per-mag-bin bias correction + LightGBM residual corrector.
3. Apply to test predictions and package.

Seed = 42.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)
PROJECT_ROOT = Path("/home/ningyd/CodingSpace/aftershock_qualification_train")
FEATURES_PATH = PROJECT_ROOT / "data/processed/qualification_features.csv"
DECOUPLED_OOF_PATH = PROJECT_ROOT / "data/models/qualification_decoupled_full/decoupled_oof_predictions.csv"
FINAL_T123_PKG = PROJECT_ROOT / "submission_package_final_t123_no_commitment"
OUTPUT_DIR = PROJECT_ROOT / "experiments/t3_hazard_candidate/package"
ZIP_PATH = PROJECT_ROOT / "experiments/t3_hazard_candidate/qualification_submission_t3_hazard_no_commitment.zip"

T3_LOWER, T3_UPPER = 72.0, 168.0

# Features for T3 correction - only static/legal
T3_CORRECTION_FEATURES = [
    "mainshock_mag", "mainshock_depth",
    "plate_type_SUB", "plate_boundary_distance_km",
    "bath_deficit", "bath_early_max_mag",
    "early_aftershock_count", "early_max_mag",
    "count_72h", "energy_72h",
    "etas_branching_ratio", "gr_b_value",
]


def load_training_data():
    feat = pd.read_csv(FEATURES_PATH)
    oof = pd.read_csv(DECOUPLED_OOF_PATH)
    t3_oof = oof[oof["window"] == "T3"].dropna(subset=["decoupled_time_raw"]).copy()
    df = feat.merge(
        t3_oof[["mainshock_id", "decoupled_time_raw"]],
        on="mainshock_id", how="inner"
    )
    df = df[df["has_T3_aftershock"] == True].dropna(subset=["target_T3_time_to_max_hours"]).copy()
    df["t3_time_residual"] = df["decoupled_time_raw"] - df["target_T3_time_to_max_hours"]

    avail = [f for f in T3_CORRECTION_FEATURES if f in df.columns]
    df[avail] = df[avail].fillna(0.0)

    print(f"[T3 Cal] Samples: {len(df)}, mean residual: {df['t3_time_residual'].mean():.1f}h, "
          f"std: {df['t3_time_residual'].std():.1f}h")
    return df, avail


def parse_line(line: str):
    parts = line.strip().split()
    if len(parts) < 7:
        return None
    return {
        "token": parts[0], "lon": float(parts[1]), "lat": float(parts[2]),
        "main_mag": float(parts[3]), "pred_mag": float(parts[4]),
        "pred_time_str": parts[-1],
    }


def token_to_hours(token: str, time_str: str) -> float:
    ms = pd.Timestamp(
        year=int(token[0:4]), month=int(token[4:6]), day=int(token[6:8]),
        hour=int(token[8:10]), minute=int(token[10:12]), second=int(token[12:14]), tz="UTC"
    )
    pt = pd.Timestamp(
        year=int(time_str[0:4]), month=int(time_str[4:6]),
        day=int(time_str[6:8]), hour=int(time_str[8:10]),
        minute=0, second=0, tz="UTC"
    )
    return (pt - ms).total_seconds() / 3600.0


def hours_to_str(token: str, hours: float) -> str:
    ms = pd.Timestamp(
        year=int(token[0:4]), month=int(token[4:6]), day=int(token[6:8]),
        hour=int(token[8:10]), minute=int(token[10:12]), second=int(token[12:14]), tz="UTC"
    )
    pt = ms + pd.Timedelta(hours=round(hours))
    return pt.strftime("%Y%m%d%H")


def load_test_features():
    feat = pd.read_csv(PROJECT_ROOT / "data/processed/test_sequences_features.csv")
    feat["mainshock_id_clean"] = feat["mainshock_id"].str.replace("_eq", "")
    return feat


def load_predictions():
    pred_dir = FINAL_T123_PKG / "predictions"
    records = []
    for fpath in sorted(pred_dir.glob("*-T1-T2.csv")):
        token = fpath.stem.replace("-T1-T2", "")
        for i, line in enumerate(fpath.read_text().strip().splitlines()):
            info = parse_line(line)
            if info:
                info["window"] = "T1" if i == 0 else "T2"
                records.append(info)
    for fpath in sorted(pred_dir.glob("*-T3.csv")):
        token = fpath.stem.replace("-T3", "")
        for line in fpath.read_text().strip().splitlines():
            info = parse_line(line)
            if info:
                info["window"] = "T3"
                records.append(info)
    return pd.DataFrame(records)


def train_correction(df, feature_cols):
    """Train T3 time correction model."""
    # Per-mag-bin bias
    df["mag_bin"] = pd.cut(df["mainshock_mag"], bins=[5, 6.5, 7.0, 7.5, 8.0, 10.0],
                           labels=["5-6.5", "6.5-7", "7-7.5", "7.5-8", "8+"])
    mag_bias = {str(k): float(v) for k, v in df.groupby("mag_bin", observed=False)["t3_time_residual"].mean().items()}
    global_bias = float(df["t3_time_residual"].mean())

    print(f"[T3 Cal] Global bias: {global_bias:.1f}h")
    print(f"[T3 Cal] Mag bias: {mag_bias}")

    # Try LightGBM residual model
    lgb_model = None
    lgb_features = None
    try:
        import lightgbm as lgb
        X = df[feature_cols].values.astype(float)
        y = np.clip(df["t3_time_residual"].values, -60, 60)
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        oof_preds = np.zeros(len(df))
        for fi, (tr, va) in enumerate(kf.split(X)):
            m = lgb.LGBMRegressor(
                n_estimators=150, learning_rate=0.05, num_leaves=31,
                min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.0, random_state=SEED+fi, verbose=-1
            )
            m.fit(X[tr], y[tr])
            oof_preds[va] = m.predict(X[va])
        corrected = df["decoupled_time_raw"] - oof_preds
        orig_mae = df["t3_time_residual"].abs().mean()
        new_mae = (corrected - df["target_T3_time_to_max_hours"]).abs().mean()
        print(f"[T3 Cal] OOF orig MAE: {orig_mae:.1f}h, corrected MAE: {new_mae:.1f}h, "
              f"gain: {orig_mae - new_mae:.1f}h")

        if new_mae < orig_mae - 0.5:
            final_m = lgb.LGBMRegressor(
                n_estimators=150, learning_rate=0.05, num_leaves=31,
                min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, verbose=-1
            )
            final_m.fit(X, y)
            lgb_model = final_m
            lgb_features = feature_cols
            print(f"[T3 Cal] ML model enabled (improvement {orig_mae-new_mae:.1f}h)")
        else:
            print(f"[T3 Cal] ML model not improving, using bias-only correction")
    except ImportError:
        print(f"[T3 Cal] sklearn/lightgbm not available, using bias-only correction")

    return {
        "global_bias": global_bias,
        "mag_bias": mag_bias,
        "lgb_model": lgb_model,
        "lgb_features": lgb_features,
    }


def apply_correction(preds, test_feat, cal):
    """Apply T3 time correction."""
    corrected = preds.copy()
    t3_mask = corrected["window"] == "T3"

    test_idx = test_feat.set_index("mainshock_id_clean")

    global_bias = cal["global_bias"]
    mag_bias = cal["mag_bias"]
    lgb_model = cal["lgb_model"]
    lgb_features = cal["lgb_features"]

    # Mag bin edges for mapping
    bin_edges = {"5-6.5": (5, 6.5), "6.5-7": (6.5, 7), "7-7.5": (7, 7.5), "7.5-8": (7.5, 8), "8+": (8, 10)}

    for idx in corrected[t3_mask].index:
        token = corrected.at[idx, "token"]
        main_mag = corrected.at[idx, "main_mag"]
        orig_time_str = corrected.at[idx, "pred_time_str"]
        orig_time_h = token_to_hours(token, orig_time_str)

        correction = -global_bias  # subtract global bias

        # Per-mag-bin adjustment
        for bin_name, (lo, hi) in bin_edges.items():
            if lo <= main_mag < hi or (bin_name == "8+" and main_mag >= 8):
                if bin_name in mag_bias:
                    correction += -mag_bias[bin_name] * 0.5
                break

        # ML correction if available
        if lgb_model is not None and lgb_features is not None and token in test_idx.index:
            feats = test_idx.loc[token][lgb_features].fillna(0.0).values.astype(float).reshape(1, -1)
            try:
                ml_corr = lgb_model.predict(feats)[0]
                correction = -(0.3 * global_bias + 0.7 * ml_corr)
            except Exception:
                pass

        new_time = orig_time_h + correction
        new_time = np.clip(new_time, T3_LOWER + 1, T3_UPPER - 1)
        corrected.at[idx, "pred_time_str"] = hours_to_str(token, new_time)

    return corrected


def write_package(pred_df, output_dir):
    pdir = output_dir / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)
    for tok in sorted(pred_df["token"].unique()):
        rows = pred_df[pred_df["token"] == tok]
        t1 = rows[rows["window"] == "T1"]
        t2 = rows[rows["window"] == "T2"]
        t3 = rows[rows["window"] == "T3"]
        if len(t1) and len(t2):
            r1, r2 = t1.iloc[0], t2.iloc[0]
            lines = [
                f"{tok} {r1['lon']:.2f} {r1['lat']:.2f} {r1['main_mag']:.1f} {r1['pred_mag']:.1f} (Ms) {r1['pred_time_str']}",
                f"{tok} {r2['lon']:.2f} {r2['lat']:.2f} {r2['main_mag']:.1f} {r2['pred_mag']:.1f} (Ms) {r2['pred_time_str']}",
            ]
            (pdir / f"{tok}-T1-T2.csv").write_text("\n".join(lines) + "\n")
        if len(t3):
            r3 = t3.iloc[0]
            (pdir / f"{tok}-T3.csv").write_text(
                f"{tok} {r3['lon']:.2f} {r3['lat']:.2f} {r3['main_mag']:.1f} {r3['pred_mag']:.1f} (Ms) {r3['pred_time_str']}\n"
            )

    # Copy technical_docs
    src = FINAL_T123_PKG / "technical_docs"
    dst = output_dir / "technical_docs"
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    # Manifest
    (output_dir / "MANIFEST.json").write_text(json.dumps({
        "package_name": "qualification_submission_t3_hazard_no_commitment",
        "description": "T3 time hazard candidate: T1/T2 unchanged, T3 time calibrated via OOF bias correction",
        "created": datetime.now(timezone.utc).isoformat(),
        "base": "qualification_submission_final_t123_no_commitment",
        "seed": SEED,
    }, indent=2))


def zip_and_hash(output_dir, zip_path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(output_dir.rglob("*")):
            if fp.is_file():
                zf.write(fp, fp.relative_to(output_dir))
    sha = hashlib.sha256()
    with open(zip_path, "rb") as f:
        while data := f.read(8192):
            sha.update(data)
    h = sha.hexdigest()
    (zip_path.parent / (zip_path.name + ".sha256")).write_text(h)
    print(f"[ZIP] {zip_path} ({zip_path.stat().st_size:,}B) SHA256: {h[:16]}...")


def main():
    print("=" * 60)
    print("T3 Hazard Candidate Enhancement")
    print("=" * 60)

    print("\n[1/4] Training T3 time correction...")
    df, feats = load_training_data()
    cal = train_correction(df, feats)

    print("\n[2/4] Loading final_t123 predictions...")
    preds = load_predictions()
    print(f"  Loaded {len(preds)} predictions")

    print("\n[3/4] Applying T3 correction...")
    test_feat = load_test_features()
    corrected = apply_correction(preds, test_feat, cal)

    # Show changes
    t3_old = preds[preds["window"] == "T3"]
    t3_new = corrected[corrected["window"] == "T3"]
    print("\n  T3 time changes:")
    for _, nr in t3_new.iterrows():
        or_ = t3_old[t3_old["token"] == nr["token"]]
        if len(or_):
            print(f"    {nr['token']}: {or_.iloc[0]['pred_time_str']} -> {nr['pred_time_str']}")

    print("\n[4/4] Writing package...")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    write_package(corrected, OUTPUT_DIR)
    zip_and_hash(OUTPUT_DIR, ZIP_PATH)
    print("\nDone!")


if __name__ == "__main__":
    main()
