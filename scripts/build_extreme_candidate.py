#!/usr/bin/env python3
"""
Extreme Risk Candidate Enhancement.

Keeps T1/T2/T3 time predictions from final_t123 unchanged.
Adds magnitude floor correction for high-risk mainshocks.

Strategy:
1. Train an extreme risk classifier on OOF data:
   - risk label: target_window_max_mag >= mainshock_mag - 2.0 OR >= 6.5
   - Features: mainshock_mag, plate_type_SUB, bath_deficit, early_aftershock_count, etc.
   - Legal window features only (no future leakage)
2. For high-risk predictions, apply magnitude floor:
   - pred_mag = max(pred_mag, floor_value)
   - floor_value = OOF-optimized via grid search on training data
3. Package into independent candidate zip.

Seed = 42 for reproducibility. No GPU required.
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
PROJECT_ROOT = Path("/home/ningyd/CodingSpace/aftershock_qualification_train")
FEATURES_PATH = PROJECT_ROOT / "data/processed/qualification_features.csv"
DECOUPLED_OOF_PATH = PROJECT_ROOT / "data/models/qualification_decoupled_full/decoupled_oof_predictions.csv"
FINAL_T123_PKG = PROJECT_ROOT / "submission_package_final_t123_no_commitment"
OUTPUT_DIR = PROJECT_ROOT / "experiments/extreme_risk_candidate/package"
ZIP_PATH = PROJECT_ROOT / "experiments/extreme_risk_candidate/qualification_submission_extreme_risk_no_commitment.zip"

# ── Per-window observation hours (legal features) ──
WINDOW_OBS_HOURS = {"T1": 0.0, "T2": 24.0, "T3": 72.0}

# ── Features used for extreme risk classification ──
# These must be LEGAL for all windows (T1 has 0h observations)
RISK_FEATURES_T1 = [
    "mainshock_mag", "mainshock_depth",
    "plate_type_SUB", "plate_boundary_distance_km",
    "bath_deficit", "bath_valid",
    "gr_b_value", "omori_p",
]

RISK_FEATURES_T2 = RISK_FEATURES_T1 + [
    "early_aftershock_count", "early_max_mag", "bath_early_max_mag",
    "count_24h", "energy_24h",
]

RISK_FEATURES_T3 = RISK_FEATURES_T1 + [
    "early_aftershock_count", "early_max_mag", "bath_early_max_mag",
    "count_72h", "energy_72h", "etas_branching_ratio",
]

WINDOW_RISK_FEATURES = {
    "T1": RISK_FEATURES_T1,
    "T2": RISK_FEATURES_T2,
    "T3": RISK_FEATURES_T3,
}


def load_training_data():
    """Load OOF and feature data, build per-window extreme risk labels."""
    feat = pd.read_csv(FEATURES_PATH)
    oof = pd.read_csv(DECOUPLED_OOF_PATH)

    # Build per-window risk labels
    windows = []
    for w in ["T1", "T2", "T3"]:
        w_oof = oof[oof["window"] == w].copy()
        w_feat = feat.copy()

        # Risk definition: max mag in window >= mainshock_mag - 2.0, OR mag >= 6.5
        target_mag_col = f"target_{w}_max_mag"
        if target_mag_col not in w_feat.columns:
            continue

        w_df = w_feat.merge(
            w_oof[["mainshock_id", "decoupled_mag_raw", "decoupled_time_raw"]],
            on="mainshock_id", how="inner"
        )

        # Risk label
        w_df["risk_extreme"] = (
            (w_df[target_mag_col] >= w_df["mainshock_mag"] - 2.0) |
            (w_df[target_mag_col] >= 6.5)
        ).astype(int)

        # Only rows with predictions
        w_df = w_df.dropna(subset=["decoupled_mag_raw"])
        w_df["window"] = w

        # Fill features
        risk_feats = [f for f in WINDOW_RISK_FEATURES[w] if f in w_df.columns]
        w_df[risk_feats] = w_df[risk_feats].fillna(0.0)

        windows.append(w_df)

    df = pd.concat(windows, ignore_index=True)
    print(f"[Extreme Risk] Total training samples: {len(df)}")
    for w in ["T1", "T2", "T3"]:
        wdf = df[df["window"] == w]
        risk_rate = wdf["risk_extreme"].mean() if len(wdf) > 0 else 0
        print(f"  {w}: {len(wdf)} rows, extreme risk rate = {risk_rate:.3f}")

    return df


def train_extreme_risk_classifier(df: pd.DataFrame):
    """Train per-window extreme risk classifiers using LightGBM."""
    classifiers = {}
    thresholds = {}
    oof_prob_thresholds = {}

    for w in ["T1", "T2", "T3"]:
        wdf = df[df["window"] == w].copy()
        if len(wdf) < 50:
            print(f"[Extreme Risk] {w}: too few samples, skipping")
            continue

        risk_feats = [f for f in WINDOW_RISK_FEATURES[w] if f in wdf.columns]
        if len(risk_feats) < 3:
            print(f"[Extreme Risk] {w}: insufficient features, skipping")
            continue

        X = wdf[risk_feats].fillna(0.0).values.astype(float)
        y = wdf["risk_extreme"].values.astype(int)
        main_mags = wdf["mainshock_mag"].values
        true_mags = wdf[f"target_{w}_max_mag"].values
        pred_mags = wdf["decoupled_mag_raw"].values

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        print(f"[Extreme Risk] {w}: {n_pos} positive, {n_neg} negative")

        try:
            from sklearn.model_selection import KFold
            import lightgbm as lgb

            kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
            oof_probs = np.zeros(len(wdf))

            for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X)):
                X_tr, X_val = X[tr_idx], X[val_idx]
                y_tr = y[tr_idx]
                # Handle class imbalance
                pos_weight = max(1.0, n_neg / max(1, n_pos))
                clf = lgb.LGBMClassifier(
                    n_estimators=150, learning_rate=0.05, num_leaves=31,
                    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                    class_weight="balanced" if pos_weight > 2 else None,
                    random_state=SEED + fold_idx, verbose=-1,
                )
                clf.fit(X_tr, y_tr)
                oof_probs[val_idx] = clf.predict_proba(X_val)[:, 1]

            # Grid search for optimal probability threshold
            best_threshold = 0.5
            best_floor = 0.0
            best_mae = float("inf")

            # Search thresholds
            for prob_thresh in np.arange(0.2, 0.8, 0.05):
                high_risk = oof_probs >= prob_thresh
                # Search floor margins
                for floor_margin in np.arange(0.0, 2.5, 0.25):
                    adjusted_mags = pred_mags.copy()
                    for i in range(len(wdf)):
                        if high_risk[i]:
                            # Floor = max(pred, main_mag - floor_margin)
                            floor_val = max(pred_mags[i], main_mags[i] - floor_margin)
                            # Blend with p75 from training
                            adjust = max(0, (main_mags[i] - pred_mags[i] - floor_margin) * 0.5)
                            adjusted_mags[i] = max(pred_mags[i], main_mags[i] - floor_margin) + adjust

                    mae = np.mean(np.abs(adjusted_mags - true_mags))
                    if mae < best_mae:
                        best_mae = mae
                        best_threshold = prob_thresh
                        best_floor = floor_margin

            print(f"[Extreme Risk] {w}: best_threshold={best_threshold:.2f}, best_floor_margin={best_floor:.2f}, OOF MAE={best_mae:.3f}")

            # Train final model
            pos_weight = max(1.0, n_neg / max(1, n_pos))
            final_clf = lgb.LGBMClassifier(
                n_estimators=150, learning_rate=0.05, num_leaves=31,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                class_weight="balanced" if pos_weight > 2 else None,
                random_state=SEED, verbose=-1,
            )
            final_clf.fit(X, y)

            classifiers[w] = final_clf
            thresholds[w] = best_threshold
            oof_prob_thresholds[w] = best_floor

        except ImportError:
            print(f"[Extreme Risk] {w}: lightgbm not available, using rule-based approach")

    # Rule-based fallback if LightGBM unavailable
    if not classifiers:
        print("\n[Extreme Risk] Using rule-based fallback classifiers")
        for w in ["T1", "T2", "T3"]:
            wdf = df[df["window"] == w].copy()
            if len(wdf) < 10:
                continue
            # Simple heuristic: extreme if mainshock_mag >= 7.5 and bath_deficit > 0
            extreme_rate = wdf["risk_extreme"].mean()
            thresholds[w] = 0.4
            oof_prob_thresholds[w] = 1.5  # floor margin
            print(f"  {w}: extreme_rate={extreme_rate:.3f}, floor_margin=1.5")

    return classifiers, thresholds, oof_prob_thresholds


def load_test_features():
    """Load test features."""
    feat = pd.read_csv(PROJECT_ROOT / "data/processed/test_sequences_features.csv")
    feat["mainshock_id_clean"] = feat["mainshock_id"].str.replace("_eq", "")
    return feat


def parse_prediction_line(line: str):
    """Parse prediction line."""
    parts = line.strip().split()
    if len(parts) < 7:
        return None
    return {
        "token": parts[0],
        "lon": float(parts[1]),
        "lat": float(parts[2]),
        "main_mag": float(parts[3]),
        "pred_mag": float(parts[4]),
        "pred_time_str": parts[-1],
    }


def load_final_t123_predictions():
    """Read all final_t123 predictions."""
    pred_dir = FINAL_T123_PKG / "predictions"
    records = []
    for fpath in sorted(pred_dir.glob("*-T1-T2.csv")):
        token = fpath.stem.replace("-T1-T2", "")
        lines = fpath.read_text().strip().splitlines()
        for i, line in enumerate(lines):
            info = parse_prediction_line(line)
            if info is None:
                continue
            info["window"] = "T1" if i == 0 else "T2"
            records.append(info)
    for fpath in sorted(pred_dir.glob("*-T3.csv")):
        token = fpath.stem.replace("-T3", "")
        lines = fpath.read_text().strip().splitlines()
        if lines:
            info = parse_prediction_line(lines[0])
            if info is None:
                continue
            info["window"] = "T3"
            records.append(info)
    return pd.DataFrame(records)


def apply_extreme_risk_correction(
    preds: pd.DataFrame,
    test_feat: pd.DataFrame,
    classifiers: dict,
    thresholds: dict,
    floor_margins: dict,
):
    """Apply extreme risk magnitude floor to predictions."""
    corrected = preds.copy()
    corrected = corrected.merge(
        test_feat[["mainshock_id_clean"]],
        left_on="token", right_on="mainshock_id_clean", how="left"
    )

    # Merge with test features
    risk_info = test_feat.copy()
    risk_info.set_index("mainshock_id_clean", inplace=True)

    for w in ["T1", "T2", "T3"]:
        if w not in classifiers:
            continue
        w_mask = corrected["window"] == w

        # Get features for this window
        risk_feats = [f for f in WINDOW_RISK_FEATURES[w] if f in risk_info.columns]
        if len(risk_feats) < 3:
            continue

        clf = classifiers.get(w)
        threshold = thresholds.get(w, 0.5)
        floor_margin = floor_margins.get(w, 1.5)

        for idx in corrected[w_mask].index:
            token = corrected.at[idx, "token"]
            main_mag = corrected.at[idx, "main_mag"]
            pred_mag = corrected.at[idx, "pred_mag"]

            if token in risk_info.index:
                row = risk_info.loc[token]
                feats = row[risk_feats].fillna(0.0).values.astype(float).reshape(1, -1)

                if clf is not None:
                    try:
                        prob = clf.predict_proba(feats)[0, 1]
                    except Exception:
                        prob = 0.0
                else:
                    # Rule-based
                    prob = 0.6 if (main_mag >= 7.5 and row.get("bath_deficit", 0) > 0) else 0.1

                if prob >= threshold:
                    # Apply floor
                    floor_val = main_mag - floor_margin
                    new_mag = max(pred_mag, floor_val)
                    if new_mag > pred_mag + 0.05:
                        corrected.at[idx, "pred_mag"] = round(new_mag, 1)
                        corrected.at[idx, "_risk_flagged"] = True
                        corrected.at[idx, "_risk_prob"] = prob
                        corrected.at[idx, "_floor_val"] = floor_val

    # Fill missing columns
    if "_risk_flagged" not in corrected.columns:
        corrected["_risk_flagged"] = False
    if "_risk_prob" not in corrected.columns:
        corrected["_risk_prob"] = 0.0

    return corrected


def write_prediction_files(pred_df: pd.DataFrame, output_dir: Path):
    """Write T1-T2 and T3 CSV files."""
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for mainshock_id in sorted(pred_df["token"].unique()):
        ms_rows = pred_df[pred_df["token"] == mainshock_id]
        t1_row = ms_rows[ms_rows["window"] == "T1"]
        t2_row = ms_rows[ms_rows["window"] == "T2"]
        t3_row = ms_rows[ms_rows["window"] == "T3"]

        if len(t1_row) > 0 and len(t2_row) > 0:
            t1 = t1_row.iloc[0]
            t2 = t2_row.iloc[0]
            t1t2_lines = []
            for row_info in [t1, t2]:
                t1t2_lines.append(
                    f"{row_info['token']} {row_info['lon']:.2f} {row_info['lat']:.2f} "
                    f"{row_info['main_mag']:.1f} {row_info['pred_mag']:.1f} (Ms) {row_info['pred_time_str']}"
                )
            (pred_dir / f"{mainshock_id}-T1-T2.csv").write_text("\n".join(t1t2_lines) + "\n")

        if len(t3_row) > 0:
            t3 = t3_row.iloc[0]
            t3_lines = [
                f"{t3['token']} {t3['lon']:.2f} {t3['lat']:.2f} "
                f"{t3['main_mag']:.1f} {t3['pred_mag']:.1f} (Ms) {t3['pred_time_str']}"
            ]
            (pred_dir / f"{mainshock_id}-T3.csv").write_text("\n".join(t3_lines) + "\n")


def copy_technical_docs(output_dir: Path):
    """Copy technical_docs from final_t123."""
    src_docs = FINAL_T123_PKG / "technical_docs"
    dst_docs = output_dir / "technical_docs"
    if src_docs.exists():
        if dst_docs.exists():
            shutil.rmtree(dst_docs)
        shutil.copytree(src_docs, dst_docs)


def create_zip(output_dir: Path, zip_path: Path):
    """Create ZIP archive."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in sorted(output_dir.rglob("*")):
            if fpath.is_file():
                arcname = fpath.relative_to(output_dir)
                zf.write(fpath, arcname)
    print(f"[ZIP] Created: {zip_path} ({zip_path.stat().st_size:,} bytes)")


def main():
    import datetime
    import json

    print("=" * 60)
    print("Extreme Risk Candidate Enhancement")
    print("=" * 60)

    # Step 1: Train extreme risk classifier
    print("\n[1/4] Training extreme risk classifier from OOF data...")
    df = load_training_data()
    classifiers, thresholds, floor_margins = train_extreme_risk_classifier(df)

    # Step 2: Load final_t123 predictions
    print("\n[2/4] Loading final_t123 predictions...")
    final_preds = load_final_t123_predictions()
    print(f"  Loaded {len(final_preds)} predictions")

    # Step 3: Load test features & apply correction
    print("\n[3/4] Applying extreme risk magnitude floor...")
    test_feat = load_test_features()
    corrected_preds = apply_extreme_risk_correction(
        final_preds, test_feat, classifiers, thresholds, floor_margins
    )

    # Print changes
    changed = corrected_preds[corrected_preds["_risk_flagged"] == True]
    print(f"\n  Risk-flagged predictions: {len(changed)}")
    for _, row in changed.iterrows():
        orig_row = final_preds[
            (final_preds["token"] == row["token"]) &
            (final_preds["window"] == row["window"])
        ]
        if len(orig_row) > 0:
            orig_mag = orig_row.iloc[0]["pred_mag"]
            print(f"    {row['token']} {row['window']}: mag {orig_mag} -> {row['pred_mag']:.1f} "
                  f"(prob={row.get('_risk_prob', 0):.2f}, floor={row.get('_floor_val', 0):.1f})")

    # Step 4: Write package
    print("\n[4/4] Writing package...")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    write_prediction_files(corrected_preds, OUTPUT_DIR)
    copy_technical_docs(OUTPUT_DIR)

    # Write MANIFEST
    manifest = {
        "package_name": "qualification_submission_extreme_risk_no_commitment",
        "description": "Extreme risk candidate: keeps final_t123 time predictions, adds extreme magnitude floor",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base_package": "qualification_submission_final_t123_no_commitment",
        "modifications": ["T1/T2/T3 magnitude predictions adjusted with extreme risk floor"],
        "seed": SEED,
    }
    (OUTPUT_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    create_zip(OUTPUT_DIR, ZIP_PATH)

    import hashlib
    sha = hashlib.sha256()
    with open(ZIP_PATH, "rb") as f:
        while True:
            data = f.read(8192)
            if not data:
                break
            sha.update(data)
    sha_str = sha.hexdigest()
    sha_path = ZIP_PATH.with_suffix(ZIP_PATH.suffix + ".sha256")
    sha_path.write_text(sha_str)
    print(f"[SHA256] {sha_str}")

    print("\nDone! Extreme risk candidate package ready.")
    print(f"  Package dir: {OUTPUT_DIR}")
    print(f"  ZIP: {ZIP_PATH}")


if __name__ == "__main__":
    main()
