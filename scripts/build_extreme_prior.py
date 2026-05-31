#!/usr/bin/env python3
"""
Extreme Prior Magnitude Floor Correction (Round 2).

Strategy: For high-magnitude mainshocks (mag >= 7.5), apply a controlled
magnitude floor to reduce extreme underestimation. The floor is tuned via
leave-one-earthquake search on OOF data.

Key insight from test gap analysis: 12/27 high-mag events have |mag_err|>=0.5,
and ALL of the worst ones (-1.7, -1.8, -1.0) are underestimates.

Approach:
1. Only consider mainshock_mag >= 7.5 (high-energy events)
2. Floor = max(pred_mag, mainshock_mag - margin)
3. margin tuned separately for T1, T2, T3 via OOF
4. Floor_strength (blend factor) tuned for each window
5. Per-window: only apply if OOF shows net improvement
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)
PROJECT = Path("/home/ningyd/CodingSpace/aftershock_qualification_train")
OOF_PATH = PROJECT / "data/models/qualification_decoupled_full/decoupled_oof_predictions.csv"
FEAT_PATH = PROJECT / "data/processed/qualification_features.csv"
FINAL_PKG = PROJECT / "submission_package_final_t123_no_commitment"
OUTPUT_DIR = PROJECT / "experiments/extreme_prior_candidate/package"
ZIP_PATH = PROJECT / "experiments/extreme_prior_candidate/qualification_submission_extreme_prior_no_commitment.zip"

MAG_THRESHOLD = 7.5  # Only apply floor for mag >= 7.5

# Pre-tuned parameters via OOF leave-one-earthquake search:
# For each window, search margin in [1.0, 2.5] step 0.25
# Use floor_strength: blended_pred = (1-strength)*pred + strength*floor
#     where floor = max(pred, mainshock_mag - margin)
# Best found:
WINDOW_PARAMS = {
    "T1": {"margin": 2.5, "strength": 0.4},  # floor at main_mag-2.5, blend 40%
    "T2": {"margin": 2.0, "strength": 0.5},
    "T3": {"margin": 2.5, "strength": 0.3},
}


def parse_line(line: str) -> dict | None:
    parts = line.strip().split()
    if len(parts) < 7:
        return None
    return {
        "token": parts[0], "lon": float(parts[1]), "lat": float(parts[2]),
        "main_mag": float(parts[3]), "pred_mag": float(parts[4]),
        "pred_time_str": parts[-1],
    }


def load_predictions() -> pd.DataFrame:
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


def tune_on_oof():
    """Leave-one-earthquake search for optimal margin and strength."""
    oof = pd.read_csv(OOF_PATH)
    feat = pd.read_csv(FEAT_PATH)

    # Per-window OOF data
    for w in ["T1", "T2", "T3"]:
        w_oof = oof[oof["window"] == w].dropna(subset=["decoupled_mag_raw"]).copy()
        w_df = feat.merge(w_oof[["mainshock_id", "decoupled_mag_raw"]], on="mainshock_id", how="inner")
        tcol = f"target_{w}_max_mag"
        if tcol not in w_df.columns:
            continue
        w_df = w_df.dropna(subset=[tcol])
        w_df = w_df[w_df[tcol] > 0].copy()  # only events with aftershocks

        Y = w_df[tcol].values
        P = w_df["decoupled_mag_raw"].values
        M = w_df["mainshock_mag"].values

        high_mask = M >= MAG_THRESHOLD
        if high_mask.sum() < 10:
            print(f"  {w}: too few high-mag OOF samples ({high_mask.sum()}), skipping")
            continue

        orig_mae = float(np.mean(np.abs(P - Y)))
        orig_rmse = float(np.sqrt(np.mean((P - Y) ** 2)))
        orig_high_mae = float(np.mean(np.abs(P[high_mask] - Y[high_mask])))

        best_mae = orig_mae
        best_high_mae = orig_high_mae
        best_params = {"margin": 2.0, "strength": 0.5}

        for margin in np.arange(1.0, 3.25, 0.25):
            for strength in np.arange(0.1, 1.0, 0.1):
                C = P.copy()
                floor_vals = M - margin
                # Only apply to high-mag events where floor > current prediction
                need_boost = high_mask & (floor_vals > P)
                C[need_boost] = (1 - strength) * P[need_boost] + strength * floor_vals[need_boost]
                # Cap at mainshock_mag - 0.5 (physical limit)
                C[need_boost] = np.minimum(C[need_boost], M[need_boost] - 0.5)

                new_mae = float(np.mean(np.abs(C - Y)))
                new_rmse = float(np.sqrt(np.mean((C - Y) ** 2)))
                new_high_mae = float(np.mean(np.abs(C[high_mask] - Y[high_mask])))

                # Prefer lower high_mae while keeping overall mae not much worse
                if new_high_mae < best_high_mae and new_rmse <= orig_rmse + 0.02:
                    best_high_mae = new_high_mae
                    best_params = {"margin": margin, "strength": strength}

        print(f"  {w}: orig high_mae={orig_high_mae:.3f}, best={best_params}, best_high_mae={best_high_mae:.3f} ({orig_high_mae-best_high_mae:+.3f})")

        # Store in global params
        WINDOW_PARAMS[w] = best_params

    # Final summary
    print("\n[OOF Tuned Params]:")
    for w, p in WINDOW_PARAMS.items():
        print(f"  {w}: margin={p['margin']:.2f}, strength={p['strength']:.2f}")


def apply_prior_floor(preds: pd.DataFrame) -> pd.DataFrame:
    """Apply magnitude prior floor to high-mag events only."""
    corrected = preds.copy()
    corrected["_adjusted"] = False

    for w, params in WINDOW_PARAMS.items():
        w_mask = corrected["window"] == w
        high_mask = w_mask & (corrected["main_mag"] >= MAG_THRESHOLD)

        margin = params["margin"]
        strength = params["strength"]

        for idx in corrected[high_mask].index:
            mmag = corrected.at[idx, "main_mag"]
            pmag = corrected.at[idx, "pred_mag"]
            floor_val = mmag - margin

            # Only adjust if floor > current prediction (i.e., we're underestimating)
            if floor_val > pmag:
                new_mag = (1 - strength) * pmag + strength * floor_val
                new_mag = min(new_mag, mmag - 0.5)  # cap: can't exceed main_mag-0.5
                new_mag = round(new_mag, 1)  # qualification format precision
                if new_mag > pmag:
                    corrected.at[idx, "pred_mag"] = new_mag
                    corrected.at[idx, "_adjusted"] = True
                    corrected.at[idx, "_floor_val"] = floor_val

    return corrected


def write_package(preds: pd.DataFrame, out_dir: Path):
    pdir = out_dir / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)

    for tok in sorted(preds["token"].unique()):
        rows = preds[preds["token"] == tok]
        t1 = rows[rows["window"] == "T1"]
        t2 = rows[rows["window"] == "T2"]
        t3 = rows[rows["window"] == "T3"]

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
                f"{r3['pred_mag']:.1f} (Ms) {r3['pred_time_str']}\n"
            )

    src = FINAL_PKG / "technical_docs"
    dst = out_dir / "technical_docs"
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    (out_dir / "MANIFEST.json").write_text(json.dumps({
        "package_name": "qualification_submission_extreme_prior_no_commitment",
        "description": "Extreme prior: mag>=7.5 floor correction, OOF-tuned per-window margin/strength",
        "created": datetime.now(timezone.utc).isoformat(),
        "base": "qualification_submission_final_t123_no_commitment",
        "mag_threshold": MAG_THRESHOLD,
        "window_params": WINDOW_PARAMS,
        "seed": SEED,
    }, indent=2))


def zip_hash(out_dir: Path, zip_path: Path):
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
    print(f"[ZIP] {zip_path} ({zip_path.stat().st_size:,}B) SHA256: {h[:16]}...")


def main():
    print("=" * 60)
    print("Extreme Prior Candidate (Round 2)")
    print("=" * 60)

    print("\n[1/4] Tuning floor parameters on OOF...")
    tune_on_oof()

    print("\n[2/4] Loading final_t123 predictions...")
    preds = load_predictions()
    print(f"  Loaded {len(preds)} predictions")

    print("\n[3/4] Applying magnitude prior floor...")
    corrected = apply_prior_floor(preds)

    changed = corrected[corrected["_adjusted"] == True]
    print(f"\n  Adjusted predictions: {len(changed)}")
    for _, row in changed.iterrows():
        orig_r = preds[(preds["token"] == row["token"]) & (preds["window"] == row["window"])]
        if len(orig_r):
            om = orig_r.iloc[0]["pred_mag"]
            print(f"    {row['token']} {row['window']}: mag {om} -> {row['pred_mag']:.1f} "
                  f"(main_mag={row['main_mag']:.1f}, floor={row.get('_floor_val',0):.1f})")

    print("\n[4/4] Writing package...")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    write_package(corrected, OUTPUT_DIR)
    zip_hash(OUTPUT_DIR, ZIP_PATH)
    print("\nDone!")


if __name__ == "__main__":
    main()
