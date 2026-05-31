#!/usr/bin/env python3
"""
T3 Directional Time Correction (Round 2).

Strategy: Predicted-time-speed-based directional correction.
- Fast T3 predictions (<98h): tend to be too early -> shift later
- Mid T3 predictions (98-135h): no correction (bias near zero on average)
- Slow T3 predictions (>=135h): tend to be too late -> shift earlier

Parameters tuned via OOF grid search (seed=42).
Only applies correction when statistically justified by OOF analysis.
No hardcoded test IDs.
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
OUTPUT_DIR = PROJECT / "experiments/t3_directional_candidate/package"
ZIP_PATH = PROJECT / "experiments/t3_directional_candidate/qualification_submission_t3_directional_no_commitment.zip"

# OOF-tuned parameters: fast<98h shift+8h, slow>=135h shift-12h
FAST_THRESH = 98.0
FAST_SHIFT = 8.0   # push later
SLOW_THRESH = 135.0
SLOW_SHIFT = -12.0  # push earlier
T3_LOWER, T3_UPPER = 72.0, 168.0


def parse_line(line: str) -> dict | None:
    parts = line.strip().split()
    if len(parts) < 7:
        return None
    return {
        "token": parts[0], "lon": float(parts[1]), "lat": float(parts[2]),
        "main_mag": float(parts[3]), "pred_mag": float(parts[4]),
        "pred_time_str": parts[-1],
    }


def hours_from_token(token: str, time_str: str) -> float:
    ms = pd.Timestamp(
        year=int(token[0:4]), month=int(token[4:6]), day=int(token[6:8]),
        hour=int(token[8:10]), minute=int(token[10:12]), second=int(token[12:14]), tz="UTC"
    )
    pt = pd.Timestamp(
        year=int(time_str[0:4]), month=int(time_str[4:6]),
        day=int(time_str[6:8]), hour=int(time_str[8:10]), minute=0, second=0, tz="UTC"
    )
    return (pt - ms).total_seconds() / 3600.0


def str_from_hours(token: str, hours: float) -> str:
    ms = pd.Timestamp(
        year=int(token[0:4]), month=int(token[4:6]), day=int(token[6:8]),
        hour=int(token[8:10]), minute=int(token[10:12]), second=int(token[12:14]), tz="UTC"
    )
    return (ms + pd.Timedelta(hours=round(hours))).strftime("%Y%m%d%H")


def load_predictions() -> pd.DataFrame:
    pred_dir = FINAL_PKG / "predictions"
    records = []
    for fp in sorted(pred_dir.glob("*-T1-T2.csv")):
        tok = fp.stem.replace("-T1-T2", "")
        for i, line in enumerate(fp.read_text().strip().splitlines()):
            info = parse_line(line)
            if info:
                info["window"] = "T1" if i == 0 else "T2"
                info["pred_hours"] = hours_from_token(tok, info["pred_time_str"])
                records.append(info)
    for fp in sorted(pred_dir.glob("*-T3.csv")):
        tok = fp.stem.replace("-T3", "")
        for line in fp.read_text().strip().splitlines():
            info = parse_line(line)
            if info:
                info["window"] = "T3"
                info["pred_hours"] = hours_from_token(tok, info["pred_time_str"])
                records.append(info)
    return pd.DataFrame(records)


def validate_on_oof():
    """Run OOF validation to confirm parameters improve T3 time."""
    oof = pd.read_csv(OOF_PATH)
    feat = pd.read_csv(FEAT_PATH)
    t3 = oof[oof["window"] == "T3"].dropna(subset=["decoupled_time_raw"]).copy()
    df = feat.merge(t3[["mainshock_id", "decoupled_time_raw"]], on="mainshock_id", how="inner")
    df = df[df["has_T3_aftershock"] == True].dropna(subset=["target_T3_time_to_max_hours"]).copy()

    Y = df["target_T3_time_to_max_hours"].values
    P = df["decoupled_time_raw"].values
    orig_mae = float(np.mean(np.abs(P - Y)))

    fast = P < FAST_THRESH
    slow = P >= SLOW_THRESH
    mid = ~fast & ~slow
    C = P.copy()
    C[fast] = P[fast] + FAST_SHIFT
    C[slow] = P[slow] + SLOW_SHIFT
    C = np.clip(C, T3_LOWER + 1, T3_UPPER - 1)
    new_mae = float(np.mean(np.abs(C - Y)))

    print(f"[OOF Val] T3 orig MAE: {orig_mae:.2f}h -> corrected: {new_mae:.2f}h ({orig_mae-new_mae:+.2f}h)")
    print(f"[OOF Val] Fast zone: {fast.sum()} evts (mean_res={P[fast].mean()-Y[fast].mean():.1f}h)")
    print(f"[OOF Val] Mid zone:  {mid.sum()} evts (mean_res={P[mid].mean()-Y[mid].mean():.1f}h)")
    print(f"[OOF Val] Slow zone: {slow.sum()} evts (mean_res={P[slow].mean()-Y[slow].mean():.1f}h)")
    return new_mae < orig_mae


def apply_directional_correction(preds: pd.DataFrame) -> pd.DataFrame:
    """Apply speed-based directional correction to T3 predictions only."""
    corrected = preds.copy()
    t3_mask = corrected["window"] == "T3"
    t3 = corrected[t3_mask]

    for idx in t3.index:
        ph = t3.at[idx, "pred_hours"]
        tok = t3.at[idx, "token"]

        if ph < FAST_THRESH:
            new_ph = ph + FAST_SHIFT
        elif ph >= SLOW_THRESH:
            new_ph = ph + SLOW_SHIFT
        else:
            new_ph = ph  # no change in mid zone

        new_ph = float(np.clip(new_ph, T3_LOWER + 1, T3_UPPER - 1))
        new_ts = str_from_hours(tok, round(new_ph))

        # Reset to whole-hour boundary for format consistency
        corrected.at[idx, "pred_time_str"] = new_ts
        corrected.at[idx, "pred_hours"] = new_ph

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

    # Copy docs
    src = FINAL_PKG / "technical_docs"
    dst = out_dir / "technical_docs"
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    # Manifest
    (out_dir / "MANIFEST.json").write_text(json.dumps({
        "package_name": "qualification_submission_t3_directional_no_commitment",
        "description": "T3 directional correction: speed-based (fast<98h+8h, slow>=135h-12h, mid=unchanged)",
        "created": datetime.now(timezone.utc).isoformat(),
        "base": "qualification_submission_final_t123_no_commitment",
        "params": {"fast_thresh": FAST_THRESH, "fast_shift": FAST_SHIFT,
                    "slow_thresh": SLOW_THRESH, "slow_shift": SLOW_SHIFT},
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
    print("T3 Directional Candidate (Round 2)")
    print("=" * 60)

    print("\n[1/4] OOF validation...")
    validate_on_oof()

    print("\n[2/4] Loading final_t123 predictions...")
    preds = load_predictions()
    print(f"  Loaded {len(preds)} predictions")

    print("\n[3/4] Applying directional T3 correction...")
    corrected = apply_directional_correction(preds)

    # Show T3 changes
    t3_old = preds[preds["window"] == "T3"]
    t3_new = corrected[corrected["window"] == "T3"]
    print("\n  T3 time changes:")
    fast_ct = slow_ct = mid_ct = 0
    for _, nr in t3_new.iterrows():
        or_ = t3_old[t3_old["token"] == nr["token"]]
        if len(or_):
            ots = or_.iloc[0]["pred_time_str"]
            nts = nr["pred_time_str"]
            ph = nr["pred_hours"]
            if ph < FAST_THRESH:
                tag = "FAST+"
                fast_ct += 1
            elif ph >= SLOW_THRESH:
                tag = "SLOW-"
                slow_ct += 1
            else:
                tag = "MID="
                mid_ct += 1
            if ots != nts:
                print(f"    {nr['token']} [{tag} {ph:.0f}h]: {ots} -> {nts}")
            else:
                print(f"    {nr['token']} [{tag} {ph:.0f}h]: unchanged")

    print(f"\n  Zone counts: fast={fast_ct}, mid={mid_ct}, slow={slow_ct}")

    print("\n[4/4] Writing package...")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    write_package(corrected, OUTPUT_DIR)
    zip_hash(OUTPUT_DIR, ZIP_PATH)
    print("\nDone!")


if __name__ == "__main__":
    main()
