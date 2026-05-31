#!/usr/bin/env python3
"""OOF-first extreme prior optimizer.

Searches high-magnitude mainshock floor parameters on OOF data ONLY.
Visible test metrics serve only as tie-break, not as optimization target.

Strategy:
- For each window (T1/T2/T3), search grid of (mainshock_mag threshold,
  margin, strength) over OOF predictions.
- Accept only params that improve OOF high-mainshock MAE AND do not
  degrade OOF overall RMSE/MAE beyond tolerance (0.001).
- Among survivors, pick best OOF high_mae; tie-break on visible ALL MagRMSE.
"""

from __future__ import annotations

import hashlib, json, shutil, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np, pandas as pd

SEED = 42
np.random.seed(SEED)
PROJECT = Path("/home/ningyd/CodingSpace/aftershock_qualification_train")
OOF_PATH = PROJECT / "data/models/qualification_decoupled_full/decoupled_oof_predictions.csv"
FEAT_PATH = PROJECT / "data/processed/qualification_features.csv"
FINAL_PKG = PROJECT / "submission_package_final_t123_no_commitment"
OUT_DIR = PROJECT / "experiments/extreme_prior_oof_first"
PKG_DIR = OUT_DIR / "package"
ZIP_PATH = OUT_DIR / "qualification_submission_extreme_prior_oof_first.zip"

OOF_TOLERANCE = 0.001  # max allowed degradation in OOF overall RMSE/MAE
MAG_CAP = 0.5  # prediction cap: can't exceed mainshock_mag - cap

# Grid search space
THRESHOLD_CANDIDATES = [round(x, 1) for x in np.arange(7.0, 8.6, 0.1)]
MARGIN_CANDIDATES = [round(x, 1) for x in np.arange(0.8, 3.7, 0.1)]
STRENGTH_CANDIDATES = [round(x, 1) for x in np.arange(0.1, 1.05, 0.1)]
WINDOWS = ["T1", "T2", "T3"]


def load_oof_data():
    oof = pd.read_csv(OOF_PATH)
    feat = pd.read_csv(FEAT_PATH)
    return oof, feat


def evaluate_on_oof(oof, feat):
    """Per-window OOF evaluation: original metrics + grid search best params."""
    results = {}
    for w in WINDOWS:
        w_oof = oof[oof["window"] == w].dropna(subset=["decoupled_mag_raw"]).copy()
        w_df = feat.merge(w_oof[["mainshock_id", "decoupled_mag_raw"]],
                          on="mainshock_id", how="inner")
        tcol = f"target_{w}_max_mag"
        if tcol not in w_df.columns:
            continue
        w_df = w_df.dropna(subset=[tcol])
        w_df = w_df[w_df[tcol] > 0].copy()

        Y = w_df[tcol].values
        P = w_df["decoupled_mag_raw"].values
        M = w_df["mainshock_mag"].values

        if len(Y) < 20:
            results[w] = None
            continue

        orig_rmse = float(np.sqrt(np.mean((P - Y) ** 2)))
        orig_mae = float(np.mean(np.abs(P - Y)))

        best = {"threshold": 7.5, "margin": 2.0, "strength": 0.5,
                "oof_high_mae": 999.0, "oof_rmse": orig_rmse, "oof_mae": orig_mae}

        for thr in THRESHOLD_CANDIDATES:
            high_mask = M >= thr
            if high_mask.sum() < 8:
                continue
            orig_high_mae = float(np.mean(np.abs(P[high_mask] - Y[high_mask])))

            for margin in MARGIN_CANDIDATES:
                for strength in STRENGTH_CANDIDATES:
                    C = P.copy()
                    floor_vals = M - margin
                    need_boost = high_mask & (floor_vals > P)
                    if not need_boost.any():
                        continue
                    C[need_boost] = (1 - strength) * P[need_boost] + strength * floor_vals[need_boost]
                    C[need_boost] = np.minimum(C[need_boost], M[need_boost] - MAG_CAP)
                    C = np.clip(C, 0.0, None)

                    new_rmse = float(np.sqrt(np.mean((C - Y) ** 2)))
                    new_mae = float(np.mean(np.abs(C - Y)))
                    new_high_mae = float(np.mean(np.abs(C[high_mask] - Y[high_mask])))

                    # Must improve high_mae AND not degrade overall
                    if new_high_mae >= orig_high_mae:
                        continue
                    if new_rmse > orig_rmse + OOF_TOLERANCE:
                        continue
                    if new_mae > orig_mae + OOF_TOLERANCE:
                        continue

                    # Better high_mae, or tie on high_mae with better rmse
                    if (new_high_mae < best["oof_high_mae"] or
                        (abs(new_high_mae - best["oof_high_mae"]) < 0.0005 and new_rmse < best["oof_rmse"])):
                        best = {"threshold": thr, "margin": margin, "strength": strength,
                                "oof_high_mae": new_high_mae, "oof_rmse": new_rmse, "oof_mae": new_mae}

        print(f"  {w}: orig_rmse={orig_rmse:.4f} ma={orig_mae:.4f} → "
              f"best thr={best['threshold']:.1f} m={best['margin']:.1f} s={best['strength']:.1f} "
              f"high_mae={best['oof_high_mae']:.4f} rmse={best['oof_rmse']:.4f}")
        results[w] = best
    return results


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


def apply_rules(preds, rules):
    corrected = preds.copy()
    corrected["_adjusted"] = False
    for w, r in rules.items():
        if r is None:
            continue
        w_mask = corrected["window"] == w
        high_mask = w_mask & (corrected["main_mag"] >= r["threshold"])
        for idx in corrected[high_mask].index:
            mmag = corrected.at[idx, "main_mag"]
            pmag = corrected.at[idx, "pred_mag"]
            floor_val = mmag - r["margin"]
            if floor_val > pmag:
                new_mag = (1 - r["strength"]) * pmag + r["strength"] * floor_val
                new_mag = min(new_mag, mmag - MAG_CAP)
                new_mag = max(new_mag, 0.0)
                new_mag = round(new_mag, 1)
                if new_mag > pmag:
                    corrected.at[idx, "pred_mag"] = new_mag
                    corrected.at[idx, "_adjusted"] = True
                    corrected.at[idx, "_floor_val"] = floor_val
    return corrected


def write_package(preds, out_dir, rules):
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
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    (out_dir / "MANIFEST.json").write_text(json.dumps({
        "package_name": "extreme_prior_oof_first",
        "description": "OOF-first extreme prior; grid search on OOF, visible tie-break only",
        "created": datetime.now(timezone.utc).isoformat(),
        "base": "qualification_submission_final_t123_no_commitment",
        "oof_tolerance": OOF_TOLERANCE,
        "rules": {w: (r if r else {}) for w, r in rules.items()},
        "seed": SEED,
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
    print(f"[ZIP] {zip_path} ({zip_path.stat().st_size:,}B) SHA256: {h[:16]}...")


def evaluate_zip(zip_path):
    import subprocess
    result = subprocess.run(
        [sys.executable, str(PROJECT / "scripts/evaluate_qualification_package.py"),
         "--zip-path", str(zip_path),
         "--output", str(OUT_DIR / "summary.csv"),
         "--package-name", "oof_first"],
        capture_output=True, text=True, cwd=str(PROJECT))
    print(result.stdout)
    if result.returncode != 0:
        print(f"[WARN] eval stderr: {result.stderr}")


def main():
    print("=" * 60)
    print("OOF-First Extreme Prior Optimizer")
    print("=" * 60)

    print("\n[1/4] Loading OOF data...")
    oof, feat = load_oof_data()
    print(f"  OOF: {len(oof)} rows, features: {len(feat)} rows")

    print("\n[2/4] Grid search on OOF (threshold × margin × strength)...")
    rules = evaluate_on_oof(oof, feat)

    print("\n[3/4] Loading final_t123 predictions & applying rules...")
    preds = load_predictions()
    corrected = apply_rules(preds, rules)
    changed = corrected[corrected["_adjusted"] == True]
    print(f"  Adjusted: {len(changed)} predictions")
    for _, row in changed.iterrows():
        orig = preds[(preds["token"] == row["token"]) & (preds["window"] == row["window"])]
        if len(orig):
            print(f"    {row['token']} {row['window']}: {orig.iloc[0]['pred_mag']:.1f} → {row['pred_mag']:.1f}")

    print("\n[4/4] Writing package & evaluating...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if PKG_DIR.exists():
        shutil.rmtree(PKG_DIR)
    PKG_DIR.mkdir(parents=True)
    write_package(corrected, PKG_DIR, rules)
    zip_hash(PKG_DIR, ZIP_PATH)

    # Save rules
    (OUT_DIR / "selected_rules.json").write_text(json.dumps({
        w: (r if r else {}) for w, r in rules.items()}, indent=2))
    # Save adjusted predictions
    corrected.to_csv(OUT_DIR / "adjusted_predictions.csv", index=False)

    # Evaluate
    evaluate_zip(ZIP_PATH)

    # Write recommendation
    summary_df = pd.read_csv(OUT_DIR / "summary.csv") if (OUT_DIR / "summary.csv").exists() else None
    lines = ["# OOF-First Extreme Prior 优化报告", "",
             f"- 生成时间: {datetime.now(timezone.utc).isoformat()}",
             f"- OOF 容忍度: RMSE/MAE degradation ≤ {OOF_TOLERANCE}",
             f"- Cap: mainshock_mag - {MAG_CAP}", "",
             "## 每窗口规则", ""]
    for w in WINDOWS:
        r = rules.get(w)
        if r:
            lines.append(f"- **{w}**: threshold≥{r['threshold']:.1f}, margin={r['margin']:.1f}, "
                         f"strength={r['strength']:.1f}, oof_high_mae={r['oof_high_mae']:.4f}")
    lines.append("")
    lines.append("## 评估指标")
    if summary_df is not None:
        lines.append("```csv")
        lines.append(summary_df.to_csv(index=False))
        lines.append("```")
    lines.extend(["", "## 推荐", "",
                  "此包基于 OOF 数据优先选择参数，可见集仅用于 tie-break。",
                  "泛化风险低于直接在可见集上优化的 R3 public_max。"])
    (OUT_DIR / "recommendation.md").write_text("\n".join(lines))
    print(f"\nDone! Results: {OUT_DIR}")


if __name__ == "__main__":
    main()
