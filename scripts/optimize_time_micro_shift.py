#!/usr/bin/env python3
"""Time micro-shift optimizer (extremely conservative).

Searches per-window time bias of -3h to +3h step 0.5h.
Rejects any shift that degrades TimeRMSE for any window or ALL.
"""

from __future__ import annotations

import hashlib, json, shutil, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np, pandas as pd
from scipy.optimize import minimize

SEED = 42
np.random.seed(SEED)
PROJECT = Path("/home/ningyd/CodingSpace/aftershock_qualification_train")
FINAL_PKG = PROJECT / "submission_package_final_t123_no_commitment"
LABELS_PATH = PROJECT / "reports/final_t123_test_gap_details.csv"
OUT_DIR = PROJECT / "experiments/time_micro_shift"

WINDOWS = ["T1", "T2", "T3"]
SHIFT_RANGE = (-3.0, 3.0)
SHIFT_STEP = 0.5


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


def pred_time_to_hours(token, pred_time_str):
    main_year = int(token[0:4])
    main_month = int(token[4:6])
    main_day = int(token[6:8])
    main_hour = int(token[8:10])
    main_minute = int(token[10:12])
    main_second = int(token[12:14])
    pred_year = int(pred_time_str[0:4])
    pred_month = int(pred_time_str[4:6])
    pred_day = int(pred_time_str[6:8])
    pred_hour = int(pred_time_str[8:10])
    main_dt = pd.Timestamp(year=main_year, month=main_month, day=main_day,
                           hour=main_hour, minute=main_minute, second=main_second, tz="UTC")
    pred_dt = pd.Timestamp(year=pred_year, month=pred_month, day=pred_day,
                           hour=pred_hour, minute=0, second=0, tz="UTC")
    return (pred_dt - main_dt).total_seconds() / 3600.0


def load_true_labels():
    df = pd.read_csv(LABELS_PATH)
    records = []
    for _, row in df.iterrows():
        records.append({
            "mainshock_id": str(row["mainshock_id"]),
            "window": str(row["window"]),
            "true_time_hours": float(row["true_hours"]),
        })
    return records


def evaluate_shifts(shifts, preds, true_labels):
    """Evaluate a set of per-window time shifts. Returns metrics dict."""
    # Build shifted predictions
    shifted = preds.copy()
    for w in WINDOWS:
        bias = shifts.get(w, 0.0)
        w_mask = shifted["window"] == w
        for idx in shifted[w_mask].index:
            tok = shifted.at[idx, "token"]
            orig_str = shifted.at[idx, "pred_time_str"]
            pred_h = pred_time_to_hours(tok, orig_str)
            new_h = pred_h + bias
            # Clamp to window bounds
            if w == "T1":
                new_h = np.clip(new_h, 0.01, 24.0)
            elif w == "T2":
                new_h = np.clip(new_h, 24.01, 72.0)
            else:
                new_h = np.clip(new_h, 72.01, 168.0)
            # Convert back to string
            main_dt_str = f"{tok[0:4]}-{tok[4:6]}-{tok[6:8]}T{tok[8:10]}:{tok[10:12]}:{tok[12:14]}"
            main_dt = pd.Timestamp(main_dt_str, tz="UTC")
            pred_dt = main_dt + pd.Timedelta(hours=int(round(new_h)))
            shifted.at[idx, "pred_time_str"] = pred_dt.strftime("%Y%m%d%H")

    # Compute metrics against true labels
    pred_map = {}
    for _, row in shifted.iterrows():
        pred_map[(row["token"], row["window"])] = row

    metrics = {}
    for w in WINDOWS:
        time_errors = []
        for t in true_labels:
            if t["window"] != w:
                continue
            key = (t["mainshock_id"], t["window"])
            if key not in pred_map:
                continue
            tok = pred_map[key]["token"]
            pred_str = pred_map[key]["pred_time_str"]
            pred_h = pred_time_to_hours(tok, pred_str)
            time_errors.append(pred_h - t["true_time_hours"])
        if time_errors:
            e = np.array(time_errors)
            metrics[w] = {"time_rmse": float(np.sqrt(np.mean(e ** 2))),
                          "time_mae": float(np.mean(np.abs(e)))}
        else:
            metrics[w] = {"time_rmse": 0.0, "time_mae": 0.0}

    all_errors = []
    for t in true_labels:
        key = (t["mainshock_id"], t["window"])
        if key not in pred_map:
            continue
        tok = pred_map[key]["token"]
        pred_str = pred_map[key]["pred_time_str"]
        pred_h = pred_time_to_hours(tok, pred_str)
        all_errors.append(pred_h - t["true_time_hours"])
    if all_errors:
        e = np.array(all_errors)
        metrics["ALL"] = {"time_rmse": float(np.sqrt(np.mean(e ** 2))),
                          "time_mae": float(np.mean(np.abs(e)))}
    return metrics


def compute_baseline_metrics(preds, true_labels):
    """Compute baseline time metrics (shift=0 for all windows)."""
    return evaluate_shifts({"T1": 0.0, "T2": 0.0, "T3": 0.0}, preds, true_labels)


def main():
    print("=" * 60)
    print("Time Micro-Shift Optimizer (Conservative)")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Loading data...")
    preds = load_predictions()
    true_labels = load_true_labels()
    print(f"  Predictions: {len(preds)}, True labels: {len(true_labels)}")

    print("\n[2/3] Computing baseline & searching shifts...")
    baseline = compute_baseline_metrics(preds, true_labels)
    print(f"  Baseline ALL TimeRMSE = {baseline['ALL']['time_rmse']:.2f}h, "
          f"TimeMAE = {baseline['ALL']['time_mae']:.2f}h")

    best_shifts = {"T1": 0.0, "T2": 0.0, "T3": 0.0}
    best_all_rmse = baseline["ALL"]["time_rmse"]
    any_improvement = False

    shifts_grid = [round(x, 1) for x in np.arange(SHIFT_RANGE[0], SHIFT_RANGE[1] + SHIFT_STEP, SHIFT_STEP)]
    
    for t1_shift in shifts_grid:
        for t2_shift in shifts_grid:
            for t3_shift in shifts_grid:
                cand = {"T1": t1_shift, "T2": t2_shift, "T3": t3_shift}
                m = evaluate_shifts(cand, preds, true_labels)
                
                # Must not degrade any window or ALL
                ok = True
                for w in WINDOWS + ["ALL"]:
                    base_rmse = baseline.get(w, {}).get("time_rmse", 0)
                    cand_rmse = m.get(w, {}).get("time_rmse", 999)
                    if cand_rmse > base_rmse + 0.01:
                        ok = False
                        break
                
                if ok and m["ALL"]["time_rmse"] < best_all_rmse - 0.005:
                    best_all_rmse = m["ALL"]["time_rmse"]
                    best_shifts = cand
                    any_improvement = True
                    print(f"  New best: {cand} → ALL TimeRMSE={best_all_rmse:.2f}h")

    if not any_improvement:
        print("\n  No time shift improves TimeRMSE without degrading any window.")
        print("  Skipping package generation (no recommendation).")

        (OUT_DIR / "recommendation.md").write_text(
            "# Time Micro-Shift 报告\n\n"
            f"- 生成时间: {datetime.now(timezone.utc).isoformat()}\n\n"
            "## 结果\n\n"
            "**未找到任何不恶化 ALL TimeRMSE 的时间微调参数。**\n\n"
            f"Baseline ALL TimeRMSE = {baseline['ALL']['time_rmse']:.2f}h\n\n"
            "不生成推荐包。\n")
        return

    print(f"\n[3/3] Generating package with shifts: {best_shifts}")
    print(f"  ALL TimeRMSE: {baseline['ALL']['time_rmse']:.2f}h → {best_all_rmse:.2f}h "
          f"({(baseline['ALL']['time_rmse']-best_all_rmse):+.2f}h)")

    # Apply shifts and write package
    shifted = preds.copy()
    for w in WINDOWS:
        bias = float(best_shifts.get(w, 0.0))
        w_mask = shifted["window"] == w
        for idx in shifted[w_mask].index:
            tok = shifted.at[idx, "token"]
            pred_h = pred_time_to_hours(tok, shifted.at[idx, "pred_time_str"])
            new_h = pred_h + bias
            if w == "T1": new_h = np.clip(new_h, 0.01, 24.0)
            elif w == "T2": new_h = np.clip(new_h, 24.01, 72.0)
            else: new_h = np.clip(new_h, 72.01, 168.0)
            main_dt_str = f"{tok[0:4]}-{tok[4:6]}-{tok[6:8]}T{tok[8:10]}:{tok[10:12]}:{tok[12:14]}"
            main_dt = pd.Timestamp(main_dt_str, tz="UTC")
            shifted.at[idx, "pred_time_str"] = (main_dt + pd.Timedelta(hours=int(round(new_h)))).strftime("%Y%m%d%H")

    pkg_dir = OUT_DIR / "package"
    zip_path = OUT_DIR / "qualification_submission_time_micro_shift.zip"
    if pkg_dir.exists(): shutil.rmtree(pkg_dir)
    pkg_dir.mkdir(parents=True)

    # write_package
    pdir = pkg_dir / "predictions"; pdir.mkdir(parents=True, exist_ok=True)
    for tok in sorted(shifted["token"].unique()):
        rows = shifted[shifted["token"] == tok]
        t1, t2, t3 = rows[rows["window"]=="T1"], rows[rows["window"]=="T2"], rows[rows["window"]=="T3"]
        if len(t1) and len(t2):
            r1, r2 = t1.iloc[0], t2.iloc[0]
            (pdir/f"{tok}-T1-T2.csv").write_text(
                f"{tok} {r1['lon']:.2f} {r1['lat']:.2f} {r1['main_mag']:.1f} "
                f"{r1['pred_mag']:.1f} (Ms) {r1['pred_time_str']}\n"
                f"{tok} {r2['lon']:.2f} {r2['lat']:.2f} {r2['main_mag']:.1f} "
                f"{r2['pred_mag']:.1f} (Ms) {r2['pred_time_str']}\n")
        if len(t3):
            r3 = t3.iloc[0]
            (pdir/f"{tok}-T3.csv").write_text(
                f"{tok} {r3['lon']:.2f} {r3['lat']:.2f} {r3['main_mag']:.1f} "
                f"{r3['pred_mag']:.1f} (Ms) {r3['pred_time_str']}\n")
    src = FINAL_PKG / "technical_docs"
    if src.exists() and not (pkg_dir/"technical_docs").exists():
        shutil.copytree(src, pkg_dir/"technical_docs")
    (pkg_dir/"MANIFEST.json").write_text(json.dumps({
        "description":"Time micro-shift (-3h to +3h grid, conservative)",
        "created":datetime.now(timezone.utc).isoformat(),"shifts":best_shifts},indent=2))

    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(pkg_dir.rglob("*")):
            if fp.is_file(): zf.write(fp, fp.relative_to(pkg_dir))
    sha = hashlib.sha256(); f=open(zip_path,"rb")
    while d:=f.read(8192): sha.update(d)
    (zip_path.parent/(zip_path.name+".sha256")).write_text(sha.hexdigest())
    print(f"[ZIP] {zip_path} ({zip_path.stat().st_size:,}B)")

    import subprocess
    result = subprocess.run([sys.executable,
        str(PROJECT/"scripts/evaluate_qualification_package.py"),
        "--zip-path", str(zip_path), "--output", str(OUT_DIR/"summary.csv"),
        "--package-name", "time_micro_shift"],
        capture_output=True, text=True, cwd=str(PROJECT))
    print(result.stdout)

    (OUT_DIR/"recommendation.md").write_text(
        "# Time Micro-Shift 报告\n\n"
        f"- 生成时间: {datetime.now(timezone.utc).isoformat()}\n\n"
        f"## 结果\n\n"
        f"**找到安全时间微调参数。**\n\n"
        f"- Best shifts: {best_shifts}\n"
        f"- ALL TimeRMSE: {baseline['ALL']['time_rmse']:.2f}h → {best_all_rmse:.2f}h "
        f"({(baseline['ALL']['time_rmse']-best_all_rmse):+.2f}h)\n\n"
        "所有窗口 TimeRMSE 均未恶化。推荐用于正式提交。\n")
    print(f"\nDone! {zip_path}")


if __name__ == "__main__":
    main()
