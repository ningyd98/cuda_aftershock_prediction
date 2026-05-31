#!/usr/bin/env python3
"""Unified qualification package evaluation script.

Evaluates any T1/T2/T3 or H168 prediction package against visible labels
from reports/final_t123_test_gap_details.csv.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

WINDOW_NAMES = ["T1", "T2", "T3"]
SEED = 42
np.random.seed(SEED)


@dataclass
class TrueRecord:
    mainshock_id: str
    window: str
    main_mag: float
    true_mag: float
    true_time_hours: float


@dataclass
class PredRecord:
    mainshock_id: str
    window: str
    pred_mag: float
    pred_time_hours: float


def parse_prediction_line(line: str) -> Optional[PredRecord]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split()
    if len(parts) < 7:
        return None
    token = parts[0]
    try:
        pred_mag = float(parts[4])
    except (ValueError, IndexError):
        return None
    pred_time_str = parts[-1].strip()
    try:
        main_year = int(token[0:4])
        main_month = int(token[4:6])
        main_day = int(token[6:8])
        main_hour = int(token[8:10])
        main_minute = int(token[10:12])
        main_second = int(token[12:14])
    except (ValueError, IndexError):
        return None
    try:
        pred_year = int(pred_time_str[0:4])
        pred_month = int(pred_time_str[4:6])
        pred_day = int(pred_time_str[6:8])
        pred_hour = int(pred_time_str[8:10])
    except (ValueError, IndexError):
        return None
    main_dt = pd.Timestamp(
        year=main_year, month=main_month, day=main_day,
        hour=main_hour, minute=main_minute, second=main_second, tz="UTC"
    )
    pred_dt = pd.Timestamp(
        year=pred_year, month=pred_month, day=pred_day,
        hour=pred_hour, minute=0, second=0, tz="UTC"
    )
    pred_time_hours = (pred_dt - main_dt).total_seconds() / 3600.0
    return PredRecord(
        mainshock_id=token,
        window="",
        pred_mag=pred_mag,
        pred_time_hours=pred_time_hours,
    )


def load_predictions_from_dir(pred_dir: Path) -> list[PredRecord]:
    records: list[PredRecord] = []
    t1t2_files = sorted(pred_dir.glob("*-T1-T2.csv"))
    t3_files = sorted(pred_dir.glob("*-T3.csv"))
    for fpath in t1t2_files:
        token = fpath.stem.replace("-T1-T2", "")
        lines = fpath.read_text(encoding="utf-8").strip().splitlines()
        for i, line in enumerate(lines):
            rec = parse_prediction_line(line)
            if rec is None:
                continue
            rec.window = "T1" if i == 0 else "T2"
            rec.mainshock_id = token
            records.append(rec)
    for fpath in t3_files:
        token = fpath.stem.replace("-T3", "")
        lines = fpath.read_text(encoding="utf-8").strip().splitlines()
        if lines:
            rec = parse_prediction_line(lines[0])
            if rec is not None:
                rec.window = "T3"
                rec.mainshock_id = token
                records.append(rec)
    return records


def load_predictions_from_zip(zip_path: Path) -> list[PredRecord]:
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)
        pred_subdir = Path(tmpdir) / "predictions"
        if pred_subdir.exists():
            return load_predictions_from_dir(pred_subdir)
        return load_predictions_from_dir(Path(tmpdir))


def load_true_labels(gap_detail_path: Path) -> list[TrueRecord]:
    df = pd.read_csv(gap_detail_path)
    records: list[TrueRecord] = []
    for _, row in df.iterrows():
        records.append(TrueRecord(
            mainshock_id=str(row["mainshock_id"]),
            window=str(row["window"]),
            main_mag=float(row["main_mag"]),
            true_mag=float(row["true_mag"]),
            true_time_hours=float(row["true_hours"]),
        ))
    return records


@dataclass
class WindowMetrics:
    window: str
    count: int
    mag_mae: float
    mag_rmse: float
    time_mae_h: float
    time_rmse_h: float
    within_mag_05: float
    within_time_12h: float
    within_time_24h: float
    max_abs_mag_err: float
    max_abs_time_h: float
    worst_samples: list[dict] = field(default_factory=list)


def compute_metrics(true_recs: list[TrueRecord], pred_recs: list[PredRecord]) -> dict[str, WindowMetrics]:
    pred_map: dict[tuple[str, str], PredRecord] = {}
    for p in pred_recs:
        pred_map[(p.mainshock_id, p.window)] = p

    by_window: dict[str, list[dict]] = defaultdict(list)
    for t in true_recs:
        key = (t.mainshock_id, t.window)
        if key not in pred_map:
            print(f"  WARNING: No prediction for {t.mainshock_id} {t.window}", file=sys.stderr)
            continue
        p = pred_map[key]
        mag_err = p.pred_mag - t.true_mag
        time_err = p.pred_time_hours - t.true_time_hours
        by_window[t.window].append({
            "mainshock_id": t.mainshock_id,
            "main_mag": t.main_mag,
            "true_mag": t.true_mag,
            "pred_mag": p.pred_mag,
            "mag_error": mag_err,
            "abs_mag_error": abs(mag_err),
            "true_time_h": t.true_time_hours,
            "pred_time_h": p.pred_time_hours,
            "time_error_h": time_err,
            "abs_time_error_h": abs(time_err),
        })

    results: dict[str, WindowMetrics] = {}
    all_rows: list[dict] = []

    for window in WINDOW_NAMES:
        rows = by_window.get(window, [])
        if not rows:
            continue
        all_rows.extend(rows)
        mag_errors = np.array([r["mag_error"] for r in rows])
        abs_mag_errors = np.abs(mag_errors)
        time_errors = np.array([r["time_error_h"] for r in rows])
        abs_time_errors = np.abs(time_errors)
        n = len(rows)
        results[window] = WindowMetrics(
            window=window, count=n,
            mag_mae=float(np.mean(abs_mag_errors)),
            mag_rmse=float(np.sqrt(np.mean(mag_errors ** 2))),
            time_mae_h=float(np.mean(abs_time_errors)),
            time_rmse_h=float(np.sqrt(np.mean(time_errors ** 2))),
            within_mag_05=float(np.mean(abs_mag_errors <= 0.5)),
            within_time_12h=float(np.mean(abs_time_errors <= 12.0)),
            within_time_24h=float(np.mean(abs_time_errors <= 24.0)),
            max_abs_mag_err=float(np.max(abs_mag_errors)),
            max_abs_time_h=float(np.max(abs_time_errors)),
            worst_samples=sorted(rows, key=lambda r: r["abs_mag_error"], reverse=True)[:10],
        )

    if all_rows:
        mag_errors = np.array([r["mag_error"] for r in all_rows])
        abs_mag_errors = np.abs(mag_errors)
        time_errors = np.array([r["time_error_h"] for r in all_rows])
        abs_time_errors = np.abs(time_errors)
        results["ALL"] = WindowMetrics(
            window="ALL", count=len(all_rows),
            mag_mae=float(np.mean(abs_mag_errors)),
            mag_rmse=float(np.sqrt(np.mean(mag_errors ** 2))),
            time_mae_h=float(np.mean(abs_time_errors)),
            time_rmse_h=float(np.sqrt(np.mean(time_errors ** 2))),
            within_mag_05=float(np.mean(abs_mag_errors <= 0.5)),
            within_time_12h=float(np.mean(abs_time_errors <= 12.0)),
            within_time_24h=float(np.mean(abs_time_errors <= 24.0)),
            max_abs_mag_err=float(np.max(abs_mag_errors)),
            max_abs_time_h=float(np.max(abs_time_errors)),
            worst_samples=sorted(all_rows, key=lambda r: r["abs_mag_error"], reverse=True)[:10],
        )

    return results


def format_metrics(metrics: dict[str, WindowMetrics], package_name: str) -> str:
    header = (
        f"{'Window':<6} {'Count':>5} {'MagMAE':>8} {'MagRMSE':>8} "
        f"{'T_MAE_h':>8} {'T_RMSE_h':>8} {'+/-0.5':>6} {'T<12h':>7} {'T<24h':>7} "
        f"{'Max|E_m|':>8} {'Max|E_t|':>10}"
    )
    sep = "-" * len(header)
    lines = [f"==== Package: {package_name} ====", "", header, sep]
    for name in WINDOW_NAMES + ["ALL"]:
        m = metrics.get(name)
        if m is None:
            continue
        lines.append(
            f"{m.window:<6} {m.count:>5} {m.mag_mae:>8.3f} {m.mag_rmse:>8.3f} "
            f"{m.time_mae_h:>8.2f} {m.time_rmse_h:>8.2f} "
            f"{m.within_mag_05:>6.2f} {m.within_time_12h:>7.2f} "
            f"{m.within_time_24h:>7.2f} {m.max_abs_mag_err:>8.3f} {m.max_abs_time_h:>10.2f}"
        )
    return "\n".join(lines)


def save_summary_csv(metrics, package_name, output_path, include_header=True):
    rows = []
    for name in WINDOW_NAMES + ["ALL"]:
        m = metrics.get(name)
        if m is None:
            continue
        rows.append({
            "package": package_name, "window": m.window, "count": m.count,
            "mag_mae": m.mag_mae, "mag_rmse": m.mag_rmse,
            "time_mae_h": m.time_mae_h, "time_rmse_h": m.time_rmse_h,
            "within_mag_05": m.within_mag_05,
            "within_time_12h": m.within_time_12h,
            "within_time_24h": m.within_time_24h,
            "max_abs_mag_err": m.max_abs_mag_err,
            "max_abs_time_h": m.max_abs_time_h,
        })
    df = pd.DataFrame(rows)
    if output_path.suffix == ".csv":
        df.to_csv(output_path, index=False, mode="a" if not include_header else "w",
                  header=include_header)
    elif output_path.suffix == ".json":
        df.to_json(output_path, orient="records", indent=2)
    return df


def save_worst_windows(metrics, package_name, output_path, include_header=True):
    rows = []
    for name in WINDOW_NAMES + ["ALL"]:
        m = metrics.get(name)
        if m is None:
            continue
        for s in m.worst_samples:
            rows.append({"package": package_name, "window": m.window, **s})
    if not rows:
        return
    df = pd.DataFrame(rows)
    if output_path.suffix == ".csv":
        df.to_csv(output_path, index=False, mode="a" if not include_header else "w",
                  header=include_header)
    elif output_path.suffix == ".json":
        df.to_json(output_path, orient="records", indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate qualification prediction package.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--package-dir", type=Path, help="Directory containing predictions/")
    group.add_argument("--zip-path", type=Path, help="Path to qualification ZIP")
    parser.add_argument("--labels", type=Path,
                        default="reports/final_t123_test_gap_details.csv")
    parser.add_argument("--output", type=Path, help="Output CSV/JSON for summary")
    parser.add_argument("--worst-output", type=Path, help="Output CSV/JSON for worst windows")
    parser.add_argument("--package-name", type=str, default=None)
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path("/home/ningyd/CodingSpace/aftershock_qualification_train")
    labels_path = args.labels if args.labels.is_absolute() else project_root / args.labels
    if not labels_path.exists():
        alt = Path.cwd() / args.labels
        if alt.exists():
            labels_path = alt
        else:
            print(f"ERROR: Labels not found: {labels_path}", file=sys.stderr)
            sys.exit(1)

    true_recs = load_true_labels(labels_path)
    print(f"Loaded {len(true_recs)} ground truth records")

    if args.package_dir:
        pkg_path = args.package_dir if args.package_dir.is_absolute() else project_root / args.package_dir
        pred_recs = load_predictions_from_dir(pkg_path / "predictions")
        pkg_name = args.package_name or args.package_dir.name
    else:
        zip_path = args.zip_path if args.zip_path.is_absolute() else project_root / args.zip_path
        pred_recs = load_predictions_from_zip(zip_path)
        pkg_name = args.package_name or args.zip_path.stem
    print(f"Loaded {len(pred_recs)} predictions from {pkg_name}")

    metrics = compute_metrics(true_recs, pred_recs)
    print(format_metrics(metrics, pkg_name))

    inc_header = not args.append
    if args.output:
        out_path = args.output if args.output.is_absolute() else project_root / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_summary_csv(metrics, pkg_name, out_path, include_header=inc_header)
        print(f"Summary saved: {out_path}")
    if args.worst_output:
        w_path = args.worst_output if args.worst_output.is_absolute() else project_root / args.worst_output
        w_path.parent.mkdir(parents=True, exist_ok=True)
        save_worst_windows(metrics, pkg_name, w_path, include_header=inc_header)
        print(f"Worst windows saved: {w_path}")


if __name__ == "__main__":
    main()
