from __future__ import annotations

"""
余震预测结果综合可视化。

从 OOF 预测和特征文件中生成 5 张 PNG 图并保存到 reports/figures/。

用法:
  python scripts/visualize_results.py \
    --oof data/models/ensemble_oof_predictions.csv \
    --features data/processed/advanced_features.csv \
    --weights data/models/ensemble_weights.json \
    --output-dir reports/figures
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

DEFAULT_FIGSIZE = (8, 6)
DEFAULT_DPI = 150


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ─── Plot 1: Mag Scatter ────────────────────────────────────────────
def plot_mag_scatter(oof_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    y_true = oof_df["target_max_mag"].to_numpy()
    y_pred = oof_df["ensemble_pred_mag"].to_numpy()
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]

    ax.scatter(y_true, y_pred, alpha=0.3, s=8, c="#1f77b4", edgecolors="none")
    lim_min = min(y_true.min(), y_pred.min()) - 0.5
    lim_max = max(y_true.max(), y_pred.max()) + 0.5
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1, label="y=x")

    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mae = np.mean(np.abs(y_pred - y_true))
    ax.set_xlabel("True max aftershock mag")
    ax.set_ylabel("Ensemble predicted mag")
    ax.set_title(f"Mag Scatter (RMSE={rmse:.3f}, MAE={mae:.3f}, N={len(y_true)})")
    ax.legend()
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    fig.tight_layout()
    fig.savefig(output_dir / "mag_scatter.png", dpi=DEFAULT_DPI)
    plt.close(fig)
    print(f"  ✓ mag_scatter.png")


# ─── Plot 2: Time Scatter (log) ─────────────────────────────────────
def plot_time_scatter_log(oof_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    y_true = oof_df["target_time_to_max_days"].to_numpy()
    y_pred = oof_df["ensemble_pred_time"].to_numpy()
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true >= 0) & (y_pred >= 0)
    y_true, y_pred = y_true[valid], y_pred[valid]

    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    ax.scatter(log_true, log_pred, alpha=0.3, s=8, c="#ff7f0e", edgecolors="none")
    lim_min = min(log_true.min(), log_pred.min()) - 0.2
    lim_max = max(log_true.max(), log_pred.max()) + 0.2
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1, label="y=x")

    rmse = np.sqrt(np.mean((log_pred - log_true) ** 2))
    mae = np.mean(np.abs(log_pred - log_true))
    ax.set_xlabel("log1p(True time to max, days)")
    ax.set_ylabel("log1p(Ensemble predicted time, days)")
    ax.set_title(f"Time Scatter (log1p) (RMSE={rmse:.3f}, MAE={mae:.3f}, N={len(y_true)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "time_scatter_log.png", dpi=DEFAULT_DPI)
    plt.close(fig)
    print(f"  ✓ time_scatter_log.png")


# ─── Plot 3: Residual Distribution ──────────────────────────────────
def plot_residual_dist(oof_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    y_true_m = oof_df["target_max_mag"].to_numpy()
    y_pred_m = oof_df["ensemble_pred_mag"].to_numpy()
    valid_m = np.isfinite(y_true_m) & np.isfinite(y_pred_m)
    res_m = y_pred_m[valid_m] - y_true_m[valid_m]

    axes[0].hist(res_m, bins=50, color="#1f77b4", alpha=0.7, edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Mag residual (pred - true)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Mag Residuals (μ={res_m.mean():.3f}, σ={res_m.std():.3f})")

    y_true_t = oof_df["target_time_to_max_days"].to_numpy()
    y_pred_t = oof_df["ensemble_pred_time"].to_numpy()
    valid_t = np.isfinite(y_true_t) & np.isfinite(y_pred_t) & (y_true_t >= 0) & (y_pred_t >= 0)
    res_t = np.log1p(y_pred_t[valid_t]) - np.log1p(y_true_t[valid_t])

    axes[1].hist(res_t, bins=50, color="#ff7f0e", alpha=0.7, edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Time residual (log1p pred - log1p true)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Time Residuals, log1p (μ={res_t.mean():.3f}, σ={res_t.std():.3f})")

    fig.tight_layout()
    fig.savefig(output_dir / "residual_dist.png", dpi=DEFAULT_DPI)
    plt.close(fig)
    print(f"  ✓ residual_dist.png")


# ─── Plot 4: Error by Plate Boundary Type ──────────────────────────
def plot_error_by_plate(oof_df: pd.DataFrame, features_df: pd.DataFrame, output_dir: Path) -> None:
    if "mainshock_id" not in oof_df.columns or "mainshock_id" not in features_df.columns:
        print("  ⚠ error_by_plate: 数据缺少 mainshock_id，跳过")
        return

    merged = oof_df.merge(
        features_df[["mainshock_id", "nearest_plate_boundary_type"]],
        on="mainshock_id", how="inner",
    )
    if merged.empty:
        print("  ⚠ error_by_plate: 合并后无数据，跳过")
        return

    groups = merged.groupby("nearest_plate_boundary_type")
    labels, mag_mae, time_asym_mae = [], [], []
    for name, grp in groups:
        if len(grp) < 5:
            continue
        labels.append(str(name))
        v_m = np.isfinite(grp["target_max_mag"]) & np.isfinite(grp["ensemble_pred_mag"])
        mag_mae.append(float(np.mean(np.abs(grp["ensemble_pred_mag"][v_m] - grp["target_max_mag"][v_m]))))

        v_t = np.isfinite(grp["target_time_to_max_days"]) & np.isfinite(grp["ensemble_pred_time"])
        te = grp["ensemble_pred_time"][v_t] - grp["target_time_to_max_days"][v_t]
        lw = 2.0
        tw = np.where(te > 0, lw, 1.0)
        time_asym_mae.append(float(np.mean(tw * np.abs(te))))

    if not labels:
        print("  ⚠ error_by_plate: 无有效分组，跳过")
        return

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, mag_mae, width, label="Mag MAE", color="#1f77b4")
    ax.bar(x + width/2, time_asym_mae, width, label="Time Asymmetric MAE", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Error")
    ax.set_title("Error by Plate Boundary Type")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "error_by_plate.png", dpi=DEFAULT_DPI)
    plt.close(fig)
    print(f"  ✓ error_by_plate.png")


# ─── Plot 5: Ensemble Weights ──────────────────────────────────────
def plot_ensemble_weights(weights_path: Path, output_dir: Path) -> None:
    if not weights_path.exists():
        print("  ⚠ ensemble_weights.json 不存在，跳过")
        return
    with open(weights_path, "r") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    targets = [("mag", "Mag Weights"), ("time", "Time Weights")]
    colors = {"baseline": "#1f77b4", "xgboost": "#ff7f0e", "dl": "#2ca02c", "gnn": "#d62728"}

    for ax, (key, title) in zip(axes, targets):
        weights = data.get(key, data)
        if not isinstance(weights, dict):
            continue
        names = list(weights.keys())
        values = [weights[n] for n in names]
        bar_colors = [colors.get(n, "#9467bd") for n in names]
        ax.bar(names, values, color=bar_colors)
        ax.set_title(title)
        ax.set_ylabel("Weight")
        ax.set_ylim(0, max(max(values, default=0.1) * 1.2, 0.02))
        for i, v in enumerate(values):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "ensemble_weights.png", dpi=DEFAULT_DPI)
    plt.close(fig)
    print(f"  ✓ ensemble_weights.png")


# ─── Main ───────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="余震预测结果综合可视化")
    parser.add_argument("--oof", type=Path, required=True, help="ensemble_oof_predictions.csv 路径")
    parser.add_argument("--features", type=Path, default=None, help="advanced_features.csv 路径")
    parser.add_argument("--weights", type=Path, default=None, help="ensemble_weights.json 路径")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    oof_path = resolve_project_path(args.oof)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    oof_df = pd.read_csv(oof_path)
    print(f"Loaded OOF: {len(oof_df)} samples, {list(oof_df.columns)}")

    # Check required columns
    required = {"target_max_mag", "target_time_to_max_days", "ensemble_pred_mag", "ensemble_pred_time"}
    missing = required - set(oof_df.columns)
    if missing:
        print(f"FATAL: 缺少列 {missing}")
        sys.exit(1)

    plot_mag_scatter(oof_df, output_dir)
    plot_time_scatter_log(oof_df, output_dir)
    plot_residual_dist(oof_df, output_dir)

    # Error by plate
    if args.features:
        feats_path = resolve_project_path(args.features)
        if feats_path.exists():
            features_df = pd.read_csv(feats_path)
            plot_error_by_plate(oof_df, features_df, output_dir)
        else:
            print("  ⚠ features 文件不存在，跳过 error_by_plate")
    else:
        print("  ⚠ 未指定 --features，跳过 error_by_plate")

    # Weights bar chart
    if args.weights:
        weights_path = resolve_project_path(args.weights)
        plot_ensemble_weights(weights_path, output_dir)
    else:
        print("  ⚠ 未指定 --weights，跳过 ensemble_weights.png")

    print(f"\n✓ All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
