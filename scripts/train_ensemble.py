from __future__ import annotations

from __future__ import annotations

"""
多模型 OOF 融合权重搜索。

读取各模型的 OOF 预测 CSV，按 mainshock_id + mainshock_time 合并，
分别为震级 (mag) 和时间 (time) 搜索最优融合权重。

震级目标: 最小化 mag_rmse
时间目标: 最小化 time_asymmetric_rmse (考虑预测偏晚惩罚)

输出:
  data/models/ensemble_weights.json  (新格式: {"mag": {...}, "time": {...}})
  data/models/ensemble_metrics.json
  data/models/ensemble_oof_predictions.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.evaluator import calculate_metrics

TARGET_COLS = ["target_max_mag", "target_time_to_max_days"]
TIME_COL = "mainshock_time"
ID_COL = "mainshock_id"

# 所有可能的模型名 -> OOF 文件名
MODEL_FILE_MAP = {
    "baseline": "oof_predictions.csv",
    "xgboost": "oof_predictions.csv",
    "dl": "dl_oof_predictions.csv",
    "gnn": "gnn_oof_predictions.csv",
}

# 每个模型 OOF CSV 中的预测列名
MODEL_PRED_COLS = {
    "baseline": ("baseline_pred_mag", "baseline_pred_time"),
    "xgboost": ("xgboost_pred_mag", "xgboost_pred_time"),
    "dl": ("dl_pred_mag", "dl_pred_time"),
    "gnn": ("gnn_pred_mag", "gnn_pred_time"),
}


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_oof_file(path: Path) -> pd.DataFrame | None:
    """加载单个 OOF 预测文件。"""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if ID_COL not in df.columns:
        print(f"  ⚠ {path.name}: 缺少 {ID_COL} 列，跳过")
        return None
    return df


def merge_all_oof(model_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """合并且所有可用的 OOF 预测。返回 (merged_df, available_models)。"""
    base_cols = [ID_COL, TIME_COL, *TARGET_COLS]
    merged: pd.DataFrame | None = None
    available: list[str] = []

    for model_name, csv_name in MODEL_FILE_MAP.items():
        path = model_dir / csv_name
        df = load_oof_file(path)
        if df is None:
            continue

        mag_col, time_col = MODEL_PRED_COLS[model_name]
        if mag_col not in df.columns or time_col not in df.columns:
            print(f"  ⚠ {csv_name}: 缺少预测列 {mag_col}/{time_col}，跳过 {model_name}")
            continue

        keep = [c for c in base_cols if c in df.columns] + [mag_col, time_col]
        sub = df[keep].copy()

        if merged is None:
            merged = sub
        else:
            merged = merged.merge(sub, on=ID_COL, how="inner", suffixes=("", f"_dup_{model_name}"))
        available.append(model_name)

    if merged is None or not available:
        raise FileNotFoundError(f"在 {model_dir} 中未找到任何可用 OOF 文件")

    # 清理 NaN 预测行
    for model_name in available:
        mag_col, time_col = MODEL_PRED_COLS[model_name]
        merged = merged.dropna(subset=[mag_col, time_col])

    print(f"  合并 {len(available)} 个模型 ({', '.join(available)})，有效样本: {len(merged)}")
    return merged, available


def search_weights_for_target(
    merged: pd.DataFrame,
    models: list[str],
    target_idx: int,
    objective: str,
    late_weight: float,
    grid_step: float,
) -> tuple[dict[str, float], float]:
    """
    为单个目标搜索融合权重。

    target_idx: 0 = mag, 1 = time
    objective: "mag_rmse" 或 "time_asymmetric_rmse"
    """
    n_models = len(models)
    if n_models == 1:
        return {models[0]: 1.0}, 0.0

    # 构建预测矩阵: (N, n_models)
    pred_cols = [MODEL_PRED_COLS[m][target_idx] for m in models]
    pred_matrix = merged[pred_cols].to_numpy(dtype=float)
    y_true = merged[TARGET_COLS[target_idx]].to_numpy(dtype=float)

    # 权重网格：n_models 维单纯形
    # 先搜索 2 模型组合（baseline+xgboost），然后固定到多模型归一化
    best_objective = float("inf")
    best_weights = {m: 1.0 / n_models for m in models}

    if n_models == 2:
        grid = np.arange(0.0, 1.0 + grid_step / 2.0, grid_step)
        for w in grid:
            wvec = np.array([w, 1.0 - w])
            pred = pred_matrix @ wvec
            if target_idx == 0:
                val = float(np.sqrt(np.mean((pred - y_true) ** 2)))
            else:
                time_err = pred - y_true
                time_w = np.where(time_err > 0, late_weight, 1.0)
                val = float(np.sqrt(np.mean(time_w * time_err ** 2)))
            if val < best_objective:
                best_objective = val
                best_weights = {models[0]: round(float(w), 4), models[1]: round(float(1 - w), 4)}
    else:
        # 多模型：仅均匀 + 两两组合搜索后归一化
        # 基准: 均匀
        wvec = np.ones(n_models) / n_models
        pred = pred_matrix @ wvec
        if target_idx == 0:
            best_objective = float(np.sqrt(np.mean((pred - y_true) ** 2)))
        else:
            time_err = pred - y_true
            time_w = np.where(time_err > 0, late_weight, 1.0)
            best_objective = float(np.sqrt(np.mean(time_w * time_err ** 2)))

        # 贪心：逐个增加/减少权重
        base_vals = np.ones(n_models) / n_models
        grid = np.arange(0.0, 1.0 + grid_step, grid_step)
        for i in range(n_models):
            for w in grid:
                trial = base_vals.copy()
                trial[i] = w
                trial = trial / trial.sum()
                pred = pred_matrix @ trial
                if target_idx == 0:
                    val = float(np.sqrt(np.mean((pred - y_true) ** 2)))
                else:
                    time_err = pred - y_true
                    time_w = np.where(time_err > 0, late_weight, 1.0)
                    val = float(np.sqrt(np.mean(time_w * time_err ** 2)))
                if val < best_objective:
                    best_objective = val
                    best_weights = {models[j]: round(float(trial[j]), 4) for j in range(n_models)}

    # 归一化确保和为 1
    total = sum(best_weights.values())
    best_weights = {k: round(v / total, 4) for k, v in best_weights.items()}
    return best_weights, best_objective


def compute_ensemble_metrics(
    merged: pd.DataFrame,
    available: list[str],
    mag_weights: dict[str, float],
    time_weights: dict[str, float],
    late_weight: float,
) -> dict:
    """计算融合后的整体指标。"""
    # 融合预测
    pred_mag = np.zeros(len(merged))
    pred_time = np.zeros(len(merged))
    for m in available:
        mag_col, time_col = MODEL_PRED_COLS[m]
        pred_mag += merged[mag_col].to_numpy() * mag_weights.get(m, 0.0)
        pred_time += merged[time_col].to_numpy() * time_weights.get(m, 0.0)

    return calculate_metrics(
        y_true_mag=merged[TARGET_COLS[0]].to_numpy(),
        y_pred_mag=pred_mag,
        y_true_time=merged[TARGET_COLS[1]].to_numpy(),
        y_pred_time=pred_time,
        late_weight=late_weight,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多模型 OOF 融合权重搜索")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "models",
        help="模型产物目录 (含 OOF CSV)",
    )
    parser.add_argument("--grid-step", type=float, default=0.02, help="权重搜索步长")
    parser.add_argument("--late-weight", type=float, default=2.0, help="预测偏晚惩罚权重")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录，默认同 --model-dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = resolve_project_path(args.model_dir)
    output_dir = resolve_project_path(args.output_dir) if args.output_dir else model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== 多模型 OOF 融合权重搜索 ===")
    print(f"模型目录: {model_dir}")
    print(f"步长: {args.grid_step}\n")

    # 1. 合并 OOF
    merged, available = merge_all_oof(model_dir)
    print(f"\n可用模型: {available}")

    # 2. 搜索震级权重
    print("\n--- 震级权重搜索 (minimize mag_rmse) ---")
    mag_weights, mag_best = search_weights_for_target(
        merged, available, target_idx=0,
        objective="mag_rmse", late_weight=args.late_weight, grid_step=args.grid_step,
    )
    for m, w in mag_weights.items():
        print(f"  {m}: {w:.4f}")
    print(f"  最优 mag_rmse: {mag_best:.4f}")

    # 3. 搜索时间权重
    print("\n--- 时间权重搜索 (minimize time_asymmetric_rmse) ---")
    time_weights, time_best = search_weights_for_target(
        merged, available, target_idx=1,
        objective="time_asymmetric_rmse", late_weight=args.late_weight, grid_step=args.grid_step,
    )
    for m, w in time_weights.items():
        print(f"  {m}: {w:.4f}")
    print(f"  最优 time_asymmetric_rmse: {time_best:.4f}")

    # 4. 计算融合指标
    ensemble_metrics = compute_ensemble_metrics(
        merged, available, mag_weights, time_weights, args.late_weight,
    )
    print(f"\n--- 融合模型指标 ---")
    for k, v in ensemble_metrics.items():
        print(f"  {k}: {v:.4f}")

    # 5. 保存
    weights_json = {
        "mag": mag_weights,
        "time": time_weights,
    }
    with (output_dir / "ensemble_weights.json").open("w", encoding="utf-8") as f:
        json.dump(weights_json, f, ensure_ascii=False, indent=2)

    metrics_json = {
        "mag_search_objective": "mag_rmse",
        "mag_best_value": round(float(mag_best), 4),
        "time_search_objective": "time_asymmetric_rmse",
        "time_best_value": round(float(time_best), 4),
        "ensemble_metrics": {k: round(float(v), 4) for k, v in ensemble_metrics.items()},
        "available_models": available,
        "grid_step": args.grid_step,
        "late_weight": args.late_weight,
    }
    with (output_dir / "ensemble_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_json, f, ensure_ascii=False, indent=2)

    # 保存融合后的 OOF 预测
    ensemble_pred_mag = np.zeros(len(merged))
    ensemble_pred_time = np.zeros(len(merged))
    for m in available:
        mag_col, time_col = MODEL_PRED_COLS[m]
        ensemble_pred_mag += merged[mag_col].to_numpy() * mag_weights.get(m, 0.0)
        ensemble_pred_time += merged[time_col].to_numpy() * time_weights.get(m, 0.0)

    oof_out = merged[[ID_COL, TIME_COL, *TARGET_COLS]].copy()
    oof_out["ensemble_pred_mag"] = ensemble_pred_mag
    oof_out["ensemble_pred_time"] = ensemble_pred_time
    # 同时保留各单模型预测
    for m in available:
        mag_col, time_col = MODEL_PRED_COLS[m]
        oof_out[f"{m}_pred_mag"] = merged[mag_col]
        oof_out[f"{m}_pred_time"] = merged[time_col]
    oof_out.to_csv(output_dir / "ensemble_oof_predictions.csv", index=False, encoding="utf-8")

    print(f"\n✓ 融合产物已保存:")
    print(f"  {output_dir / 'ensemble_weights.json'}")
    print(f"  {output_dir / 'ensemble_metrics.json'}")
    print(f"  {output_dir / 'ensemble_oof_predictions.csv'}")

    # 6. 单模型 vs 融合对比
    print(f"\n{'='*60}")
    print("单模型 vs 融合 (OOF 指标)")
    print(f"{'='*60}")
    for m in available:
        mag_col, time_col = MODEL_PRED_COLS[m]
        m_metrics = calculate_metrics(
            y_true_mag=merged[TARGET_COLS[0]].to_numpy(),
            y_pred_mag=merged[mag_col].to_numpy(),
            y_true_time=merged[TARGET_COLS[1]].to_numpy(),
            y_pred_time=merged[time_col].to_numpy(),
            late_weight=args.late_weight,
        )
        print(f"\n  [{m}]")
        print(f"    mag_rmse={m_metrics['mag_rmse']:.4f}  time_rmse={m_metrics['time_rmse']:.4f}  time_asymmetric_rmse={m_metrics['time_asymmetric_rmse']:.4f}")

    print(f"\n  [ENSEMBLE]")
    print(f"    mag_rmse={ensemble_metrics['mag_rmse']:.4f}  time_rmse={ensemble_metrics['time_rmse']:.4f}  time_asymmetric_rmse={ensemble_metrics['time_asymmetric_rmse']:.4f}")


if __name__ == "__main__":
    main()
