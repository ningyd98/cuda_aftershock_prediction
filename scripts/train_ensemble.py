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
from tqdm import tqdm

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


def discover_oof_inputs(model_dir: Path) -> list[tuple[str, str, pd.DataFrame]]:
    """先扫描所有 OOF 文件，确认预测列可用，再统一决定 merge key。"""
    inputs: list[tuple[str, str, pd.DataFrame]] = []
    for model_name, csv_name in tqdm(
        MODEL_FILE_MAP.items(),
        desc="扫描 OOF 文件",
        unit="model",
        leave=False,
    ):
        path = model_dir / csv_name
        df = load_oof_file(path)
        if df is None:
            continue

        mag_col, time_col = MODEL_PRED_COLS[model_name]
        if mag_col not in df.columns or time_col not in df.columns:
            print(f"  ⚠ {csv_name}: 缺少预测列 {mag_col}/{time_col}，跳过 {model_name}")
            continue
        inputs.append((model_name, csv_name, df))
    return inputs


def merge_all_oof(model_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """合并所有可用的 OOF 预测。返回 (merged_df, available_models)。

    优先使用 [mainshock_id, mainshock_time] 作为 join key；
    若缺失 mainshock_time 则退化为仅用 mainshock_id，并打印 warning。
    """
    inputs = discover_oof_inputs(model_dir)
    if not inputs:
        raise FileNotFoundError(f"在 {model_dir} 中未找到任何可用 OOF 文件")

    use_time_key = all(TIME_COL in df.columns for _, _, df in inputs)
    join_keys = [ID_COL, TIME_COL] if use_time_key else [ID_COL]
    if not use_time_key:
        missing_time = [csv_name for _, csv_name, df in inputs if TIME_COL not in df.columns]
        print(f"  ⚠ 以下 OOF 缺少 {TIME_COL}: {sorted(set(missing_time))}，统一使用 {ID_COL} 合并")

    target_input = next(
        ((model_name, csv_name, df) for model_name, csv_name, df in inputs if all(c in df.columns for c in TARGET_COLS)),
        None,
    )
    if target_input is None:
        raise ValueError("可用 OOF 文件均缺少目标列，无法计算融合指标。")

    _, target_csv, target_df = target_input
    base_keep = list(join_keys)
    if TIME_COL in target_df.columns and TIME_COL not in base_keep:
        base_keep.append(TIME_COL)
    base_keep += TARGET_COLS
    merged = target_df[base_keep].copy()
    available: list[str] = []

    for model_name, csv_name, df in tqdm(
        inputs,
        desc="合并 OOF 预测",
        unit="model",
        leave=False,
    ):
        mag_col, time_col = MODEL_PRED_COLS[model_name]
        keep = list(join_keys) + [mag_col, time_col]
        sub = df[keep].copy()
        merged = merged.merge(sub, on=join_keys, how="inner", validate="one_to_one")
        available.append(model_name)

    # 清理 NaN 预测行
    merged = merged.dropna(subset=TARGET_COLS)
    for model_name in available:
        mag_col, time_col = MODEL_PRED_COLS[model_name]
        merged = merged.dropna(subset=[mag_col, time_col])

    print(f"  基准目标文件: {target_csv}")
    print(f"  合并键: {join_keys}")
    print(f"  合并 {len(available)} 个模型 ({', '.join(available)})，有效样本: {len(merged)}")
    return merged, available


def search_weights_for_target(
    merged: pd.DataFrame,
    models: list[str],
    target_idx: int,
    late_weight: float,
    grid_step: float,
) -> tuple[dict[str, float], float]:
    """
    完整 simplex 网格搜索，支持 1/2/3/4 个可用模型。

    target_idx: 0 = mag (最小化 mag_rmse), 1 = time (最小化 time_asymmetric_rmse)
    单模型时也真实计算 objective，不返回 0。
    """
    pred_cols = [MODEL_PRED_COLS[m][target_idx] for m in models]
    pred_matrix = merged[pred_cols].to_numpy(dtype=float)
    y_true = merged[TARGET_COLS[target_idx]].to_numpy(dtype=float)
    n_models = len(models)
    target_name = "震级" if target_idx == 0 else "时间"

    def _compute_objective(wvec: np.ndarray) -> float:
        pred = pred_matrix @ wvec
        if target_idx == 0:
            return float(np.sqrt(np.mean((pred - y_true) ** 2)))
        time_err = pred - y_true
        time_w = np.where(time_err > 0, late_weight, 1.0)
        return float(np.sqrt(np.mean(time_w * time_err ** 2)))

    # 单模型：真实计算 objective
    if n_models == 1:
        obj = _compute_objective(np.array([1.0]))
        return {models[0]: 1.0}, obj

    # 2 模型：完整 1D 网格 (w, 1-w)
    if n_models == 2:
        best_objective = float("inf")
        best_weights = {models[0]: 0.5, models[1]: 0.5}
        grid = np.arange(0.0, 1.0 + grid_step / 2.0, grid_step)
        for w in tqdm(
            grid,
            desc=f"{target_name} 2模型权重搜索",
            unit="weight",
            leave=False,
        ):
            wvec = np.array([w, 1.0 - w])
            obj = _compute_objective(wvec)
            if obj < best_objective:
                best_objective = obj
                best_weights = {models[0]: round(float(w), 4), models[1]: round(float(1 - w), 4)}
        total = sum(best_weights.values())
        best_weights = {k: round(v / total, 4) for k, v in best_weights.items()}
        return best_weights, float(best_objective)

    # 3+ 模型：simplex 采样网格
    # 生成所有和为 1 的 n 元组 (步长 grid_step)
    best_objective = float("inf")
    best_weights = {m: 1.0 / n_models for m in models}

    def _gen_simplex(n: int, step: float):
        """迭代生成 n 维单纯形上的网格点 (和为 1, 步长 step)。"""
        n_pts = int(1.0 / step)
        if n == 3:
            for i in range(n_pts + 1):
                for j in range(n_pts + 1 - i):
                    k = n_pts - i - j
                    yield np.array([i, j, k], dtype=float) / n_pts
        elif n == 4:
            for i in range(n_pts + 1):
                for j in range(n_pts + 1 - i):
                    for k in range(n_pts + 1 - i - j):
                        l = n_pts - i - j - k
                        yield np.array([i, j, k, l], dtype=float) / n_pts

    def _simplex_grid_total(n: int, step: float) -> int | None:
        """计算当前 3/4 模型单纯形网格点数量，用于准确显示进度。"""
        n_pts = int(1.0 / step)
        if n == 3:
            return (n_pts + 1) * (n_pts + 2) // 2
        if n == 4:
            return (n_pts + 1) * (n_pts + 2) * (n_pts + 3) // 6
        return None

    simplex_iter = tqdm(
        _gen_simplex(n_models, grid_step),
        total=_simplex_grid_total(n_models, grid_step),
        desc=f"{target_name} simplex 权重搜索",
        unit="weight",
        leave=False,
    )
    for wvec in simplex_iter:
        obj = _compute_objective(wvec)
        if obj < best_objective:
            best_objective = obj
            best_weights = {models[i]: round(float(wvec[i]), 4) for i in range(n_models)}

    total = sum(best_weights.values())
    best_weights = {k: round(v / total, 4) for k, v in best_weights.items()}
    return best_weights, float(best_objective)


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
    for m in tqdm(available, desc="计算融合预测", unit="model", leave=False):
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


def search_ridge_stacking_weights(
    merged: pd.DataFrame,
    available: list[str],
    target_idx: int,
    late_weight: float,
) -> tuple[dict[str, float], float]:
    """使用 Ridge 回归 (带非负约束) 学习 stacking 权重。

    相比 simplex 网格搜索，Ridge stacking 可以学到更精细的权重分配，
    且通过 L2 正则化防止过拟合。

    Args:
        merged: 合并的 OOF DataFrame
        available: 可用模型列表
        target_idx: 0=mag, 1=time
        late_weight: 时间目标预测偏晚惩罚

    Returns:
        (权重 dict, 最优 objective 值)
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    pred_cols = [MODEL_PRED_COLS[m][target_idx] for m in available]
    X_stack = merged[pred_cols].to_numpy(dtype=float)
    y_true = merged[TARGET_COLS[target_idx]].to_numpy(dtype=float)

    if len(available) <= 1:
        return {available[0]: 1.0}, float(
            np.sqrt(np.mean((X_stack[:, 0] - y_true) ** 2))
        )

    # 标准化输入
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_stack)

    # 时间目标: 对正误差加权
    if target_idx == 1:  # time
        sample_weight = np.ones(len(y_true))
        residuals = np.zeros(len(y_true))
        # 先用等权融合的残差方向来确定 sample_weight
        mean_pred = X_stack.mean(axis=1)
        residuals = mean_pred - y_true
        sample_weight = np.where(residuals > 0, late_weight, 1.0)
    else:
        sample_weight = np.ones(len(y_true))

    # Ridge 回归，对正系数非负约束 (通过多次尝试不同的 alpha)
    best_weights = None
    best_objective = float("inf")

    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        ridge = Ridge(alpha=alpha, fit_intercept=False, positive=True)
        ridge.fit(X_scaled, y_true, sample_weight=sample_weight)
        # 恢复原始尺度的系数
        w_raw = ridge.coef_ / scaler.scale_
        w_sum = w_raw.sum()
        if w_sum <= 0:
            continue
        w_norm = w_raw / w_sum
        pred = X_stack @ w_norm

        if target_idx == 0:
            obj = float(np.sqrt(np.mean((pred - y_true) ** 2)))
        else:
            time_err = pred - y_true
            time_w = np.where(time_err > 0, late_weight, 1.0)
            obj = float(np.sqrt(np.mean(time_w * time_err ** 2)))

        if obj < best_objective:
            best_objective = obj
            best_weights = {available[i]: round(float(w_norm[i]), 4)
                           for i in range(len(available))}

    # Fallback: 如果 Ridge 没有找到更好结果，用等权
    if best_weights is None:
        best_weights = {m: round(1.0 / len(available), 4) for m in available}
        pred = X_stack @ np.array(list(best_weights.values()))
        if target_idx == 0:
            best_objective = float(np.sqrt(np.mean((pred - y_true) ** 2)))
        else:
            time_err = pred - y_true
            time_w = np.where(time_err > 0, late_weight, 1.0)
            best_objective = float(np.sqrt(np.mean(time_w * time_err ** 2)))

    return best_weights, best_objective


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

    # 验证 grid_step 能否整除 1.0
    if 1.0 % args.grid_step > 1e-9 and abs(1.0 % args.grid_step - args.grid_step) > 1e-9:
        # 不能整除，自动修整为最接近的整除数
        n_pts = max(1, int(1.0 / args.grid_step))
        corrected = 1.0 / n_pts
        print(f"⚠ grid_step={args.grid_step} 不能整除 1.0，已自动修正为 {corrected:.4f}")
        args.grid_step = corrected

    print("=== 多模型 OOF 融合权重搜索 ===")
    print(f"模型目录: {model_dir}")
    print(f"步长: {args.grid_step}\n")

    # 1. 合并 OOF
    merged, available = merge_all_oof(model_dir)
    print(f"\n可用模型: {available}")

    # 2. 搜索震级权重
    print("\n--- 震级权重搜索 (minimize mag_rmse) ---")

    # 同时运行两种方法
    mag_weights_grid, mag_best_grid = search_weights_for_target(
        merged, available, target_idx=0,
        late_weight=args.late_weight, grid_step=args.grid_step,
    )
    mag_weights_ridge, mag_best_ridge = search_ridge_stacking_weights(
        merged, available, target_idx=0, late_weight=args.late_weight,
    )

    # 选择更优的方法
    mag_weights = mag_weights_grid if mag_best_grid <= mag_best_ridge else mag_weights_ridge
    mag_best = min(mag_best_grid, mag_best_ridge)
    mag_method = "grid" if mag_best_grid <= mag_best_ridge else "ridge"
    print(f"  Grid:  {mag_best_grid:.4f}  {mag_weights_grid}")
    print(f"  Ridge: {mag_best_ridge:.4f}  {mag_weights_ridge}")
    print(f"  选用 [{mag_method}] 权重: {mag_weights}")
    print(f"  最优 mag_rmse: {mag_best:.4f}")

    # 3. 搜索时间权重
    print("\n--- 时间权重搜索 (minimize time_asymmetric_rmse) ---")

    time_weights_grid, time_best_grid = search_weights_for_target(
        merged, available, target_idx=1,
        late_weight=args.late_weight, grid_step=args.grid_step,
    )
    time_weights_ridge, time_best_ridge = search_ridge_stacking_weights(
        merged, available, target_idx=1, late_weight=args.late_weight,
    )

    time_weights = time_weights_grid if time_best_grid <= time_best_ridge else time_weights_ridge
    time_best = min(time_best_grid, time_best_ridge)
    time_method = "grid" if time_best_grid <= time_best_ridge else "ridge"
    print(f"  Grid:  {time_best_grid:.4f}  {time_weights_grid}")
    print(f"  Ridge: {time_best_ridge:.4f}  {time_weights_ridge}")
    print(f"  选用 [{time_method}] 权重: {time_weights}")
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
    for m in tqdm(available, desc="保存单模型 OOF 列", unit="model", leave=False):
        mag_col, time_col = MODEL_PRED_COLS[m]
        oof_out[f"{m}_pred_mag"] = merged[mag_col]
        oof_out[f"{m}_pred_time"] = merged[time_col]
    oof_out.to_csv(output_dir / "ensemble_oof_predictions.csv", index=False, encoding="utf-8")

    print(f"\n✓ 融合产物已保存:")
    print(f"  {output_dir / 'ensemble_weights.json'}")
    print(f"  {output_dir / 'ensemble_metrics.json'}")
    print(f"  {output_dir / 'ensemble_oof_predictions.csv'}")

    # 6. 单模型 vs 融合对比 (同时收集单模型指标写入 metrics_json)
    per_model_metrics: dict[str, dict] = {}
    print(f"\n{'='*60}")
    print("单模型 vs 融合 (OOF 指标)")
    print(f"{'='*60}")
    for m in tqdm(available, desc="单模型指标对比", unit="model", leave=False):
        mag_col, time_col = MODEL_PRED_COLS[m]
        m_metrics = calculate_metrics(
            y_true_mag=merged[TARGET_COLS[0]].to_numpy(),
            y_pred_mag=merged[mag_col].to_numpy(),
            y_true_time=merged[TARGET_COLS[1]].to_numpy(),
            y_pred_time=merged[time_col].to_numpy(),
            late_weight=args.late_weight,
        )
        per_model_metrics[m] = {k: round(float(v), 4) for k, v in m_metrics.items()}
        print(f"\n  [{m}]")
        print(f"    mag_rmse={m_metrics['mag_rmse']:.4f}  time_rmse={m_metrics['time_rmse']:.4f}  time_asymmetric_rmse={m_metrics['time_asymmetric_rmse']:.4f}")

    print(f"\n  [ENSEMBLE]")
    print(f"    mag_rmse={ensemble_metrics['mag_rmse']:.4f}  time_rmse={ensemble_metrics['time_rmse']:.4f}  time_asymmetric_rmse={ensemble_metrics['time_asymmetric_rmse']:.4f}")

    # 更新 metrics_json 包含单模型指标
    metrics_json["per_model_metrics"] = per_model_metrics
    with (output_dir / "ensemble_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_json, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
