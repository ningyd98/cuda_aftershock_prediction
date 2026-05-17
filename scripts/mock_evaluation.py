from __future__ import annotations

"""
模拟线上评测系统 (Mock Evaluation System)

对应 project_plan 第 5.2 节：
"团队内部需构建一个'伪实时'评测系统，模拟接收到主震后动态预测的过程。"

功能:
1. 从历史目录中按时间顺序 replay 地震序列
2. 对每条主震，仅用"主震发生时刻之前"的数据进行训练和预测
3. 累积评估指标，跟踪模型表现
4. 生成评估报告

用法:
  python scripts/mock_evaluation.py --data data/processed/advanced_features.csv --output data/processed/mock_eval_report.csv
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.evaluator import calculate_metrics
from src.models import BaselineLGBM
from src.utils import set_random_seed

TARGET_COLS = ["target_max_mag", "target_time_to_max_days"]
TIME_COL = "mainshock_time"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="模拟线上评测系统")
    parser.add_argument(
        "--data", type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "advanced_features.csv",
    )
    parser.add_argument(
        "--model-dir", type=Path,
        default=PROJECT_ROOT / "data" / "models",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "mock_eval_report.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-train-samples", type=int, default=500,
        help="最少需要多少训练样本才开始评测",
    )
    parser.add_argument(
        "--stride", type=int, default=200,
        help="每次新增多少样本重训一次模型",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=100,
        help="轻量训练树数（模拟评测需频繁重训）",
    )
    return parser.parse_args()


def select_feature_cols(df: pd.DataFrame) -> list[str]:
    """筛选数值型特征列（与训练脚本保持一致）。"""
    FEATURE_PREFIXES = (
        "early_", "gr_", "omori_", "anisotropy_", "plate_type_",
        "count_", "energy_", "etas_", "bath_",
    )
    EXPLICIT = {
        "mainshock_mag", "mainshock_depth", "advanced_early_event_count",
        "plate_boundary_distance_km",
    }
    EXCLUDE = {
        "mainshock_id", "mainshock_time", "mainshock_lat", "mainshock_lon",
        "nearest_plate_boundary_type", *TARGET_COLS,
    }
    candidates = []
    for col in df.columns:
        if col in EXCLUDE:
            continue
        if col in EXPLICIT or col.startswith(FEATURE_PREFIXES):
            if pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].astype(int)
            if pd.api.types.is_numeric_dtype(df[col]):
                candidates.append(col)
    return candidates


def create_expanding_window(
    df_sorted: pd.DataFrame,
    min_train: int,
    stride: int,
) -> list[tuple[int, int, int, int]]:
    """
    生成扩展窗口切分方案。

    返回 list of (train_start, train_end, valid_start, valid_end) 索引。
    第 i 个窗口的验证集是第 i+1 个 stride 块。
    """
    n = len(df_sorted)
    windows = []
    train_end = min_train
    while train_end + stride <= n:
        valid_start = train_end
        valid_end = min(train_end + stride, n)
        if valid_end - valid_start < 10:
            break
        windows.append((0, train_end, valid_start, valid_end))
        train_end = valid_end
    return windows


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    df = pd.read_csv(args.data)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce")
    df = df.dropna(subset=[TIME_COL, *TARGET_COLS]).sort_values(TIME_COL).reset_index(drop=True)

    feature_cols = select_feature_cols(df)
    if not feature_cols:
        raise ValueError("未找到可训练特征列")

    print(f"模拟线上评测")
    print(f"  总样本: {len(df)}")
    print(f"  特征数: {len(feature_cols)}")
    print(f"  最少训练样本: {args.min_train_samples}")
    print(f"  步长: {args.stride}")
    print(f"  轻量树数: {args.n_estimators}")

    windows = create_expanding_window(df, args.min_train_samples, args.stride)
    print(f"  评测窗口数: {len(windows)}\n")

    all_metrics: list[dict] = []
    cumulative_preds: list[np.ndarray] = []
    cumulative_targets: list[np.ndarray] = []

    for win_idx, (tr_s, tr_e, vl_s, vl_e) in enumerate(
        tqdm(windows, desc="模拟评测进度")
    ):
        X_train = df.iloc[tr_s:tr_e][feature_cols]
        y_train = df.iloc[tr_s:tr_e][TARGET_COLS]
        X_valid = df.iloc[vl_s:vl_e][feature_cols]
        y_valid = df.iloc[vl_s:vl_e][TARGET_COLS]

        # 轻量训练
        model = BaselineLGBM(
            random_state=args.seed,
            n_estimators=args.n_estimators,
            learning_rate=0.05,
        )
        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_valid), a_min=0.0, a_max=None)

        # 窗口指标
        win_metrics = calculate_metrics(
            y_true_mag=y_valid.iloc[:, 0].to_numpy(),
            y_pred_mag=preds[:, 0],
            y_true_time=y_valid.iloc[:, 1].to_numpy(),
            y_pred_time=preds[:, 1],
            late_weight=2.0,
        )
        train_start_time = df.iloc[tr_s][TIME_COL]
        train_end_time = df.iloc[tr_e - 1][TIME_COL]
        valid_start_time = df.iloc[vl_s][TIME_COL]
        valid_end_time = df.iloc[vl_e - 1][TIME_COL]

        all_metrics.append({
            "window": win_idx + 1,
            "train_start": str(train_start_time)[:10],
            "train_end": str(train_end_time)[:10],
            "valid_start": str(valid_start_time)[:10],
            "valid_end": str(valid_end_time)[:10],
            "train_size": tr_e - tr_s,
            "valid_size": vl_e - vl_s,
            **win_metrics,
        })
        cumulative_preds.append(preds)
        cumulative_targets.append(y_valid.to_numpy())

    # 累积指标
    all_preds = np.concatenate(cumulative_preds, axis=0)
    all_targets = np.concatenate(cumulative_targets, axis=0)
    cumulative_metrics = calculate_metrics(
        y_true_mag=all_targets[:, 0], y_pred_mag=all_preds[:, 0],
        y_true_time=all_targets[:, 1], y_pred_time=all_preds[:, 1],
        late_weight=2.0,
    )

    report_df = pd.DataFrame(all_metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(args.output, index=False, encoding="utf-8")

    print(f"\n{'='*60}")
    print("模拟线上评测报告")
    print(f"{'='*60}")
    print(report_df.to_string(index=False))
    print(f"\n{'='*60}")
    print("累积评测指标")
    print(f"{'='*60}")
    for k, v in cumulative_metrics.items():
        print(f"  {k}: {v:.4f}")

    # 保存最终模型
    save_dir = args.model_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    final_model = BaselineLGBM(random_state=args.seed, n_estimators=300, learning_rate=0.03)
    final_model.fit(df[feature_cols], df[TARGET_COLS])
    joblib.dump(final_model, save_dir / "mock_eval_model.joblib")
    with (save_dir / "feature_cols.json").open("w") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"\n报告已保存: {args.output}")
    print(f"模型已保存: {save_dir / 'mock_eval_model.joblib'}")


if __name__ == "__main__":
    main()
