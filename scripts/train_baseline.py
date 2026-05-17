from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models import BaselineLGBM, BaselineXGBoost
from src.trainer import time_series_cv_train
from src.utils import set_random_seed


TARGET_COLS = ["target_max_mag", "target_time_to_max_days"]
TIME_COL = "mainshock_time"
FEATURE_PREFIXES = (
    "early_",
    "gr_",
    "omori_",
    "anisotropy_",
    "plate_type_",
    "count_",
    "energy_",
    "etas_",
)
EXPLICIT_FEATURES = {
    "mainshock_mag",
    "mainshock_depth",
    "advanced_early_event_count",
    "plate_boundary_distance_km",
}
EXCLUDE_COLS = {
    "mainshock_id",
    "mainshock_time",
    "mainshock_lat",
    "mainshock_lon",
    "nearest_plate_boundary_type",
    *TARGET_COLS,
}


def resolve_project_path(path_value: str | Path) -> Path:
    """将相对路径解析到项目根目录。"""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """自动筛选阶段一产生的数值特征列。"""
    candidates: list[str] = []
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
        if col in EXPLICIT_FEATURES or col.startswith(FEATURE_PREFIXES):
            candidates.append(col)

    # 布尔特征转成 0/1，类别特征不进入模型。
    numeric_cols: list[str] = []
    for col in candidates:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    return numeric_cols


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """清理训练数据：目标和时间必须存在，特征缺失保留给树模型处理。"""
    cleaned_df = df.copy()
    cleaned_df[TIME_COL] = pd.to_datetime(
        cleaned_df[TIME_COL],
        utc=True,
        errors="coerce",
        format="mixed",
    )
    cleaned_df = cleaned_df.dropna(subset=[TIME_COL, *TARGET_COLS]).reset_index(drop=True)
    return cleaned_df


def parse_args() -> argparse.Namespace:
    """解析训练参数。"""
    parser = argparse.ArgumentParser(description="训练余震预测 LightGBM 基线模型")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "advanced_features.csv",
        help="阶段一高级特征 CSV 路径",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="时间序列 CV 折数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--late-weight", type=float, default=2.0, help="预测偏晚惩罚权重")
    parser.add_argument("--n-estimators", type=int, default=300, help="树模型迭代轮数")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="学习率")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="若提供，则在该目录保存全量训练后的模型与特征列",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="lightgbm",
        choices=["lightgbm", "xgboost", "both"],
        help="模型类型: lightgbm, xgboost, both",
    )
    return parser.parse_args()


def save_training_artifacts(
    save_dir: Path,
    model,
    feature_cols: list[str],
    fold_metrics_df: pd.DataFrame,
    mean_metrics: dict,
    args: argparse.Namespace,
    model_filename: str = "baseline_model.joblib",
) -> None:
    """保存比赛推理需要的 baseline 模型、特征列和元信息。"""
    save_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, save_dir / model_filename)

    with (save_dir / "feature_cols.json").open("w", encoding="utf-8") as file:
        json.dump(feature_cols, file, ensure_ascii=False, indent=2)

    ensemble_weights_path = save_dir / "ensemble_weights.json"
    if not ensemble_weights_path.exists():
        with ensemble_weights_path.open("w", encoding="utf-8") as file:
            json.dump({"baseline": 1.0, "dl": 0.0}, file, ensure_ascii=False, indent=2)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": model.backend,
        "target_cols": TARGET_COLS,
        "time_col": TIME_COL,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "late_weight": args.late_weight,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "mean_metrics": mean_metrics,
    }
    with (save_dir / "model_meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)

    fold_metrics_df.to_csv(save_dir / "cv_metrics.csv", index=False, encoding="utf-8")
    print(f"\n模型产物已保存: {save_dir}")


def main() -> None:
    """读取高级特征并运行时间序列交叉验证。"""
    args = parse_args()
    set_random_seed(args.seed)

    data_path = resolve_project_path(args.data)
    df = prepare_training_frame(pd.read_csv(data_path))
    feature_cols = select_feature_columns(df)

    if not feature_cols:
        raise ValueError("未筛选到任何可训练特征列，请检查 advanced_features.csv。")

    print(f"训练数据: {data_path}")
    print(f"样本数: {len(df)}")
    print(f"特征数: {len(feature_cols)}")
    print("特征列:")
    print(", ".join(feature_cols))

    def model_factory():
        return BaselineLGBM(
            random_state=args.seed,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
        )

    fold_metrics_df, mean_metrics = time_series_cv_train(
        df=df,
        feature_cols=feature_cols,
        target_cols=TARGET_COLS,
        n_splits=args.n_splits,
        model_factory=model_factory,
        time_col=TIME_COL,
        late_weight=args.late_weight,
    )

    display_cols = [
        "fold",
        "backend",
        "train_size",
        "valid_size",
        "train_start",
        "train_end",
        "valid_start",
        "valid_end",
        "mag_rmse",
        "mag_mae",
        "time_rmse",
        "time_mae",
        "time_asymmetric_mae",
        "time_asymmetric_rmse",
    ]

    print("\n每折验证指标:")
    print(fold_metrics_df[display_cols].to_string(index=False))

    print("\n平均验证指标:")
    for key, value in mean_metrics.items():
        print(f"{key}: {value:.6f}")

    if args.save_dir is not None:
        save_dir = resolve_project_path(args.save_dir)

        # --- LightGBM ---
        if args.model_type in ("lightgbm", "both"):
            print("\n--- 全量训练 LightGBM 模型 ---")
            lgbm_model = BaselineLGBM(
                random_state=args.seed,
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
            )
            lgbm_model.fit(df[feature_cols], df[TARGET_COLS])
            save_training_artifacts(
                save_dir=save_dir,
                model=lgbm_model,
                feature_cols=feature_cols,
                fold_metrics_df=fold_metrics_df,
                mean_metrics=mean_metrics,
                args=args,
                model_filename="baseline_model.joblib",
            )

        # --- XGBoost ---
        if args.model_type in ("xgboost", "both"):
            print("\n--- 全量训练 XGBoost 模型 ---")
            xgb_model = BaselineXGBoost(
                random_state=args.seed,
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
            )
            xgb_model.fit(df[feature_cols], df[TARGET_COLS])
            save_training_artifacts(
                save_dir=save_dir,
                model=xgb_model,
                feature_cols=feature_cols,
                fold_metrics_df=fold_metrics_df,
                mean_metrics=mean_metrics,
                args=args,
                model_filename="xgboost_model.joblib",
            )

            # 更新融合权重
            ensemble_path = save_dir / "ensemble_weights.json"
            with ensemble_path.open("w", encoding="utf-8") as f:
                json.dump({"baseline": 0.6, "xgboost": 0.4, "dl": 0.0}, f, ensure_ascii=False, indent=2)
            print(f"双模型融合权重已保存: baseline=0.6, xgboost=0.4")


if __name__ == "__main__":
    main()
