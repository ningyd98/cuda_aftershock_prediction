from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.evaluator import calculate_metrics
from src.models import BaselineLGBM, BaselineXGBoost
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
MODEL_FILE_NAMES = {
    "baseline": "baseline_model.joblib",
    "xgboost": "xgboost_model.joblib",
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
    return cleaned_df.dropna(subset=[TIME_COL, *TARGET_COLS]).reset_index(drop=True)


def requested_model_names(model_type: str) -> list[str]:
    """把命令行模型类型转换为内部模型名称。"""
    if model_type == "lightgbm":
        return ["baseline"]
    if model_type == "xgboost":
        return ["xgboost"]
    return ["baseline", "xgboost"]


def build_model(model_name: str, args: argparse.Namespace):
    """按模型名称创建一个全新的模型实例。"""
    common_kwargs = {
        "random_state": args.seed,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
    }
    if model_name == "baseline":
        return BaselineLGBM(**common_kwargs)
    if model_name == "xgboost":
        return BaselineXGBoost(**common_kwargs)
    raise ValueError(f"未知模型名称: {model_name}")


def parse_args() -> argparse.Namespace:
    """解析训练参数。"""
    parser = argparse.ArgumentParser(description="训练余震预测树模型基线")
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
    parser.add_argument(
        "--ensemble-grid-step",
        type=float,
        default=0.02,
        help="LightGBM/XGBoost OOF 融合权重搜索步长",
    )
    return parser.parse_args()


def run_oof_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_names: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    """按时间滚动验证，并为每个树模型生成 OOF 预测。"""
    train_df = df.sort_values(TIME_COL).reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=args.n_splits)
    X = train_df[feature_cols]
    y = train_df[TARGET_COLS]

    oof_preds = {
        model_name: np.full((len(train_df), len(TARGET_COLS)), np.nan, dtype=float)
        for model_name in model_names
    }
    fold_records: list[dict] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
        train_end_time = train_df.loc[train_idx[-1], TIME_COL]
        valid_start_time = train_df.loc[valid_idx[0], TIME_COL]
        if valid_start_time <= train_end_time:
            raise RuntimeError("时间序列切分异常：验证集时间未晚于训练集。")

        for model_name in model_names:
            model = build_model(model_name, args)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = np.asarray(model.predict(X.iloc[valid_idx]), dtype=float)
            preds = np.clip(preds, a_min=0.0, a_max=None)
            oof_preds[model_name][valid_idx] = preds

            metrics = calculate_metrics(
                y_true_mag=y.iloc[valid_idx, 0].to_numpy(),
                y_pred_mag=preds[:, 0],
                y_true_time=y.iloc[valid_idx, 1].to_numpy(),
                y_pred_time=preds[:, 1],
                late_weight=args.late_weight,
            )
            fold_records.append(
                {
                    "fold": fold_idx,
                    "model": model_name,
                    "backend": getattr(model, "backend", model.__class__.__name__),
                    "train_size": int(len(train_idx)),
                    "valid_size": int(len(valid_idx)),
                    "train_start": str(train_df.loc[train_idx[0], TIME_COL])[:10],
                    "train_end": str(train_end_time)[:10],
                    "valid_start": str(valid_start_time)[:10],
                    "valid_end": str(train_df.loc[valid_idx[-1], TIME_COL])[:10],
                    **metrics,
                }
            )

    fold_metrics_df = pd.DataFrame(fold_records)
    mean_metrics_by_model = {
        model_name: {
            metric: float(value)
            for metric, value in (
                fold_metrics_df.loc[fold_metrics_df["model"] == model_name]
                .select_dtypes(include=[np.number])
                .drop(columns=["fold", "train_size", "valid_size"], errors="ignore")
                .mean()
                .items()
            )
        }
        for model_name in model_names
    }

    oof_df = train_df[["mainshock_id", TIME_COL, *TARGET_COLS]].copy()
    for model_name, preds in oof_preds.items():
        oof_df[f"{model_name}_pred_mag"] = preds[:, 0]
        oof_df[f"{model_name}_pred_time"] = preds[:, 1]
    return fold_metrics_df, oof_df, mean_metrics_by_model


def _valid_oof_mask(oof_df: pd.DataFrame, model_names: list[str]) -> np.ndarray:
    """选出所有参与模型都已有 OOF 预测的验证样本。"""
    mask = np.ones(len(oof_df), dtype=bool)
    for model_name in model_names:
        mask &= oof_df[f"{model_name}_pred_mag"].notna().to_numpy()
        mask &= oof_df[f"{model_name}_pred_time"].notna().to_numpy()
    return mask


def search_tree_ensemble_weights(
    oof_df: pd.DataFrame,
    model_names: list[str],
    late_weight: float,
    grid_step: float,
) -> tuple[dict[str, float], dict]:
    """基于 OOF 预测搜索树模型融合权重。"""
    weights = {"baseline": 0.0, "xgboost": 0.0, "dl": 0.0, "gnn": 0.0}
    active_names = [name for name in model_names if name in {"baseline", "xgboost"}]
    if len(active_names) == 1:
        weights[active_names[0]] = 1.0
        mask = _valid_oof_mask(oof_df, active_names)
        pred_mag = oof_df.loc[mask, f"{active_names[0]}_pred_mag"].to_numpy()
        pred_time = oof_df.loc[mask, f"{active_names[0]}_pred_time"].to_numpy()
        metrics = calculate_metrics(
            oof_df.loc[mask, TARGET_COLS[0]].to_numpy(),
            pred_mag,
            oof_df.loc[mask, TARGET_COLS[1]].to_numpy(),
            pred_time,
            late_weight=late_weight,
        )
        metrics["ensemble_objective"] = float(metrics["mag_rmse"] + metrics["time_asymmetric_rmse"])
        return weights, metrics

    mask = _valid_oof_mask(oof_df, active_names)
    if not mask.any():
        weights["baseline"] = 1.0
        return weights, {"ensemble_objective": float("nan")}

    y_mag = oof_df.loc[mask, TARGET_COLS[0]].to_numpy()
    y_time = oof_df.loc[mask, TARGET_COLS[1]].to_numpy()
    baseline_mag = oof_df.loc[mask, "baseline_pred_mag"].to_numpy()
    baseline_time = oof_df.loc[mask, "baseline_pred_time"].to_numpy()
    xgb_mag = oof_df.loc[mask, "xgboost_pred_mag"].to_numpy()
    xgb_time = oof_df.loc[mask, "xgboost_pred_time"].to_numpy()

    grid = np.arange(0.0, 1.0 + grid_step / 2.0, grid_step)
    best_metrics: dict | None = None
    best_weight = 1.0
    for baseline_weight in grid:
        xgb_weight = 1.0 - baseline_weight
        pred_mag = baseline_weight * baseline_mag + xgb_weight * xgb_mag
        pred_time = baseline_weight * baseline_time + xgb_weight * xgb_time
        metrics = calculate_metrics(
            y_true_mag=y_mag,
            y_pred_mag=pred_mag,
            y_true_time=y_time,
            y_pred_time=pred_time,
            late_weight=late_weight,
        )
        objective = metrics["mag_rmse"] + metrics["time_asymmetric_rmse"]
        metrics["ensemble_objective"] = float(objective)
        if best_metrics is None or objective < best_metrics["ensemble_objective"]:
            best_metrics = metrics
            best_weight = float(baseline_weight)

    weights["baseline"] = round(best_weight, 4)
    weights["xgboost"] = round(1.0 - best_weight, 4)
    return weights, best_metrics or {"ensemble_objective": float("nan")}


def train_full_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_names: list[str],
    args: argparse.Namespace,
) -> dict[str, str]:
    """在全量历史样本上训练最终模型，并返回模型后端信息。"""
    save_dir = resolve_project_path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    backends: dict[str, str] = {}
    for model_name in model_names:
        model = build_model(model_name, args)
        model.fit(df[feature_cols], df[TARGET_COLS])
        joblib.dump(model, save_dir / MODEL_FILE_NAMES[model_name])
        backends[model_name] = getattr(model, "backend", model.__class__.__name__)
    return backends


def save_training_artifacts(
    save_dir: Path,
    feature_cols: list[str],
    fold_metrics_df: pd.DataFrame,
    oof_df: pd.DataFrame,
    mean_metrics_by_model: dict[str, dict],
    ensemble_weights: dict[str, float],
    ensemble_metrics: dict,
    model_backends: dict[str, str],
    args: argparse.Namespace,
    data_path: Path,
) -> None:
    """保存比赛推理需要的模型产物、特征列、OOF 预测和元信息。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "feature_cols.json").open("w", encoding="utf-8") as file:
        json.dump(feature_cols, file, ensure_ascii=False, indent=2)
    with (save_dir / "ensemble_weights.json").open("w", encoding="utf-8") as file:
        json.dump(ensemble_weights, file, ensure_ascii=False, indent=2)

    fold_metrics_df.to_csv(save_dir / "cv_metrics.csv", index=False, encoding="utf-8")
    oof_df.to_csv(save_dir / "oof_predictions.csv", index=False, encoding="utf-8")

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_data": str(data_path),
        "feature_count": len(feature_cols),
        "target_cols": TARGET_COLS,
        "time_col": TIME_COL,
        "model_type": args.model_type,
        "model_backends": model_backends,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "late_weight": args.late_weight,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "mean_metrics_by_model": mean_metrics_by_model,
        "ensemble_weights": ensemble_weights,
        "ensemble_metrics": ensemble_metrics,
    }
    with (save_dir / "model_meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)


def main() -> None:
    """读取高级特征，运行时间序列 CV，并保存最终树模型产物。"""
    args = parse_args()
    set_random_seed(args.seed)

    data_path = resolve_project_path(args.data)
    df = prepare_training_frame(pd.read_csv(data_path))
    feature_cols = select_feature_columns(df)
    model_names = requested_model_names(args.model_type)

    if args.save_dir is not None:
        args.save_dir = resolve_project_path(args.save_dir)

    if not feature_cols:
        raise ValueError("未筛选到任何可训练特征列，请检查 advanced_features.csv。")
    if len(df) <= args.n_splits:
        raise ValueError("样本数量必须大于 n_splits。")

    print(f"训练数据: {data_path}")
    print(f"样本数: {len(df)}")
    print(f"特征数: {len(feature_cols)}")
    print(f"模型: {', '.join(model_names)}")
    print("特征列:")
    print(", ".join(feature_cols))

    fold_metrics_df, oof_df, mean_metrics_by_model = run_oof_cv(
        df=df,
        feature_cols=feature_cols,
        model_names=model_names,
        args=args,
    )
    ensemble_weights, ensemble_metrics = search_tree_ensemble_weights(
        oof_df=oof_df,
        model_names=model_names,
        late_weight=args.late_weight,
        grid_step=args.ensemble_grid_step,
    )

    display_cols = [
        "fold",
        "model",
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
    for model_name, metrics in mean_metrics_by_model.items():
        print(f"[{model_name}]")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

    print("\nOOF 融合权重:")
    print(json.dumps(ensemble_weights, ensure_ascii=False, indent=2))
    print("OOF 融合指标:")
    for key, value in ensemble_metrics.items():
        print(f"  {key}: {value:.6f}")

    if args.save_dir is not None:
        model_backends = train_full_models(df, feature_cols, model_names, args)
        save_training_artifacts(
            save_dir=args.save_dir,
            feature_cols=feature_cols,
            fold_metrics_df=fold_metrics_df,
            oof_df=oof_df,
            mean_metrics_by_model=mean_metrics_by_model,
            ensemble_weights=ensemble_weights,
            ensemble_metrics=ensemble_metrics,
            model_backends=model_backends,
            args=args,
            data_path=data_path,
        )
        print(f"\n模型产物已保存: {args.save_dir}")


if __name__ == "__main__":
    main()
