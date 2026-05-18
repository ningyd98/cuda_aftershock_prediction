from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

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
    "bath_",
    "fault_type_",
    "productivity_",
    "mag_ratio_",
    "mag_diff_",
    "energy_per_",
    "log_energy_",
    "count_ratio_",
    "energy_ratio_",
    "omori_p_",
    "omori_decay_",
    "etas_p_",
    "aniso_",
    "plate_dist_",
    "log_plate_",
    "b_value_",
    "log_depth",
    "depth_mag_",
    "productivity_per_",
)
EXPLICIT_FEATURES = {
    "mainshock_mag",
    "mainshock_depth",
    "advanced_early_event_count",
    "plate_boundary_distance_km",
    "strike1",
    "dip1",
    "rake1",
    "strike2",
    "dip2",
    "rake2",
    "plunge_P",
    "trend_P",
    "plunge_T",
    "trend_T",
    "f_clvd",
    "gcmt_time_diff_seconds",
    "gcmt_distance_km",
    "focal_mechanism_valid",
}
EXCLUDE_COLS = {
    "mainshock_id",
    "mainshock_time",
    "mainshock_lat",
    "mainshock_lon",
    "nearest_plate_boundary_type",
    "has_target_aftershock",
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


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加交互特征和衍生特征以提升预测性能。"""
    df = df.copy()

    # 震级差异比率特征
    if "mainshock_mag" in df.columns and "early_max_mag" in df.columns:
        df["mag_ratio_early_main"] = df["early_max_mag"] / df["mainshock_mag"].clip(lower=1.0)
        df["mag_diff_main_early"] = df["mainshock_mag"] - df["early_max_mag"]

    # 能量释放速率
    if "early_energy_sum" in df.columns and "early_aftershock_count" in df.columns:
        df["energy_per_event"] = df["early_energy_sum"] / df["early_aftershock_count"].clip(lower=1)
        df["log_energy_sum"] = np.log1p(df["early_energy_sum"])

    # 时间衰减特征 (早期 vs 晚期活动比)
    if "count_1h" in df.columns and "count_72h" in df.columns:
        df["count_ratio_1h_72h"] = df["count_1h"] / df["count_72h"].clip(lower=1)
        df["count_ratio_6h_72h"] = df.get("count_6h", 0) / df["count_72h"].clip(lower=1)
        df["count_ratio_24h_72h"] = df.get("count_24h", 0) / df["count_72h"].clip(lower=1)

    if "energy_1h" in df.columns and "energy_72h" in df.columns:
        df["energy_ratio_1h_72h"] = df["energy_1h"] / df["energy_72h"].clip(lower=1e-10)
        df["energy_ratio_24h_72h"] = df.get("energy_24h", 0) / df["energy_72h"].clip(lower=1e-10)

    # Omori-Utsu 参数交互
    if "omori_p" in df.columns and "omori_c" in df.columns:
        df["omori_p_times_c"] = df["omori_p"] * df["omori_c"]
        df["omori_decay_rate"] = df["omori_p"] / df["omori_c"].clip(lower=1e-6)

    # ETAS 参数交互
    if "etas_p" in df.columns and "etas_alpha" in df.columns:
        df["etas_p_alpha_ratio"] = df["etas_p"] / df["etas_alpha"].clip(lower=1e-6)

    # 空间各向异性与震级交互
    if "anisotropy_major_axis_km" in df.columns and "mainshock_mag" in df.columns:
        df["aniso_area_proxy"] = df["anisotropy_major_axis_km"] * df.get("anisotropy_minor_axis_km", 0)
        df["aniso_per_mag"] = df["anisotropy_major_axis_km"] / df["mainshock_mag"].clip(lower=1.0)

    # 板块边界距离与震级交互
    if "plate_boundary_distance_km" in df.columns and "mainshock_mag" in df.columns:
        df["plate_dist_per_mag"] = df["plate_boundary_distance_km"] / df["mainshock_mag"].clip(lower=1.0)
        df["log_plate_dist"] = np.log1p(df["plate_boundary_distance_km"])

    # b值与震级交互
    if "gr_b_value" in df.columns and "mainshock_mag" in df.columns:
        df["b_value_times_mag"] = df["gr_b_value"] * df["mainshock_mag"]

    # 深度特征
    if "mainshock_depth" in df.columns:
        df["log_depth"] = np.log1p(df["mainshock_depth"].clip(lower=0))
        df["depth_mag_ratio"] = df["mainshock_depth"] / df["mainshock_mag"].clip(lower=1.0)

    # 生产力指数与早期事件数交互
    if "productivity_index" in df.columns and "early_aftershock_count" in df.columns:
        df["productivity_per_event"] = df["productivity_index"] / df["early_aftershock_count"].clip(lower=1)

    return df


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """清理训练数据：目标和时间必须存在，特征缺失保留给树模型处理。"""
    cleaned_df = df.copy()
    cleaned_df[TIME_COL] = pd.to_datetime(
        cleaned_df[TIME_COL],
        utc=True,
        errors="coerce",
        format="mixed",
    )
    before = len(cleaned_df)
    cleaned_df = cleaned_df.dropna(subset=[TIME_COL, *TARGET_COLS]).reset_index(drop=True)
    after = len(cleaned_df)
    if after < before:
        print(f"训练数据过滤: {before} → {after} 条 (剔除无未来余震的 NaN 目标样本)")
    return cleaned_df


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
        "transform_time_target": True,
    }
    if model_name == "baseline":
        return BaselineLGBM(
            **common_kwargs,
            use_asymmetric_time_objective=args.use_asymmetric_time_objective,
            late_weight=args.late_weight,
        )
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
    parser.add_argument(
        "--use-asymmetric-time-objective",
        action="store_true",
        help="LightGBM 时间目标使用预测偏晚惩罚的自定义 MSE objective",
    )
    parser.add_argument(
        "--purge-days",
        type=float,
        default=30.0,
        help="每折训练集中剔除距离验证集开始时间不足此天数的样本 (默认 30)",
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

    purge_delta = pd.Timedelta(days=float(getattr(args, "purge_days", 30.0)))

    oof_preds = {
        model_name: np.full((len(train_df), len(TARGET_COLS)), np.nan, dtype=float)
        for model_name in model_names
    }
    fold_records: list[dict] = []

    fold_iter = tqdm(
        enumerate(splitter.split(X), start=1),
        total=args.n_splits,
        desc="树模型 OOF folds",
        unit="fold",
    )
    for fold_idx, (train_idx, valid_idx) in fold_iter:
        train_end_time = train_df.loc[train_idx[-1], TIME_COL]
        valid_start_time = train_df.loc[valid_idx[0], TIME_COL]
        if valid_start_time <= train_end_time:
            raise RuntimeError("时间序列切分异常：验证集时间未晚于训练集。")
        fold_iter.set_postfix(
            train=len(train_idx),
            valid=len(valid_idx),
            start=str(valid_start_time)[:10],
        )

        # ---- purge: 剔除训练集中距验证集开始时间不足 purge_days 的样本 ----
        purge_cutoff = valid_start_time - purge_delta
        purge_mask = train_df.loc[train_idx, TIME_COL] <= purge_cutoff
        train_idx_purged = train_idx[purge_mask.values]
        if len(train_idx_purged) < max(10, len(train_idx) * 0.3):
            print(
                f"  警告: fold {fold_idx} purge 后训练样本仅 {len(train_idx_purged)}，"
                f"跳过 purge"
            )
            train_idx_purged = train_idx

        model_iter = tqdm(
            model_names,
            desc=f"Fold {fold_idx} 模型训练",
            unit="model",
            leave=False,
        )
        for model_name in model_iter:
            model_iter.set_postfix(model=model_name)
            model = build_model(model_name, args)
            model.fit(X.iloc[train_idx_purged], y.iloc[train_idx_purged])
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
                    "train_size": int(len(train_idx_purged)),
                    "valid_size": int(len(valid_idx)),
                    "purge_days": float(getattr(args, "purge_days", 30.0)),
                    "train_start": str(train_df.loc[train_idx_purged[0], TIME_COL])[:10],
                    "train_end": str(train_df.loc[train_idx_purged[-1], TIME_COL])[:10],
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
    """基于 OOF 预测搜索树模型融合权重（双目标独立搜索）。"""
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

    # 独立搜索震级最优权重
    best_mag_rmse = float("inf")
    best_mag_weight = 0.5
    for baseline_weight in tqdm(grid, desc="树融合震级权重搜索", unit="weight", leave=False):
        xgb_weight = 1.0 - baseline_weight
        pred_mag = baseline_weight * baseline_mag + xgb_weight * xgb_mag
        mag_rmse = float(np.sqrt(np.mean((pred_mag - y_mag) ** 2)))
        if mag_rmse < best_mag_rmse:
            best_mag_rmse = mag_rmse
            best_mag_weight = float(baseline_weight)

    # 独立搜索时间最优权重
    best_time_obj = float("inf")
    best_time_weight = 0.5
    for baseline_weight in tqdm(grid, desc="树融合时间权重搜索", unit="weight", leave=False):
        xgb_weight = 1.0 - baseline_weight
        pred_time = baseline_weight * baseline_time + xgb_weight * xgb_time
        time_error = pred_time - y_time
        abs_time_error = np.abs(time_error)
        time_weights = np.where(time_error > 0, late_weight, 1.0)
        time_asym_rmse = float(np.sqrt(np.mean(time_weights * time_error**2)))
        if time_asym_rmse < best_time_obj:
            best_time_obj = time_asym_rmse
            best_time_weight = float(baseline_weight)

    # 使用各自最优权重计算最终融合指标
    final_pred_mag = best_mag_weight * baseline_mag + (1.0 - best_mag_weight) * xgb_mag
    final_pred_time = best_time_weight * baseline_time + (1.0 - best_time_weight) * xgb_time
    best_metrics = calculate_metrics(
        y_true_mag=y_mag,
        y_pred_mag=final_pred_mag,
        y_true_time=y_time,
        y_pred_time=final_pred_time,
        late_weight=late_weight,
    )
    best_metrics["ensemble_objective"] = float(best_metrics["mag_rmse"] + best_metrics["time_asymmetric_rmse"])

    # 保存为双目标独立权重格式（兼容 make_submission 的新格式）
    weights["baseline"] = round(best_mag_weight, 4)
    weights["xgboost"] = round(1.0 - best_mag_weight, 4)
    # 额外保存双目标权重信息
    weights["_mag_baseline_w"] = round(best_mag_weight, 4)
    weights["_time_baseline_w"] = round(best_time_weight, 4)
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
    for model_name in tqdm(model_names, desc="全量树模型训练", unit="model"):
        model = build_model(model_name, args)
        model.fit(df[feature_cols], df[TARGET_COLS])
        joblib.dump(model, save_dir / MODEL_FILE_NAMES[model_name])
        backends[model_name] = getattr(model, "backend", model.__class__.__name__)
    return backends


def build_gating_classifier(args: argparse.Namespace):
    """创建 has-aftershock 门控分类器，返回 (model, backend, fit_uses_sample_weight)。"""
    try:
        from lightgbm import LGBMClassifier

        return (
            LGBMClassifier(
                n_estimators=min(args.n_estimators, 200),
                learning_rate=args.learning_rate,
                random_state=args.seed,
                verbosity=-1,
                class_weight="balanced",
            ),
            "lightgbm",
            False,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        # HistGradientBoostingClassifier 的 class_weight 参数不是所有 sklearn
        # 版本都支持；用 sample_weight 实现 balanced 权重更稳。
        return (
            HistGradientBoostingClassifier(
                max_iter=min(args.n_estimators, 200),
                learning_rate=args.learning_rate,
                random_state=args.seed,
            ),
            "sklearn_hist_gradient_boosting",
            True,
        )


def fit_gating_classifier(model, X: pd.DataFrame, y: pd.Series, use_sample_weight: bool):
    """按需传入 balanced sample_weight 训练分类器。"""
    if use_sample_weight:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        return model.fit(X, y, sample_weight=sample_weight)
    return model.fit(X, y)


def positive_class_probability(model, X: pd.DataFrame) -> np.ndarray:
    """提取 has_aftershock=1 的概率，兼容单类模型。"""
    if not hasattr(model, "predict_proba"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))

    probs = np.asarray(model.predict_proba(X), dtype=float)
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        return probs[:, classes.index(1)]
    if True in classes:
        return probs[:, classes.index(True)]
    if len(classes) == 1 and classes[0] in (0, False):
        return np.zeros(len(X), dtype=float)
    if probs.ndim == 2 and probs.shape[1] >= 2:
        return probs[:, 1]
    raise RuntimeError("分类器无法提供 has_aftershock=1 的概率。")


def fbeta_score_from_counts(tp: int, fp: int, fn: int, beta: float = 2.0) -> float:
    """根据混淆矩阵计数计算 F-beta。"""
    beta2 = beta**2
    denom = (1.0 + beta2) * tp + beta2 * fn + fp
    return 0.0 if denom == 0 else float((1.0 + beta2) * tp / denom)


def search_gating_threshold(
    y_true: np.ndarray,
    prob: np.ndarray,
    beta: float = 2.0,
    step: float = 0.01,
) -> tuple[float, dict]:
    """在 OOF 概率上搜索门控阈值，默认用 F2 偏重召回。"""
    valid = np.isfinite(prob)
    y = y_true[valid].astype(int)
    p = prob[valid]
    if len(y) == 0:
        return 0.5, {"f2": np.nan, "precision": np.nan, "recall": np.nan, "accuracy": np.nan}

    thresholds = np.arange(step, 1.0, step)
    best_threshold = 0.5
    best_fbeta = -1.0
    best_stats: dict = {}
    for threshold in tqdm(thresholds, desc="门控阈值 OOF 搜索", unit="threshold", leave=False):
        pred = (p >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        accuracy = (tp + tn) / max(len(y), 1)
        fbeta = fbeta_score_from_counts(tp, fp, fn, beta=beta)

        # 同分时取更低阈值，减少 false negative 风险。
        if fbeta > best_fbeta + 1e-12:
            best_fbeta = fbeta
            best_threshold = float(threshold)
            best_stats = {
                "f2": float(fbeta),
                "precision": float(precision),
                "recall": float(recall),
                "accuracy": float(accuracy),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
    return best_threshold, best_stats


def train_two_stage_classifier(
    raw_df: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
) -> None:
    """
    训练二阶段零膨胀门控分类器。

    分类标签: has_target_aftershock (若不存在则 fallback 到 target_max_mag.notna())
    使用与回归器相同的 feature_cols。
    保存 classifier_meta.json 和 classifier_model.joblib。
    """
    save_dir = resolve_project_path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df_cls = add_derived_features(raw_df.copy())
    # 确定分类标签
    if "has_target_aftershock" in df_cls.columns:
        y_cls = df_cls["has_target_aftershock"].astype(int)
    elif "target_max_mag" in df_cls.columns:
        y_cls = df_cls["target_max_mag"].notna().astype(int)
    else:
        print("⚠ 无法确定分类标签，跳过分类器训练")
        return

    pos_rate = float(y_cls.mean())
    print(f"\n两阶段分类器: positive_rate={pos_rate:.4f}, 正样本数={y_cls.sum()}/{len(y_cls)}")

    df_cls[TIME_COL] = pd.to_datetime(df_cls[TIME_COL], utc=True, errors="coerce", format="mixed")
    valid_time_mask = df_cls[TIME_COL].notna()
    df_cls = df_cls.loc[valid_time_mask].sort_values(TIME_COL).reset_index(drop=True)
    y_cls = y_cls.loc[valid_time_mask].reset_index(drop=True)

    for col in feature_cols:
        if col not in df_cls.columns:
            df_cls[col] = np.nan
    X_cls = df_cls[feature_cols].fillna(0.0)
    for col in X_cls.columns:
        if pd.api.types.is_bool_dtype(X_cls[col]):
            X_cls[col] = X_cls[col].astype(int)

    clf_probe, clf_backend, _ = build_gating_classifier(args)
    del clf_probe

    splitter = TimeSeriesSplit(n_splits=args.n_splits)
    purge_delta = pd.Timedelta(days=float(getattr(args, "purge_days", 30.0)))
    oof_prob = np.full(len(df_cls), np.nan, dtype=float)
    fold_iter = tqdm(
        enumerate(splitter.split(X_cls), start=1),
        total=args.n_splits,
        desc="门控分类器 OOF folds",
        unit="fold",
    )
    for fold_idx, (train_idx, valid_idx) in fold_iter:
        valid_start_time = df_cls.loc[valid_idx[0], TIME_COL]
        fold_iter.set_postfix(
            train=len(train_idx),
            valid=len(valid_idx),
            start=str(valid_start_time)[:10],
        )
        purge_cutoff = valid_start_time - purge_delta
        purge_mask = df_cls.loc[train_idx, TIME_COL] <= purge_cutoff
        train_idx_purged = train_idx[purge_mask.values]
        if len(train_idx_purged) < max(20, len(train_idx) * 0.3):
            train_idx_purged = train_idx

        y_train = y_cls.iloc[train_idx_purged]
        if y_train.nunique() < 2:
            oof_prob[valid_idx] = float(y_train.mean())
            continue

        clf_fold, _, use_sample_weight = build_gating_classifier(args)
        fit_gating_classifier(
            clf_fold,
            X_cls.iloc[train_idx_purged],
            y_train,
            use_sample_weight=use_sample_weight,
        )
        oof_prob[valid_idx] = positive_class_probability(clf_fold, X_cls.iloc[valid_idx])

    valid_oof = np.isfinite(oof_prob)
    if valid_oof.any() and y_cls.loc[valid_oof].nunique() == 2:
        cv_auc = float(roc_auc_score(y_cls.loc[valid_oof], oof_prob[valid_oof]))
    else:
        cv_auc = None
    threshold, threshold_stats = search_gating_threshold(
        y_cls.to_numpy(dtype=int),
        oof_prob,
        beta=2.0,
        step=0.01,
    )
    print(f"  OOF AUC: {cv_auc:.4f}" if cv_auc is not None else "  OOF AUC: N/A")
    print(f"  OOF F2 最优阈值: {threshold:.2f}, F2={threshold_stats.get('f2', np.nan):.4f}")

    # 全量训练
    clf, clf_backend, use_sample_weight = build_gating_classifier(args)
    fit_gating_classifier(clf, X_cls, y_cls, use_sample_weight=use_sample_weight)
    joblib.dump(clf, save_dir / "aftershock_classifier.joblib")

    classifier_oof = df_cls[["mainshock_id", TIME_COL]].copy()
    classifier_oof["has_target_aftershock"] = y_cls.astype(int)
    classifier_oof["oof_prob_has_aftershock"] = oof_prob
    classifier_oof["oof_pred_has_aftershock"] = (oof_prob >= threshold).astype(float)
    classifier_oof.loc[~np.isfinite(oof_prob), "oof_pred_has_aftershock"] = np.nan
    classifier_oof.to_csv(save_dir / "classifier_oof_predictions.csv", index=False, encoding="utf-8")

    meta = {
        "positive_rate": round(pos_rate, 4),
        "feature_cols": feature_cols,
        "threshold": round(float(threshold), 4),
        "threshold_objective": "f2",
        "oof_auc": round(cv_auc, 4) if cv_auc is not None else None,
        "oof_f2": round(float(threshold_stats.get("f2", np.nan)), 4),
        "oof_precision": round(float(threshold_stats.get("precision", np.nan)), 4),
        "oof_recall": round(float(threshold_stats.get("recall", np.nan)), 4),
        "oof_accuracy": round(float(threshold_stats.get("accuracy", np.nan)), 4),
        "backend": clf_backend,
    }
    with (save_dir / "classifier_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 分类器已保存: {save_dir / 'aftershock_classifier.joblib'}")


def prepare_regression_subset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    仅保留 has_target_aftershock=True 且目标列非 NaN 的样本用于回归训练。
    若 has_target_aftershock 不存在，则 fallback 到 target_max_mag.notna()。
    """
    df = raw_df.copy()
    if "has_target_aftershock" in df.columns:
        mask = df["has_target_aftershock"].astype(bool)
    else:
        mask = df["target_max_mag"].notna()
    return df.loc[mask].dropna(subset=TARGET_COLS).reset_index(drop=True)


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

    # 转换为双目标独立权重格式 (make_submission.py 支持)
    mag_baseline_w = ensemble_weights.get("_mag_baseline_w", ensemble_weights.get("baseline", 0.5))
    time_baseline_w = ensemble_weights.get("_time_baseline_w", ensemble_weights.get("baseline", 0.5))
    dual_weights = {
        "mag": {
            "baseline": round(float(mag_baseline_w), 4),
            "xgboost": round(1.0 - float(mag_baseline_w), 4),
            "dl": float(ensemble_weights.get("dl", 0.0)),
            "gnn": float(ensemble_weights.get("gnn", 0.0)),
        },
        "time": {
            "baseline": round(float(time_baseline_w), 4),
            "xgboost": round(1.0 - float(time_baseline_w), 4),
            "dl": float(ensemble_weights.get("dl", 0.0)),
            "gnn": float(ensemble_weights.get("gnn", 0.0)),
        },
    }
    with (save_dir / "ensemble_weights.json").open("w", encoding="utf-8") as file:
        json.dump(dual_weights, file, ensure_ascii=False, indent=2)

    fold_metrics_df.to_csv(save_dir / "cv_metrics.csv", index=False, encoding="utf-8")
    oof_df.to_csv(save_dir / "oof_predictions.csv", index=False, encoding="utf-8")

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_data": str(data_path),
        "feature_count": len(feature_cols),
        "target_cols": TARGET_COLS,
        "time_target_transform": "log1p",
        "time_col": TIME_COL,
        "model_type": args.model_type,
        "model_backends": model_backends,
        "n_splits": args.n_splits,
        "purge_days": float(getattr(args, "purge_days", 30.0)),
        "seed": args.seed,
        "late_weight": args.late_weight,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "use_asymmetric_time_objective": args.use_asymmetric_time_objective,
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
    raw_df = pd.read_csv(data_path)
    # 回归训练用正样本子集
    df = prepare_regression_subset(raw_df)
    df = prepare_training_frame(df)
    df = add_derived_features(df)
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
        # 训练两阶段分类器 (使用全量原始数据)
        train_two_stage_classifier(raw_df, feature_cols, args)
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
