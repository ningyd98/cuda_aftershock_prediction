from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, StandardScaler
from tqdm import tqdm

from src.evaluator import calculate_metrics
from src.models import BaselineLGBM

TIME_TARGET_INDEX = 1


def _default_model_factory(device: str = "cpu"):
    """默认模型工厂：每个 fold 都创建一个全新模型。"""
    return BaselineLGBM(random_state=42, device=device)


def _format_time(value) -> str:
    """将时间戳格式化为便于日志阅读的日期字符串。"""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def transform_targets_for_training(y: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    """震级保持原尺度，仅对时间目标做 log1p 长尾压缩。"""
    if isinstance(y, pd.DataFrame):
        transformed = y.copy()
        time_col = transformed.columns[TIME_TARGET_INDEX]
        transformed[time_col] = np.log1p(
            pd.to_numeric(transformed[time_col], errors="coerce").clip(lower=0.0)
        )
        return transformed

    transformed = np.asarray(y, dtype=float).copy()
    transformed[:, TIME_TARGET_INDEX] = np.log1p(
        np.clip(transformed[:, TIME_TARGET_INDEX], a_min=0.0, a_max=None)
    )
    return transformed


def inverse_transform_predictions(preds: np.ndarray) -> np.ndarray:
    """将模型输出从训练尺度还原到真实物理尺度。"""
    restored = np.asarray(preds, dtype=float).copy()
    restored[:, 0] = np.clip(restored[:, 0], a_min=0.0, a_max=None)
    restored[:, TIME_TARGET_INDEX] = np.expm1(
        np.clip(restored[:, TIME_TARGET_INDEX], a_min=-50.0, a_max=50.0)
    )
    restored[:, TIME_TARGET_INDEX] = np.clip(
        restored[:, TIME_TARGET_INDEX],
        a_min=0.0,
        a_max=None,
    )
    return restored


def build_fold_scaler(scaler_type: str | None):
    """为需要归一化的模型创建 fold 内 scaler；树模型默认不使用。"""
    if scaler_type is None:
        return None

    normalized = scaler_type.lower()
    if normalized == "standard":
        return StandardScaler()
    if normalized == "robust":
        return RobustScaler()
    raise ValueError("scaler_type 必须为 None、'standard' 或 'robust'。")


def _disable_model_internal_time_transform(model: object) -> None:
    """
    trainer.py 已在外部对时间目标做 log1p，因此关闭内置二次转换。

    对 LightGBM 自定义目标，仍保持 objective 在 log 空间解释 y_true/y_pred。
    """
    if hasattr(model, "transform_time_target"):
        setattr(model, "transform_time_target", False)

    time_model = getattr(model, "time_model", None)
    objective = getattr(time_model, "objective", None)
    if hasattr(objective, "log_space"):
        setattr(objective, "log_space", True)


def time_series_cv_train(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    n_splits: int = 5,
    model_factory: Callable[[], object] | None = None,
    time_col: str = "mainshock_time",
    late_weight: float = 2.0,
    scaler_type: str | None = None,
    device: str = "cpu",
) -> tuple[pd.DataFrame, dict]:
    """
    按主震时间排序后进行滚动时间序列交叉验证。

    target_cols 必须按 [震级目标, 时间目标] 排列。
    """
    if len(target_cols) != 2:
        raise ValueError("target_cols 必须包含两个目标列：[震级, 时间]。")

    required_cols = [time_col, *feature_cols, *target_cols]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"训练数据缺少必要字段: {missing_cols}")

    train_df = df.copy()
    train_df[time_col] = pd.to_datetime(
        train_df[time_col],
        utc=True,
        errors="coerce",
        format="mixed",
    )
    train_df = train_df.dropna(subset=[time_col, *target_cols])
    train_df = train_df.sort_values(time_col).reset_index(drop=True)

    if len(train_df) <= n_splits:
        raise ValueError("样本数量必须大于 n_splits。")

    factory = model_factory or (lambda: _default_model_factory(device=device))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_records: list[dict] = []

    X = train_df[list(feature_cols)]
    y = train_df[list(target_cols)]

    fold_iter = tqdm(
        enumerate(splitter.split(X), start=1),
        total=n_splits,
        desc="通用 TimeSeries CV folds",
        unit="fold",
    )
    for fold_idx, (train_idx, valid_idx) in fold_iter:
        X_train_raw = X.iloc[train_idx]
        X_valid_raw = X.iloc[valid_idx]
        y_train_raw = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]
        fold_iter.set_postfix(train=len(train_idx), valid=len(valid_idx))

        scaler = build_fold_scaler(scaler_type)
        if scaler is None:
            X_train = X_train_raw
            X_valid = X_valid_raw
        else:
            X_train = pd.DataFrame(
                scaler.fit_transform(X_train_raw),
                columns=list(feature_cols),
                index=X_train_raw.index,
            )
            X_valid = pd.DataFrame(
                scaler.transform(X_valid_raw),
                columns=list(feature_cols),
                index=X_valid_raw.index,
            )

        y_train = transform_targets_for_training(y_train_raw)

        model = factory()
        _disable_model_internal_time_transform(model)
        model.fit(X_train, y_train)
        preds_model_scale = np.asarray(model.predict(X_valid), dtype=float)
        preds = inverse_transform_predictions(preds_model_scale)

        metrics = calculate_metrics(
            y_true_mag=y_valid.iloc[:, 0].to_numpy(),
            y_pred_mag=preds[:, 0],
            y_true_time=y_valid.iloc[:, 1].to_numpy(),
            y_pred_time=preds[:, 1],
            late_weight=late_weight,
        )

        train_end_time = train_df.loc[train_idx[-1], time_col]
        valid_start_time = train_df.loc[valid_idx[0], time_col]
        if valid_start_time <= train_end_time:
            raise RuntimeError("时间序列切分异常：验证集时间未晚于训练集。")

        fold_records.append(
            {
                "fold": fold_idx,
                "backend": getattr(model, "backend", model.__class__.__name__),
                "scaler_type": scaler_type or "none",
                "train_size": int(len(train_idx)),
                "valid_size": int(len(valid_idx)),
                "train_start": _format_time(train_df.loc[train_idx[0], time_col]),
                "train_end": _format_time(train_end_time),
                "valid_start": _format_time(valid_start_time),
                "valid_end": _format_time(train_df.loc[valid_idx[-1], time_col]),
                **metrics,
            }
        )

    fold_metrics_df = pd.DataFrame(fold_records)
    metric_cols = [
        col
        for col in fold_metrics_df.columns
        if col
        not in {
            "fold",
            "backend",
            "scaler_type",
            "train_size",
            "valid_size",
            "train_start",
            "train_end",
            "valid_start",
            "valid_end",
        }
    ]
    mean_metrics = {
        col: float(fold_metrics_df[col].mean())
        for col in metric_cols
        if pd.api.types.is_numeric_dtype(fold_metrics_df[col])
    }

    return fold_metrics_df, mean_metrics


def train(*args, **kwargs):
    """兼容旧入口，转发到 time_series_cv_train。"""
    return time_series_cv_train(*args, **kwargs)
