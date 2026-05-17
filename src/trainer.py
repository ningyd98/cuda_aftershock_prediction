from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.evaluator import calculate_metrics
from src.models import BaselineLGBM


def _default_model_factory():
    """默认模型工厂：每个 fold 都创建一个全新模型。"""
    return BaselineLGBM(random_state=42)


def _format_time(value) -> str:
    """将时间戳格式化为便于日志阅读的日期字符串。"""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def time_series_cv_train(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    n_splits: int = 5,
    model_factory: Callable[[], object] | None = None,
    time_col: str = "mainshock_time",
    late_weight: float = 2.0,
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

    factory = model_factory or _default_model_factory
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_records: list[dict] = []

    X = train_df[list(feature_cols)]
    y = train_df[list(target_cols)]

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        model = factory()
        model.fit(X_train, y_train)
        preds = np.asarray(model.predict(X_valid), dtype=float)
        preds = np.clip(preds, a_min=0.0, a_max=None)

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
