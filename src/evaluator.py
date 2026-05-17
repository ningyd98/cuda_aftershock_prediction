from __future__ import annotations

import numpy as np


def _as_float_array(values) -> np.ndarray:
    """将输入统一转为一维浮点数组。"""
    return np.asarray(values, dtype=float).reshape(-1)


def calculate_metrics(
    y_true_mag,
    y_pred_mag,
    y_true_time,
    y_pred_time,
    late_weight: float = 2.0,
) -> dict:
    """
    计算震级、时间常规指标与非对称时间惩罚。

    当预测时间晚于实际发生时间时，说明预警滞后，误差权重乘 late_weight。
    """
    true_mag = _as_float_array(y_true_mag)
    pred_mag = np.clip(_as_float_array(y_pred_mag), a_min=0.0, a_max=None)
    true_time = _as_float_array(y_true_time)
    pred_time = np.clip(_as_float_array(y_pred_time), a_min=0.0, a_max=None)

    if not (
        len(true_mag) == len(pred_mag) == len(true_time) == len(pred_time)
    ):
        raise ValueError("真实值和预测值长度必须一致。")

    mag_error = pred_mag - true_mag
    time_error = pred_time - true_time
    abs_time_error = np.abs(time_error)
    time_weights = np.where(time_error > 0, late_weight, 1.0)

    return {
        "mag_rmse": float(np.sqrt(np.mean(mag_error**2))),
        "mag_mae": float(np.mean(np.abs(mag_error))),
        "time_rmse": float(np.sqrt(np.mean(time_error**2))),
        "time_mae": float(np.mean(abs_time_error)),
        "time_asymmetric_mae": float(np.mean(time_weights * abs_time_error)),
        "time_asymmetric_rmse": float(
            np.sqrt(np.mean(time_weights * time_error**2))
        ),
    }


def evaluate(*args, **kwargs):
    """兼容旧入口，转发到 calculate_metrics。"""
    return calculate_metrics(*args, **kwargs)
