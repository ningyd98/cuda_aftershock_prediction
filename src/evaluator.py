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
    mainshock_mag=None,
) -> dict:
    """
    计算震级、时间常规指标、非对称时间惩罚，以及物理一致性检验。

    Args:
        y_true_mag: 真实最大余震震级
        y_pred_mag: 预测最大余震震级
        y_true_time: 真实时间 (天)
        y_pred_time: 预测时间 (天)
        late_weight: 预测偏晚的惩罚倍数
        mainshock_mag: 主震震级（可选，用于 Båth 定律检验）

    Returns:
        dict with mag_rmse, mag_mae, mag_medae, time_rmse, time_mae,
        time_medae, time_asymmetric_rmse, time_asymmetric_mae,
        bath_true_delta_mean, bath_pred_delta_mean, bath_deviation
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

    # 1. Log-Time RMSE
    log_time_error = np.log10(pred_time + 1) - np.log10(true_time + 1)
    time_log_rmse = float(np.sqrt(np.mean(log_time_error**2)))

    # 2. Hit/Miss Window Metric (Tolerance: ±20% of true_time or ±0.5 days, whichever is larger to avoid division by 0 on 0 days)
    tolerance = np.maximum(0.2 * true_time, 0.5)
    hits = np.abs(time_error) <= tolerance
    time_hit_rate = float(np.mean(hits))

    # 3. Extreme Magnitude Weighted RMSE
    # Weight exponentially based on true magnitude, e.g. W_i = exp(true_mag - 4.5)
    mag_weights = np.exp(true_mag - 4.5)
    mag_weighted_rmse = float(np.sqrt(np.average(mag_error**2, weights=mag_weights)))

    result = {
        # 震级指标
        "mag_rmse": float(np.sqrt(np.mean(mag_error**2))),
        "mag_mae": float(np.mean(np.abs(mag_error))),
        "mag_medae": float(np.median(np.abs(mag_error))),
        "mag_weighted_rmse": mag_weighted_rmse,
        # 时间指标
        "time_rmse": float(np.sqrt(np.mean(time_error**2))),
        "time_mae": float(np.mean(abs_time_error)),
        "time_medae": float(np.median(abs_time_error)),
        "time_log_rmse": time_log_rmse,
        "time_hit_rate": time_hit_rate,
        # 非对称惩罚
        "time_asymmetric_mae": float(np.mean(time_weights * abs_time_error)),
        "time_asymmetric_rmse": float(
            np.sqrt(np.mean(time_weights * time_error**2))
        ),
    }

    # 能量比: 震级误差对应的能量倍数
    # 能量 ∝ 10^(1.5M)，所以能量比 = 10^(1.5 × |ΔM|)
    energy_ratios = 10 ** (1.5 * np.abs(mag_error))
    result["mag_energy_ratio_median"] = float(np.median(energy_ratios))
    result["mag_energy_ratio_mean"] = float(np.mean(energy_ratios))

    # Båth 定律检验: ΔM = M_main - M_max_aftershock ≈ 1.2
    if mainshock_mag is not None:
        ms_mag = _as_float_array(mainshock_mag)
        if len(ms_mag) == len(true_mag):
            bath_ref = 1.2
            true_delta = np.clip(ms_mag - true_mag, 0, None)
            pred_delta = np.clip(ms_mag - pred_mag, 0, None)
            result["bath_true_delta_mean"] = float(np.mean(true_delta))
            result["bath_true_delta_std"] = float(np.std(true_delta))
            result["bath_pred_delta_mean"] = float(np.mean(pred_delta))
            result["bath_pred_delta_std"] = float(np.std(pred_delta))
            result["bath_deviation"] = float(np.mean(np.abs(
                true_delta - pred_delta
            )))

    return result


def evaluate(*args, **kwargs):
    """兼容旧入口，转发到 calculate_metrics。"""
    return calculate_metrics(*args, **kwargs)


def format_metrics_table(metrics: dict, title: str = "Evaluation Metrics") -> str:
    """将指标字典格式化为可读的多行表格字符串。"""
    lines = [f"{'='*50}", f"  {title}", f"{'='*50}"]

    # 震级指标
    lines.append("  📊 Magnitude Prediction:")
    lines.append(f"    RMSE : {metrics.get('mag_rmse', float('nan')):.4f}  (root mean squared, magnitude units)")
    lines.append(f"    MAE  : {metrics.get('mag_mae', float('nan')):.4f}  (mean absolute error)")
    lines.append(f"    MedAE: {metrics.get('mag_medae', float('nan')):.4f}  (median absolute, robust to outliers)")
    if "mag_weighted_rmse" in metrics:
        lines.append(f"    W-RMSE: {metrics['mag_weighted_rmse']:.4f}  (weighted by exp(true_mag - 4.5))")
    # 能量比
    if "mag_energy_ratio_median" in metrics:
        lines.append(f"    EnergyRatio (median): {metrics['mag_energy_ratio_median']:.2f}×  (典型能量偏差倍数)")
        lines.append(f"    EnergyRatio (mean):   {metrics.get('mag_energy_ratio_mean', 0):.2f}×")

    # 时间指标
    lines.append("  ⏱️  Timing Prediction (days):")
    lines.append(f"    RMSE : {metrics.get('time_rmse', float('nan')):.4f}")
    if "time_log_rmse" in metrics:
        lines.append(f"    LogRMSE : {metrics.get('time_log_rmse', float('nan')):.4f}")
    lines.append(f"    MAE  : {metrics.get('time_mae', float('nan')):.4f}")
    lines.append(f"    MedAE: {metrics.get('time_medae', float('nan')):.4f}")
    if "time_hit_rate" in metrics:
        lines.append(f"    HitRate: {metrics.get('time_hit_rate', float('nan'))*100:.2f}% (tolerance: max(±20%, ±0.5d))")
    lines.append(f"    AsymRMSE (late×2): {metrics.get('time_asymmetric_rmse', float('nan')):.4f}")
    lines.append(f"    AsymMAE  (late×2): {metrics.get('time_asymmetric_mae', float('nan')):.4f}")

    # Båth 定律
    if "bath_true_delta_mean" in metrics:
        lines.append("  🔬 Båth's Law (ΔM = M_main − M_max_aftershock, reference ≈ 1.2):")
        lines.append(f"    True  ΔM: {metrics['bath_true_delta_mean']:.3f} ± {metrics.get('bath_true_delta_std', 0):.3f}")
        lines.append(f"    Pred  ΔM: {metrics['bath_pred_delta_mean']:.3f} ± {metrics.get('bath_pred_delta_std', 0):.3f}")
        lines.append(f"    ΔM Deviation: {metrics['bath_deviation']:.4f}  (lower = better physics)")

    # 综合分
    if "combined_score" in metrics:
        lines.append(f"  🎯 Combined Score: {metrics['combined_score']:.4f}")

    lines.append(f"{'='*50}")
    return "\n".join(lines)
