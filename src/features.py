from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _round_to_bin(values: np.ndarray, bin_width: float) -> np.ndarray:
    """将震级归入指定精度的 bin，减少浮点误差对统计结果的影响。"""
    decimals = max(0, int(np.ceil(-np.log10(bin_width))))
    return np.round(np.round(values / bin_width) * bin_width, decimals)


def _empty_gr_result(n_events: int, mc: float = np.nan) -> dict:
    """返回统一格式的 Gutenberg-Richter 空结果。"""
    return {
        "gr_b_value": np.nan,
        "gr_a_value": np.nan,
        "gr_mc": mc,
        "gr_n": int(n_events),
        "gr_valid": False,
    }


def _empty_omori_result(n_events: int, nll: float = np.nan) -> dict:
    """返回统一格式的大森-宇津空结果。"""
    return {
        "omori_p": np.nan,
        "omori_c": np.nan,
        "omori_k": np.nan,
        "omori_nll": nll,
        "omori_n": int(n_events),
        "omori_p_boundary_hit": False,
        "omori_valid": False,
    }


def estimate_mc_maxc(magnitudes: np.ndarray, bin_width: float = 0.1) -> float:
    """
    使用最大曲率法 MAXC 估计完整性震级 Mc。

    做法：按 bin_width 对震级做直方图，频次最高的震级 bin 即为 Mc；
    若多个 bin 频次并列最高，则取最小的震级 bin。
    """
    mags = magnitudes[np.isfinite(magnitudes)]
    if len(mags) == 0:
        return np.nan

    binned_mags = _round_to_bin(mags, bin_width)
    unique_bins, counts = np.unique(binned_mags, return_counts=True)
    mode_bins = unique_bins[counts == counts.max()]
    return float(mode_bins.min())


def estimate_gr_b_value(
    events: pd.DataFrame,
    mag_col: str = "mag",
    mc: float | None = None,
    bin_width: float = 0.1,
    min_events: int = 5,
) -> dict:
    """
    使用 Aki-Utsu 极大似然估计 Gutenberg-Richter 定律 b 值。

    若未显式传入 mc，则使用 MAXC 自动估计完整性震级。
    返回 a 值、b 值、Mc、参与拟合事件数和有效性标记。
    """
    if mag_col not in events.columns:
        return _empty_gr_result(0)

    mags = events[mag_col].dropna().astype(float).to_numpy()
    if len(mags) < min_events:
        return _empty_gr_result(len(mags))

    mc_hat = estimate_mc_maxc(mags, bin_width) if mc is None else float(mc)
    if not np.isfinite(mc_hat):
        return _empty_gr_result(len(mags), mc=mc_hat)

    complete_mags = mags[mags >= mc_hat]
    if len(complete_mags) < min_events:
        return _empty_gr_result(len(complete_mags), mc=mc_hat)

    denominator = complete_mags.mean() - (mc_hat - bin_width / 2.0)
    if denominator <= 0 or not np.isfinite(denominator):
        return _empty_gr_result(len(complete_mags), mc=mc_hat)

    b_value = np.log10(np.e) / denominator
    a_value = np.log10(len(complete_mags)) + b_value * mc_hat

    return {
        "gr_b_value": float(b_value),
        "gr_a_value": float(a_value),
        "gr_mc": float(mc_hat),
        "gr_n": int(len(complete_mags)),
        "gr_valid": bool(np.isfinite(b_value) and np.isfinite(a_value)),
    }


def fit_omori_utsu(
    events: pd.DataFrame,
    mainshock_time,
    time_col: str = "time",
    obs_days: float = 3.0,
    min_events: int = 8,
    p_bounds: Sequence[float] = (0.2, 2.5),
    c_bounds: Sequence[float] = (1e-4, 10.0),
    initial_p: float = 1.0,
    initial_c: float = 0.05,
    boundary_tol: float = 1e-4,
    fit_start_day: float = 0.0,
) -> dict:
    """
    使用非齐次泊松过程 MLE 拟合大森-宇津定律。

    强度函数为 lambda(t) = K / (t + c)^p。若 p_hat 非常贴近优化边界，
    通常说明优化器撞到边界而非找到稳定极值，因此标记 omori_valid=False。
    """
    if time_col not in events.columns:
        return _empty_omori_result(0)

    if events.empty or len(events) < min_events:
        return _empty_omori_result(len(events))

    p_min, p_max = map(float, p_bounds)
    c_min, c_max = map(float, c_bounds)

    if p_min >= p_max or c_min <= 0 or c_min >= c_max:
        raise ValueError("大森定律拟合边界非法，请检查 p_bounds 与 c_bounds。")

    times = pd.to_datetime(events[time_col], utc=True, errors="coerce")
    mainshock_time = pd.to_datetime(mainshock_time, utc=True)
    elapsed_days = (times - mainshock_time).dt.total_seconds().to_numpy() / 86400.0
    elapsed_days = elapsed_days[np.isfinite(elapsed_days)]
    elapsed_days = elapsed_days[
        (elapsed_days > fit_start_day) & (elapsed_days <= obs_days)
    ]

    if len(elapsed_days) < min_events:
        return _empty_omori_result(len(elapsed_days))

    n_events = len(elapsed_days)

    def integral_term(p_value: float, c_value: float) -> float:
        """计算拟合窗口内强度函数积分项。"""
        start = fit_start_day + c_value
        end = obs_days + c_value
        if start <= 0 or end <= 0:
            return np.nan
        if abs(p_value - 1.0) < boundary_tol:
            return float(np.log(end / start))
        return float(
            (end ** (1.0 - p_value) - start ** (1.0 - p_value))
            / (1.0 - p_value)
        )

    def negative_log_likelihood(params: np.ndarray) -> float:
        """大森-宇津模型的负对数似然。"""
        p_value, log_c_value = params
        c_value = float(np.exp(log_c_value))

        integ = integral_term(float(p_value), c_value)
        if integ <= 0 or not np.isfinite(integ):
            return np.inf

        # 固定 p 与 c 后，K 的极大似然估计有解析解。
        k_value = n_events / integ
        if k_value <= 0 or not np.isfinite(k_value):
            return np.inf

        log_lambda = np.log(k_value) - p_value * np.log(elapsed_days + c_value)
        nll = -(np.sum(log_lambda) - k_value * integ)
        return float(nll) if np.isfinite(nll) else np.inf

    result = minimize(
        negative_log_likelihood,
        x0=np.array([initial_p, np.log(initial_c)], dtype=float),
        bounds=[(p_min, p_max), (np.log(c_min), np.log(c_max))],
        method="L-BFGS-B",
    )

    if not result.success:
        nll = float(result.fun) if np.isfinite(result.fun) else np.nan
        return _empty_omori_result(n_events, nll=nll)

    p_hat, log_c_hat = result.x
    c_hat = float(np.exp(log_c_hat))
    integ_hat = integral_term(float(p_hat), c_hat)
    k_hat = float(n_events / integ_hat) if integ_hat > 0 else np.nan

    p_boundary_hit = (
        abs(float(p_hat) - p_min) < boundary_tol
        or abs(float(p_hat) - p_max) < boundary_tol
    )

    return {
        "omori_p": float(p_hat),
        "omori_c": c_hat,
        "omori_k": k_hat,
        "omori_nll": float(result.fun),
        "omori_n": int(n_events),
        "omori_p_boundary_hit": bool(p_boundary_hit),
        "omori_valid": bool(
            result.success
            and not p_boundary_hit
            and np.isfinite(k_hat)
            and np.isfinite(result.fun)
        ),
    }
