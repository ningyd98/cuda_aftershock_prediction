"""OOF 融合模块。

基于 OOF 预测学习震级/时间/极端风险的融合权重。
支持：网格搜索、非负约束、和为1、缺模型自动剔除、DL/GNN fallback。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def _rmse(e: np.ndarray, w: np.ndarray | None = None) -> float:
    if w is None:
        return float(np.sqrt(np.mean(np.square(e))))
    w = np.asarray(w, dtype=float)
    return float(np.sqrt(np.sum(w * np.square(e)) / np.sum(w)))


def _mae(e: np.ndarray) -> float:
    return float(np.mean(np.abs(e)))


def fit_oof_fusion_weights(
    oof_df: pd.DataFrame,
    target_col: str,
    pred_cols: Sequence[str],
    objective: str = "rmse",
    grid_step: float = 0.02,
    non_negative: bool = True,
    sum_to_one: bool = True,
    weights_init: dict[str, float] | None = None,
) -> dict[str, float]:
    """在 OOF 上搜索最优融合权重。

    Args:
        oof_df: OOF 预测 DataFrame，含 target_col 和 pred_cols
        target_col: 真实目标列名
        pred_cols: 候选预测列名列表
        objective: "rmse" | "mae" | "mae+rmse" | "time_mae" | "time_mae+rmse"
        grid_step: simplex 网格步长（仅 ≤3 个模型时用精确网格，否则用随机抽样）
        non_negative: 权重非负
        sum_to_one: 权重和为 1
        weights_init: 初始权重（仅用于提供缺省值，实际会被覆盖）

    Returns:
        {col_name: weight} dict
    """
    # 过滤可用列——任一列为 NaN 时剔除该样本
    available_cols = [c for c in pred_cols if c in oof_df.columns]
    if not available_cols:
        return {}

    valid_mask = oof_df[target_col].notna()
    for c in available_cols:
        valid_mask &= oof_df[c].notna()

    if valid_mask.sum() < 5:
        # 数据太少，等权平均
        w = 1.0 / len(available_cols)
        return {c: w for c in available_cols}

    y = oof_df.loc[valid_mask, target_col].to_numpy(dtype=float)
    preds = {c: oof_df.loc[valid_mask, c].to_numpy(dtype=float) for c in available_cols}

    # 候选数量决定搜索策略
    n = len(available_cols)
    if n == 1:
        return {available_cols[0]: 1.0}

    if n == 2:
        # 一维网格
        grid = np.arange(0.0, 1.0 + grid_step / 2, grid_step)
        best_score = float("inf")
        best_w = 0.5
        c0, c1 = available_cols[0], available_cols[1]
        p0, p1 = preds[c0], preds[c1]
        for w in grid:
            fused = w * p0 + (1.0 - w) * p1
            score = _compute_score(y, fused, objective)
            if score < best_score:
                best_score = score
                best_w = float(w)
        return {c0: best_w, c1: 1.0 - best_w}

    if n == 3:
        # 二维网格
        grid = np.arange(0.0, 1.0 + grid_step / 2, grid_step)
        best_score = float("inf")
        best_weights = (1.0 / 3, 1.0 / 3, 1.0 / 3)
        c0, c1, c2 = available_cols[0], available_cols[1], available_cols[2]
        p0, p1, p2 = preds[c0], preds[c1], preds[c2]
        for w0 in grid:
            for w1 in grid:
                if w0 + w1 > 1.0:
                    continue
                w2 = 1.0 - w0 - w1
                fused = w0 * p0 + w1 * p1 + w2 * p2
                score = _compute_score(y, fused, objective)
                if score < best_score:
                    best_score = score
                    best_weights = (float(w0), float(w1), float(w2))
        return {c0: best_weights[0], c1: best_weights[1], c2: best_weights[2]}

    # n >= 4: 用随机抽样 + Dirichlet
    rng = np.random.RandomState(42)
    n_samples = max(500, n * 200)
    best_score = float("inf")
    best_weights = {c: 1.0 / n for c in available_cols}
    for _ in range(n_samples):
        raw = rng.exponential(1.0, n)
        raw /= raw.sum()
        w_dict = {c: float(raw[i]) for i, c in enumerate(available_cols)}
        fused = np.zeros_like(y)
        for c, w in w_dict.items():
            fused += w * preds[c]
        score = _compute_score(y, fused, objective)
        if score < best_score:
            best_score = score
            best_weights = w_dict

    return best_weights


def _compute_score(y_true: np.ndarray, y_pred: np.ndarray, objective: str) -> float:
    e = y_pred - y_true
    if objective == "mae":
        return _mae(e)
    if objective == "mae+rmse":
        return _mae(e) + _rmse(e)
    if objective == "time_mae":  # v2: 时间专用 — 大误差 (T3) 额外惩罚
        late_weight = np.where(e > 0, 2.0, 1.0)
        return float(np.mean(late_weight * np.abs(e)))
    if objective == "time_mae+rmse":  # v2: 时间混合目标
        late_weight = np.where(e > 0, 2.0, 1.0)
        return float(np.mean(late_weight * np.abs(e))) + _rmse(e)
    return _rmse(e)  # default rmse


def apply_fusion(
    preds_dict: dict[str, np.ndarray | float],
    weights: dict[str, float],
) -> float:
    """应用融合权重得到最终预测值。"""
    total = 0.0
    result = 0.0
    for name, weight in weights.items():
        if name in preds_dict and weight > 0:
            val = preds_dict[name]
            result += float(val) * float(weight)
            total += float(weight)
    if total > 0:
        return result / total
    # fallback: mean of available
    vals = [float(v) for v in preds_dict.values() if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """归一化权重使之和为 1。缺 key 或全 0 时等权。"""
    total = sum(max(0.0, v) for v in weights.values())
    if total <= 0:
        n = max(len(weights), 1)
        return {k: 1.0 / n for k in weights}
    return {k: max(0.0, v) / total for k, v in weights.items()}


def fuse_oof_dataframe(
    oof_df: pd.DataFrame,
    target_col: str,
    pred_cols: Sequence[str],
    objective: str = "rmse",
    grid_step: float = 0.02,
) -> tuple[np.ndarray, dict[str, float]]:
    """一键融合：返回 fused_predictions 和 weights。"""
    weights = fit_oof_fusion_weights(oof_df, target_col, pred_cols, objective, grid_step)
    fused = np.zeros(len(oof_df))
    for c, w in weights.items():
        if c in oof_df.columns:
            fused += w * oof_df[c].fillna(0.0).to_numpy(dtype=float)
    return fused, weights


def save_fusion_weights(weights: dict[str, dict[str, dict[str, float]]],
                        path: Path) -> None:
    """保存融合权重 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights, indent=2, ensure_ascii=False), encoding="utf-8")


def load_fusion_weights(path: Path) -> dict:
    """加载融合权重 JSON。"""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)
