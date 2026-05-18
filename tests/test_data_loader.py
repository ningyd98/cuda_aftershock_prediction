"""test_data_loader.py — 测试主震-余震序列构建的基本逻辑。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import build_earthquake_sequences


def _make_fake_catalog(tmp_path: Path) -> Path:
    """构造最小化地震目录：M6+ 主震 + 早期余震 + 目标窗口余震 + 无余震主震。"""
    rows = [
        # 主震 1 (有未来余震) — 时间最早
        {"time": "2000-01-01T00:00:00Z", "latitude": 35.0, "longitude": 140.0, "depth": 30, "mag": 6.5},
        {"time": "2000-01-01T06:00:00Z", "latitude": 35.01, "longitude": 140.01, "depth": 10, "mag": 5.0},
        {"time": "2000-01-02T00:00:00Z", "latitude": 35.02, "longitude": 140.02, "depth": 15, "mag": 4.5},
        {"time": "2000-01-05T00:00:00Z", "latitude": 35.01, "longitude": 140.00, "depth": 12, "mag": 5.5},
        # 主震 2 (仅有早期余震，无未来余震) — 空间位置远离其他事件
        {"time": "2010-01-01T00:00:00Z", "latitude": -30.0, "longitude": -70.0, "depth": 20, "mag": 6.8},
        {"time": "2010-01-01T01:00:00Z", "latitude": -29.99, "longitude": -70.01, "depth": 10, "mag": 4.5},
    ]
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "fake_catalog.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_has_target_aftershock_field(tmp_path):
    """验证新字段 has_target_aftershock 的存在和正确性。"""
    csv_path = _make_fake_catalog(tmp_path)
    df = build_earthquake_sequences(csv_path)

    assert "has_target_aftershock" in df.columns, "缺少 has_target_aftershock 字段"

    # 至少有 2 个序列
    assert len(df) >= 2, f"期望至少 2 个序列，实际 {len(df)}"

    # 主震 1 (Mw 6.5): 有未来余震 (Mw 5.5 at 01/05)
    ms1_candidates = df[df["mainshock_mag"] == 6.5]
    if len(ms1_candidates) > 0:
        ms1 = ms1_candidates.iloc[0]
        assert bool(ms1["has_target_aftershock"]) is True, f"期望 True, 实际 {ms1['has_target_aftershock']}"
        assert np.isfinite(ms1["target_max_mag"])
        assert ms1["target_max_mag"] > 0

    # 主震 2 (Mw 6.8): 仅有早期余震，无未来余震
    ms2_candidates = df[df["mainshock_mag"] == 6.8]
    if len(ms2_candidates) > 0:
        ms2 = ms2_candidates.iloc[0]
        assert bool(ms2["has_target_aftershock"]) is False, f"期望 False, 实际 {ms2['has_target_aftershock']}"
        assert pd.isna(ms2["target_max_mag"]), f"应为 NaN, 实际 {ms2['target_max_mag']}"
        assert pd.isna(ms2["target_time_to_max_days"])


def test_target_nan_for_no_future_aftershocks(tmp_path):
    """无未来余震样本的目标列必须是 NaN 而非 0.0。"""
    csv_path = _make_fake_catalog(tmp_path)
    df = build_earthquake_sequences(csv_path)

    no_future = df[~df["has_target_aftershock"]]
    for _, row in no_future.iterrows():
        assert pd.isna(row["target_max_mag"]), f"target_max_mag 应为 NaN: {row['target_max_mag']}"
        assert pd.isna(row["target_time_to_max_days"]), f"target_time_to_max_days 应为 NaN"

    with_future = df[df["has_target_aftershock"]]
    for _, row in with_future.iterrows():
        assert np.isfinite(row["target_max_mag"])
        assert np.isfinite(row["target_time_to_max_days"])


def test_sample_count(tmp_path):
    """验证序列构建的基本统计。"""
    csv_path = _make_fake_catalog(tmp_path)
    df = build_earthquake_sequences(csv_path)
    assert len(df) >= 2, f"样本数应 >=2, 实际 {len(df)}"
    assert "mainshock_id" in df.columns
    assert "early_aftershock_count" in df.columns


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        test_has_target_aftershock_field(tp)
        test_target_nan_for_no_future_aftershocks(tp)
        test_sample_count(tp)
    print("✓ test_data_loader.py 全部通过")
