"""test_features.py — 测试特征工程核心函数。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (
    _load_gcmt_catalog,
    estimate_etas_parameters,
    estimate_gr_b_value,
    fit_omori_utsu,
    match_focal_mechanism,
)


def test_gcmt_matching_empty_catalog():
    """空 GCMT 目录返回稳定空特征。"""
    result = match_focal_mechanism(
        mainshock_time="2011-03-11 05:46:24",
        mainshock_lat=38.3,
        mainshock_lon=142.37,
        gcmt_df=None,
    )
    assert result["focal_mechanism_valid"] is False
    assert result["fault_type_UNK"] == 1
    assert result["fault_type_NF"] == 0
    assert result["fault_type_SS"] == 0
    assert result["fault_type_TF"] == 0
    assert not np.isfinite(result["strike1"])


def test_gcmt_matching_in_range_only():
    """空间范围外的候选不会误匹配。"""
    gcmt = pd.DataFrame([
        {
            "time": pd.to_datetime("2011-03-11 05:46:24", utc=True),
            "latitude": -50.0,  # 远在南半球
            "longitude": 140.0,  # 同样经度但距离 > 8000 km
            "strike1": 180.0, "dip1": 45.0, "rake1": 90.0,
            "fault_type": "TF",
        },
        {
            "time": pd.to_datetime("2011-03-12 00:00:00", utc=True),
            "latitude": 38.3,  # 距离很近
            "longitude": 142.5,
            "strike1": 30.0, "dip1": 60.0, "rake1": -90.0,
            "fault_type": "NF",
        },
    ])
    result = match_focal_mechanism(
        mainshock_time="2011-03-11 05:46:24",
        mainshock_lat=38.3,
        mainshock_lon=142.37,
        gcmt_df=gcmt,
        time_window_days=1.0,
        spatial_radius_km=200.0,
    )
    # 第二条候选在空间范围内（约15km），第一条虽然时间匹配但空间远
    assert result["focal_mechanism_valid"] is True
    assert np.isfinite(result["strike1"])
    # 验证选中的是 NF 而非 TF（因为在范围内）
    assert result["fault_type_NF"] == 1  # 空间范围内是 NF


def test_gr_b_value_min_events():
    """事件数不足时返回无效结果。"""
    events = pd.DataFrame({"mag": [4.0, 4.5, 4.2, 4.8]})
    result = estimate_gr_b_value(events, min_events=5)
    assert result["gr_valid"] is False
    assert result["gr_n"] == 4


def test_gr_b_value_basic():
    """足够事件时返回有效 b 值。"""
    rng = np.random.default_rng(42)
    mags = rng.exponential(scale=1.0, size=100) + 3.0
    events = pd.DataFrame({"mag": mags})
    result = estimate_gr_b_value(events, min_events=10)
    assert bool(result["gr_valid"]) is True
    assert result["gr_b_value"] > 0
    assert np.isfinite(result["gr_a_value"])


def test_omori_fit_min_events():
    """事件数不足时返回无效结果。"""
    events = pd.DataFrame({
        "time": pd.to_datetime([
            "2000-01-01 01:00", "2000-01-01 02:00", "2000-01-01 03:00",
        ], utc=True),
    })
    result = fit_omori_utsu(events, mainshock_time="2000-01-01 00:00", min_events=8)
    assert result["omori_valid"] is False


def test_etas_with_mainshock_mag():
    """传入真实主震震级的 ETAS 正常返回。"""
    # 构造足够的早期余震
    n_events = 30
    times = [pd.to_datetime("2000-01-01 00:00", utc=True) + pd.Timedelta(hours=i * 2) for i in range(n_events)]
    mags = np.random.default_rng(42).uniform(3.0, 5.5, n_events)
    events = pd.DataFrame({"time": times, "mag": mags})

    result = estimate_etas_parameters(
        events,
        mainshock_time="2000-01-01 00:00",
        mainshock_mag=6.5,
        min_events=10,
        max_events=200,
    )
    # 基本字段检查
    assert "etas_mu" in result
    assert "etas_valid" in result
    assert result["etas_n"] > 0


def test_etas_fallback_no_mainshock_mag():
    """未传入主震震级时回退到早期余震最大震级（向后兼容）。"""
    n_events = 30
    times = [pd.to_datetime("2000-01-01 00:00", utc=True) + pd.Timedelta(hours=i * 2) for i in range(n_events)]
    mags = np.random.default_rng(43).uniform(3.0, 5.5, n_events)
    events = pd.DataFrame({"time": times, "mag": mags})

    result = estimate_etas_parameters(
        events,
        mainshock_time="2000-01-01 00:00",
        mainshock_mag=None,
        min_events=10,
        max_events=200,
    )
    assert result["etas_n"] > 0  # 不应崩溃


if __name__ == "__main__":
    test_gcmt_matching_empty_catalog()
    test_gcmt_matching_in_range_only()
    test_gr_b_value_min_events()
    test_gr_b_value_basic()
    test_omori_fit_min_events()
    test_etas_with_mainshock_mag()
    test_etas_fallback_no_mainshock_mag()
    print("✓ test_features.py 全部通过")
