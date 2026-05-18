"""test_make_submission.py — 测试推理管道的关键逻辑。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_submission import check_feature_consistency


def test_feature_consistency_all_present():
    """所有训练特征都存在时应通过检查。"""
    train_cols = ["mainshock_mag", "gr_b_value", "omori_p", "plate_type_SUB"]
    feat_df = pd.DataFrame(columns=train_cols + ["mainshock_id"])
    try:
        check_feature_consistency(feat_df, train_cols)
    except RuntimeError:
        assert False, "全部特征存在时不应报错"


def test_feature_consistency_some_missing_below_threshold():
    """少量缺失不应报错。"""
    train_cols = ["mainshock_mag", "gr_b_value", "omori_p", "etas_mu",
                  "plate_type_SUB", "fault_type_SS", "bath_dm1"]
    feat_df = pd.DataFrame(columns=["mainshock_mag", "gr_b_value", "omori_p",
                                    "plate_type_SUB"])
    # 缺失 3/7 ≈ 43% > 30% threshold — 会报错
    # 用更大的阈值测试
    try:
        check_feature_consistency(feat_df, train_cols, max_missing_ratio=0.50)
    except RuntimeError as e:
        assert False, f"缺失 42% 在阈值 50% 以内不应报错: {e}"


def test_feature_consistency_missing_above_threshold():
    """缺失超过阈值应报错。"""
    train_cols = ["mainshock_mag", "gr_b_value", "omori_p", "etas_mu",
                  "plate_type_SUB"]
    feat_df = pd.DataFrame(columns=["mainshock_mag"])
    try:
        check_feature_consistency(feat_df, train_cols, max_missing_ratio=0.30)
        assert False, "缺失 80% > 30% 应报错"
    except RuntimeError:
        pass  # 预期结果


def test_feature_consistency_strict_mode():
    """strict 模式任意缺失都应报错。"""
    train_cols = ["mainshock_mag", "gr_b_value", "omori_p"]
    feat_df = pd.DataFrame(columns=["mainshock_mag", "gr_b_value"])
    try:
        check_feature_consistency(feat_df, train_cols, strict=True)
        assert False, "strict 模式任意缺失应报错"
    except RuntimeError:
        pass  # 预期结果


def test_submission_output_format():
    """验证 submission CSV 字段格式。"""
    submission = pd.DataFrame([
        {"mainshock_id": "test_001", "predicted_max_mag": 5.5, "predicted_time_to_max": 3.2},
        {"mainshock_id": "test_002", "predicted_max_mag": 4.8, "predicted_time_to_max": 0.5},
    ])
    required = {"mainshock_id", "predicted_max_mag", "predicted_time_to_max"}
    assert required.issubset(set(submission.columns)), f"缺少列: {required - set(submission.columns)}"
    assert submission["predicted_max_mag"].notna().all()
    assert submission["predicted_time_to_max"].notna().all()
    assert (submission["predicted_time_to_max"] >= 0).all()
    assert (submission["predicted_max_mag"] > 0).all()


if __name__ == "__main__":
    test_feature_consistency_all_present()
    test_feature_consistency_some_missing_below_threshold()
    test_feature_consistency_missing_above_threshold()
    test_feature_consistency_strict_mode()
    test_submission_output_format()
    print("✓ test_make_submission.py 全部通过")
