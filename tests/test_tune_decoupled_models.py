"""调参脚本测试 v2。"""

from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np, pandas as pd, pytest


class TestPerTargetScoring:
    def test_mag_score_positive(self):
        from scripts.tune_decoupled_models import _mag_score
        y = np.array([5.0, 6.0]); oof = np.array([5.1, 6.2]); ef = np.array([True, False])
        Args = type('Args', (), {'w_mag_mae': 1.0, 'w_mag_rmse': 1.0, 'w_extreme': 0.5})
        assert _mag_score(y, oof, ef, Args()) > 0

    def test_time_score_positive(self):
        from scripts.tune_decoupled_models import _time_score
        y = np.array([10.0, 50.0]); oof = np.array([12.0, 55.0])
        Args = type('Args', (), {'w_time_mae': 0.03, 'w_time_rmse': 0.03, 'w_late': 0.3, 'w_t1_bonus': 0.2})
        assert _time_score(y, oof, Args()) > 0


class TestSearchSpaces:
    def test_mag_nonempty(self):
        from scripts.tune_decoupled_models import _mag_space
        s = _mag_space(True); assert len(s) > 0
        for v in s.values(): assert len(v) > 0

    def test_fast_smaller(self):
        from scripts.tune_decoupled_models import _mag_space
        assert sum(len(v) for v in _mag_space(True).values()) < sum(len(v) for v in _mag_space(False).values())

    def test_bucket_keys(self):
        from scripts.tune_decoupled_models import _bucket_space
        for k in ['n_estimators', 'learning_rate', 'num_leaves']:
            assert k in _bucket_space(True)

    def test_postproc_biases(self):
        from scripts.tune_decoupled_models import _postproc_space
        s = _postproc_space(True)
        for k in ['mag_bias_T1', 'time_bias_T1']: assert k in s


class TestSampling:
    def test_prefix(self):
        from scripts.tune_decoupled_models import _sample
        rng = np.random.RandomState(42)
        clean, _ = _sample({'a': [1, 2]}, rng, 'test')
        assert 'a' in clean


class TestParseArgs:
    def test_defaults(self):
        old = sys.argv[:]; sys.argv = ["tune.py"]
        try:
            from scripts.tune_decoupled_models import parse_args
            a = parse_args()
            assert a.mag_trials == 80; assert a.n_splits == 5; assert a.seed == 42
        finally: sys.argv = old

    def test_weight_args(self):
        old = sys.argv[:]; sys.argv = ['tune.py', '--w-mag-mae', '2.0', '--w-t1-bonus', '0.5']
        try:
            from scripts.tune_decoupled_models import parse_args
            a = parse_args()
            assert a.w_mag_mae == 2.0; assert a.w_t1_bonus == 0.5; assert a.w_time_mae == 0.03
        finally: sys.argv = old


class TestNewCLIFlags:
    def test_target_flag(self):
        old = sys.argv[:]; sys.argv = ['tune.py', '--target', 'mag', '--separate-target-tuning', 'true']
        try:
            from scripts.tune_decoupled_models import parse_args
            a = parse_args()
            assert a.target == 'mag'; assert a.separate_target_tuning == True
        finally: sys.argv = old

    def test_fusion_flag(self):
        old = sys.argv[:]; sys.argv = ['tune.py', '--enable-oof-fusion', 'true', '--fusion-grid-step', '0.05']
        try:
            from scripts.tune_decoupled_models import parse_args
            a = parse_args()
            assert a.enable_oof_fusion == True; assert a.fusion_grid_step == 0.05
        finally: sys.argv = old

    def test_per_target_trials(self):
        old = sys.argv[:]; sys.argv = ['tune.py', '--mag-trials', '20', '--time-trials', '30', '--extreme-trials', '10']
        try:
            from scripts.tune_decoupled_models import parse_args
            a = parse_args()
            assert a.mag_trials == 20; assert a.time_trials == 30; assert a.extreme_trials == 10
        finally: sys.argv = old


class TestOofFusion:
    def test_fusion_two_models(self):
        from scripts.oof_fusion import fit_oof_fusion_weights
        rng = np.random.RandomState(42)
        true = rng.randn(100)
        df = pd.DataFrame({'target': true, 'm1': true + 0.1*rng.randn(100), 'm2': true + 0.5*rng.randn(100)})
        w = fit_oof_fusion_weights(df, 'target', ['m1', 'm2'], 'rmse')
        assert 'm1' in w and 'm2' in w
        assert abs(sum(w.values()) - 1.0) < 0.001
        assert w['m1'] > 0.5

    def test_fusion_missing_model(self):
        from scripts.oof_fusion import fit_oof_fusion_weights
        df = pd.DataFrame({'target': [1, 2, 3], 'm1': [np.nan, np.nan, np.nan], 'm2': [1.1, 2.2, 3.3]})
        w = fit_oof_fusion_weights(df, 'target', ['m1', 'm2'], 'rmse')
        assert 'm2' in w

    def test_normalize_weights(self):
        from scripts.oof_fusion import normalize_weights
        w = normalize_weights({'a': 2, 'b': 3, 'c': 0})
        assert abs(sum(w.values()) - 1.0) < 0.001; assert w['c'] == 0


# --------------- end-to-end smoke ---------------

def _make_mock_csv(tmp_path):
    rng = np.random.RandomState(42); n = 30
    df = pd.DataFrame({
        "mainshock_id": [f"m{i:04d}" for i in range(n)],
        "mainshock_time": pd.date_range("2000-01-01", periods=n, freq="180D"),
        "mainshock_lat": rng.uniform(-30, 30, n), "mainshock_lon": rng.uniform(-180, 180, n),
        "mainshock_mag": rng.uniform(6.0, 8.0, n), "mainshock_depth": rng.uniform(5, 50, n),
        "early_aftershock_count": rng.randint(0, 20, n), "early_energy_sum": rng.uniform(1e10, 1e15, n),
        "count_1h": rng.randint(0, 5, n), "count_24h": rng.randint(0, 15, n), "count_72h": rng.randint(0, 30, n),
        "energy_1h": rng.uniform(0, 1e14, n), "energy_24h": rng.uniform(0, 1e15, n), "energy_72h": rng.uniform(0, 1e16, n),
        "plate_boundary_distance_km": rng.uniform(0, 500, n),
        "target_T1_max_mag": rng.uniform(0, 6.5, n), "target_T1_time_to_max_hours": rng.uniform(0, 24, n),
        "target_T2_max_mag": rng.uniform(0, 7.0, n), "target_T2_time_to_max_hours": rng.uniform(24, 72, n),
        "target_T3_max_mag": rng.uniform(0, 7.5, n), "target_T3_time_to_max_hours": rng.uniform(72, 168, n),
    })
    p = tmp_path / "mock.csv"; df.to_csv(p, index=False)
    return p


def test_tuning_smoke(tmp_path):
    dp = _make_mock_csv(tmp_path); out = tmp_path / "tune_smoke"
    old = sys.argv[:]
    sys.argv = ["tune.py", "--data", str(dp), "--mag-trials", "2", "--time-trials", "2",
                "--extreme-trials", "2", "--n-splits", "2", "--fast", "--device", "cpu",
                "--windows", "T1,T2", "--output-dir", str(out)]
    try:
        from scripts.tune_decoupled_models import main; main()
    finally: sys.argv = old

    assert (out / "best_params.json").exists()
    assert (out / "tuning_summary.json").exists()
    with open(out / "best_params.json") as f: bp = json.load(f)
    assert "windows" in bp; assert "postprocessing" in bp; assert "global" in bp
    for wn in ("T1", "T2"):
        w = bp["windows"][wn]
        assert "mag_model" in w
        assert "time_model" in w
        assert "extreme_model" in w
