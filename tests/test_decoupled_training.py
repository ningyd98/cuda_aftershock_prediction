"""Decoupled 训练脚本测试。"""

from __future__ import annotations

import json, tempfile, sys
from pathlib import Path

import numpy as np, pandas as pd, pytest


class TestBestParamsLoading:
    def test_load_existing(self):
        from scripts.train_decoupled_window_models import load_best_params
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"windows": {"T1": {"mag_model": {"n_estimators": 100}}}, "postprocessing": {"mag_bias_T1": -0.1}}, f)
            p = Path(f.name)
        try:
            r = load_best_params(p)
            assert r["windows"]["T1"]["mag_model"]["n_estimators"] == 100
        finally:
            p.unlink()

    def test_load_none(self):
        from scripts.train_decoupled_window_models import load_best_params
        assert load_best_params(None) == {}

    def test_load_nonexistent(self):
        from scripts.train_decoupled_window_models import load_best_params
        assert load_best_params(Path("/nonexistent.json")) == {}


class TestFuseMag:
    def test_avg(self):
        from scripts.train_decoupled_window_models import fuse_mag
        m = {"a": np.array([4.0, 5.0]), "b": np.array([4.2, 4.8])}
        np.testing.assert_allclose(fuse_mag(m), np.array([4.1, 4.9]))

    def test_single(self):
        from scripts.train_decoupled_window_models import fuse_mag
        np.testing.assert_array_equal(fuse_mag({"a": np.array([3.0, 4.0])}), np.array([3.0, 4.0]))


class TestMetricsHelpers:
    def test_rmse(self):
        from scripts.train_decoupled_window_models import _rmse
        assert _rmse(np.array([1.0, -1.0, 0.0])) == pytest.approx(np.sqrt(2.0 / 3.0))

    def test_calc_metrics(self):
        from scripts.train_decoupled_window_models import calc_metrics
        m = calc_metrics(np.array([5.0, 6.0]), np.array([10.0, 50.0]),
                         np.array([5.0, 6.0]), np.array([10.0, 50.0]))
        assert m["mag_rmse"] == pytest.approx(0.0)
        assert m["time_hour_mae"] == pytest.approx(0.0)


class TestParseArgs:
    def test_defaults(self):
        old = sys.argv[:]
        sys.argv = ["train.py"]
        try:
            from scripts.train_decoupled_window_models import parse_args
            a = parse_args()
            assert a.n_splits == 5
            assert a.seed == 42
        finally:
            sys.argv = old


# --------------- 端到端 smoke ---------------

def _mock_csv(tmp_path):
    n = 30
    np.random.seed(42)
    df = pd.DataFrame({
        "mainshock_id": [f"s{i:04d}" for i in range(n)],
        "mainshock_time": pd.date_range("2000-01-01", periods=n, freq="180D"),
        "mainshock_lat": np.random.uniform(-30, 30, n),
        "mainshock_lon": np.random.uniform(-180, 180, n),
        "mainshock_mag": np.random.uniform(6.0, 8.0, n),
        "mainshock_depth": np.random.uniform(5, 50, n),
        "early_aftershock_count": np.random.randint(0, 20, n),
        "early_energy_sum": np.random.uniform(1e10, 1e15, n),
        "count_1h": np.random.randint(0, 5, n),
        "count_24h": np.random.randint(0, 15, n),
        "count_72h": np.random.randint(0, 30, n),
        "energy_1h": np.random.uniform(0, 1e14, n),
        "energy_24h": np.random.uniform(0, 1e15, n),
        "energy_72h": np.random.uniform(0, 1e16, n),
        "plate_boundary_distance_km": np.random.uniform(0, 500, n),
        "target_T1_max_mag": np.random.uniform(0, 6.5, n),
        "target_T1_time_to_max_hours": np.random.uniform(0, 24, n),
        "target_T2_max_mag": np.random.uniform(0, 7.0, n),
        "target_T2_time_to_max_hours": np.random.uniform(24, 72, n),
        "target_T3_max_mag": np.random.uniform(0, 7.5, n),
        "target_T3_time_to_max_hours": np.random.uniform(72, 168, n),
    })
    p = tmp_path / "m.csv"
    df.to_csv(p, index=False)
    return p


def test_train_smoke(tmp_path):
    dp = _mock_csv(tmp_path)
    sd = tmp_path / "models"
    bp_path = tmp_path / "bp.json"
    bp = {
        "windows": {
            wn: {"mag_model": {"n_estimators": 10}, "bucket_model": {"n_estimators": 10},
                 "extreme_model": {"n_estimators": 10}}
            for wn in ("T1", "T2", "T3")
        },
        "postprocessing": {"mag_bias_T1": -0.1, "time_bias_T1": 1.0},
    }
    bp_path.write_text(json.dumps(bp))

    old = sys.argv[:]
    sys.argv = ["train.py", "--data", str(dp), "--save-dir", str(sd),
                "--n-splits", "2", "--device", "cpu", "--n-estimators", "5",
                "--learning-rate", "0.1", "--model-type", "lightgbm",
                "--best-params", str(bp_path)]
    try:
        from scripts.train_decoupled_window_models import main
        main()
    finally:
        sys.argv = old

    mp = sd / "qualification_decoupled_models.joblib"
    assert mp.exists()
    import joblib
    art = joblib.load(mp)
    assert art["artifact_type"] == "qualification_decoupled_v2"
    assert art["postprocessing"]["mag_bias_T1"] == -0.1
    for wn in ("T1", "T2", "T3"):
        w = art["windows"][wn]
        assert "mag_models" in w
        assert "postprocessing" in w
        assert "mag_bias" in w["postprocessing"]
    # 验证 metrics 有 raw/postprocessed
    mp2 = sd / "decoupled_metrics.json"
    assert mp2.exists()
    with open(mp2) as f:
        met = json.load(f)
    for wn in ("T1", "T2", "T3"):
        assert "raw" in met["window_metrics"][wn]
        assert "postprocessed" in met["window_metrics"][wn]


class TestExtremePostprocessing:
    def test_t1_early_bonus(self):
        from scripts.train_decoupled_window_models import apply_extreme_postprocessing
        import numpy as np
        mag = np.array([5.0, 4.0])
        time = np.array([5.0, 15.0])
        ep = np.array([0.8, 0.9])  # both high risk
        mm = np.array([7.0, 7.0])
        pp = {'extreme_prob_threshold': 0.5, 'high_risk_mag_quantile_weight': 0.5,
              'extreme_margin': 1.2, 'early_time_shift_strength': 0.1,
              't1_early_delta_bonus': 0.3}
        mag2, _ = apply_extreme_postprocessing(mag, time, ep, mm, 'T1', pp)
        # With bonus, T1 high-risk mags should be higher
        assert mag2[0] > mag[0]
        assert mag2[1] > mag[1]

    def test_t1_no_bonus_when_zero(self):
        from scripts.train_decoupled_window_models import apply_extreme_postprocessing
        import numpy as np
        mag = np.array([5.0])
        time = np.array([5.0])
        ep = np.array([0.8])
        mm = np.array([7.0])
        pp_no = {'extreme_prob_threshold': 0.5, 't1_early_delta_bonus': 0.0}
        mag_no, _ = apply_extreme_postprocessing(mag, time, ep, mm, 'T1', pp_no)
        pp_yes = {**pp_no, 't1_early_delta_bonus': 0.3}
        mag_yes, _ = apply_extreme_postprocessing(mag, time, ep, mm, 'T1', pp_yes)
        assert mag_yes[0] > mag_no[0]
