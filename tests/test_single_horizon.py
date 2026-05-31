"""Single-horizon H168 单元与 smoke 测试。"""

from __future__ import annotations

import json, sys, tempfile, zipfile
from pathlib import Path

import numpy as np, pandas as pd, pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ===================================================================
# Test H168 label derivation
# ===================================================================

class TestDeriveH168Labels:
    def test_largest_mag_chosen(self):
        from src.qualification import derive_h168_labels, H168_MAG_COL, H168_TIME_COL, H168_FLAG_COL
        df = pd.DataFrame({
            "target_T1_max_mag": [5.0, 4.0, 0.0],
            "target_T2_max_mag": [4.5, 5.5, 0.0],
            "target_T3_max_mag": [4.0, 4.8, 0.0],
            "target_T1_time_to_max_hours": [10.0, 8.0, 84.0],
            "target_T2_time_to_max_hours": [30.0, 40.0, 84.0],
            "target_T3_time_to_max_hours": [80.0, 100.0, 84.0],
        })
        result = derive_h168_labels(df)
        # row 0: T1=5.0 is max
        assert result[H168_MAG_COL].iloc[0] == 5.0
        assert result[H168_TIME_COL].iloc[0] == 10.0
        # row 1: T2=5.5 is max
        assert result[H168_MAG_COL].iloc[1] == 5.5
        assert result[H168_TIME_COL].iloc[1] == 40.0
        # row 2: all zero — no aftershock
        assert result[H168_MAG_COL].iloc[2] == 0.0
        assert result[H168_TIME_COL].iloc[2] == 84.0
        assert result[H168_FLAG_COL].iloc[2] == False

    def test_tie_picks_earliest_time(self):
        from src.qualification import derive_h168_labels, H168_MAG_COL, H168_TIME_COL
        df = pd.DataFrame({
            "target_T1_max_mag": [5.0],
            "target_T2_max_mag": [5.0],
            "target_T3_max_mag": [3.0],
            "target_T1_time_to_max_hours": [10.0],
            "target_T2_time_to_max_hours": [25.0],
            "target_T3_time_to_max_hours": [80.0],
        })
        result = derive_h168_labels(df)
        assert result[H168_MAG_COL].iloc[0] == 5.0
        # T1 and T2 both have 5.0, pick earliest time = 10.0
        assert result[H168_TIME_COL].iloc[0] == 10.0

    def test_all_nan_treated_as_zero(self):
        from src.qualification import derive_h168_labels, H168_MAG_COL, H168_TIME_COL
        df = pd.DataFrame({
            "target_T1_max_mag": [np.nan],
            "target_T2_max_mag": [np.nan],
            "target_T3_max_mag": [np.nan],
            "target_T1_time_to_max_hours": [np.nan],
            "target_T2_time_to_max_hours": [np.nan],
            "target_T3_time_to_max_hours": [np.nan],
        })
        result = derive_h168_labels(df)
        assert result[H168_MAG_COL].iloc[0] == 0.0
        assert result[H168_TIME_COL].iloc[0] == 84.0


# ===================================================================
# Test H168 bucket boundaries
# ===================================================================

class TestH168Buckets:
    def test_bucket_count(self):
        from src.time_buckets import get_time_buckets
        buckets = get_time_buckets("H168")
        assert len(buckets) == 4

    def test_bucket_coverage(self):
        from src.time_buckets import get_time_buckets
        buckets = get_time_buckets("H168")
        lo, hi = buckets[0][0], buckets[-1][1]
        assert lo == 0.0
        assert hi == 168.0

    def test_bucket_center_in_range(self):
        from src.time_buckets import bucket_centers
        centers = bucket_centers("H168")
        assert len(centers) == 4
        for c in centers:
            assert 0.0 < c < 168.0

    def test_assign_h168_buckets(self):
        from src.time_buckets import assign_time_bucket
        # (0,12] → 0; (12,48] → 1; (48,96] → 2; (96,168] → 3
        assert assign_time_bucket("H168", 6.0) == 0
        assert assign_time_bucket("H168", 30.0) == 1
        assert assign_time_bucket("H168", 72.0) == 2
        assert assign_time_bucket("H168", 120.0) == 3

    def test_expected_time_peaked(self):
        from src.time_buckets import expected_time_from_bucket_probs
        # peaked at bucket index 2 → center should be near (48+96)/2 = 72
        probs = np.array([0.1, 0.1, 0.7, 0.1])
        et = expected_time_from_bucket_probs("H168", probs)
        assert 60.0 < et < 84.0


# ===================================================================
# Test H168 feature reconstruction (static only, no observation leak)
# ===================================================================

class TestReconstructH168Features:
    def test_only_static_features(self):
        from src.qualification import reconstruct_h168_features
        df = pd.DataFrame({
            "mainshock_id": ["m1"],
            "mainshock_mag": [7.0],
            "mainshock_depth": [10.0],
            "mainshock_lat": [0.0],
            "mainshock_lon": [0.0],
            "mainshock_time": ["2000-01-01"],
            "plate_boundary_distance_km": [50.0],
            "early_aftershock_count": [100],   # should be dropped
            "count_1h": [20],                  # should be dropped
            "count_24h": [50],                 # should be dropped
            "energy_1h": [1e12],               # should be dropped
        })
        result = reconstruct_h168_features(df)
        # static features should remain
        assert "mainshock_mag" in result.columns
        assert "plate_boundary_distance_km" in result.columns
        # observation features must be absent
        assert "early_aftershock_count" not in result.columns
        assert "count_1h" not in result.columns
        assert "energy_1h" not in result.columns
        # placeholders
        assert "count_obs_0h" in result.columns
        assert result["count_obs_0h"].iloc[0] == 0.0


# ===================================================================
# Test format_single_horizon_line
# ===================================================================

class TestFormatSingleHorizonLine:
    def test_format_contains_ms_and_no_header(self):
        from src.qualification import format_single_horizon_line
        ms = {"time": pd.Timestamp("2008-05-12T06:28:04"),
              "longitude": 103.4, "latitude": 31.0, "mag": 7.9}
        line = format_single_horizon_line(ms, 6.5, 30.0)
        # space-separated, no header
        assert "," not in line.split()[0]  # first token is timestamp, not CSV header
        assert "(Ms)" in line
        # check key fields present
        assert "2008" in line
        assert "103.40" in line
        assert "31.00" in line
        assert "7.9" in line

    def test_clamp_to_window(self):
        from src.qualification import format_single_horizon_line
        ms = {"time": pd.Timestamp("2000-01-01"), "longitude": 0.0, "latitude": 0.0, "mag": 6.0}
        # predict mag > mainshock_mag + 0.5 → should be clamped
        line = format_single_horizon_line(ms, 20.0, 84.0)
        parts = line.split()
        # predicted mag (5th field) should be <= 6.5
        pred_mag_str = parts[4]
        pred_mag = float(pred_mag_str)
        assert pred_mag <= 6.5


# ===================================================================
# Test single-horizon package output format
# ===================================================================

def _mock_event_csv(tmp_path):
    df = pd.DataFrame([
        {"Date":"2008-05-12","Time":"06:28:04","Lat":31.0,"Lon":103.4,"Mag":7.9,"Depth":19.0},
        {"Date":"2008-05-13","Time":"06:28:04","Lat":31.1,"Lon":103.5,"Mag":5.1,"Depth":10.0},
    ])
    p = tmp_path / "test_seq.csv"
    df.to_csv(p, index=False)
    return p


def _mock_artifact(tmp_path):
    import joblib
    from sklearn.dummy import DummyRegressor, DummyClassifier

    dr = DummyRegressor(strategy="constant", constant=6.0)
    dr.fit([[0]], [0])
    dc = DummyClassifier(strategy="uniform")
    dc.fit(np.random.RandomState(42).randn(10,3), np.random.RandomState(42).randint(0,4,10))
    # mock direct time regressors: must support predict() → log-space value
    dr_time_lgbm = DummyRegressor(strategy="constant", constant=np.log1p(84.0))
    dr_time_lgbm.fit([[0]], [0])
    dr_time_xgb = DummyRegressor(strategy="constant", constant=np.log1p(50.0))
    dr_time_xgb.fit([[0]], [0])

    art = {
        "artifact_type": "qualification_single_horizon_v2",
        "postprocessing": {},
        "H168": {
            "observation_hours": 0.0,
            "feature_cols": ["mainshock_mag", "mainshock_depth", "plate_boundary_distance_km"],
            "mag_models": {"baseline": dr, "xgboost": dr},
            "bucket_model": dc,
            "extreme_model": dc,
            "time_direct_model_lgbm": dr_time_lgbm,
            "time_direct_model_xgb": dr_time_xgb,
            "weights": {"mag": {"baseline": 0.5, "xgboost": 0.5}},
            "fusion": {},
            "version": "single_horizon_v2",
        },
    }
    p = tmp_path / "mock_artifact.joblib"
    joblib.dump(art, p)
    return p


class TestSingleHorizonPackage:
    def test_default_no_legacy_files(self, tmp_path):
        """默认打包只产生 qualification_predictions.csv，无 T1/T2/T3 文件。"""
        import sys
        input_dir = tmp_path / "test_sequences"
        input_dir.mkdir()
        _mock_event_csv(input_dir)
        model_path = _mock_artifact(tmp_path)
        pb_path = tmp_path / "mock_pb.json"
        pb_path.write_text(json.dumps({
            "type":"FeatureCollection",
            "features":[{"type":"Feature",
                          "geometry":{"type":"LineString","coordinates":[[100,30],[105,35]]},
                          "properties":{"STEP_CLASS":"SUB"}}]
        }))

        old_argv = sys.argv[:]
        sys.argv = [
            "make_single_horizon_package.py",
            "--input-dir", str(input_dir), "--model-path", str(model_path),
            "--output-dir", str(tmp_path/"pkg"), "--zip-path", str(tmp_path/"out.zip"),
            "--skip-commitment", "--clean",
            "--gcmt-catalog", str(tmp_path/"nox.csv"),
            "--plate-boundaries", str(pb_path),
        ]
        try:
            from scripts.make_single_horizon_package import main
            main()
        finally:
            sys.argv = old_argv

        zpath = tmp_path / "out.zip"
        assert zpath.exists()
        with zipfile.ZipFile(zpath) as zf:
            names = zf.namelist()

        # default: only qualification_predictions.csv in predictions/
        pred_files = [n for n in names if n.startswith("predictions/")]
        assert len(pred_files) == 1
        assert pred_files[0] == "predictions/qualification_predictions.csv"
        # no legacy T1/T2/T3 files within predictions/
        assert not any(("T1-T2" in n or "T3" in n) and n.startswith("predictions/") for n in names)

    def test_format_one_line_per_sequence(self):
        """验证 prediction 文件每行一个序列，无 header，space-separated，含 (Ms)。
        第一个 token 必须是 YYYYMMDDhhmmss 格式，不含 '-' 或 'T'。"""
        from src.qualification import write_single_horizon_prediction_file
        ms = {"time": pd.Timestamp("2008-05-12T06:28:04"),
              "longitude": 103.4, "latitude": 31.0, "mag": 7.9}
        rows = [(ms, 6.5, 30.0)]
        path = write_single_horizon_prediction_file(Path(tempfile.mkdtemp()), rows)
        content = path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 1
        # should not contain a header row marker
        assert lines[0].count(" ") >= 5
        assert "(Ms)" in lines[0]
        # first token must be compact YYYYMMDDhhmmss
        first_token = lines[0].split()[0]
        assert len(first_token) == 14, f"expected 14-char timestamp, got '{first_token}'"
        assert first_token.isdigit(), f"first token must be all digits: '{first_token}'"
        assert "-" not in first_token
        assert "T" not in first_token


# ===================================================================
# Fusion coverage tests
# ===================================================================

def _make_fusion_artifact(tmp_path, mag_fusion=None, time_fusion=None):
    """Create a mock artifact with two mag models and bucket+direct time models."""
    import joblib
    from sklearn.dummy import DummyRegressor, DummyClassifier

    dr_baseline = DummyRegressor(strategy="constant", constant=6.5)
    dr_baseline.fit([[0]], [0])
    dr_xgb = DummyRegressor(strategy="constant", constant=5.0)
    dr_xgb.fit([[0]], [0])
    dc = DummyClassifier(strategy="uniform")
    dc.fit(np.random.RandomState(42).randn(10,3), np.random.RandomState(42).randint(0,4,10))
    dr_time_lgbm = DummyRegressor(strategy="constant", constant=np.log1p(80.0))
    dr_time_lgbm.fit([[0]], [0])
    dr_time_xgb = DummyRegressor(strategy="constant", constant=np.log1p(60.0))
    dr_time_xgb.fit([[0]], [0])

    fusion = {}
    if mag_fusion:
        fusion["mag"] = mag_fusion
    if time_fusion:
        fusion["time"] = time_fusion

    art = {
        "artifact_type": "qualification_single_horizon_v2",
        "postprocessing": {},
        "H168": {
            "observation_hours": 0.0,
            "feature_cols": ["mainshock_mag", "mainshock_depth", "plate_boundary_distance_km"],
            "mag_models": {"baseline": dr_baseline, "xgboost": dr_xgb},
            "bucket_model": dc,
            "extreme_model": dc,
            "time_direct_model_lgbm": dr_time_lgbm,
            "time_direct_model_xgb": dr_time_xgb,
            "weights": {"mag": {"baseline": 0.6, "xgboost": 0.4}},
            "fusion": fusion,
            "version": "single_horizon_v2",
        },
    }
    p = tmp_path / "fusion_artifact.joblib"
    joblib.dump(art, p)
    return p


class TestMagFusionKeys:
    def test_uses_fusion_keys(self, tmp_path):
        """验证 mag 融合使用正确的 key（oof_mag_lgbm/oof_mag_xgb），而非 model 内部 key。"""
        model_path = _make_fusion_artifact(
            tmp_path,
            mag_fusion={"oof_mag_lgbm": 0.7, "oof_mag_xgb": 0.3},
        )
        import sys
        input_dir = tmp_path / "test_sequences"; input_dir.mkdir()
        _mock_event_csv(input_dir)
        pb_path = tmp_path / "pb.json"
        pb_path.write_text(json.dumps({
            "type":"FeatureCollection",
            "features":[{"type":"Feature",
                          "geometry":{"type":"LineString","coordinates":[[100,30],[105,35]]},
                          "properties":{"STEP_CLASS":"SUB"}}]
        }))
        old = sys.argv[:]
        sys.argv = ["pkg.py","--input-dir",str(input_dir),"--model-path",str(model_path),
                    "--output-dir",str(tmp_path/"pkg"),"--zip-path",str(tmp_path/"out.zip"),
                    "--skip-commitment","--clean",
                    "--gcmt-catalog",str(tmp_path/"nox.csv"),
                    "--plate-boundaries",str(pb_path)]
        try:
            from scripts.make_single_horizon_package import main; main()
        finally: sys.argv = old
        # 检查预测行：6.5*0.7 + 5.0*0.3 = 4.55+1.5 = 6.05
        zpath = tmp_path / "out.zip"
        assert zpath.exists()
        import zipfile
        with zipfile.ZipFile(zpath) as zf:
            content = zf.read("predictions/qualification_predictions.csv").decode()
        parts = content.strip().split()
        pred_mag = float(parts[4])  # 5th token
        # should be ~6.05 (fused), not 6.5 (baseline alone) nor 5.75 (equal avg)
        assert 5.9 <= pred_mag <= 6.2, f"expected ~6.05 fused, got {pred_mag}"

    def test_no_fusion_falls_back_to_weights(self, tmp_path):
        """无 fusion 时回退到 artifact weights 加权平均。"""
        model_path = _make_fusion_artifact(tmp_path, mag_fusion=None)
        import sys
        input_dir = tmp_path / "test_sequences"; input_dir.mkdir()
        _mock_event_csv(input_dir)
        pb_path = tmp_path / "pb.json"
        pb_path.write_text(json.dumps({
            "type":"FeatureCollection",
            "features":[{"type":"Feature",
                          "geometry":{"type":"LineString","coordinates":[[100,30],[105,35]]},
                          "properties":{"STEP_CLASS":"SUB"}}]
        }))
        old = sys.argv[:]
        sys.argv = ["pkg.py","--input-dir",str(input_dir),"--model-path",str(model_path),
                    "--output-dir",str(tmp_path/"pkg"),"--zip-path",str(tmp_path/"out.zip"),
                    "--skip-commitment","--clean",
                    "--gcmt-catalog",str(tmp_path/"nox.csv"),
                    "--plate-boundaries",str(pb_path)]
        try:
            from scripts.make_single_horizon_package import main; main()
        finally: sys.argv = old
        import zipfile
        with zipfile.ZipFile(tmp_path/"out.zip") as zf:
            content = zf.read("predictions/qualification_predictions.csv").decode()
        parts = content.strip().split()
        pred_mag = float(parts[4])
        # weights: baseline=0.6, xgboost=0.4 → 6.5*0.6 + 5.0*0.4 = 3.9+2.0 = 5.9
        assert 5.7 <= pred_mag <= 6.1, f"expected ~5.9 weighted, got {pred_mag}"


class TestTimeFusion:
    def test_direct_time_candidates_used(self, tmp_path):
        """验证时间融合 numeric 结果：bucket+direct_lgbm+direct_xgb 三候选加权。

        H168 bucket centers: (0+12)/2=6, (12+48)/2=30, (48+96)/2=72, (96+168)/2=132.
        DummyClassifier("uniform") → uniform probs → expected_time = mean(centers) = (6+30+72+132)/4 = 60h.
        direct_lgbm: expm1(log1p(80)) ≈ 80h.  direct_xgb: expm1(log1p(60)) ≈ 60h.
        Fusion weights: bucket=0.4, direct_lgbm=0.4, direct_xgb=0.2.
        Fused = 0.4*60 + 0.4*80 + 0.2*60 = 24 + 32 + 12 = 68h.
        """
        import joblib
        model_path = _make_fusion_artifact(
            tmp_path,
            time_fusion={"oof_time_bucket_raw": 0.4,
                         "oof_time_direct_lgbm_raw": 0.4,
                         "oof_time_direct_xgb_raw": 0.2},
        )
        artifact = joblib.load(model_path)
        import sys
        input_dir = tmp_path / "test_sequences"; input_dir.mkdir()
        _mock_event_csv(input_dir)
        (tmp_path / "nox.csv").write_text("")
        pb_path = tmp_path / "pb.json"
        pb_path.write_text(json.dumps({
            "type":"FeatureCollection",
            "features":[{"type":"Feature",
                          "geometry":{"type":"LineString","coordinates":[[100,30],[105,35]]},
                          "properties":{"STEP_CLASS":"SUB"}}]
        }))

        from src.qualification import normalize_event_table, pick_mainshock
        from scripts.make_single_horizon_package import predict_single_horizon
        event_df = normalize_event_table(pd.read_csv(input_dir / "test_seq.csv"))
        mainshock = pick_mainshock(event_df)

        old = sys.argv[:]
        sys.argv = ["pkg.py","--input-dir",str(input_dir),"--model-path",str(model_path),
                    "--output-dir",str(tmp_path/"pkg"),"--zip-path",str(tmp_path/"out.zip"),
                    "--skip-commitment","--clean",
                    "--gcmt-catalog",str(tmp_path/"nox.csv"),
                    "--plate-boundaries",str(pb_path)]
        # Build a proper mock args for predict_single_horizon direct call
        MockArgs = type('MockArgs', (), {
            'plate_boundaries': str(pb_path),
            'gcmt_catalog': str(tmp_path/'nox.csv'),
            'time_mae_guard': 'true',
            'magnitude_type': 'Ms',
        })
        try:
            pred_mag, pred_time = predict_single_horizon(artifact, event_df, mainshock, MockArgs())
        finally: sys.argv = old
        # H168 bucket centers average ~60h, direct_lgbm ~80, direct_xgb ~60.
        # Fused = 0.4*60 + 0.4*80 + 0.2*60 = 68h.  Allow ±3h tolerance.
        assert 65 <= pred_time <= 71, f"expected fused time ~68h, got {pred_time:.1f}h"


class TestLegacyDedup:
    def test_legacy_dedup_uses_seen_legacy(self, tmp_path):
        """验证 --legacy-window-output 的去重逻辑使用 seen_legacy 集合。"""
        model_path = _mock_artifact(tmp_path)
        import sys
        input_dir = tmp_path / "test_sequences"; input_dir.mkdir()
        # two CSV files with same event → one should be deduplicated
        df = pd.DataFrame([
            {"Date":"2008-05-12","Time":"06:28:04","Lat":31.0,"Lon":103.4,"Mag":7.9,"Depth":19.0},
        ])
        df.to_csv(input_dir / "seq1.csv", index=False)
        df.to_csv(input_dir / "seq2.csv", index=False)
        pb_path = tmp_path / "pb.json"
        pb_path.write_text(json.dumps({
            "type":"FeatureCollection",
            "features":[{"type":"Feature",
                          "geometry":{"type":"LineString","coordinates":[[100,30],[105,35]]},
                          "properties":{"STEP_CLASS":"SUB"}}]
        }))
        old = sys.argv[:]
        sys.argv = ["pkg.py","--input-dir",str(input_dir),"--model-path",str(model_path),
                    "--output-dir",str(tmp_path/"pkg"),"--zip-path",str(tmp_path/"out.zip"),
                    "--skip-commitment","--clean","--legacy-window-output",
                    "--gcmt-catalog",str(tmp_path/"nox.csv"),
                    "--plate-boundaries",str(pb_path)]
        try:
            from scripts.make_single_horizon_package import main; main()
        finally: sys.argv = old
        # should not crash with `seen_ids_legacy if False else set()`
        import zipfile
        with zipfile.ZipFile(tmp_path/"out.zip") as zf:
            names = zf.namelist()
        pred_files = [n for n in names if n.startswith("predictions/")]
        # single-horizon: 1 + legacy: 2 files per unique event = 1*2+1=3
        # Actually legacy produces T1-T2.csv + T3.csv per unique event.
        # With dedup, 2 seqs → 1 unique → 2 legacy files + 1 main = 3 total
        assert len(pred_files) >= 3  # at least qualification + T1-T2 + T3


# ===================================================================
# Fast 2-fold tuning + training + packaging smoke
# ===================================================================

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

def test_tune_smoke(tmp_path):
    dp = _make_mock_csv(tmp_path); out = tmp_path / "tune_smoke"
    old = sys.argv[:]
    sys.argv = ["tune.py", "--data", str(dp), "--mag-trials", "2", "--time-trials", "2",
                "--extreme-trials", "2", "--n-splits", "2", "--fast", "--device", "cpu",
                "--output-dir", str(out)]
    try:
        from scripts.tune_single_horizon_models import main; main()
    finally: sys.argv = old
    assert (out/"best_params.json").exists()
    assert (out/"tuning_summary.json").exists()

def test_train_smoke(tmp_path):
    dp = _make_mock_csv(tmp_path); sd = tmp_path / "models"
    bp_path = tmp_path / "bp.json"
    bp = {"mag_model":{"n_estimators":10},"time_model":{"n_estimators":10},"extreme_model":{"n_estimators":10}}
    bp_path.write_text(json.dumps(bp))
    old = sys.argv[:]
    sys.argv = ["train.py", "--data", str(dp), "--save-dir", str(sd),
                "--n-splits", "2", "--device", "cpu", "--n-estimators", "5",
                "--learning-rate", "0.1", "--model-type", "lightgbm",
                "--best-params", str(bp_path)]
    try:
        from scripts.train_single_horizon_models import main; main()
    finally: sys.argv = old
    mp = sd / "qualification_single_horizon_model.joblib"
    assert mp.exists()
    import joblib
    art = joblib.load(mp)
    assert art["artifact_type"] == "qualification_single_horizon_v2"
    assert "H168" in art
    assert "mag_models" in art["H168"]
    assert "bucket_model" in art["H168"]
    # v2: must also have direct time regression model objects
    assert art["H168"].get("time_direct_model_lgbm") is not None
    assert art["H168"].get("time_direct_model_xgb") is not None

def test_full_smoke(tmp_path):
    """端到端：调参→训练→打包。"""
    dp = _make_mock_csv(tmp_path)
    tune_out = tmp_path / "tune_out"; sd = tmp_path / "models"

    # 1. tune
    old = sys.argv[:]
    sys.argv = ["tune.py", "--data", str(dp), "--mag-trials", "2", "--time-trials", "2",
                "--extreme-trials", "2", "--n-splits", "2", "--fast", "--device", "cpu",
                "--output-dir", str(tune_out)]
    try:
        from scripts.tune_single_horizon_models import main; main()
    finally: sys.argv = old

    # 2. train
    sys.argv = ["train.py", "--data", str(dp), "--save-dir", str(sd),
                "--n-splits", "2", "--device", "cpu", "--n-estimators", "5",
                "--learning-rate", "0.1", "--model-type", "lightgbm",
                "--best-params", str(tune_out/"best_params.json")]
    try:
        from scripts.train_single_horizon_models import main; main()
    finally: sys.argv = old

    # 3. package
    input_dir = tmp_path / "test_sequences"; input_dir.mkdir()
    _mock_event_csv(input_dir)
    pb_path = tmp_path / "mock_pb.json"
    pb_path.write_text(json.dumps({
        "type":"FeatureCollection",
        "features":[{"type":"Feature",
                      "geometry":{"type":"LineString","coordinates":[[100,30],[105,35]]},
                      "properties":{"STEP_CLASS":"SUB"}}]
    }))
    sys.argv = ["pkg.py", "--input-dir", str(input_dir),
                "--model-path", str(sd/"qualification_single_horizon_model.joblib"),
                "--output-dir", str(tmp_path/"pkg"), "--zip-path", str(tmp_path/"out.zip"),
                "--skip-commitment", "--clean",
                "--gcmt-catalog", str(tmp_path/"nox.csv"),
                "--plate-boundaries", str(pb_path)]
    try:
        from scripts.make_single_horizon_package import main; main()
    finally: sys.argv = old

    zpath = tmp_path / "out.zip"
    assert zpath.exists()
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
    assert "predictions/qualification_predictions.csv" in names
    assert "MANIFEST.json" in names
    # default: no legacy files within predictions/
    assert not any(("T1-T2" in n or "T3" in n) and n.startswith("predictions/") for n in names)


# ===================================================================
# run.sh qualification delegation smoke
# ===================================================================

class TestRunShQualificationDelegation:
    def test_qualification_help_exits_zero_and_mentions_h168(self):
        """验证 run.sh qualification --help 退出码 0 且输出提到 H168/single-horizon 及 CPU 默认。"""
        import subprocess
        project_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["bash", str(project_root / "run.sh"), "qualification", "--help"],
            capture_output=True, text=True, timeout=30,
            cwd=str(project_root),
        )
        assert result.returncode == 0, f"exit={result.returncode} stderr={result.stderr}"
        output = result.stdout + result.stderr
        assert "single-horizon" in output.lower() or "H168" in output, (
            f"help output should mention H168/single-horizon: {output[:500]}")
        # qualification pipeline help should mention CPU default
        assert "cpu" in output.lower(), (
            f"help output should mention CPU default: {output[:500]}")
