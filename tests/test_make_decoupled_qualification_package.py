"""Decoupled 打包脚本测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


class TestParseCalibration:
    """三态校准解析。"""

    def test_artifact_default(self):
        from scripts.make_decoupled_qualification_package import _parse_calibration
        mode, _ = _parse_calibration(None)
        assert mode == "artifact"
        mode2, _ = _parse_calibration("artifact")
        assert mode2 == "artifact"

    def test_none_disables(self):
        from scripts.make_decoupled_qualification_package import _parse_calibration
        mode, vals = _parse_calibration("none")
        assert mode == "none"
        assert vals == {}
        mode2, _ = _parse_calibration("off")
        assert mode2 == "none"

    def test_override(self):
        from scripts.make_decoupled_qualification_package import _parse_calibration
        mode, vals = _parse_calibration("T1:-0.1,T2:0.2")
        assert mode == "override"
        assert vals == {"T1": -0.1, "T2": 0.2}


class TestApplyCLICalibration:
    """CLI 校准覆盖。"""

    def test_empty_no_change(self):
        from scripts.make_decoupled_qualification_package import apply_cli_calibration
        preds = {"T1": (5.0, 10.0)}
        r = apply_cli_calibration(preds, 7.0, {}, {})
        assert r["T1"][0] == pytest.approx(5.0)

    def test_explicit_mag(self):
        from scripts.make_decoupled_qualification_package import apply_cli_calibration
        preds = {"T1": (5.0, 10.0)}
        r = apply_cli_calibration(preds, 7.0, {"T1": 0.5}, {})
        assert r["T1"][0] == pytest.approx(5.5)

    def test_clamp(self):
        from scripts.make_decoupled_qualification_package import apply_cli_calibration
        preds = {"T1": (5.0, 100.0)}
        r = apply_cli_calibration(preds, 7.0, {}, {})
        assert r["T1"][1] <= 24.0
        assert r["T1"][1] > 0.0


# --------------- end-to-end smoke ---------------

def _make_mock_event_csv(tmp_path):
    df = pd.DataFrame([
        {"Date": "2008-05-12", "Time": "06:28:04", "Lat": 31.0, "Lon": 103.4, "Mag": 7.9, "Depth": 19.0},
        {"Date": "2008-05-13", "Time": "06:28:04", "Lat": 31.1, "Lon": 103.5, "Mag": 5.1, "Depth": 10.0},
        {"Date": "2008-05-15", "Time": "06:28:04", "Lat": 31.1, "Lon": 103.5, "Mag": 5.8, "Depth": 10.0},
        {"Date": "2008-05-18", "Time": "06:28:04", "Lat": 31.1, "Lon": 103.5, "Mag": 6.0, "Depth": 10.0},
    ])
    p = tmp_path / "test_seq.csv"
    df.to_csv(p, index=False)
    return p


def _make_mock_artifact(tmp_path):
    import joblib
    from sklearn.dummy import DummyRegressor, DummyClassifier
    art = {
        "artifact_type": "qualification_decoupled_v1",
        "postprocessing": {
            "mag_bias_T1": 0.0, "mag_bias_T2": 0.0, "mag_bias_T3": 0.0,
            "time_bias_T1": 0.0, "time_bias_T2": 0.0, "time_bias_T3": 0.0,
        },
        "windows": {},
    }
    for wn in ("T1", "T2", "T3"):
        dr = DummyRegressor(strategy="constant", constant=5.0)
        dr.fit([[0]], [0])
        from sklearn.utils import check_random_state
        rng = check_random_state(42)
        X = rng.randn(10, 3)
        y = rng.randint(0, 4, 10)
        dc = DummyClassifier(strategy="uniform")
        dc.fit(X, y)
        art["windows"][wn] = {
            "observation_hours": {"T1": 0.0, "T2": 24.0, "T3": 72.0}[wn],
            "feature_cols": ["mainshock_mag", "mainshock_depth", "plate_boundary_distance_km"],
            "mag_models": {"baseline": dr},
            "bucket_model": dc, "extreme_model": dc,
            "weights": {"mag": {"baseline": 1.0}},
            "postprocessing": {"mag_bias": 0.0, "time_bias": 0.0},
        }
    p = tmp_path / "mock_artifact.joblib"
    joblib.dump(art, p)
    return p


def test_package_smoke(tmp_path):
    """端到端：mock artifact + mock 震例 → 打包 → 验证 ZIP 结构。"""
    import zipfile, sys

    input_dir = tmp_path / "test_sequences"
    input_dir.mkdir()
    _make_mock_event_csv(input_dir)

    model_path = _make_mock_artifact(tmp_path)
    pb_path = tmp_path / "mock_pb.json"
    pb_path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{"type": "Feature",
                       "geometry": {"type": "LineString", "coordinates": [[100, 30], [105, 35]]},
                       "properties": {"STEP_CLASS": "SUB"}}]
    }))

    old_argv = sys.argv[:]
    sys.argv = [
        "make_decoupled_qualification_package.py",
        "--input-dir", str(input_dir), "--model-path", str(model_path),
        "--output-dir", str(tmp_path / "pkg"), "--zip-path", str(tmp_path / "out.zip"),
        "--skip-commitment", "--clean",
        "--gcmt-catalog", str(tmp_path / "nox.csv"),
        "--plate-boundaries", str(pb_path),
        "--mag-calibration", "none", "--time-calibration-hours", "none",
    ]
    try:
        from scripts.make_decoupled_qualification_package import main
        main()
    finally:
        sys.argv = old_argv

    zpath = tmp_path / "out.zip"
    assert zpath.exists()
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
    assert any(n.startswith("predictions/") and n.endswith(".csv") for n in names)
    assert "MANIFEST.json" in names
