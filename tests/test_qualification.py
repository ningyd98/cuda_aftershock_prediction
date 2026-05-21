from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from src.qualification import (
    extract_window_targets,
    format_qualification_line,
    mainshock_token,
    normalize_event_table,
    pick_mainshock,
    reconstruct_legal_window_features,
)


def test_window_boundaries_and_mainshock_token():
    raw = pd.DataFrame(
        [
            {"Date": "2008-05-12", "Time": "06:28:04", "Lat": 31.0, "Lon": 103.4, "Mag": 7.9, "Depth": 19.0},
            {"Date": "2008-05-13", "Time": "06:28:04", "Lat": 31.1, "Lon": 103.5, "Mag": 5.1, "Depth": 10.0},
            {"Date": "2008-05-13", "Time": "06:28:05", "Lat": 31.1, "Lon": 103.5, "Mag": 5.7, "Depth": 10.0},
            {"Date": "2008-05-15", "Time": "06:28:04", "Lat": 31.1, "Lon": 103.5, "Mag": 5.8, "Depth": 10.0},
            {"Date": "2008-05-18", "Time": "06:28:04", "Lat": 31.1, "Lon": 103.5, "Mag": 6.0, "Depth": 10.0},
        ]
    )
    events = normalize_event_table(raw)
    mainshock = pick_mainshock(events)
    targets = extract_window_targets(events, mainshock)

    assert mainshock_token(mainshock) == "20080512062804"
    assert targets["target_T1_max_mag"] == 5.1
    assert targets["target_T2_max_mag"] == 5.8
    assert targets["target_T3_max_mag"] == 6.0


def test_submission_line_format():
    mainshock = {
        "time": pd.Timestamp("2008-05-12T06:28:04Z"),
        "longitude": 103.4,
        "latitude": 31.0,
        "mag": 7.95,
    }
    line = format_qualification_line(mainshock, "T1", 5.44, 17.5)
    assert line == "20080512062804 103.40 31.00 8.0 5.4 (Ms) 2008051300"


def test_legal_window_feature_reconstruction_blocks_future_observations():
    features = pd.DataFrame(
        [
            {
                "mainshock_id": "x",
                "mainshock_time": "2020-01-01T00:00:00Z",
                "mainshock_mag": 7.0,
                "mainshock_depth": 10.0,
                "count_1h": 2,
                "count_24h": 8,
                "count_72h": 50,
                "energy_1h": 10.0,
                "energy_24h": 100.0,
                "energy_72h": 900.0,
                "early_max_mag": 6.5,
                "gr_b_value": 0.8,
                "omori_p": 1.1,
                "target_T1_max_mag": 5.0,
                "target_T1_time_to_max_hours": 2.0,
                "target_T2_max_mag": 5.5,
                "target_T2_time_to_max_hours": 30.0,
                "target_T3_max_mag": 5.8,
                "target_T3_time_to_max_hours": 90.0,
            }
        ]
    )

    t1 = reconstruct_legal_window_features(features, "T1")
    t2 = reconstruct_legal_window_features(features, "T2")
    t3 = reconstruct_legal_window_features(features, "T3")

    assert "count_24h" not in t1.columns
    assert "early_max_mag" not in t1.columns
    assert "count_24h" in t2.columns
    assert "count_72h" not in t2.columns
    assert "early_max_mag" not in t2.columns
    assert t2.loc[0, "early_aftershock_count"] == 8
    assert "count_72h" in t3.columns
    assert "early_max_mag" in t3.columns
