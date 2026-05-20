from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from src.qualification import (
    extract_window_targets,
    format_qualification_line,
    mainshock_token,
    normalize_event_table,
    pick_mainshock,
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
