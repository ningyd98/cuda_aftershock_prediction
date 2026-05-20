from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualificationWindow:
    name: str
    lower_hours: float
    upper_hours: float

    @property
    def mag_col(self) -> str:
        return f"target_{self.name}_max_mag"

    @property
    def time_col(self) -> str:
        return f"target_{self.name}_time_to_max_hours"

    @property
    def flag_col(self) -> str:
        return f"has_{self.name}_aftershock"

    @property
    def midpoint_hours(self) -> float:
        return (self.lower_hours + self.upper_hours) / 2.0


QUALIFICATION_WINDOWS: tuple[QualificationWindow, ...] = (
    QualificationWindow("T1", 0.0, 24.0),
    QualificationWindow("T2", 24.0, 72.0),
    QualificationWindow("T3", 72.0, 168.0),
)
WINDOW_BY_NAME = {window.name: window for window in QUALIFICATION_WINDOWS}


def qualification_target_cols() -> list[str]:
    cols: list[str] = []
    for window in QUALIFICATION_WINDOWS:
        cols.extend([window.mag_col, window.time_col])
    return cols


def qualification_aux_cols() -> list[str]:
    return [window.flag_col for window in QUALIFICATION_WINDOWS]


def normalize_event_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df = df.rename(
        columns={
            "Lat": "latitude",
            "Lon": "longitude",
            "Mag": "mag",
            "Depth": "depth",
        }
    )

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce", format="mixed")
    elif {"Date", "Time"}.issubset(df.columns):
        df["time"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            utc=True,
            errors="coerce",
            format="mixed",
        )
    elif {"Year", "Month", "Day", "Hour", "Minute", "Second"}.issubset(df.columns):
        df["time"] = pd.to_datetime(
            dict(
                year=df["Year"],
                month=df["Month"],
                day=df["Day"],
                hour=df["Hour"],
                minute=df["Minute"],
                second=df["Second"],
            ),
            utc=True,
            errors="coerce",
        )
    else:
        raise ValueError("Input events must include time, Date/Time, or Year/Month/Day fields.")

    required = ["time", "latitude", "longitude", "mag", "depth"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Input events are missing required columns: {missing}")

    df = df.dropna(subset=required).sort_values("time").reset_index(drop=True)
    if "id" not in df.columns:
        df["id"] = df["time"].dt.strftime("%Y%m%d%H%M%S").astype(str) + "_eq"
    return df


def pick_mainshock(event_df: pd.DataFrame) -> pd.Series:
    if event_df.empty:
        raise ValueError("Cannot pick a mainshock from an empty event table.")
    max_mag = event_df["mag"].max()
    return event_df.loc[event_df["mag"] == max_mag].sort_values("time").iloc[0]


def mainshock_token(mainshock: pd.Series | dict) -> str:
    timestamp = pd.Timestamp(mainshock["time"])
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.strftime("%Y%m%d%H%M%S")


def _haversine_km(
    lat1: float,
    lon1: float,
    lat2: np.ndarray,
    lon2: np.ndarray,
    earth_radius_km: float = 6371.0,
) -> np.ndarray:
    lat1_rad, lon1_rad = np.radians([lat1, lon1])
    lat2_rad = np.radians(lat2.astype(float))
    lon2_rad = np.radians(lon2.astype(float))
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    return earth_radius_km * 2.0 * np.arcsin(np.sqrt(a))


def aftershock_candidates(
    catalog_df: pd.DataFrame,
    mainshock: pd.Series | dict,
    spatial_radius_km: float = 100.0,
    earth_radius_km: float = 6371.0,
    max_hours: float = 168.0,
) -> pd.DataFrame:
    main_time = pd.Timestamp(mainshock["time"])
    main_lat = float(mainshock["latitude"])
    main_lon = float(mainshock["longitude"])
    max_end = main_time + pd.Timedelta(hours=max_hours)

    candidates = catalog_df.loc[
        (catalog_df["time"] > main_time) & (catalog_df["time"] <= max_end)
    ].copy()
    if candidates.empty:
        candidates["elapsed_hours"] = []
        candidates["distance_km"] = []
        return candidates

    candidates["distance_km"] = _haversine_km(
        main_lat,
        main_lon,
        candidates["latitude"].to_numpy(),
        candidates["longitude"].to_numpy(),
        earth_radius_km=earth_radius_km,
    )
    candidates = candidates.loc[candidates["distance_km"] <= spatial_radius_km].copy()
    candidates["elapsed_hours"] = (
        candidates["time"] - main_time
    ).dt.total_seconds() / 3600.0
    return candidates


def _window_mask(candidates: pd.DataFrame, window: QualificationWindow) -> pd.Series:
    elapsed = candidates["elapsed_hours"]
    lower_ok = elapsed > window.lower_hours
    if window.lower_hours == 0:
        lower_ok = elapsed > 0.0
    return lower_ok & (elapsed <= window.upper_hours)


def extract_window_targets(
    catalog_df: pd.DataFrame,
    mainshock: pd.Series | dict,
    spatial_radius_km: float = 100.0,
    earth_radius_km: float = 6371.0,
) -> dict[str, float | bool]:
    candidates = aftershock_candidates(
        catalog_df,
        mainshock,
        spatial_radius_km=spatial_radius_km,
        earth_radius_km=earth_radius_km,
        max_hours=max(window.upper_hours for window in QUALIFICATION_WINDOWS),
    )
    labels: dict[str, float | bool] = {}
    for window in QUALIFICATION_WINDOWS:
        in_window = candidates.loc[_window_mask(candidates, window)].copy()
        if in_window.empty:
            labels[window.flag_col] = False
            labels[window.mag_col] = 0.0
            labels[window.time_col] = float(window.midpoint_hours)
            continue

        best = in_window.sort_values(["mag", "time"], ascending=[False, True]).iloc[0]
        labels[window.flag_col] = True
        labels[window.mag_col] = float(best["mag"])
        labels[window.time_col] = float(best["elapsed_hours"])
    return labels


def append_qualification_targets(
    mainshock_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    spatial_radius_km: float = 100.0,
    earth_radius_km: float = 6371.0,
) -> pd.DataFrame:
    catalog = normalize_event_table(catalog_df)
    result = mainshock_df.copy()
    if "mainshock_time" not in result.columns:
        raise ValueError("mainshock_df must include mainshock_time.")

    records: list[dict[str, float | bool]] = []
    for _, row in result.iterrows():
        mainshock = {
            "time": pd.to_datetime(row["mainshock_time"], utc=True, errors="coerce"),
            "latitude": float(row["mainshock_lat"]),
            "longitude": float(row["mainshock_lon"]),
            "mag": float(row["mainshock_mag"]),
            "depth": float(row.get("mainshock_depth", 0.0)),
        }
        records.append(
            extract_window_targets(
                catalog,
                mainshock,
                spatial_radius_km=spatial_radius_km,
                earth_radius_km=earth_radius_km,
            )
        )
    return pd.concat([result.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def build_qualification_samples_from_catalog(
    catalog_df: pd.DataFrame,
    min_mainshock_mag: float = 6.0,
    max_depth_km: float = 70.0,
    spatial_radius_km: float = 100.0,
    earth_radius_km: float = 6371.0,
) -> pd.DataFrame:
    catalog = normalize_event_table(catalog_df)
    mainshocks = catalog.loc[
        (catalog["mag"] >= min_mainshock_mag) & (catalog["depth"] <= max_depth_km)
    ].copy()
    rows: list[dict[str, object]] = []
    processed_aftershock_ids: set[str] = set()

    for _, mainshock in mainshocks.iterrows():
        if str(mainshock["id"]) in processed_aftershock_ids:
            continue
        candidates = aftershock_candidates(
            catalog,
            mainshock,
            spatial_radius_km=spatial_radius_km,
            earth_radius_km=earth_radius_km,
        )
        if not candidates.empty and float(candidates["mag"].max()) > float(mainshock["mag"]):
            continue
        processed_aftershock_ids.update(candidates["id"].astype(str).tolist())
        record: dict[str, object] = {
            "mainshock_id": str(mainshock["id"]),
            "mainshock_time": mainshock["time"],
            "mainshock_lat": float(mainshock["latitude"]),
            "mainshock_lon": float(mainshock["longitude"]),
            "mainshock_mag": float(mainshock["mag"]),
            "mainshock_depth": float(mainshock["depth"]),
        }
        record.update(
            extract_window_targets(
                catalog,
                mainshock,
                spatial_radius_km=spatial_radius_km,
                earth_radius_km=earth_radius_km,
            )
        )
        rows.append(record)
    return pd.DataFrame(rows)


def clamp_prediction_to_window(
    window_name: str,
    mag: float,
    time_hours: float,
    mainshock_mag: float,
) -> tuple[float, float]:
    window = WINDOW_BY_NAME[window_name]
    safe_mag = 0.0 if not np.isfinite(mag) else float(mag)
    safe_time = window.midpoint_hours if not np.isfinite(time_hours) else float(time_hours)
    safe_mag = min(max(safe_mag, 0.0), float(mainshock_mag) + 0.5)
    safe_time = min(max(safe_time, window.lower_hours + 1e-6), window.upper_hours)
    return safe_mag, safe_time


def rule_window_prediction(window_name: str, mainshock_mag: float) -> tuple[float, float]:
    window = WINDOW_BY_NAME[window_name]
    decay = {"T1": 1.15, "T2": 1.35, "T3": 1.55}.get(window_name, 1.35)
    return clamp_prediction_to_window(
        window_name,
        mag=max(0.0, float(mainshock_mag) - decay),
        time_hours=window.midpoint_hours,
        mainshock_mag=mainshock_mag,
    )


def format_prediction_time(mainshock_time: pd.Timestamp, time_hours: float) -> str:
    timestamp = pd.Timestamp(mainshock_time)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    predicted = timestamp + pd.Timedelta(hours=int(round(float(time_hours))))
    return predicted.strftime("%Y%m%d%H")


def format_qualification_line(
    mainshock: pd.Series | dict,
    window_name: str,
    mag: float,
    time_hours: float,
    magnitude_type: str = "Ms",
) -> str:
    safe_mag, safe_time = clamp_prediction_to_window(
        window_name,
        mag=mag,
        time_hours=time_hours,
        mainshock_mag=float(mainshock["mag"]),
    )
    token = mainshock_token(mainshock)
    pred_time = format_prediction_time(pd.Timestamp(mainshock["time"]), safe_time)
    return (
        f"{token} "
        f"{float(mainshock['longitude']):.2f} "
        f"{float(mainshock['latitude']):.2f} "
        f"{float(mainshock['mag']):.1f} "
        f"{safe_mag:.1f} ({magnitude_type}) "
        f"{pred_time}"
    )


def write_qualification_prediction_files(
    output_dir: Path,
    mainshock: pd.Series | dict,
    predictions: dict[str, tuple[float, float]],
    magnitude_type: str = "Ms",
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    token = mainshock_token(mainshock)

    t1_t2_lines = [
        format_qualification_line(mainshock, name, *predictions[name], magnitude_type=magnitude_type)
        for name in ("T1", "T2")
    ]
    t3_lines = [
        format_qualification_line(mainshock, "T3", *predictions["T3"], magnitude_type=magnitude_type)
    ]

    t1_t2_path = output_dir / f"{token}-T1-T2.csv"
    t3_path = output_dir / f"{token}-T3.csv"
    t1_t2_path.write_text("\n".join(t1_t2_lines) + "\n", encoding="utf-8")
    t3_path.write_text("\n".join(t3_lines) + "\n", encoding="utf-8")
    return [t1_t2_path, t3_path]


def iter_prediction_files(predictions_dir: Path) -> Iterable[Path]:
    yield from sorted(predictions_dir.glob("*-T1-T2.csv"))
    yield from sorted(predictions_dir.glob("*-T3.csv"))
