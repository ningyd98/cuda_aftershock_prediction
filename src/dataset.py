from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.utils import haversine_km


DOMAIN_PRIOR_FILL_VALUES = {
    "gr_b_value": 1.0,
    "omori_p": 1.0,
    "omori_c": 0.05,
    "etas_p": 1.0,
    "etas_c": 0.05,
}


@dataclass
class SequenceBuildConfig:
    """早期余震序列构建参数。"""

    obs_days: float = 3.0
    spatial_radius_km: float = 100.0
    earth_radius_km: float = 6371.0
    max_seq_len: int = 256


@dataclass
class DatasetPreprocessors:
    """深度学习输入预处理器，可用 joblib 序列化保存。"""

    seq_scaler: RobustScaler | StandardScaler
    global_scaler: RobustScaler | StandardScaler
    global_fill_values: dict[str, float]
    global_indicator_cols: list[str]
    scaler_type: str = "robust"
    add_missing_indicators: bool = True


def _build_scaler(scaler_type: str):
    """按名称创建 sklearn scaler。"""
    normalized = scaler_type.lower()
    if normalized == "robust":
        return RobustScaler()
    if normalized == "standard":
        return StandardScaler()
    raise ValueError("scaler_type 必须为 'robust' 或 'standard'。")


class EarthquakeSequenceDataset(Dataset):
    """
    将主震样本行转换为“早期余震事件序列 + 全局手工特征”的训练样本。

    每个样本返回:
    - seq_x: (seq_len, event_feature_dim)
    - global_x: 阶段一提取的全局特征
    - y: [target_max_mag, target_time_to_max_days]
    - metadata: 主震 ID 和时间
    """

    def __init__(
        self,
        sequence_df: pd.DataFrame,
        event_catalog_df: pd.DataFrame,
        global_feature_cols: Sequence[str],
        target_cols: Sequence[str] = ("target_max_mag", "target_time_to_max_days"),
        config: SequenceBuildConfig | None = None,
        preprocessors: DatasetPreprocessors | None = None,
        fit_preprocessors: bool = True,
        scaler_type: str = "robust",
        add_missing_indicators: bool = True,
    ) -> None:
        self.sequence_df = sequence_df.copy().reset_index(drop=True)
        self.event_catalog_df = self._normalize_event_catalog(event_catalog_df)
        self.global_feature_cols = list(global_feature_cols)
        self.target_cols = list(target_cols)
        self.config = config or SequenceBuildConfig()
        self.preprocessors = preprocessors
        self.fit_preprocessors = fit_preprocessors
        self.scaler_type = scaler_type
        self.add_missing_indicators = add_missing_indicators

        self.sequence_df["mainshock_time"] = pd.to_datetime(
            self.sequence_df["mainshock_time"],
            utc=True,
            errors="coerce",
            format="mixed",
        )

        self.event_feature_cols = [
            "dt_days",
            "log_dt_days",
            "rel_x_km",
            "rel_y_km",
            "distance_km",
            "depth",
            "mag",
        ]
        self._validate_columns()
        self._initialize_preprocessors()
        self.global_x_matrix = self._build_global_matrix()
        self.global_feature_dim = int(self.global_x_matrix.shape[1])

    def __len__(self) -> int:
        return len(self.sequence_df)

    def __getitem__(self, idx: int) -> dict:
        row = self.sequence_df.iloc[idx]
        early_events = self._extract_early_events(row)
        seq_x = self._build_event_tensor(early_events, row)
        seq_x = self._transform_event_tensor(seq_x)
        global_x = self.global_x_matrix[idx]
        y = row[self.target_cols].astype(float).to_numpy(dtype=np.float32)

        return {
            "seq_x": torch.tensor(seq_x, dtype=torch.float32),
            "global_x": torch.tensor(global_x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "metadata": {
                "mainshock_id": row["mainshock_id"],
                "mainshock_time": str(row["mainshock_time"]),
            },
        }

    def _validate_columns(self) -> None:
        """检查构建 Dataset 所需的关键列。"""
        required_sequence_cols = [
            "mainshock_id",
            "mainshock_time",
            "mainshock_lat",
            "mainshock_lon",
            *self.global_feature_cols,
            *self.target_cols,
        ]
        missing_sequence_cols = [
            col for col in required_sequence_cols if col not in self.sequence_df.columns
        ]
        if missing_sequence_cols:
            raise ValueError(f"主震样本表缺少必要字段: {missing_sequence_cols}")

    def _initialize_preprocessors(self) -> None:
        """拟合或复用序列和全局特征预处理器。"""
        if self.preprocessors is not None and not self.fit_preprocessors:
            return

        seq_scaler = _build_scaler(self.scaler_type)
        seq_rows = []
        for _, row in self.sequence_df.iterrows():
            events = self._extract_early_events(row)
            seq = self._build_event_tensor(events, row)
            if len(seq):
                seq_rows.append(seq)
        seq_fit_matrix = (
            np.vstack(seq_rows)
            if seq_rows
            else np.zeros((1, len(self.event_feature_cols)), dtype=np.float32)
        )
        seq_scaler.fit(seq_fit_matrix)

        global_fill_values, indicator_cols, filled_global = self._fit_global_fill_values()
        global_scaler = _build_scaler(self.scaler_type)
        global_scaler.fit(filled_global.to_numpy(dtype=float))

        self.preprocessors = DatasetPreprocessors(
            seq_scaler=seq_scaler,
            global_scaler=global_scaler,
            global_fill_values=global_fill_values,
            global_indicator_cols=indicator_cols,
            scaler_type=self.scaler_type,
            add_missing_indicators=self.add_missing_indicators,
        )

    def _fit_global_fill_values(self) -> tuple[dict[str, float], list[str], pd.DataFrame]:
        """根据领域先验和训练集中位数确定全局特征填充值。"""
        numeric_df = self._raw_global_dataframe()
        fill_values: dict[str, float] = {}
        indicator_cols: list[str] = []

        for col in self.global_feature_cols:
            series = numeric_df[col]
            if self.add_missing_indicators and series.isna().any():
                indicator_cols.append(col)

            if col in DOMAIN_PRIOR_FILL_VALUES:
                fill_values[col] = float(DOMAIN_PRIOR_FILL_VALUES[col])
                continue

            median_value = series.median(skipna=True)
            fill_values[col] = float(median_value) if np.isfinite(median_value) else 0.0

        filled_global = numeric_df.fillna(fill_values).fillna(0.0)
        return fill_values, indicator_cols, filled_global

    def _raw_global_dataframe(self) -> pd.DataFrame:
        """抽取并数值化全局特征表，布尔值转 0/1。"""
        global_df = self.sequence_df[self.global_feature_cols].copy()
        for col in global_df.columns:
            if pd.api.types.is_bool_dtype(global_df[col]):
                global_df[col] = global_df[col].astype(int)
        return global_df.apply(pd.to_numeric, errors="coerce")

    def _build_global_matrix(self) -> np.ndarray:
        """填充、缩放全局特征，并追加 missing indicator。"""
        if self.preprocessors is None:
            raise RuntimeError("Dataset preprocessors 尚未初始化。")

        raw_global = self._raw_global_dataframe()
        filled_global = (
            raw_global
            .fillna(self.preprocessors.global_fill_values)
            .fillna(0.0)
        )
        scaled_global = self.preprocessors.global_scaler.transform(
            filled_global.to_numpy(dtype=float)
        )

        if self.preprocessors.add_missing_indicators and self.preprocessors.global_indicator_cols:
            indicators = raw_global[
                self.preprocessors.global_indicator_cols
            ].isna().astype(float).to_numpy(dtype=np.float32)
            global_matrix = np.hstack([scaled_global, indicators])
        else:
            global_matrix = scaled_global

        return np.nan_to_num(
            global_matrix,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

    def _transform_event_tensor(self, seq_x: np.ndarray) -> np.ndarray:
        """对真实事件序列逐维缩放；空序列保持 0 行。"""
        if self.preprocessors is None or len(seq_x) == 0:
            return seq_x.astype(np.float32)

        transformed = self.preprocessors.seq_scaler.transform(seq_x.astype(float))
        return np.nan_to_num(
            transformed,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

    def _normalize_event_catalog(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化事件目录，供每个主震重建早期余震序列。"""
        required_event_cols = ["time", "latitude", "longitude", "mag", "depth"]
        missing_event_cols = [col for col in required_event_cols if col not in df.columns]
        if missing_event_cols:
            raise ValueError(f"事件目录缺少必要字段: {missing_event_cols}")

        event_df = df.copy()
        event_df["time"] = pd.to_datetime(
            event_df["time"],
            utc=True,
            errors="coerce",
            format="mixed",
        )
        event_df = event_df.dropna(subset=required_event_cols)
        return event_df.sort_values("time").reset_index(drop=True)

    def _extract_early_events(self, row: pd.Series) -> pd.DataFrame:
        """截取单个主震观测窗口内、空间半径内的早期余震事件。"""
        main_time = row["mainshock_time"]
        obs_end = main_time + pd.Timedelta(days=self.config.obs_days)

        mask = (
            (self.event_catalog_df["time"] > main_time)
            & (self.event_catalog_df["time"] <= obs_end)
        )
        candidates = self.event_catalog_df.loc[mask].copy()
        if candidates.empty:
            return candidates

        candidates["distance_km"] = haversine_km(
            row["mainshock_lat"],
            row["mainshock_lon"],
            candidates["latitude"].to_numpy(),
            candidates["longitude"].to_numpy(),
            earth_radius_km=self.config.earth_radius_km,
        )
        candidates = candidates.loc[
            candidates["distance_km"] <= self.config.spatial_radius_km
        ].copy()

        return candidates.sort_values("time").head(self.config.max_seq_len)

    def _build_event_tensor(self, events: pd.DataFrame, row: pd.Series) -> np.ndarray:
        """把早期余震事件表转换为 Transformer 输入张量。"""
        if events.empty:
            return np.zeros((0, len(self.event_feature_cols)), dtype=np.float32)

        dt_days = (
            events["time"] - row["mainshock_time"]
        ).dt.total_seconds().to_numpy() / 86400.0

        lat0 = np.radians(float(row["mainshock_lat"]))
        lon0 = np.radians(float(row["mainshock_lon"]))
        lat = np.radians(events["latitude"].to_numpy(dtype=float))
        lon = np.radians(events["longitude"].to_numpy(dtype=float))

        rel_x_km = self.config.earth_radius_km * np.cos(lat0) * (lon - lon0)
        rel_y_km = self.config.earth_radius_km * (lat - lat0)

        seq = np.column_stack(
            [
                dt_days,
                np.log1p(dt_days),
                rel_x_km,
                rel_y_km,
                events["distance_km"].to_numpy(dtype=float),
                events["depth"].to_numpy(dtype=float),
                events["mag"].to_numpy(dtype=float),
            ]
        )
        return np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def earthquake_collate_fn(batch: list[dict]) -> dict:
    """
    对不等长事件序列进行 Padding，并生成 Transformer padding mask。

    seq_padding_mask 中 True 表示 padding 位置。
    """
    batch_size = len(batch)
    seq_dim = batch[0]["seq_x"].shape[-1]
    max_len = max(item["seq_x"].shape[0] for item in batch)
    max_len = max(max_len, 1)

    seq_x = torch.zeros(batch_size, max_len, seq_dim, dtype=torch.float32)
    seq_padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)

    for idx, item in enumerate(batch):
        cur_len = item["seq_x"].shape[0]
        if cur_len > 0:
            seq_x[idx, :cur_len] = item["seq_x"]
            seq_padding_mask[idx, :cur_len] = False

    return {
        "seq_x": seq_x,
        "seq_padding_mask": seq_padding_mask,
        "global_x": torch.stack([item["global_x"] for item in batch]),
        "y": torch.stack([item["y"] for item in batch]),
        "metadata": [item["metadata"] for item in batch],
    }
