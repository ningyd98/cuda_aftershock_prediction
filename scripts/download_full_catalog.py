from __future__ import annotations

import argparse
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fetch_usgs_catalog(
    start_year: int,
    end_year: int,
    min_mag: float = 4.5,
    max_depth: float = 70.0,
    request_sleep_seconds: float = 1.5,
) -> pd.DataFrame:
    """通过 USGS API 按年拉取全球目录，用于余震特征提取。"""
    base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    all_frames: list[pd.DataFrame] = []

    print(f"开始获取 {start_year}-{end_year} Mw≥{min_mag} 全球地震目录")
    print(f"数据用于余震序列特征提取（G-R b值、大森定律等）\n")

    for year in range(start_year, end_year + 1):
        print(f"  [{year}] 正在拉取 ...", end=" ", flush=True)
        params = {
            "format": "csv",
            "starttime": f"{year}-01-01",
            "endtime": f"{year}-12-31",
            "minmagnitude": min_mag,
            "maxdepth": max_depth,
            "orderby": "time",
        }
        try:
            resp = requests.get(base_url, params=params, timeout=60)
            resp.raise_for_status()
            df_year = pd.read_csv(StringIO(resp.text))
            all_frames.append(df_year)
            print(f"成功, {len(df_year)} 条")
        except requests.RequestException as exc:
            print(f"失败: {exc}")
        time.sleep(request_sleep_seconds)

    if not all_frames:
        raise RuntimeError("未获取到任何数据，请检查网络或 USGS API 可用性。")

    final_df = pd.concat(all_frames, ignore_index=True)
    final_df["time"] = pd.to_datetime(final_df["time"], utc=True, errors="coerce")
    final_df = final_df.dropna(subset=["time", "latitude", "longitude", "mag", "depth"])
    final_df = final_df.sort_values("time").reset_index(drop=True)
    return final_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载 USGS 完整地震目录用于余震特征提取")
    parser.add_argument("--start-year", type=int, default=1970)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--min-mag", type=float, default=4.5)
    parser.add_argument("--max-depth", type=float, default=70.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = fetch_usgs_catalog(
        start_year=args.start_year,
        end_year=args.end_year,
        min_mag=args.min_mag,
        max_depth=args.max_depth,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fname = (
        f"USGS_Mw{args.min_mag}_Depth{args.max_depth:g}_"
        f"{args.start_year}-{args.end_year}.csv"
    )
    output_path = args.output_dir / fname
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n完整目录已保存: {output_path}")
    print(f"事件总数: {len(df)}")
    print(f"时间跨度: {df['time'].min()} ~ {df['time'].max()}")
    print(f"震级范围: {df['mag'].min():.1f} ~ {df['mag'].max():.1f}")


if __name__ == "__main__":
    main()
