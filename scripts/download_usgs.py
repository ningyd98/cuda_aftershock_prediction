from __future__ import annotations

import argparse
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fetch_usgs_earthquakes(
    start_year: int,
    end_year: int,
    min_mag: float = 6.0,
    max_depth: float = 70.0,
    request_sleep_seconds: float = 1.0,
) -> pd.DataFrame:
    """通过 USGS API 获取指定年份范围内的全球浅源强震目录。"""
    base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    all_data_frames: list[pd.DataFrame] = []

    print(f"开始获取 {start_year} 年至 {end_year} 年的全球强震数据...")

    for year in range(start_year, end_year + 1):
        print(f"正在拉取 {year} 年的数据...", end=" ")
        params = {
            "format": "csv",
            "starttime": f"{year}-01-01",
            "endtime": f"{year}-12-31",
            "minmagnitude": min_mag,
            "maxdepth": max_depth,
            "orderby": "time",
        }

        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            df_year = pd.read_csv(StringIO(response.text))
            all_data_frames.append(df_year)
            print(f"成功！获取到 {len(df_year)} 条记录。")
        except requests.RequestException as exc:
            print(f"请求失败: {exc}")

        time.sleep(request_sleep_seconds)

    if not all_data_frames:
        print("未获取到任何数据。")
        return pd.DataFrame()

    final_df = pd.concat(all_data_frames, ignore_index=True)
    final_df["time"] = pd.to_datetime(final_df["time"], utc=True, errors="coerce")
    return final_df


def parse_args() -> argparse.Namespace:
    """解析下载参数。"""
    parser = argparse.ArgumentParser(description="下载 USGS 全球浅源强震目录")
    parser.add_argument("--start-year", type=int, default=1970)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--min-mag", type=float, default=6.0)
    parser.add_argument("--max-depth", type=float, default=70.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="原始数据保存目录",
    )
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    df_earthquakes = fetch_usgs_earthquakes(
        start_year=args.start_year,
        end_year=args.end_year,
        min_mag=args.min_mag,
        max_depth=args.max_depth,
    )

    if df_earthquakes.empty:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        args.output_dir / f"USGS_Mw{args.min_mag}_Depth{args.max_depth:g}_"
        f"{args.start_year}-{args.end_year}.csv"
    )
    df_earthquakes.to_csv(output_path, index=False, encoding="utf-8")

    print("\n--- 数据获取完成 ---")
    print(f"总计获取地震记录数: {len(df_earthquakes)}")
    print(f"完整数据已保存: {output_path}")


if __name__ == "__main__":
    main()
