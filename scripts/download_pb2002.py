from __future__ import annotations

import argparse
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PB2002_URL = (
    "https://raw.githubusercontent.com/fraxen/tectonicplates/master/"
    "GeoJSON/PB2002_boundaries.json"
)


def download_bird_plate_boundaries(output_path: str | Path) -> Path:
    """下载 Peter Bird (2003) 全球板块边界 GeoJSON 数据。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("正在下载 Peter Bird 板块边界数据...")
    print(f"数据源链接: {PB2002_URL}")

    response = requests.get(PB2002_URL, timeout=30)
    response.raise_for_status()
    output_path.write_bytes(response.content)

    print("下载成功！")
    print(f"文件大小: {len(response.content) / 1024:.2f} KB")
    print(f"保存路径: {output_path.resolve()}")
    return output_path


def parse_args() -> argparse.Namespace:
    """解析下载参数。"""
    parser = argparse.ArgumentParser(description="下载 PB2002 全球板块边界数据")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "PB2002_boundaries.json",
        help="GeoJSON 输出路径",
    )
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    download_bird_plate_boundaries(args.output)


if __name__ == "__main__":
    main()
