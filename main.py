from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


SCRIPT_MAP = {
    "build-sequences": PROJECT_ROOT / "src" / "data_loader.py",
    "build-features": PROJECT_ROOT / "scripts" / "build_features.py",
    "download-usgs": PROJECT_ROOT / "scripts" / "download_usgs.py",
    "download-full-catalog": PROJECT_ROOT / "scripts" / "download_full_catalog.py",
    "download-pb2002": PROJECT_ROOT / "scripts" / "download_pb2002.py",
    "download-gcmt": PROJECT_ROOT / "scripts" / "download_gcmt.py",
    "make-submission": PROJECT_ROOT / "scripts" / "make_submission.py",
    "mock-evaluation": PROJECT_ROOT / "scripts" / "mock_evaluation.py",
    "train-baseline": PROJECT_ROOT / "scripts" / "train_baseline.py",
    "train-dl": PROJECT_ROOT / "scripts" / "train_dl.py",
    "train-gnn": PROJECT_ROOT / "scripts" / "train_gnn.py",
}


def run_entry(script_path: Path, forwarded_args: list[str]) -> None:
    """把子命令参数转交给对应脚本，保持单一项目入口。"""
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(script_path), *forwarded_args]
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv


def parse_args() -> argparse.Namespace:
    """解析统一入口参数。"""
    parser = argparse.ArgumentParser(description="余震预测项目统一入口")
    parser.add_argument("command", choices=sorted(SCRIPT_MAP))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    """统一调度数据下载、基础样本构建与高级特征生成。"""
    args = parse_args()
    run_entry(SCRIPT_MAP[args.command], args.args)


if __name__ == "__main__":
    main()
