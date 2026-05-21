from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


SCRIPT_MAP = {
    "analyze-transformer": PROJECT_ROOT / "scripts" / "analyze_transformer.py",
    "build-features": PROJECT_ROOT / "scripts" / "build_features.py",
    "build-qualification-labels": PROJECT_ROOT / "scripts" / "build_qualification_labels.py",
    "build-sequences": PROJECT_ROOT / "src" / "data_loader.py",
    "download-full-catalog": PROJECT_ROOT / "scripts" / "download_full_catalog.py",
    "download-gcmt": PROJECT_ROOT / "scripts" / "download_gcmt.py",
    "download-pb2002": PROJECT_ROOT / "scripts" / "download_pb2002.py",
    "download-usgs": PROJECT_ROOT / "scripts" / "download_usgs.py",
    "generate-experiment-report": PROJECT_ROOT / "scripts" / "generate_experiment_report.py",
    "make-qualification-package": PROJECT_ROOT / "scripts" / "make_qualification_package.py",
    "make-submission": PROJECT_ROOT / "scripts" / "make_submission.py",
    "mock-evaluation": PROJECT_ROOT / "scripts" / "mock_evaluation.py",
    "train-baseline": PROJECT_ROOT / "scripts" / "train_baseline.py",
    "train-dl": PROJECT_ROOT / "scripts" / "train_dl.py",
    "train-ensemble": PROJECT_ROOT / "scripts" / "train_ensemble.py",
    "train-gnn": PROJECT_ROOT / "scripts" / "train_gnn.py",
    "train-legal-fusion": PROJECT_ROOT / "scripts" / "train_legal_fusion.py",
    "train-window-baseline": PROJECT_ROOT / "scripts" / "train_window_baseline.py",
}


def run_entry(script_path: Path, forwarded_args: list[str]) -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(script_path), *forwarded_args]
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aftershock prediction project entrypoint")
    parser.add_argument("command", choices=sorted(SCRIPT_MAP))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_entry(SCRIPT_MAP[args.command], args.args)


if __name__ == "__main__":
    main()
