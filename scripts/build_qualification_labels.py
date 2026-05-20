from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.qualification import (
    append_qualification_targets,
    build_qualification_samples_from_catalog,
    qualification_target_cols,
)


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build T1/T2/T3 qualification labels for aftershock prediction.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "USGS_Mw4.0_Depth70_1970-2023.csv",
        help="Full earthquake catalog used to extract aftershocks.",
    )
    parser.add_argument(
        "--base-features",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "advanced_features.csv",
        help="Existing per-mainshock feature table. If missing, labels are built from catalog only.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "qualification_features.csv",
        help="Output CSV with T1/T2/T3 targets.",
    )
    parser.add_argument("--radius-km", type=float, default=100.0)
    parser.add_argument("--earth-radius-km", type=float, default=6371.0)
    parser.add_argument("--min-mainshock-mag", type=float, default=6.0)
    parser.add_argument("--max-depth-km", type=float, default=70.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog_path = resolve_project_path(args.catalog)
    base_features_path = resolve_project_path(args.base_features)
    output_path = resolve_project_path(args.output)

    catalog_df = pd.read_csv(catalog_path)
    if base_features_path.exists():
        base_df = pd.read_csv(base_features_path)
        result = append_qualification_targets(
            base_df,
            catalog_df,
            spatial_radius_km=args.radius_km,
            earth_radius_km=args.earth_radius_km,
        )
        source = f"base features: {base_features_path}"
    else:
        result = build_qualification_samples_from_catalog(
            catalog_df,
            min_mainshock_mag=args.min_mainshock_mag,
            max_depth_km=args.max_depth_km,
            spatial_radius_km=args.radius_km,
            earth_radius_km=args.earth_radius_km,
        )
        source = f"catalog only: {catalog_path}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Qualification labels saved: {output_path}")
    print(f"Source: {source}")
    print(f"Rows: {len(result)}")
    print(f"Target columns: {', '.join(qualification_target_cols())}")


if __name__ == "__main__":
    main()
