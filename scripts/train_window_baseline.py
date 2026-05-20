from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_baseline import add_derived_features, build_model, select_feature_columns
from src.qualification import QUALIFICATION_WINDOWS, qualification_target_cols


TIME_COL = "mainshock_time"


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train independent T1/T2/T3 tree baselines for qualification submission.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "qualification_features.csv",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--late-weight", type=float, default=2.0)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument(
        "--model-type",
        choices=["lightgbm", "xgboost", "both"],
        default="both",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="cuda",
    )
    parser.add_argument("--gpu-use-dp", action="store_true")
    parser.add_argument("--use-asymmetric-time-objective", action="store_true")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "models",
    )
    return parser.parse_args()


def requested_model_names(model_type: str) -> list[str]:
    if model_type == "lightgbm":
        return ["baseline"]
    if model_type == "xgboost":
        return ["xgboost"]
    return ["baseline", "xgboost"]


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error))))


def calculate_window_metrics(y_true: pd.DataFrame, preds: np.ndarray) -> dict[str, float]:
    mag_error = preds[:, 0] - y_true.iloc[:, 0].to_numpy(dtype=float)
    time_error = preds[:, 1] - y_true.iloc[:, 1].to_numpy(dtype=float)
    late_weights = np.where(time_error > 0.0, 2.0, 1.0)
    return {
        "mag_rmse": _rmse(mag_error),
        "mag_mae": float(np.mean(np.abs(mag_error))),
        "time_hour_rmse": _rmse(time_error),
        "time_hour_mae": float(np.mean(np.abs(time_error))),
        "time_hour_asymmetric_rmse": float(
            np.sqrt(np.mean(late_weights * np.square(time_error)))
        ),
        "time_hour_hit_rate": float(
            np.mean(np.abs(time_error) <= np.maximum(0.2 * y_true.iloc[:, 1], 3.0))
        ),
    }


def fit_predict_oof(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    model_name: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    train_df = df.sort_values(TIME_COL).reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=args.n_splits)
    X = train_df[feature_cols]
    y = train_df[target_cols]
    oof = np.full((len(train_df), 2), np.nan, dtype=float)
    records: list[dict[str, object]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
        model = build_model(model_name, args)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = np.asarray(model.predict(X.iloc[valid_idx]), dtype=float)
        preds[:, 0] = np.clip(preds[:, 0], 0.0, None)
        preds[:, 1] = np.clip(preds[:, 1], 0.0, None)
        oof[valid_idx] = preds
        metrics = calculate_window_metrics(y.iloc[valid_idx], preds)
        records.append(
            {
                "fold": fold_idx,
                "model": model_name,
                "train_size": int(len(train_idx)),
                "valid_size": int(len(valid_idx)),
                "valid_start": str(train_df.loc[valid_idx[0], TIME_COL])[:10],
                "valid_end": str(train_df.loc[valid_idx[-1], TIME_COL])[:10],
                **metrics,
            }
        )
    return oof, records


def train_final_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    model_name: str,
    args: argparse.Namespace,
):
    model = build_model(model_name, args)
    model.fit(df[feature_cols], df[target_cols])
    return model


def main() -> None:
    args = parse_args()
    data_path = resolve_project_path(args.data)
    save_dir = resolve_project_path(args.save_dir)
    df = pd.read_csv(data_path)

    missing_targets = [col for col in qualification_target_cols() if col not in df.columns]
    if missing_targets:
        raise ValueError(
            "Qualification target columns are missing. Run build-qualification-labels first: "
            + ", ".join(missing_targets)
        )

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce", format="mixed")
    df = add_derived_features(df)
    df = df.dropna(subset=[TIME_COL, *qualification_target_cols()]).reset_index(drop=True)
    feature_cols = select_feature_columns(df)
    if not feature_cols:
        raise ValueError("No usable numeric feature columns found.")

    model_names = requested_model_names(args.model_type)
    save_dir.mkdir(parents=True, exist_ok=True)
    artifact: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": str(data_path),
        "feature_cols": feature_cols,
        "target_unit": "hours_since_mainshock",
        "windows": {},
    }
    all_records: list[dict[str, object]] = []

    for window in QUALIFICATION_WINDOWS:
        target_cols = [window.mag_col, window.time_col]
        window_models: dict[str, object] = {}
        window_metrics: dict[str, dict[str, float]] = {}
        for model_name in model_names:
            oof, records = fit_predict_oof(df, feature_cols, target_cols, model_name, args)
            all_records.extend({"window": window.name, **record} for record in records)
            valid_mask = np.isfinite(oof).all(axis=1)
            if valid_mask.any():
                window_metrics[model_name] = calculate_window_metrics(
                    df.loc[valid_mask, target_cols],
                    oof[valid_mask],
                )
            window_models[model_name] = train_final_model(
                df,
                feature_cols,
                target_cols,
                model_name,
                args,
            )
        artifact["windows"][window.name] = {
            "models": window_models,
            "metrics": window_metrics,
            "weights": {
                "mag": {name: 1.0 / len(model_names) for name in model_names},
                "time": {name: 1.0 / len(model_names) for name in model_names},
            },
        }

    model_path = save_dir / "qualification_window_models.joblib"
    metrics_path = save_dir / "qualification_window_metrics.json"
    folds_path = save_dir / "qualification_window_oof_metrics.csv"

    joblib.dump(artifact, model_path)
    metrics_payload = {
        "created_at": artifact["created_at"],
        "data": str(data_path),
        "feature_count": len(feature_cols),
        "models": model_names,
        "window_metrics": {
            window: payload["metrics"]
            for window, payload in artifact["windows"].items()
        },
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(all_records).to_csv(folds_path, index=False, encoding="utf-8")

    print(f"Qualification model artifact saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    print(f"Fold metrics saved: {folds_path}")


if __name__ == "__main__":
    main()
