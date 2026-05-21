from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_baseline import add_derived_features, build_model, select_feature_columns
from scripts.train_window_baseline import requested_model_names
from src.qualification import (
    QUALIFICATION_WINDOWS,
    WINDOW_BY_NAME,
    observation_hours_for_window,
    qualification_target_cols,
    reconstruct_legal_window_features,
)


TIME_COL = "mainshock_time"


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train legal-window qualification models with extreme-aftershock risk "
            "adjustment and OOF fusion."
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "qualification_features.csv",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--late-weight", type=float, default=2.0)
    parser.add_argument("--n-estimators", type=int, default=500)
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
        "--extreme-margin",
        type=float,
        default=1.2,
        help="Extreme flag threshold: target_mag >= mainshock_mag - margin.",
    )
    parser.add_argument(
        "--extreme-weight",
        type=float,
        default=4.0,
        help="OOF weight multiplier for extreme aftershock examples.",
    )
    parser.add_argument(
        "--ensemble-grid-step",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "models" / "qualification_legal_fusion",
    )
    return parser.parse_args()


def _rmse(error: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    if sample_weight is None:
        return float(np.sqrt(np.mean(np.square(error))))
    weight = np.asarray(sample_weight, dtype=float)
    return float(np.sqrt(np.sum(weight * np.square(error)) / np.sum(weight)))


def _asymmetric_time_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    late_weight: float,
    sample_weight: np.ndarray | None = None,
) -> float:
    error = y_pred - y_true
    weight = np.where(error > 0.0, float(late_weight), 1.0)
    if sample_weight is not None:
        weight = weight * np.asarray(sample_weight, dtype=float)
    return _rmse(error, sample_weight=weight)


def calculate_metrics(
    y_mag: np.ndarray,
    y_time: np.ndarray,
    pred_mag: np.ndarray,
    pred_time: np.ndarray,
    late_weight: float,
) -> dict[str, float]:
    mag_error = pred_mag - y_mag
    time_error = pred_time - y_time
    return {
        "mag_rmse": _rmse(mag_error),
        "mag_mae": float(np.mean(np.abs(mag_error))),
        "time_hour_rmse": _rmse(time_error),
        "time_hour_mae": float(np.mean(np.abs(time_error))),
        "time_hour_asymmetric_rmse": _asymmetric_time_rmse(
            y_time,
            pred_time,
            late_weight=late_weight,
        ),
        "time_hour_hit_rate": float(
            np.mean(np.abs(time_error) <= np.maximum(0.2 * y_time, 3.0))
        ),
    }


def positive_class_probability(model, X: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))

    probs = np.asarray(model.predict_proba(X), dtype=float)
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        return probs[:, classes.index(1)]
    if True in classes:
        return probs[:, classes.index(True)]
    if len(classes) == 1 and classes[0] in (0, False):
        return np.zeros(len(X), dtype=float)
    if len(classes) == 1 and classes[0] in (1, True):
        return np.ones(len(X), dtype=float)
    if probs.ndim == 2 and probs.shape[1] >= 2:
        return probs[:, 1]
    return np.zeros(len(X), dtype=float)


def build_risk_classifier(args: argparse.Namespace, y: pd.Series):
    if y.nunique(dropna=False) < 2:
        constant = int(y.iloc[0]) if len(y) else 0
        return DummyClassifier(strategy="constant", constant=constant)

    try:
        from lightgbm import LGBMClassifier

        params = {
            "n_estimators": min(int(args.n_estimators), 400),
            "learning_rate": float(args.learning_rate),
            "num_leaves": 31,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_lambda": 2.0,
            "random_state": int(args.seed),
            "class_weight": "balanced",
            "verbosity": -1,
        }
        if args.device == "cuda":
            params["device"] = "cuda"
        return LGBMClassifier(**params)
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=min(int(args.n_estimators), 300),
            learning_rate=float(args.learning_rate),
            random_state=int(args.seed),
        )


def fit_predict_oof_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    model_names: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    splitter = TimeSeriesSplit(n_splits=args.n_splits)
    X = df[feature_cols]
    y = df[target_cols]
    oof = {
        model_name: np.full((len(df), 2), np.nan, dtype=float)
        for model_name in model_names
    }
    records: list[dict[str, object]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
        for model_name in model_names:
            model = build_model(model_name, args)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = np.asarray(model.predict(X.iloc[valid_idx]), dtype=float)
            preds[:, 0] = np.clip(preds[:, 0], 0.0, None)
            preds[:, 1] = np.clip(preds[:, 1], 0.0, None)
            oof[model_name][valid_idx] = preds
            metrics = calculate_metrics(
                y.iloc[valid_idx, 0].to_numpy(dtype=float),
                y.iloc[valid_idx, 1].to_numpy(dtype=float),
                preds[:, 0],
                preds[:, 1],
                late_weight=args.late_weight,
            )
            records.append(
                {
                    "fold": fold_idx,
                    "model": model_name,
                    "train_size": int(len(train_idx)),
                    "valid_size": int(len(valid_idx)),
                    "valid_start": str(df.loc[valid_idx[0], TIME_COL])[:10],
                    "valid_end": str(df.loc[valid_idx[-1], TIME_COL])[:10],
                    **metrics,
                }
            )
    return oof, records


def fit_predict_oof_risk(
    df: pd.DataFrame,
    feature_cols: list[str],
    y: pd.Series,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    splitter = TimeSeriesSplit(n_splits=args.n_splits)
    X = df[feature_cols]
    oof = np.full(len(df), np.nan, dtype=float)
    records: list[dict[str, object]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
        model = build_risk_classifier(args, y.iloc[train_idx])
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        prob = positive_class_probability(model, X.iloc[valid_idx])
        oof[valid_idx] = prob
        valid_y = y.iloc[valid_idx].to_numpy(dtype=int)
        pred_y = prob >= 0.5
        tp = int(np.sum((pred_y == 1) & (valid_y == 1)))
        fp = int(np.sum((pred_y == 1) & (valid_y == 0)))
        fn = int(np.sum((pred_y == 0) & (valid_y == 1)))
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        try:
            auc = float(roc_auc_score(valid_y, prob)) if len(np.unique(valid_y)) > 1 else float("nan")
        except ValueError:
            auc = float("nan")
        records.append(
            {
                "fold": fold_idx,
                "model": "extreme_risk",
                "positive_rate": float(np.mean(valid_y)),
                "prob_mean": float(np.mean(prob)),
                "precision_at_0_5": float(precision),
                "recall_at_0_5": float(recall),
                "roc_auc": auc,
            }
        )
    return oof, records


def train_final_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    model_names: list[str],
    args: argparse.Namespace,
) -> dict[str, object]:
    models: dict[str, object] = {}
    for model_name in model_names:
        model = build_model(model_name, args)
        model.fit(df[feature_cols], df[target_cols])
        models[model_name] = model
    return models


def train_final_risk_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    y: pd.Series,
    args: argparse.Namespace,
):
    model = build_risk_classifier(args, y)
    model.fit(df[feature_cols], y)
    return model


def search_model_weights(
    y_true: np.ndarray,
    pred_map: dict[str, np.ndarray],
    objective: str,
    args: argparse.Namespace,
    sample_weight: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray, float]:
    names = list(pred_map)
    if len(names) == 1:
        return {names[0]: 1.0}, pred_map[names[0]], 0.0
    if len(names) != 2:
        weight = 1.0 / len(names)
        pred = sum(pred_map[name] * weight for name in names)
        return {name: weight for name in names}, pred, float("nan")

    best_score = float("inf")
    best_weight = 0.5
    best_pred = pred_map[names[0]]
    grid = np.arange(0.0, 1.0 + args.ensemble_grid_step / 2.0, args.ensemble_grid_step)
    for weight_a in grid:
        pred = weight_a * pred_map[names[0]] + (1.0 - weight_a) * pred_map[names[1]]
        if objective == "mag":
            score = _rmse(pred - y_true, sample_weight=sample_weight)
        else:
            score = _asymmetric_time_rmse(
                y_true,
                pred,
                late_weight=args.late_weight,
                sample_weight=sample_weight,
            )
        if score < best_score:
            best_score = score
            best_weight = float(weight_a)
            best_pred = pred
    return {names[0]: best_weight, names[1]: 1.0 - best_weight}, best_pred, best_score


def search_risk_mag_adjustment(
    base_pred: np.ndarray,
    main_mag: np.ndarray,
    risk_prob: np.ndarray,
    y_true: np.ndarray,
    extreme_flag: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    sample_weight = np.where(extreme_flag, float(args.extreme_weight), 1.0)
    best_pred = base_pred.copy()
    best_score = _rmse(best_pred - y_true, sample_weight=sample_weight)
    best_params: dict[str, float | bool] = {
        "enabled": False,
        "objective": best_score,
    }

    risk_values = risk_prob[np.isfinite(risk_prob)]
    if len(risk_values) == 0:
        return best_pred, best_params
    thresholds = sorted(
        set(
            [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
            + [float(np.quantile(risk_values, q)) for q in (0.60, 0.70, 0.80, 0.90)]
        )
    )
    margins = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    weights = np.arange(0.1, 1.0 + 1e-9, 0.1)

    for threshold in thresholds:
        active = risk_prob >= threshold
        if not active.any():
            continue
        for margin in margins:
            risk_floor = np.clip(main_mag - margin, a_min=0.0, a_max=None)
            raised = np.maximum(base_pred, risk_floor)
            for weight in weights:
                candidate = base_pred.copy()
                candidate[active] = (
                    (1.0 - weight) * base_pred[active] + weight * raised[active]
                )
                candidate = np.minimum(candidate, main_mag + 0.5)
                score = _rmse(candidate - y_true, sample_weight=sample_weight)
                if score < best_score:
                    best_score = score
                    best_pred = candidate
                    best_params = {
                        "enabled": True,
                        "threshold": float(threshold),
                        "margin": float(margin),
                        "weight": float(weight),
                        "objective": float(score),
                        "active_rate": float(np.mean(active)),
                    }
    return best_pred, best_params


def search_risk_time_adjustment(
    base_pred: np.ndarray,
    risk_prob: np.ndarray,
    y_true: np.ndarray,
    extreme_flag: np.ndarray,
    window_name: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    sample_weight = np.where(extreme_flag, float(args.extreme_weight), 1.0)
    window = WINDOW_BY_NAME[window_name]
    best_pred = base_pred.copy()
    best_score = _asymmetric_time_rmse(
        y_true,
        best_pred,
        late_weight=args.late_weight,
        sample_weight=sample_weight,
    )
    best_params: dict[str, float | bool] = {
        "enabled": False,
        "objective": best_score,
    }

    risk_values = risk_prob[np.isfinite(risk_prob)]
    if len(risk_values) == 0:
        return best_pred, best_params
    thresholds = sorted(
        set(
            [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
            + [float(np.quantile(risk_values, q)) for q in (0.60, 0.70, 0.80, 0.90)]
        )
    )
    fractions = [0.05, 0.10, 0.20, 0.35, 0.50]
    weights = np.arange(0.1, 1.0 + 1e-9, 0.1)

    for threshold in thresholds:
        active = risk_prob >= threshold
        if not active.any():
            continue
        for fraction in fractions:
            anchor = window.lower_hours + (window.upper_hours - window.lower_hours) * fraction
            early = np.minimum(base_pred, anchor)
            for weight in weights:
                candidate = base_pred.copy()
                candidate[active] = (
                    (1.0 - weight) * base_pred[active] + weight * early[active]
                )
                candidate = np.clip(
                    candidate,
                    window.lower_hours + 1e-6,
                    window.upper_hours,
                )
                score = _asymmetric_time_rmse(
                    y_true,
                    candidate,
                    late_weight=args.late_weight,
                    sample_weight=sample_weight,
                )
                if score < best_score:
                    best_score = score
                    best_pred = candidate
                    best_params = {
                        "enabled": True,
                        "threshold": float(threshold),
                        "fraction": float(fraction),
                        "weight": float(weight),
                        "objective": float(score),
                        "active_rate": float(np.mean(active)),
                    }
    return best_pred, best_params


def apply_risk_mag_adjustment(
    base_pred: np.ndarray,
    main_mag: np.ndarray,
    risk_prob: np.ndarray,
    params: dict[str, object],
) -> np.ndarray:
    if not params.get("enabled"):
        return base_pred
    threshold = float(params["threshold"])
    margin = float(params["margin"])
    weight = float(params["weight"])
    active = risk_prob >= threshold
    result = base_pred.copy()
    floor = np.clip(main_mag - margin, a_min=0.0, a_max=None)
    raised = np.maximum(base_pred, floor)
    result[active] = (1.0 - weight) * base_pred[active] + weight * raised[active]
    return np.minimum(result, main_mag + 0.5)


def apply_risk_time_adjustment(
    base_pred: np.ndarray,
    risk_prob: np.ndarray,
    params: dict[str, object],
    window_name: str,
) -> np.ndarray:
    if not params.get("enabled"):
        return base_pred
    window = WINDOW_BY_NAME[window_name]
    threshold = float(params["threshold"])
    fraction = float(params["fraction"])
    weight = float(params["weight"])
    active = risk_prob >= threshold
    result = base_pred.copy()
    anchor = window.lower_hours + (window.upper_hours - window.lower_hours) * fraction
    early = np.minimum(base_pred, anchor)
    result[active] = (1.0 - weight) * base_pred[active] + weight * early[active]
    return np.clip(result, window.lower_hours + 1e-6, window.upper_hours)


def risk_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(prob)
    y = y_true[valid].astype(int)
    p = prob[valid]
    if len(y) == 0:
        return {}
    pred = p >= 0.5
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    try:
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    return {
        "positive_rate": float(np.mean(y)),
        "prob_mean": float(np.mean(p)),
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "roc_auc": auc,
    }


def main() -> None:
    args = parse_args()
    data_path = resolve_project_path(args.data)
    save_dir = resolve_project_path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    model_names = requested_model_names(args.model_type)
    artifact: dict[str, object] = {
        "artifact_type": "qualification_legal_fusion_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": str(data_path),
        "target_unit": "hours_since_mainshock",
        "extreme_margin": float(args.extreme_margin),
        "extreme_weight": float(args.extreme_weight),
        "windows": {},
    }
    metrics_payload: dict[str, object] = {
        "created_at": artifact["created_at"],
        "artifact_type": artifact["artifact_type"],
        "data": str(data_path),
        "models": model_names,
        "extreme_margin": float(args.extreme_margin),
        "extreme_weight": float(args.extreme_weight),
        "window_metrics": {},
    }
    all_fold_records: list[dict[str, object]] = []
    all_oof_rows: list[pd.DataFrame] = []

    for window in QUALIFICATION_WINDOWS:
        legal_df = reconstruct_legal_window_features(df, window.name)
        legal_df[TIME_COL] = df[TIME_COL]
        legal_df = legal_df.dropna(subset=[TIME_COL, window.mag_col, window.time_col]).reset_index(drop=True)
        feature_cols = select_feature_columns(legal_df)
        if not feature_cols:
            raise ValueError(f"No legal feature columns found for {window.name}.")

        target_cols = [window.mag_col, window.time_col]
        oof_by_model, model_records = fit_predict_oof_models(
            legal_df,
            feature_cols,
            target_cols,
            model_names,
            args,
        )
        all_fold_records.extend({"window": window.name, **record} for record in model_records)

        y_mag = legal_df[window.mag_col].to_numpy(dtype=float)
        y_time = legal_df[window.time_col].to_numpy(dtype=float)
        main_mag = legal_df["mainshock_mag"].to_numpy(dtype=float)
        extreme_flag = (y_mag > 0.0) & (y_mag >= main_mag - float(args.extreme_margin))
        extreme_y = pd.Series(extreme_flag.astype(int), index=legal_df.index)

        risk_oof, risk_records = fit_predict_oof_risk(
            legal_df,
            feature_cols,
            extreme_y,
            args,
        )
        all_fold_records.extend({"window": window.name, **record} for record in risk_records)

        valid_mask = np.isfinite(risk_oof)
        for preds in oof_by_model.values():
            valid_mask &= np.isfinite(preds).all(axis=1)
        if not valid_mask.any():
            raise RuntimeError(f"No valid OOF rows for {window.name}.")

        mag_pred_map = {
            name: preds[valid_mask, 0] for name, preds in oof_by_model.items()
        }
        time_pred_map = {
            name: preds[valid_mask, 1] for name, preds in oof_by_model.items()
        }
        sample_weight = np.where(extreme_flag[valid_mask], float(args.extreme_weight), 1.0)
        mag_weights, base_mag_pred, _ = search_model_weights(
            y_mag[valid_mask],
            mag_pred_map,
            objective="mag",
            args=args,
            sample_weight=sample_weight,
        )
        time_weights, base_time_pred, _ = search_model_weights(
            y_time[valid_mask],
            time_pred_map,
            objective="time",
            args=args,
            sample_weight=sample_weight,
        )
        adjusted_mag_pred, mag_adjustment = search_risk_mag_adjustment(
            base_mag_pred,
            main_mag[valid_mask],
            risk_oof[valid_mask],
            y_mag[valid_mask],
            extreme_flag[valid_mask],
            args,
        )
        adjusted_time_pred, time_adjustment = search_risk_time_adjustment(
            base_time_pred,
            risk_oof[valid_mask],
            y_time[valid_mask],
            extreme_flag[valid_mask],
            window.name,
            args,
        )

        per_model_metrics = {
            name: calculate_metrics(
                y_mag[valid_mask],
                y_time[valid_mask],
                preds[valid_mask, 0],
                preds[valid_mask, 1],
                late_weight=args.late_weight,
            )
            for name, preds in oof_by_model.items()
        }
        base_metrics = calculate_metrics(
            y_mag[valid_mask],
            y_time[valid_mask],
            base_mag_pred,
            base_time_pred,
            late_weight=args.late_weight,
        )
        fused_metrics = calculate_metrics(
            y_mag[valid_mask],
            y_time[valid_mask],
            adjusted_mag_pred,
            adjusted_time_pred,
            late_weight=args.late_weight,
        )
        fused_metrics["extreme_mag_mae"] = float(
            np.mean(np.abs(adjusted_mag_pred[extreme_flag[valid_mask]] - y_mag[valid_mask][extreme_flag[valid_mask]]))
        ) if extreme_flag[valid_mask].any() else float("nan")
        fused_metrics["extreme_count"] = int(np.sum(extreme_flag[valid_mask]))

        final_models = train_final_models(
            legal_df,
            feature_cols,
            target_cols,
            model_names,
            args,
        )
        final_risk_model = train_final_risk_model(
            legal_df,
            feature_cols,
            extreme_y,
            args,
        )

        artifact["windows"][window.name] = {
            "observation_hours": observation_hours_for_window(window.name),
            "feature_cols": feature_cols,
            "models": final_models,
            "risk_model": final_risk_model,
            "weights": {
                "mag": mag_weights,
                "time": time_weights,
            },
            "risk_target": {
                "margin": float(args.extreme_margin),
                "definition": "target_max_mag >= mainshock_mag - margin",
            },
            "risk_adjustment": {
                "mag": mag_adjustment,
                "time": time_adjustment,
            },
            "metrics": {
                "models": per_model_metrics,
                "base_fusion": base_metrics,
                "legal_risk_fusion": fused_metrics,
                "risk": risk_metrics(extreme_y.to_numpy(dtype=int), risk_oof),
            },
        }
        metrics_payload["window_metrics"][window.name] = artifact["windows"][window.name]["metrics"]

        oof_rows = legal_df[["mainshock_id", TIME_COL, "mainshock_mag", window.mag_col, window.time_col]].copy()
        oof_rows["window"] = window.name
        oof_rows["extreme_flag"] = extreme_flag
        oof_rows["risk_prob"] = risk_oof
        oof_rows["base_fused_mag"] = np.nan
        oof_rows["base_fused_time"] = np.nan
        oof_rows["risk_fused_mag"] = np.nan
        oof_rows["risk_fused_time"] = np.nan
        oof_rows.loc[valid_mask, "base_fused_mag"] = base_mag_pred
        oof_rows.loc[valid_mask, "base_fused_time"] = base_time_pred
        oof_rows.loc[valid_mask, "risk_fused_mag"] = adjusted_mag_pred
        oof_rows.loc[valid_mask, "risk_fused_time"] = adjusted_time_pred
        for name, preds in oof_by_model.items():
            oof_rows[f"{name}_pred_mag"] = preds[:, 0]
            oof_rows[f"{name}_pred_time"] = preds[:, 1]
        all_oof_rows.append(oof_rows)

    model_path = save_dir / "qualification_window_models.joblib"
    metrics_path = save_dir / "qualification_window_metrics.json"
    folds_path = save_dir / "qualification_window_oof_metrics.csv"
    oof_path = save_dir / "qualification_window_oof_predictions.csv"

    joblib.dump(artifact, model_path)
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(all_fold_records).to_csv(folds_path, index=False, encoding="utf-8")
    pd.concat(all_oof_rows, ignore_index=True).to_csv(oof_path, index=False, encoding="utf-8")

    print(f"Legal fusion artifact saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    print(f"Fold metrics saved: {folds_path}")
    print(f"OOF predictions saved: {oof_path}")


if __name__ == "__main__":
    main()
