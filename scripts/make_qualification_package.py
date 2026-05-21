from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.qualification import (
    QUALIFICATION_WINDOWS,
    clamp_prediction_to_window,
    mainshock_token,
    normalize_event_table,
    reconstruct_legal_window_features,
    pick_mainshock,
    rule_window_prediction,
    write_qualification_prediction_files,
)


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build qualification ZIP package with T1/T2 and T3 prediction files.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "test_sequences",
        help="Directory containing test sequence CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "submission_package",
        help="Working directory for package contents.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=PROJECT_ROOT / "qualification_submission.zip",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "models" / "qualification_window_models.joblib",
    )
    parser.add_argument(
        "--commitment-template",
        type=Path,
        default=None,
        help="Official commitment letter template/file. Required unless discoverable locally.",
    )
    parser.add_argument("--plate-boundaries", type=Path, default=PROJECT_ROOT / "data" / "raw" / "PB2002_boundaries.json")
    parser.add_argument("--gcmt-catalog", type=Path, default=PROJECT_ROOT / "data" / "raw" / "GlobalCMT_1976-2024.csv")
    parser.add_argument("--magnitude-type", type=str, default="Ms")
    parser.add_argument("--allow-rule-fallback", action="store_true")
    parser.add_argument(
        "--skip-commitment",
        action="store_true",
        help="Build an inspection package without copying the official commitment letter.",
    )
    parser.add_argument("--clean", action="store_true", help="Remove an existing output directory first.")
    return parser.parse_args()


def discover_commitment_template() -> Path | None:
    patterns = ["*commitment*", "*promise*", "*承诺*", "*承诺书*"]
    suffixes = {".doc", ".docx", ".pdf", ".txt", ".md"}
    for root in (PROJECT_ROOT / "data", PROJECT_ROOT):
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.rglob(pattern):
                if path.is_file() and path.suffix.lower() in suffixes:
                    return path
    return None


def copy_commitment(template_path: Path | None, commitment_dir: Path) -> Path:
    source = template_path if template_path is not None else discover_commitment_template()
    if source is None or not source.exists():
        raise RuntimeError(
            "Official commitment letter template was not found. "
            "Pass --commitment-template /path/to/template before building the final ZIP."
        )
    commitment_dir.mkdir(parents=True, exist_ok=True)
    dest = commitment_dir / source.name
    shutil.copy2(source, dest)
    return dest


def copy_code_snapshot(destination: Path, package_dir: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)

    excluded_dirs = {
        ".git",
        ".conda",
        "__pycache__",
        ".pytest_cache",
        "submission_package",
        package_dir.name,
    }
    excluded_suffixes = {
        ".pt",
        ".pth",
        ".joblib",
        ".pkl",
        ".zip",
        ".sqlite",
        ".db",
    }

    def ignore(dir_path: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        current = Path(dir_path)
        for name in names:
            path = current / name
            if name in excluded_dirs:
                ignored.add(name)
            elif path.is_file() and path.suffix.lower() in excluded_suffixes:
                ignored.add(name)
            elif path.is_dir() and name == "data":
                ignored.add(name)
        return ignored

    shutil.copytree(PROJECT_ROOT, destination, ignore=ignore)


def ensure_report(report_dir: Path, metrics_path: Path | None) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "experiment_report.md"
    if report_path.exists():
        return report_path

    lines = [
        "# Aftershock Qualification Experiment Report",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Submission Target",
        "",
        "This package follows the qualification format: each test mainshock emits two files, ",
        "`{YYYYMMDDhhmmss}-T1-T2.csv` and `{YYYYMMDDhhmmss}-T3.csv`, covering T1 ",
        "(0-24h), T2 (24-72h), and T3 (72-168h).",
        "",
        "## Reproduction Commands",
        "",
        "```bash",
        "python main.py build-qualification-labels",
        "python main.py train-window-baseline --model-type both --device cuda",
        "python main.py make-qualification-package --commitment-template /path/to/official_template",
        "```",
        "",
    ]
    if metrics_path is not None and metrics_path.exists():
        lines.extend(["## Metrics Artifact", "", f"Metrics file: `{metrics_path}`", ""])
    else:
        lines.extend([
            "## Metrics Artifact",
            "",
            "Full CUDA metrics were not found in this workspace when the package was generated.",
            "",
        ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def load_artifact(model_path: Path):
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def make_model_matrix_from_frame(feature_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    model_df = feature_df.copy()
    for col in feature_cols:
        if col not in model_df.columns:
            model_df[col] = np.nan
        if pd.api.types.is_bool_dtype(model_df[col]):
            model_df[col] = model_df[col].astype(int)
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    return model_df[feature_cols]


def build_feature_matrix(
    event_df: pd.DataFrame,
    args: argparse.Namespace,
    feature_cols: list[str],
    window_name: str | None = None,
    observation_hours: float | None = None,
) -> pd.DataFrame:
    from scripts.make_submission import (
        add_derived_features,
        build_single_sequence_features,
        make_model_matrix,
    )

    obs_days = 3.0 if observation_hours is None else max(float(observation_hours), 0.0) / 24.0
    feature_df, _ = build_single_sequence_features(
        event_df,
        plate_boundaries_path=resolve_project_path(args.plate_boundaries),
        gcmt_catalog_path=resolve_project_path(args.gcmt_catalog),
        obs_days=obs_days,
    )
    if window_name is not None:
        legal_df = reconstruct_legal_window_features(add_derived_features(feature_df), window_name)
        return make_model_matrix_from_frame(legal_df, feature_cols)
    return make_model_matrix(add_derived_features(feature_df), feature_cols)


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


def apply_risk_mag_adjustment(
    base_pred: float,
    mainshock_mag: float,
    risk_prob: float,
    params: dict[str, object],
) -> float:
    if not params.get("enabled"):
        return float(base_pred)
    if float(risk_prob) < float(params["threshold"]):
        return float(base_pred)
    floor = max(0.0, float(mainshock_mag) - float(params["margin"]))
    raised = max(float(base_pred), floor)
    weight = float(params["weight"])
    return min((1.0 - weight) * float(base_pred) + weight * raised, float(mainshock_mag) + 0.5)


def apply_risk_time_adjustment(
    base_pred: float,
    risk_prob: float,
    params: dict[str, object],
    window_name: str,
) -> float:
    if not params.get("enabled"):
        return float(base_pred)
    if float(risk_prob) < float(params["threshold"]):
        return float(base_pred)
    window = next(window for window in QUALIFICATION_WINDOWS if window.name == window_name)
    anchor = window.lower_hours + (window.upper_hours - window.lower_hours) * float(params["fraction"])
    early = min(float(base_pred), anchor)
    weight = float(params["weight"])
    return (1.0 - weight) * float(base_pred) + weight * early


def predict_with_artifact(
    artifact: dict,
    event_df: pd.DataFrame,
    mainshock: pd.Series,
    args: argparse.Namespace,
) -> dict[str, tuple[float, float]]:
    if artifact.get("artifact_type") == "qualification_legal_fusion_v1":
        return predict_with_legal_fusion_artifact(artifact, event_df, mainshock, args)

    feature_cols = artifact["feature_cols"]
    X = build_feature_matrix(event_df, args, feature_cols)
    predictions: dict[str, tuple[float, float]] = {}

    for window in QUALIFICATION_WINDOWS:
        payload = artifact["windows"].get(window.name, {})
        models = payload.get("models", {})
        weights = payload.get("weights", {})
        mag_weights = weights.get("mag", {})
        time_weights = weights.get("time", {})
        mag_total = 0.0
        time_total = 0.0
        mag_pred = 0.0
        time_pred = 0.0

        for name, model in models.items():
            pred = np.asarray(model.predict(X), dtype=float).reshape(1, 2)
            w_mag = max(float(mag_weights.get(name, 0.0)), 0.0)
            w_time = max(float(time_weights.get(name, 0.0)), 0.0)
            if w_mag > 0:
                mag_pred += float(pred[0, 0]) * w_mag
                mag_total += w_mag
            if w_time > 0:
                time_pred += float(pred[0, 1]) * w_time
                time_total += w_time

        if mag_total <= 0 or time_total <= 0:
            predictions[window.name] = rule_window_prediction(window.name, float(mainshock["mag"]))
            continue

        predictions[window.name] = clamp_prediction_to_window(
            window.name,
            mag=mag_pred / mag_total,
            time_hours=time_pred / time_total,
            mainshock_mag=float(mainshock["mag"]),
        )
    return predictions


def predict_with_legal_fusion_artifact(
    artifact: dict,
    event_df: pd.DataFrame,
    mainshock: pd.Series,
    args: argparse.Namespace,
) -> dict[str, tuple[float, float]]:
    predictions: dict[str, tuple[float, float]] = {}
    for window in QUALIFICATION_WINDOWS:
        payload = artifact["windows"].get(window.name, {})
        feature_cols = payload.get("feature_cols", [])
        X = build_feature_matrix(
            event_df,
            args,
            feature_cols,
            window_name=window.name,
            observation_hours=payload.get("observation_hours", 72.0),
        )
        models = payload.get("models", {})
        weights = payload.get("weights", {})
        mag_weights = weights.get("mag", {})
        time_weights = weights.get("time", {})
        mag_total = 0.0
        time_total = 0.0
        mag_pred = 0.0
        time_pred = 0.0

        for name, model in models.items():
            pred = np.asarray(model.predict(X), dtype=float).reshape(1, 2)
            w_mag = max(float(mag_weights.get(name, 0.0)), 0.0)
            w_time = max(float(time_weights.get(name, 0.0)), 0.0)
            if w_mag > 0:
                mag_pred += float(pred[0, 0]) * w_mag
                mag_total += w_mag
            if w_time > 0:
                time_pred += float(pred[0, 1]) * w_time
                time_total += w_time

        if mag_total <= 0 or time_total <= 0:
            predictions[window.name] = rule_window_prediction(window.name, float(mainshock["mag"]))
            continue

        mag_pred = mag_pred / mag_total
        time_pred = time_pred / time_total
        risk_model = payload.get("risk_model")
        if risk_model is not None:
            risk_prob = float(positive_class_probability(risk_model, X)[0])
            adjustment = payload.get("risk_adjustment", {})
            mag_pred = apply_risk_mag_adjustment(
                mag_pred,
                float(mainshock["mag"]),
                risk_prob,
                adjustment.get("mag", {}),
            )
            time_pred = apply_risk_time_adjustment(
                time_pred,
                risk_prob,
                adjustment.get("time", {}),
                window.name,
            )

        predictions[window.name] = clamp_prediction_to_window(
            window.name,
            mag=mag_pred,
            time_hours=time_pred,
            mainshock_mag=float(mainshock["mag"]),
        )
    return predictions


def rule_predictions(mainshock: pd.Series) -> dict[str, tuple[float, float]]:
    return {
        window.name: rule_window_prediction(window.name, float(mainshock["mag"]))
        for window in QUALIFICATION_WINDOWS
    }


def zip_directory(source_dir: Path, zip_path: Path) -> list[str]:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                arcname = path.relative_to(source_dir).as_posix()
                files.append(arcname)
                zf.write(path, arcname)
    return files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    input_dir = resolve_project_path(args.input_dir)
    output_dir = resolve_project_path(args.output_dir)
    zip_path = resolve_project_path(args.zip_path)
    model_path = resolve_project_path(args.model_path)
    commitment_template = (
        resolve_project_path(args.commitment_template)
        if args.commitment_template is not None
        else None
    )

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    predictions_dir = output_dir / "predictions"
    technical_docs_dir = output_dir / "technical_docs"
    commitment_dir = output_dir / "commitment"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    technical_docs_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV test sequences found in {input_dir}")

    artifact = load_artifact(model_path)
    if artifact is None and not args.allow_rule_fallback:
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. "
            "Train it first or pass --allow-rule-fallback for a non-final smoke package."
        )

    summary_rows: list[dict[str, object]] = []
    for csv_file in csv_files:
        event_df = normalize_event_table(pd.read_csv(csv_file))
        mainshock = pick_mainshock(event_df)
        if artifact is None:
            predictions = rule_predictions(mainshock)
            source = "rule_fallback"
        else:
            try:
                predictions = predict_with_artifact(artifact, event_df, mainshock, args)
                source = "qualification_window_models"
            except Exception:
                if not args.allow_rule_fallback:
                    raise
                predictions = rule_predictions(mainshock)
                source = "rule_fallback_after_model_error"

        written = write_qualification_prediction_files(
            predictions_dir,
            mainshock,
            predictions,
            magnitude_type=args.magnitude_type,
        )
        summary_rows.append(
            {
                "sequence": str(csv_file),
                "mainshock_id": mainshock_token(mainshock),
                "source": source,
                "files": [path.name for path in written],
            }
        )

    if args.skip_commitment:
        commitment_dir.mkdir(parents=True, exist_ok=True)
        commitment_file = None
    else:
        commitment_file = copy_commitment(commitment_template, commitment_dir)
    metrics_path = model_path.parent / "qualification_window_metrics.json"
    report_path = ensure_report(technical_docs_dir, metrics_path if metrics_path.exists() else None)
    copy_code_snapshot(technical_docs_dir / "code", output_dir)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prediction_file_count": len(list(predictions_dir.glob("*.csv"))),
        "sequence_count": len(summary_rows),
        "commitment_file": commitment_file.name if commitment_file is not None else None,
        "commitment_skipped": bool(args.skip_commitment),
        "report_file": report_path.relative_to(output_dir).as_posix(),
        "model_artifact": str(model_path) if artifact is not None else None,
        "rows": summary_rows,
    }
    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    files = zip_directory(output_dir, zip_path)
    sha = sha256_file(zip_path)
    sha_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    sha_path.write_text(f"{sha}  {zip_path.name}\n", encoding="utf-8")

    print(f"Qualification package directory: {output_dir}")
    print(f"ZIP package: {zip_path}")
    print(f"SHA256: {sha}")
    print(f"Files in ZIP: {len(files)}")


if __name__ == "__main__":
    main()
