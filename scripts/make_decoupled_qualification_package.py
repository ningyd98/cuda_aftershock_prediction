"""Decoupled 资格赛提交打包脚本。

读取 qualification_decoupled_models.joblib，对 test_sequences 预测，
生成符合资格赛格式的 predictions/ + technical_docs/ + MANIFEST.json，并打 ZIP。

校准三态控制：
- 默认 / --mag-calibration artifact：使用 artifact 内置 postprocessing
- --mag-calibration none：关闭 artifact 校准，不加 CLI 校准
- --mag-calibration T1:-0.1,T2:0.1,T3:0.1：不用 artifact 校准，改用 CLI
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_qualification_package import (
    copy_code_snapshot,
    load_artifact,
    sha256_file,
    zip_directory,
    build_feature_matrix,
)
from src.qualification import (
    QUALIFICATION_WINDOWS,
    clamp_prediction_to_window,
    mainshock_token,
    normalize_event_table,
    pick_mainshock,
    rule_window_prediction,
    write_qualification_prediction_files,
)
from scripts.oof_fusion import apply_fusion, normalize_weights
from src.time_buckets import (
    align_bucket_probabilities, expected_time_from_bucket_probs,
    safe_extreme_probability,
)


# ---------------------------------------------------------------------------
# 帮助函数
# ---------------------------------------------------------------------------

def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build decoupled qualification ZIP package.",
    )
    parser.add_argument("--input-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "test_sequences")
    parser.add_argument("--model-path", type=Path,
                        default=PROJECT_ROOT / "data" / "models" / "qualification_decoupled"
                        / "qualification_decoupled_models.joblib")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "submission_package_decoupled")
    parser.add_argument("--zip-path", type=Path,
                        default=PROJECT_ROOT / "qualification_submission_decoupled.zip")
    parser.add_argument("--commitment-template", type=Path, default=None)
    parser.add_argument("--skip-commitment", action="store_true")
    parser.add_argument("--plate-boundaries", type=Path,
                        default=PROJECT_ROOT / "data" / "raw" / "PB2002_boundaries.json")
    parser.add_argument("--gcmt-catalog", type=Path,
                        default=PROJECT_ROOT / "data" / "raw" / "GlobalCMT_1976-2024.csv")
    parser.add_argument("--magnitude-type", type=str, default="Ms")
    parser.add_argument(
        "--mag-calibration", type=str, default="artifact",
        help="'artifact' (use built-in, default), 'none' (disable), "
             "or 'T1:-0.1,T2:0.1,T3:0.1' (CLI override).",
    )
    parser.add_argument(
        "--time-calibration-hours", type=str, default="artifact",
        help="'artifact', 'none', or 'T1:0,T2:-2,T3:9'.",
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--time-mae-guard", type=str, default="true",
        help="Enforce T2/T3 time calibration guard (max shift ratio 1.3). "
             "'true' (default), 'false' to disable.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 校准语义：三态解析
# ---------------------------------------------------------------------------

def _parse_calibration(value: str | None) -> tuple[str, dict[str, float]]:
    """返回 (mode, values)。
    mode ∈ {"artifact", "none", "override"}
    - "artifact": 使用 artifact 内置校准
    - "none": 不使用任何校准
    - "override": 使用 CLI 指定的 values
    """
    if value is None:
        return ("artifact", {})
    v = value.strip().lower()
    if v in {"", "artifact"}:
        return ("artifact", {})
    if v in {"none", "off", "false", "0"}:
        return ("none", {})
    # 否则解析为显式 dict
    result: dict[str, float] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError(f"Invalid calibration item: {item!r}")
        wn, raw = item.split(":", 1)
        result[wn.strip().upper()] = float(raw)
    return ("override", result)


# ---------------------------------------------------------------------------
# 预测核心
# ---------------------------------------------------------------------------

def predict_with_decoupled_artifact(
    artifact: dict,
    event_df: pd.DataFrame,
    mainshock: pd.Series,
    args: argparse.Namespace,
    *,
    use_artifact_mag: bool = True,
    use_artifact_time: bool = True,
) -> dict[str, tuple[float, float]]:
    """使用 decoupled artifact 做多窗口预测 (v2: 含时间校准约束)。

    Args:
        use_artifact_mag: 是否应用 artifact 震级 bias
        use_artifact_time: 是否应用 artifact 时间 bias
    """
    predictions: dict[str, tuple[float, float]] = {}
    postproc_global = artifact.get("postprocessing", {})
    # 解析时间 MAE guard 开关
    time_mae_guard_enabled = getattr(args, "time_mae_guard", "true").lower() in ("true", "1", "yes")

    for window in QUALIFICATION_WINDOWS:
        wn = window.name
        payload = artifact["windows"].get(wn, {})
        if not payload:
            predictions[wn] = rule_window_prediction(wn, float(mainshock["mag"]))
            continue

        feature_cols = payload.get("feature_cols", [])
        X = build_feature_matrix(
            event_df, args, feature_cols,
            window_name=wn,
            observation_hours=payload.get("observation_hours", 72.0),
        )

        mag_models = payload.get("mag_models", {})
        bucket_model = payload.get("bucket_model")
        extreme_model = payload.get("extreme_model")
        weights = payload.get("weights", {})
        mag_weights = weights.get("mag", {})

        # ── 震级预测（加权平均） ──
        mag_total = 0.0
        mag_pred = 0.0
        mag_preds_list = []
        for name, model in mag_models.items():
            p = float(np.asarray(model.predict(X), dtype=float).ravel()[0])
            w = float(mag_weights.get(name, 1.0 / max(1, len(mag_models))))
            mag_pred += p * w
            mag_total += w
            mag_preds_list.append(p)

        if mag_total > 0:
            mag_pred = mag_pred / mag_total
        elif mag_preds_list:
            mag_pred = float(np.mean(mag_preds_list))

        # Apply fusion weights if available in artifact
        fw_wn = payload.get("fusion", {}).get("mag", {})
        if fw_wn and mag_preds_list:
            fused = 0.0; total_w = 0.0
            for i, (n, m) in enumerate(mag_models.items()):
                w = float(fw_wn.get(n, 1.0/len(mag_models)) if len(mag_models)>0 else 1.0)
                fused += float(np.asarray(m.predict(X),dtype=float).ravel()[0]) * w
                total_w += w
            if total_w > 0: mag_pred = fused / total_w

        # ── 极端大余震概率 ──
        extreme_prob = 0.0
        if extreme_model is not None:
            try:
                extreme_prob = float(safe_extreme_probability(extreme_model, X)[0])
            except Exception:
                extreme_prob = 0.0

        # ── 极端大余震后处理 ──
        threshold = float(postproc_global.get("extreme_prob_threshold", 0.5))
        if extreme_prob >= threshold:
            uplift = float(postproc_global.get("high_risk_mag_quantile_weight", 0.5))
            margin = float(postproc_global.get("extreme_margin", 1.2))
            floor = max(0.0, float(mainshock["mag"]) - margin)
            raised = max(mag_pred, floor)
            mag_pred = (1.0 - uplift) * mag_pred + uplift * raised
            # T1 early bonus
            if wn == 'T1':
                bonus = float(postproc_global.get('t1_early_delta_bonus', 0.0))
                if bonus > 0:
                    mag_pred += bonus

        # ── 时间预测（时间桶概率加权） ──
        time_pred_raw = window.midpoint_hours
        if bucket_model is not None:
            try:
                raw = np.asarray(bucket_model.predict_proba(X), dtype=float)
                probs = align_bucket_probabilities(bucket_model, raw)
                time_pred_raw = float(expected_time_from_bucket_probs(wn, probs[0]))
            except Exception:
                pass

        time_pred = time_pred_raw

        # ── 高风险时间早期移动 ──
        if extreme_prob >= threshold:
            shift = float(postproc_global.get("early_time_shift_strength", 0.1))
            early = window.lower_hours + (window.upper_hours - window.lower_hours) * 0.3
            time_pred = (1.0 - shift) * time_pred + shift * early

        # ── 窗口级 bias：仅当对应 flag 为 True 时应用 ──
        win_post = payload.get("postprocessing", {})
        if use_artifact_mag:
            mag_pred += float(win_post.get("mag_bias", 0.0))
        if use_artifact_time:
            time_bias = float(win_post.get("time_bias", 0.0))
            # v2: T2/T3 时间校准 guard — 避免过度偏移导致 MAE 退化
            if time_mae_guard_enabled and wn in ("T2", "T3") and abs(time_bias) > 0:
                # 限制偏移不超过窗口宽度的 20%
                max_shift = (window.upper_hours - window.lower_hours) * 0.20
                if abs(time_bias) > max_shift:
                    time_bias = max_shift if time_bias > 0 else -max_shift
            time_pred += time_bias

        predictions[wn] = clamp_prediction_to_window(
            wn,
            mag=float(mag_pred),
            time_hours=float(time_pred),
            mainshock_mag=float(mainshock["mag"]),
        )

    return predictions


def apply_cli_calibration(
    predictions: dict[str, tuple[float, float]],
    mainshock_mag: float,
    mag_calibration: dict[str, float],
    time_calibration: dict[str, float],
) -> dict[str, tuple[float, float]]:
    """仅应用 CLI 显式校准覆盖。artifact 校准已在 predict 阶段处理。"""
    calibrated: dict[str, tuple[float, float]] = {}
    for wn, (mag, th) in predictions.items():
        mb = float(mag_calibration.get(wn, 0.0))
        tb = float(time_calibration.get(wn, 0.0))
        calibrated[wn] = clamp_prediction_to_window(
            wn,
            mag=float(mag) + mb,
            time_hours=float(th) + tb,
            mainshock_mag=mainshock_mag,
        )
    return calibrated


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def write_report(technical_docs_dir: Path, manifest: dict) -> Path:
    report_path = technical_docs_dir / "experiment_report.md"
    lines = [
        "# Decoupled 资格赛余震预测实验报告",
        "",
        f"生成时间：{manifest['created_at']}",
        "",
        "## 一、策略说明",
        "",
        "本提交包采用 decoupled pipeline：震级预测和时间预测使用分离的独立模型。",
        "时间预测使用时间桶分类（每窗口 4 桶），以降低 T3 窗口的时间预测误差。",
        "同时包含极端大余震检测器，用于上尾震级校正和早期时间偏移。",
        "",
        "## 二、校准策略",
        "",
        f"- 震级校准模式：`{manifest.get('mag_calibration_mode', 'artifact')}`",
        f"- 时间校准模式：`{manifest.get('time_calibration_mode', 'artifact')}`",
    ]
    if manifest.get("mag_calibration_mode") == "override":
        lines.append(f"- CLI 震级校准：`{manifest.get('cli_mag_calibration', {})}`")
    if manifest.get("time_calibration_mode") == "override":
        lines.append(f"- CLI 时间校准：`{manifest.get('cli_time_calibration', {})}`")
    lines.extend([
        "",
        "## 三、复现命令",
        "",
        "```bash",
        "python main.py make-decoupled-qualification-package --skip-commitment",
        "```",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    input_dir = resolve_project_path(args.input_dir)
    output_dir = resolve_project_path(args.output_dir)
    zip_path = resolve_project_path(args.zip_path)
    model_path = resolve_project_path(args.model_path)

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
    if artifact is None:
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    # 三态校准解析
    mag_mode, mag_cli = _parse_calibration(args.mag_calibration)
    time_mode, time_cli = _parse_calibration(args.time_calibration_hours)

    use_artifact_mag = (mag_mode == "artifact")
    use_artifact_time = (time_mode == "artifact")

    summary_rows: list[dict] = []
    seen_ids: set[str] = set()

    for csv_file in csv_files:
        event_df = normalize_event_table(pd.read_csv(csv_file))
        mainshock = pick_mainshock(event_df)
        ms_id = mainshock_token(mainshock)
        if ms_id in seen_ids:
            continue
        seen_ids.add(ms_id)

        # predict：artifact 校准根据 flag 决定是否应用
        raw_predictions = predict_with_decoupled_artifact(
            artifact, event_df, mainshock, args,
            use_artifact_mag=use_artifact_mag,
            use_artifact_time=use_artifact_time,
        )
        # CLI 校准仅在 override 模式下应用
        cli_mag = mag_cli if mag_mode == "override" else {}
        cli_time = time_cli if time_mode == "override" else {}
        predictions = apply_cli_calibration(
            raw_predictions,
            mainshock_mag=float(mainshock["mag"]),
            mag_calibration=cli_mag,
            time_calibration=cli_time,
        )

        written = write_qualification_prediction_files(
            predictions_dir, mainshock, predictions,
            magnitude_type=args.magnitude_type,
        )
        summary_rows.append({
            "sequence": str(csv_file),
            "mainshock_id": ms_id,
            "T1_mag": predictions["T1"][0],
            "T1_time": predictions["T1"][1],
            "T2_mag": predictions["T2"][0],
            "T2_time": predictions["T2"][1],
            "T3_mag": predictions["T3"][0],
            "T3_time": predictions["T3"][1],
            "files": [p.name for p in written],
        })

    # Copy collateral
    metrics_src = model_path.parent / "decoupled_metrics.json"
    if metrics_src.exists():
        shutil.copy2(metrics_src, technical_docs_dir / "decoupled_metrics.json")

    for report_name in ("architecture_review.md", "decoupled_tuning_plan.md"):
        src = PROJECT_ROOT / "reports" / report_name
        if src.exists():
            shutil.copy2(src, technical_docs_dir / report_name)

    # Manifest
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy": "decoupled_mag_time_with_buckets",
        "mag_calibration_mode": mag_mode,
        "time_calibration_mode": time_mode,
        "sequence_count": len(summary_rows),
        "prediction_file_count": len(list(predictions_dir.glob("*.csv"))),
        "model_artifact": str(model_path),
        "skip_commitment": bool(args.skip_commitment),
        "rows": summary_rows,
    }
    if mag_mode == "override":
        manifest["cli_mag_calibration"] = mag_cli
    if time_mode == "override":
        manifest["cli_time_calibration"] = time_cli
    # 记录 artifact 内的后处理参数（供审阅）
    manifest["artifact_postprocessing"] = artifact.get("postprocessing", {})

    report_path = write_report(technical_docs_dir, manifest)
    manifest["report_file"] = report_path.relative_to(output_dir).as_posix()

    copy_code_snapshot(technical_docs_dir / "code", output_dir)

    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.skip_commitment:
        commitment_dir.mkdir(parents=True, exist_ok=True)
    else:
        from scripts.make_qualification_package import copy_commitment
        copy_commitment(args.commitment_template, commitment_dir)

    files = zip_directory(output_dir, zip_path)
    sha = sha256_file(zip_path)
    zip_path.with_suffix(zip_path.suffix + ".sha256").write_text(
        f"{sha}  {zip_path.name}\n", encoding="utf-8",
    )

    print(f"Decoupled package directory: {output_dir}")
    print(f"ZIP: {zip_path}")
    print(f"SHA256: {sha}")
    print(f"Files in ZIP: {len(files)}")


if __name__ == "__main__":
    main()
