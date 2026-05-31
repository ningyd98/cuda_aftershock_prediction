"""Single-horizon 资格赛提交打包脚本。

读取 qualification_single_horizon_model.joblib，对 test_sequences 预测，
生成 qualification_predictions.csv（一行一个主震，无 header），打 ZIP。

默认不输出 T1/T2/T3 文件。使用 --legacy-window-output 可同时输出旧格式。
"""

from __future__ import annotations

import argparse, json, shutil, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np, pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_qualification_package import (
    copy_code_snapshot, load_artifact, sha256_file, zip_directory, build_feature_matrix,
)
from src.qualification import (
    H168_WINDOW, QUALIFICATION_WINDOWS, WINDOW_BY_NAME,
    clamp_prediction_to_window, mainshock_token, normalize_event_table,
    pick_mainshock, rule_window_prediction,
    write_qualification_prediction_files,
    write_single_horizon_prediction_file,
)
from src.time_buckets import (
    align_bucket_probabilities, expected_time_from_bucket_probs,
    safe_extreme_probability,
)

WIN = H168_WINDOW
WNAME = "H168"


def resolve_project_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(description="Build single-horizon qualification ZIP package.")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT/"data"/"test_sequences")
    parser.add_argument("--model-path", type=Path,
                        default=PROJECT_ROOT/"data"/"models"/"single_horizon"/"qualification_single_horizon_model.joblib")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT/"submission_package_single_horizon")
    parser.add_argument("--zip-path", type=Path, default=PROJECT_ROOT/"qualification_submission_single_horizon.zip")
    parser.add_argument("--commitment-template", type=Path, default=None)
    parser.add_argument("--skip-commitment", action="store_true")
    parser.add_argument("--plate-boundaries", type=Path, default=PROJECT_ROOT/"data"/"raw"/"PB2002_boundaries.json")
    parser.add_argument("--gcmt-catalog", type=Path, default=PROJECT_ROOT/"data"/"raw"/"GlobalCMT_1976-2024.csv")
    parser.add_argument("--magnitude-type", type=str, default="Ms")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--legacy-window-output", action="store_true",
                        help="Also produce legacy *-T1-T2.csv / *-T3.csv files.")
    return parser.parse_args()


def predict_single_horizon(artifact, event_df, mainshock, args):
    """H168 单窗口预测 (v2: 支持 mag/time 多候选融合)。"""
    payload = artifact.get("H168", {})
    if not payload:
        mag, th = rule_window_prediction(WNAME, float(mainshock["mag"]))
        return mag, th

    feature_cols = payload.get("feature_cols", [])
    X = build_feature_matrix(event_df, args, feature_cols, window_name=WNAME,
                             observation_hours=payload.get("observation_hours", 0.0))

    postproc_global = artifact.get("postprocessing", {})
    mag_models = payload.get("mag_models", {})
    bucket_model = payload.get("bucket_model")
    extreme_model = payload.get("extreme_model")

    # ── 震级候选：映射 mag_models 中的 "baseline"/"xgboost" → fusion key "oof_mag_lgbm"/"oof_mag_xgb" ──
    mag_candidates = {}
    if "baseline" in mag_models:
        try:
            mag_candidates["oof_mag_lgbm"] = float(np.asarray(
                mag_models["baseline"].predict(X), dtype=float).ravel()[0])
        except Exception:
            pass
    if "xgboost" in mag_models:
        try:
            mag_candidates["oof_mag_xgb"] = float(np.asarray(
                mag_models["xgboost"].predict(X), dtype=float).ravel()[0])
        except Exception:
            pass
    # 应用 mag 融合权重
    fw_mag = payload.get("fusion", {}).get("mag", {})
    if fw_mag and mag_candidates:
        fused = 0.0; total_w = 0.0
        for key, w in fw_mag.items():
            if key in mag_candidates and w > 0:
                fused += mag_candidates[key] * w
                total_w += w
        if total_w > 0:
            mag_pred = fused / total_w
        else:
            mag_pred = float(np.mean(list(mag_candidates.values())))
    elif mag_candidates:
        weights = payload.get("weights", {}).get("mag", {})
        total = 0.0; mag_pred = 0.0
        for nm in mag_models:
            w = float(weights.get(nm, 1.0 / max(1, len(mag_models))))
            if nm in mag_candidates or nm in ("baseline", "xgboost"):
                key = f"oof_mag_{'xgb' if nm == 'xgboost' else 'lgbm'}"
                if key in mag_candidates:
                    mag_pred += mag_candidates[key] * w
                    total += w
        mag_pred = mag_pred / total if total > 0 else float(np.mean(list(mag_candidates.values())))
    else:
        mag_pred = 0.0
    mag_pred = np.clip(float(mag_pred), 0.0, None)

    # ── 时间候选：bucket + direct_lgbm + direct_xgb ──
    time_candidates = {}
    if bucket_model is not None:
        try:
            raw = np.asarray(bucket_model.predict_proba(X), dtype=float)
            probs = align_bucket_probabilities(bucket_model, raw)
            time_candidates["oof_time_bucket_raw"] = float(
                expected_time_from_bucket_probs(WNAME, probs[0]))
        except Exception:
            pass
    for model_key, cand_key, model_attr in [
        ("time_direct_model_lgbm", "oof_time_direct_lgbm_raw", "time_direct_model_lgbm"),
        ("time_direct_model_xgb", "oof_time_direct_xgb_raw", "time_direct_model_xgb"),
    ]:
        model = payload.get(model_attr)
        if model is not None:
            try:
                pred_log = float(np.asarray(model.predict(X), dtype=float).ravel()[0])
                pred_hours = float(np.expm1(pred_log))
                time_candidates[cand_key] = float(np.clip(
                    pred_hours, WIN.lower_hours + 1e-6, WIN.upper_hours))
            except Exception:
                pass
    # 应用时间融合权重
    fw_time = payload.get("fusion", {}).get("time", {})
    if fw_time and time_candidates:
        fused = 0.0; total_w = 0.0
        for key, w in fw_time.items():
            if key in time_candidates and w > 0:
                fused += time_candidates[key] * w
                total_w += w
        if total_w > 0:
            time_pred = fused / total_w
        else:
            time_pred = float(np.mean(list(time_candidates.values())))
    elif time_candidates:
        time_pred = float(np.mean(list(time_candidates.values())))
    else:
        time_pred = WIN.midpoint_hours

    # ── 极端风险 ──
    extreme_prob = 0.0
    if extreme_model is not None:
        try:
            extreme_prob = float(safe_extreme_probability(extreme_model, X)[0])
        except Exception:
            pass

    thr = float(postproc_global.get("extreme_prob_threshold", 0.5))
    if extreme_prob >= thr:
        uplift = float(postproc_global.get("high_risk_mag_quantile_weight", 0.5))
        margin = float(postproc_global.get("extreme_margin", 1.2))
        floor = max(0.0, float(mainshock["mag"]) - margin)
        mag_pred = (1.0 - uplift) * mag_pred + uplift * max(mag_pred, floor)
        shift = float(postproc_global.get("early_time_shift_strength", 0.1))
        early = WIN.lower_hours + (WIN.upper_hours - WIN.lower_hours) * 0.3
        time_pred = (1.0 - shift) * time_pred + shift * early

    return clamp_prediction_to_window(WNAME, mag=float(mag_pred), time_hours=float(time_pred),
                                       mainshock_mag=float(mainshock["mag"]))


def write_technical_doc(technical_docs_dir, manifest):
    report_path = technical_docs_dir / "prediction_summary.md"
    lines = [
        "# Single-Horizon 余震预测实验报告",
        "",
        f"生成时间：{manifest['created_at']}",
        "",
        "## 策略说明",
        "",
        "本提交采用 single-horizon (0-168h) 最大余震预测：震级和时间使用独立模型预测，",
        "时间预测用 4 桶分类 + 直接回归双路径。仅使用主震静态特征，不使用余震观测。",
        "",
        f"## 预测行数: {manifest['sequence_count']}",
        "",
        "## 复现命令",
        "",
        "```bash",
        "python main.py make-single-horizon-package --skip-commitment",
        "```",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
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

    summary_rows = []
    seen_ids = set()
    single_horizon_rows = []

    for csv_file in csv_files:
        event_df = normalize_event_table(pd.read_csv(csv_file))
        mainshock = pick_mainshock(event_df)
        ms_id = mainshock_token(mainshock)
        if ms_id in seen_ids: continue
        seen_ids.add(ms_id)

        pred_mag, pred_time = predict_single_horizon(artifact, event_df, mainshock, args)
        single_horizon_rows.append((mainshock, pred_mag, pred_time))

        summary_rows.append({
            "sequence": str(csv_file),
            "mainshock_id": ms_id,
            "H168_mag": pred_mag,
            "H168_time": pred_time,
        })

    # 写单窗口 prediction 文件（默认唯一输出）
    write_single_horizon_prediction_file(
        predictions_dir, single_horizon_rows,
        magnitude_type=args.magnitude_type,
    )

    # 旧版窗口文件（仅在 --legacy-window-output 启用时）
    if args.legacy_window_output:
        print("[INFO] --legacy-window-output enabled: also generating T1/T2/T3 files.")
        seen_legacy = set()
        for csv_file in csv_files:
            event_df = normalize_event_table(pd.read_csv(csv_file))
            mainshock = pick_mainshock(event_df)
            ms_id = mainshock_token(mainshock)
            if ms_id in seen_legacy:
                continue
            seen_legacy.add(ms_id)
            # fallback: rule-based for T1/T2/T3
            preds = {}
            for w in QUALIFICATION_WINDOWS:
                preds[w.name] = rule_window_prediction(w.name, float(mainshock["mag"]))
            write_qualification_prediction_files(predictions_dir, mainshock, preds,
                                                  magnitude_type=args.magnitude_type)

    # technical docs
    metrics_src = model_path.parent / "single_horizon_metrics.json"
    if metrics_src.exists():
        shutil.copy2(metrics_src, technical_docs_dir / "single_horizon_metrics.json")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy": "single_horizon_168h",
        "sequence_count": len(summary_rows),
        "prediction_file_count": len(list(predictions_dir.glob("*.csv"))),
        "model_artifact": str(model_path),
        "skip_commitment": bool(args.skip_commitment),
        "legacy_window_output": bool(args.legacy_window_output),
        "rows": summary_rows,
    }

    report_path = write_technical_doc(technical_docs_dir, manifest)
    manifest["report_file"] = report_path.relative_to(output_dir).as_posix()

    copy_code_snapshot(technical_docs_dir / "code", output_dir)

    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.skip_commitment:
        commitment_dir.mkdir(parents=True, exist_ok=True)
    else:
        from scripts.make_qualification_package import copy_commitment
        copy_commitment(args.commitment_template, commitment_dir)

    files = zip_directory(output_dir, zip_path)
    sha = sha256_file(zip_path)
    zip_path.with_suffix(zip_path.suffix + ".sha256").write_text(f"{sha}  {zip_path.name}\n", encoding="utf-8")

    print(f"Single-horizon package directory: {output_dir}")
    print(f"ZIP: {zip_path}")
    print(f"Files in ZIP: {len(files)}")


if __name__ == "__main__":
    main()
