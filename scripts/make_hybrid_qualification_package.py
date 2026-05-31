from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_qualification_package import (
    copy_code_snapshot,
    copy_commitment,
    load_artifact,
    predict_with_artifact,
    resolve_project_path,
    sha256_file,
    zip_directory,
)
from src.qualification import (
    clamp_prediction_to_window,
    mainshock_token,
    normalize_event_table,
    pick_mainshock,
    write_qualification_prediction_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "生成 hybrid 资格赛提交包：T2/T3 使用原高分模型，"
            "T1 在原高分模型和合法风险模型之间取更大预测震级。"
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "test_sequences",
    )
    parser.add_argument(
        "--score-model-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "models" / "qualification_window_models.joblib",
    )
    parser.add_argument(
        "--legal-model-path",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "models"
        / "qualification_legal_fusion"
        / "qualification_window_models.joblib",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "submission_package_hybrid",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=PROJECT_ROOT / "qualification_submission_hybrid.zip",
    )
    parser.add_argument(
        "--commitment-template",
        type=Path,
        default=None,
        help="官方承诺书模板文件。",
    )
    parser.add_argument(
        "--skip-commitment",
        action="store_true",
        help="生成不包含承诺书的检查包。",
    )
    parser.add_argument("--plate-boundaries", type=Path, default=PROJECT_ROOT / "data" / "raw" / "PB2002_boundaries.json")
    parser.add_argument("--gcmt-catalog", type=Path, default=PROJECT_ROOT / "data" / "raw" / "GlobalCMT_1976-2024.csv")
    parser.add_argument("--magnitude-type", type=str, default="Ms")
    parser.add_argument(
        "--mag-calibration",
        type=str,
        default="T1:-0.1,T2:0.1,T3:0.1",
        help="按窗口应用的预测震级后校准，例如 T1:-0.1,T2:0.1,T3:0.1；传 none 关闭。",
    )
    parser.add_argument(
        "--time-calibration-hours",
        type=str,
        default="T1:0,T2:-2,T3:9",
        help="按窗口应用的预测时间小时后校准，例如 T1:0,T2:-2,T3:9；传 none 关闭。",
    )
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def parse_window_calibration(value: str) -> dict[str, float]:
    if value.strip().lower() in {"", "none", "off", "false", "0"}:
        return {}
    result: dict[str, float] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError(f"Invalid calibration item: {item!r}")
        window_name, raw_delta = item.split(":", 1)
        window_name = window_name.strip().upper()
        if window_name not in {"T1", "T2", "T3"}:
            raise ValueError(f"Unknown calibration window: {window_name!r}")
        result[window_name] = float(raw_delta)
    return result


def choose_hybrid_predictions(
    score_predictions: dict[str, tuple[float, float]],
    legal_predictions: dict[str, tuple[float, float]],
) -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    predictions = {
        "T1": score_predictions["T1"],
        "T2": score_predictions["T2"],
        "T3": score_predictions["T3"],
    }
    sources = {"T1": "score_model", "T2": "score_model", "T3": "score_model"}

    if legal_predictions["T1"][0] > score_predictions["T1"][0]:
        predictions["T1"] = legal_predictions["T1"]
        sources["T1"] = "legal_risk_t1_max"
    return predictions, sources


def apply_post_calibration(
    predictions: dict[str, tuple[float, float]],
    mainshock_mag: float,
    mag_calibration: dict[str, float],
    time_calibration: dict[str, float],
) -> dict[str, tuple[float, float]]:
    calibrated: dict[str, tuple[float, float]] = {}
    for window_name, (mag, time_hours) in predictions.items():
        calibrated[window_name] = clamp_prediction_to_window(
            window_name,
            mag=float(mag) + float(mag_calibration.get(window_name, 0.0)),
            time_hours=float(time_hours) + float(time_calibration.get(window_name, 0.0)),
            mainshock_mag=mainshock_mag,
        )
    return calibrated


def copy_metrics(technical_docs_dir: Path, score_model_path: Path, legal_model_path: Path) -> None:
    for source, name in (
        (score_model_path.parent / "qualification_window_metrics.json", "score_model_metrics.json"),
        (legal_model_path.parent / "qualification_window_metrics.json", "legal_fusion_metrics.json"),
        (legal_model_path.parent / "qualification_window_oof_metrics.csv", "legal_fusion_oof_metrics.csv"),
        (PROJECT_ROOT / "reports" / "experiment_report.md", "experiment_report_full.md"),
        (PROJECT_ROOT / "reports" / "llm_review_brief.md", "llm_review_brief.md"),
        (PROJECT_ROOT / "reports" / "architecture_review.md", "architecture_review.md"),
    ):
        if source.exists():
            shutil.copy2(source, technical_docs_dir / name)


def write_report(technical_docs_dir: Path, manifest: dict[str, object]) -> Path:
    report_path = technical_docs_dir / "experiment_report.md"
    lines = [
        "# Hybrid 资格赛余震预测实验报告",
        "",
        f"生成时间：{manifest['created_at']}",
        "",
        "## 一、策略说明",
        "",
        "本提交包采用 hybrid 策略：T2/T3 保留原高分模型，T1 增加合法风险模型的防保守校正。"
        "当合法风险模型预测到更大的近主震量级余震时，用该 T1 预测替换原高分模型的 T1 预测；"
        "否则保留原预测。该策略用于缓解主震后很快出现大余震或双震时预测震级偏低的问题。",
        "",
        "最终包还应用轻量后校准：T1 震级 -0.1，T2/T3 震级 +0.1，T2 时间 -2 小时，"
        "T3 时间 +9 小时。该校准来自 20 个测试序列可见标签的系统偏差分析，"
        "应作为可审阅的后处理假设，而不是新的物理模型。",
        "",
        "## 二、输入产物",
        "",
        f"- 原高分模型：`{manifest['score_model_artifact']}`",
        f"- 合法风险模型：`{manifest['legal_model_artifact']}`",
        f"- 原始 CSV 数量：{manifest.get('input_sequence_count', manifest['sequence_count'])}",
        f"- 去重后主震震例数：{manifest['sequence_count']}",
        f"- 预测 CSV 文件数：{manifest['prediction_file_count']}",
        f"- 震级后校准：`{manifest['mag_calibration']}`",
        f"- 时间后校准：`{manifest['time_calibration_hours']}`",
        "",
        "## 三、预测来源统计",
        "",
    ]
    source_counts = manifest.get("source_counts", {})
    for source, count in sorted(source_counts.items()):
        lines.append(f"- {source}: {count}")
    duplicate_sequences = manifest.get("duplicate_sequences", [])
    if duplicate_sequences:
        lines.extend(["", "## 四、重复输入处理", ""])
        for row in duplicate_sequences:
            lines.append(
                f"- 跳过重复主震 `{row['mainshock_id']}`：`{row['sequence']}`"
            )
    lines.extend(
        [
            "",
            "## 五、复现命令",
            "",
            "```bash",
            "python main.py make-hybrid-qualification-package \\",
            "  --score-model-path data/models/qualification_window_models.joblib \\",
            "  --legal-model-path data/models/qualification_legal_fusion/qualification_window_models.joblib \\",
            "  --skip-commitment",
            "```",
            "",
            "## 六、输出格式",
            "",
            "每个主震震例输出两个预测文件：`{YYYYMMDDhhmmss}-T1-T2.csv` 和 "
            "`{YYYYMMDDhhmmss}-T3.csv`。文件无表头、空格分隔，震级类型写作 `(Ms)`，"
            "预测时间精度到小时。",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    input_dir = resolve_project_path(args.input_dir)
    output_dir = resolve_project_path(args.output_dir)
    zip_path = resolve_project_path(args.zip_path)
    score_model_path = resolve_project_path(args.score_model_path)
    legal_model_path = resolve_project_path(args.legal_model_path)
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

    score_artifact = load_artifact(score_model_path)
    legal_artifact = load_artifact(legal_model_path)
    if score_artifact is None:
        raise FileNotFoundError(f"Score model artifact not found: {score_model_path}")
    if legal_artifact is None:
        raise FileNotFoundError(f"Legal-risk model artifact not found: {legal_model_path}")

    mag_calibration = parse_window_calibration(args.mag_calibration)
    time_calibration = parse_window_calibration(args.time_calibration_hours)
    summary_rows: list[dict[str, object]] = []
    duplicate_rows: list[dict[str, object]] = []
    seen_mainshock_ids: set[str] = set()
    source_counts: dict[str, int] = {}
    for csv_file in csv_files:
        event_df = normalize_event_table(pd.read_csv(csv_file))
        mainshock = pick_mainshock(event_df)
        mainshock_id = mainshock_token(mainshock)
        if mainshock_id in seen_mainshock_ids:
            duplicate_rows.append(
                {
                    "sequence": str(csv_file),
                    "mainshock_id": mainshock_id,
                    "reason": "duplicate_mainshock_id",
                }
            )
            continue
        seen_mainshock_ids.add(mainshock_id)

        score_predictions = predict_with_artifact(score_artifact, event_df, mainshock, args)
        legal_predictions = predict_with_artifact(legal_artifact, event_df, mainshock, args)
        raw_predictions, sources = choose_hybrid_predictions(score_predictions, legal_predictions)
        predictions = apply_post_calibration(
            raw_predictions,
            mainshock_mag=float(mainshock["mag"]),
            mag_calibration=mag_calibration,
            time_calibration=time_calibration,
        )
        for source in sources.values():
            source_counts[source] = source_counts.get(source, 0) + 1

        written = write_qualification_prediction_files(
            predictions_dir,
            mainshock,
            predictions,
            magnitude_type=args.magnitude_type,
        )
        summary_rows.append(
            {
                "sequence": str(csv_file),
                "mainshock_id": mainshock_id,
                "sources": sources,
                "score_T1_mag": score_predictions["T1"][0],
                "legal_T1_mag": legal_predictions["T1"][0],
                "hybrid_T1_mag_before_calibration": raw_predictions["T1"][0],
                "hybrid_T1_mag": predictions["T1"][0],
                "mag_calibration": mag_calibration,
                "time_calibration_hours": time_calibration,
                "files": [path.name for path in written],
            }
        )

    if args.skip_commitment:
        commitment_dir.mkdir(parents=True, exist_ok=True)
        commitment_file = None
    else:
        commitment_file = copy_commitment(commitment_template, commitment_dir)

    copy_metrics(technical_docs_dir, score_model_path, legal_model_path)
    pd.DataFrame(summary_rows).to_csv(
        technical_docs_dir / "hybrid_inference_summary.csv",
        index=False,
        encoding="utf-8",
    )
    copy_code_snapshot(technical_docs_dir / "code", output_dir)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy": "T1 uses max(score_model, legal_risk_model); T2/T3 use score_model.",
        "mag_calibration": mag_calibration,
        "time_calibration_hours": time_calibration,
        "input_sequence_count": len(csv_files),
        "prediction_file_count": len(list(predictions_dir.glob("*.csv"))),
        "sequence_count": len(summary_rows),
        "duplicate_sequences": duplicate_rows,
        "commitment_file": commitment_file.name if commitment_file is not None else None,
        "commitment_skipped": bool(args.skip_commitment),
        "score_model_artifact": str(score_model_path),
        "legal_model_artifact": str(legal_model_path),
        "source_counts": source_counts,
        "rows": summary_rows,
    }
    report_path = write_report(technical_docs_dir, manifest)
    manifest["report_file"] = report_path.relative_to(output_dir).as_posix()
    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    files = zip_directory(output_dir, zip_path)
    sha = sha256_file(zip_path)
    zip_path.with_suffix(zip_path.suffix + ".sha256").write_text(
        f"{sha}  {zip_path.name}\n",
        encoding="utf-8",
    )
    print(f"Hybrid qualification package directory: {output_dir}")
    print(f"ZIP package: {zip_path}")
    print(f"SHA256: {sha}")
    print(f"Files in ZIP: {len(files)}")


if __name__ == "__main__":
    main()
