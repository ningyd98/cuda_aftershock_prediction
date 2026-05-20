#!/usr/bin/env python3
"""
批量 OOF 推理脚本 —— 对赛事官方提供的 20 条测试序列运行全部可用模型的独立预测与集成融合。

功能:
1. 扫描 --input-dir 中的 *_eq.csv 测试序列
2. 对每条序列：构建特征 → 依次运行 baseline / xgboost / DL / GNN 模型
3. 按 ensemble_weights.json 计算双目标加权融合
4. 输出:
   - batch_oof_predictions.csv     每条序列的融合结果
   - batch_oof_predictions_full.csv 含各单模型预测 + 一致性统计
   - batch_oof_summary.json         模型可用性、一致性、权重等汇总信息

用法:
  # 批量推理所有测试序列
  python scripts/batch_oof_inference.py \
      --input-dir data/test_sequences \
      --model-dir data/models \
      --output-dir reports/batch_oof_inference

  # 单条推理（等价于增强版 make_submission）
  python scripts/batch_oof_inference.py \
      --input data/test_sequences/20230206011734_eq.csv \
      --model-dir data/models \
      --output-dir reports/batch_oof_inference
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# ── 复用 make_submission 中的核心函数 ────────────────────────────────────
from scripts.make_submission import (
    add_derived_features,
    build_single_sequence_features,
    load_ensemble_weights,
    load_feature_cols,
    make_model_matrix,
    normalize_event_table,
    predict_with_baseline,
    predict_with_dl,
    predict_with_gnn,
    resolve_project_path,
    postprocess_prediction,
    check_feature_consistency,
)


def _model_name_to_key(name: str) -> str:
    """统一模型名映射。"""
    mapping = {
        "baseline": "baseline",
        "xgboost": "xgboost",
        "dl": "dl",
        "gnn": "gnn",
    }
    return mapping.get(name.lower(), name.lower())


def run_single_inference(
    event_csv: Path,
    model_dir: Path,
    plate_boundaries_path: Path,
    gcmt_catalog_path: Path | None,
    obs_days: float,
    spatial_radius_km: float,
    earth_radius_km: float,
    gcmt_time_tolerance: float,
    gcmt_spatial_radius: float,
) -> dict:
    """对单条测试序列运行全部可用模型 + 融合推理，返回完整字典。"""
    result = {
        "mainshock_id": "",
        "mainshock_mag": float("nan"),
        "mainshock_depth": float("nan"),
        "mainshock_time": "",
        "early_event_count": 0,
        "models_available": [],
        "models_failed": [],
    }

    # ── 读取 & 特征构建 ─────────────────────────────────────────────
    try:
        event_df = normalize_event_table(pd.read_csv(event_csv))
        feature_df, early_events = build_single_sequence_features(
            event_df,
            plate_boundaries_path=plate_boundaries_path,
            gcmt_catalog_path=gcmt_catalog_path,
            obs_days=obs_days,
            spatial_radius_km=spatial_radius_km,
            earth_radius_km=earth_radius_km,
            gcmt_time_tolerance_seconds=gcmt_time_tolerance,
            gcmt_spatial_radius_km=gcmt_spatial_radius,
        )
    except Exception as exc:
        result["error"] = f"特征构建失败: {exc}"
        return result

    mainshock_id = str(feature_df.loc[0, "mainshock_id"])
    mainshock_mag = float(feature_df.loc[0, "mainshock_mag"])
    mainshock_depth = float(feature_df.loc[0, "mainshock_depth"])
    mainshock_time = str(feature_df.loc[0, "mainshock_time"])
    early_count = int(len(early_events))

    result.update(
        mainshock_id=mainshock_id,
        mainshock_mag=mainshock_mag,
        mainshock_depth=mainshock_depth,
        mainshock_time=mainshock_time,
        early_event_count=early_count,
    )

    enriched_df = add_derived_features(feature_df.copy())

    # ── 加载融合权重 ─────────────────────────────────────────────────
    weights_path = model_dir / "ensemble_weights.json"
    all_weights = load_ensemble_weights(weights_path)
    mag_weights = all_weights.get("mag", all_weights)
    time_weights = all_weights.get("time", all_weights)

    # ── 特征矩阵 (树模型共用) ──────────────────────────────────────
    feature_cols_path = model_dir / "feature_cols.json"
    try:
        feature_cols = load_feature_cols(feature_cols_path)
        check_feature_consistency(enriched_df, feature_cols, max_missing_ratio=0.50)
        X = make_model_matrix(enriched_df, feature_cols)
    except Exception as exc:
        result["error"] = f"特征矩阵构建失败: {exc}"
        return result

    # 存储中间特征快照（可选，供调试）
    result["feature_cols_count"] = len(feature_cols)
    result["feature_cols_available"] = len([c for c in feature_cols if c in X.columns])

    # ── 依次运行各模型 ───────────────────────────────────────────────
    model_preds: dict[str, dict] = {}

    # 1) Baseline (LightGBM)
    baseline_path = model_dir / "baseline_model.joblib"
    if baseline_path.exists():
        try:
            raw = predict_with_baseline(baseline_path, X)
            model_preds["baseline"] = {
                "pred_mag": float(raw[0, 0]),
                "pred_time_days": float(raw[0, 1]),
            }
            result["models_available"].append("baseline")
        except Exception as exc:
            result["models_failed"].append(f"baseline: {exc}")

    # 2) XGBoost
    xgb_path = model_dir / "xgboost_model.joblib"
    if xgb_path.exists():
        try:
            raw = predict_with_baseline(xgb_path, X)
            model_preds["xgboost"] = {
                "pred_mag": float(raw[0, 0]),
                "pred_time_days": float(raw[0, 1]),
            }
            result["models_available"].append("xgboost")
        except Exception as exc:
            result["models_failed"].append(f"xgboost: {exc}")

    # 3) Transformer (DL)
    dl_path = model_dir / "dl_model.pt"
    dl_meta_path = model_dir / "dl_meta.json"
    if dl_path.exists() and dl_meta_path.exists():
        try:
            raw = predict_with_dl(dl_path, dl_meta_path, event_df, enriched_df)
            if raw is not None:
                model_preds["dl"] = {
                    "pred_mag": float(raw[0, 0]),
                    "pred_time_days": float(raw[0, 1]),
                }
                result["models_available"].append("dl")
            else:
                result["models_failed"].append("dl: 预处理器缺失")
        except Exception as exc:
            result["models_failed"].append(f"dl: {exc}")

    # 4) ST-GNN
    gnn_path = model_dir / "gnn_model.pt"
    gnn_meta_path = model_dir / "gnn_meta.json"
    if gnn_path.exists() and gnn_meta_path.exists():
        try:
            raw = predict_with_gnn(gnn_path, gnn_meta_path, event_df, enriched_df)
            if raw is not None:
                model_preds["gnn"] = {
                    "pred_mag": float(raw[0, 0]),
                    "pred_time_days": float(raw[0, 1]),
                }
                result["models_available"].append("gnn")
            else:
                result["models_failed"].append("gnn: 预处理器缺失")
        except Exception as exc:
            result["models_failed"].append(f"gnn: {exc}")

    if not model_preds:
        result["error"] = "没有任何可用模型"
        return result

    # ── 写入单模型预测 ──────────────────────────────────────────────
    for model_name, preds in model_preds.items():
        result[f"{model_name}_pred_mag"] = preds["pred_mag"]
        result[f"{model_name}_pred_time_days"] = preds["pred_time_days"]

    # ── 震级/时间分别加权融合 ───────────────────────────────────────
    fused_mag = 0.0
    fused_time = 0.0
    total_mag_w = 0.0
    total_time_w = 0.0

    for name, preds in model_preds.items():
        w_mag = max(float(mag_weights.get(name, 0.0)), 0.0)
        w_time = max(float(time_weights.get(name, 0.0)), 0.0)
        if w_mag > 0:
            fused_mag += preds["pred_mag"] * w_mag
            total_mag_w += w_mag
        if w_time > 0:
            fused_time += preds["pred_time_days"] * w_time
            total_time_w += w_time

    if total_mag_w > 0:
        fused_mag /= total_mag_w
    else:
        fused_mag = next(iter(model_preds.values()))["pred_mag"]

    if total_time_w > 0:
        fused_time /= total_time_w
    else:
        fused_time = next(iter(model_preds.values()))["pred_time_days"]

    # 后处理
    raw_pred = np.array([[fused_mag, fused_time]], dtype=float)
    final_mag, final_time = postprocess_prediction(
        raw_pred, mainshock_mag=mainshock_mag, early_count=early_count,
    )

    result["predicted_max_mag"] = final_mag
    result["predicted_time_to_max"] = final_time
    result["prediction_source"] = "ensemble"

    # ── 模型间一致性统计 ────────────────────────────────────────────
    mag_values = [p["pred_mag"] for p in model_preds.values()]
    time_values = [p["pred_time_days"] for p in model_preds.values()]
    if len(mag_values) >= 2:
        result["mag_agreement_std"] = round(float(np.std(mag_values)), 4)
        result["mag_agreement_range"] = round(float(np.ptp(mag_values)), 4)
    if len(time_values) >= 2:
        result["time_agreement_std"] = round(float(np.std(time_values)), 4)
        result["time_agreement_range"] = round(float(np.ptp(time_values)), 4)

    return result


def parse_args() -> argparse.Namespace:
    """命令行参数。"""
    parser = argparse.ArgumentParser(
        description="批量 OOF 推理 —— 对测试序列集运行全部模型并融合",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-dir", type=Path, default=None, help="测试序列目录")
    group.add_argument("--input", type=Path, default=None, help="单条测试序列 CSV")

    parser.add_argument(
        "--model-dir", type=Path,
        default=PROJECT_ROOT / "data" / "models",
        help="模型产物目录",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "reports" / "batch_oof_inference",
        help="输出目录",
    )
    parser.add_argument(
        "--plate-boundaries", type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "PB2002_boundaries.json",
    )
    parser.add_argument(
        "--gcmt-catalog", type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "GlobalCMT_1976-2024.csv",
        help="GCMT 震源机制解 CSV；不存在时自动跳过",
    )
    parser.add_argument("--obs-days", type=float, default=3.0)
    parser.add_argument("--spatial-radius-km", type=float, default=100.0)
    parser.add_argument("--earth-radius-km", type=float, default=6371.0)
    parser.add_argument("--gcmt-time-tolerance-seconds", type=float, default=60.0)
    parser.add_argument("--gcmt-spatial-radius-km", type=float, default=50.0)
    parser.add_argument(
        "--device", type=str, default="auto",
        help="深度学习推理设备: auto / cuda / cpu",
    )
    return parser.parse_args()


def print_divider(char: str = "─", width: int = 70) -> None:
    """打印分隔线。"""
    print(char * width)


def main() -> None:
    args = parse_args()
    model_dir = resolve_project_path(args.model_dir)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plate_boundaries_path = resolve_project_path(args.plate_boundaries)
    gcmt_catalog_path = resolve_project_path(args.gcmt_catalog)
    if not gcmt_catalog_path.exists():
        print(f"ℹ GCMT 目录不可用 ({gcmt_catalog_path})，将跳过震源机制解特征")
        gcmt_catalog_path = None

    # ── 收集测试序列 ────────────────────────────────────────────────
    if args.input:
        csv_files = [resolve_project_path(args.input)]
    else:
        input_dir = resolve_project_path(args.input_dir)
        csv_files = sorted(input_dir.glob("*_eq.csv"))
        if not csv_files:
            print(f"错误: {input_dir} 中未找到任何 *_eq.csv 文件")
            sys.exit(1)

    # ── 打印环境信息 ────────────────────────────────────────────────
    print_divider()
    print("🔬 批量 OOF 推理")
    print_divider()
    print(f"  测试序列数: {len(csv_files)}")
    print(f"  模型目录:   {model_dir}")
    print(f"  输出目录:   {output_dir}")

    # 检测可用模型
    available = []
    if (model_dir / "baseline_model.joblib").exists():
        available.append("baseline")
    if (model_dir / "xgboost_model.joblib").exists():
        available.append("xgboost")
    if (model_dir / "dl_model.pt").exists():
        available.append("dl")
    if (model_dir / "gnn_model.pt").exists():
        available.append("gnn")
    print(f"  检测到模型: {', '.join(available) if available else '(无)'}")

    weights_path = model_dir / "ensemble_weights.json"
    if weights_path.exists():
        w = load_ensemble_weights(weights_path)
        print(f"  融合权重 (mag): {w.get('mag', {})}")
        print(f"  融合权重 (time): {w.get('time', {})}")
    print_divider()

    # ── 逐序列推理 ────────────────────────────────────────────────────
    rows = []
    t_start = time.perf_counter()
    for i, csv_path in enumerate(csv_files, 1):
        seq_name = csv_path.stem  # e.g. 20010126031640_eq
        print(f"\n[{i:2d}/{len(csv_files)}] {seq_name} …", end=" ", flush=True)
        t_seq = time.perf_counter()

        result = run_single_inference(
            event_csv=csv_path,
            model_dir=model_dir,
            plate_boundaries_path=plate_boundaries_path,
            gcmt_catalog_path=gcmt_catalog_path,
            obs_days=args.obs_days,
            spatial_radius_km=args.spatial_radius_km,
            earth_radius_km=args.earth_radius_km,
            gcmt_time_tolerance=args.gcmt_time_tolerance_seconds,
            gcmt_spatial_radius=args.gcmt_spatial_radius_km,
        )

        elapsed = time.perf_counter() - t_seq
        if "error" in result:
            print(f"✗ 失败 ({elapsed:.1f}s): {result['error']}")
            rows.append(result)
            continue

        print(
            f"✓ Mw={result['mainshock_mag']:.1f} → "
            f"mag={result['predicted_max_mag']:.2f} "
            f"time={result['predicted_time_to_max']:.2f}d "
            f"({len(result['models_available'])}模型, {elapsed:.1f}s)"
        )
        rows.append(result)

    total_elapsed = time.perf_counter() - t_start
    success_count = sum(1 for r in rows if "error" not in r)
    error_count = sum(1 for r in rows if "error" in r)

    # ── 构建输出 DataFrame ──────────────────────────────────────────────
    if not rows:
        print("\n错误: 没有生成任何推理结果")
        sys.exit(1)

    # 基础列（始终存在）
    base_cols = [
        "mainshock_id",
        "mainshock_mag",
        "mainshock_depth",
        "mainshock_time",
        "early_event_count",
        "predicted_max_mag",
        "predicted_time_to_max",
        "prediction_source",
        "models_available",
        "models_failed",
    ]

    # 单模型预测列（仅当存在时添加）
    model_keys = ["baseline", "xgboost", "dl", "gnn"]
    extra_cols = []
    for key in model_keys:
        mag_key = f"{key}_pred_mag"
        time_key = f"{key}_pred_time_days"
        if any(mag_key in r for r in rows if "error" not in r):
            extra_cols.append(mag_key)
        if any(time_key in r for r in rows if "error" not in r):
            extra_cols.append(time_key)

    # 一致性列
    extra_cols += [
        "mag_agreement_std",
        "mag_agreement_range",
        "time_agreement_std",
        "time_agreement_range",
    ]

    # 错误列
    extra_cols.append("error")

    all_cols = base_cols + extra_cols
    df = pd.DataFrame(rows)

    # 确保所有列存在
    for col in all_cols:
        if col not in df.columns:
            df[col] = np.nan

    # 将 list 列序列化为字符串以便 CSV 保存
    for list_col in ["models_available", "models_failed"]:
        if list_col in df.columns:
            df[list_col] = df[list_col].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else str(x) if pd.notna(x) else ""
            )

    df = df[all_cols]

    # ── 保存合并结果 ────────────────────────────────────────────────────
    full_path = output_dir / "batch_oof_predictions_full.csv"
    df.to_csv(full_path, index=False, encoding="utf-8")
    print(f"\n📄 详细结果: {full_path} ({len(df)} 行)")

    # 精简版（仅融合结果 + 主要字段）
    slim_cols = [
        "mainshock_id",
        "mainshock_mag",
        "early_event_count",
        "predicted_max_mag",
        "predicted_time_to_max",
        "models_available",
    ]
    slim_df = df[[c for c in slim_cols if c in df.columns]].copy()
    slim_path = output_dir / "batch_oof_predictions.csv"
    slim_df.to_csv(slim_path, index=False, encoding="utf-8")
    print(f"📄 精简结果: {slim_path}")

    # ── 汇总 JSON ────────────────────────────────────────────────────────
    success_df = df[df["error"].isna() | (df["error"] == "")]
    summary = {
        "run_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "total_sequences": len(csv_files),
        "success_count": success_count,
        "error_count": error_count,
        "total_elapsed_seconds": round(total_elapsed, 1),
        "average_seconds_per_sequence": round(total_elapsed / max(len(csv_files), 1), 2),
        "models_detected": available,
        "ensemble_weights": (
            load_ensemble_weights(weights_path) if weights_path.exists() else {}
        ),
        "predictions_summary": {
            "avg_predicted_mag": round(float(success_df["predicted_max_mag"].mean()), 4)
            if len(success_df) > 0 else None,
            "std_predicted_mag": round(float(success_df["predicted_max_mag"].std()), 4)
            if len(success_df) > 0 else None,
            "avg_predicted_time_days": round(float(success_df["predicted_time_to_max"].mean()), 4)
            if len(success_df) > 0 else None,
            "std_predicted_time_days": round(float(success_df["predicted_time_to_max"].std()), 4)
            if len(success_df) > 0 else None,
        },
        "model_agreement": {},
        "errors": [
            {"mainshock_id": r.get("mainshock_id", "?"), "error": r.get("error", "")}
            for r in rows if "error" in r
        ],
    }

    # 模型间平均一致性
    if "mag_agreement_std" in success_df.columns:
        summary["model_agreement"]["avg_mag_std_across_models"] = round(
            float(success_df["mag_agreement_std"].mean()), 4
        )
    if "time_agreement_std" in success_df.columns:
        summary["model_agreement"]["avg_time_std_across_models"] = round(
            float(success_df["time_agreement_std"].mean()), 4
        )

    summary_path = output_dir / "batch_oof_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"📄 汇总信息: {summary_path}")

    # ── 完成 ─────────────────────────────────────────────────────────────
    print_divider()
    print(f"✅ 批量推理完成: {success_count}/{len(csv_files)} 成功, "
          f"{error_count} 失败, 总耗时 {total_elapsed:.1f}s")
    print_divider()

    if success_count > 0:
        print("\n预测概览:")
        print(slim_df.to_string(index=False))


if __name__ == "__main__":
    main()
