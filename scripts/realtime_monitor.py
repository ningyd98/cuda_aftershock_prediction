#!/usr/bin/env python
# ============================================================
#  余震预测实时监控与自动推理工具
#
#  功能:
#   1. 定时轮询 USGS FDSN API，自动发现全球新发生的 Mw≥6.0 强震
#   2. 对每个新主震，自动查询 3 天观测窗口内的早期余震
#   3. 调用已训练的集成模型（LightGBM + XGBoost + Transformer + ST-GNN）
#      进行实时余震预测
#   4. 生成结构化预测报告（JSON + CSV），支持持续追加
#
#  使用:
#    python scripts/realtime_monitor.py                    # 前台运行，每 5 分钟轮询
#    python scripts/realtime_monitor.py --once             # 单次运行
#    python scripts/realtime_monitor.py --interval 300     # 自定义轮询间隔 (秒)
#    python scripts/realtime_monitor.py --backfill 30      # 回填最近 30 天内的强震
#    python scripts/realtime_monitor.py --min-mag 5.5      # 降低震级阈值
#
#  产物:
#    data/processed/realtime_state.json      # 处理状态跟踪
#    data/processed/realtime_predictions.csv  # 历史预测记录
#    data/processed/realtime_alerts.json      # 预警级别事件
# ============================================================

from __future__ import annotations

import argparse
import json
import sys
import time
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ── 项目路径 ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import set_random_seed, setup_cuda

# ── 常量 ───────────────────────────────────────────────────
USGS_FDSN_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
DEFAULT_MIN_MAG = 6.0
DEFAULT_POLL_INTERVAL = 300  # 秒
DEFAULT_OBS_DAYS = 3.0
DEFAULT_SPATIAL_RADIUS_KM = 100.0
DEFAULT_EARTH_RADIUS_KM = 6371.0

STATE_FILE = PROJECT_ROOT / "data" / "processed" / "realtime_state.json"
PREDICTIONS_FILE = PROJECT_ROOT / "data" / "processed" / "realtime_predictions.csv"
ALERTS_FILE = PROJECT_ROOT / "data" / "processed" / "realtime_alerts.json"

MODEL_DIR = PROJECT_ROOT / "data" / "models"
PLATE_PATH = PROJECT_ROOT / "data" / "raw" / "PB2002_boundaries.json"
GCMT_PATH = PROJECT_ROOT / "data" / "raw" / "GlobalCMT_1976-2024.csv"

# 预警阈值
ALERT_MAG_THRESHOLD = 6.5    # 预测最大余震 ≥ 此震级 → 强余震预警
ALERT_TIME_THRESHOLD = 1.0   # 预测时间 ≤ 此天数 → 紧迫预警


# ============================================================
#  USGS API 查询
# ============================================================

def query_usgs_events(
    start_time: datetime,
    end_time: datetime | None = None,
    min_magnitude: float = DEFAULT_MIN_MAG,
    max_results: int = 500,
) -> pd.DataFrame:
    """从 USGS FDSN API 查询地震事件。

    Args:
        start_time: 查询起始时间 (UTC)
        end_time: 查询结束时间 (UTC)，默认当前时刻
        min_magnitude: 最小震级
        max_results: 最大返回数

    Returns:
        含 time/latitude/longitude/mag/depth 的 DataFrame
    """
    if end_time is None:
        end_time = datetime.now(timezone.utc)

    params = {
        "format": "geojson",
        "starttime": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": min_magnitude,
        "orderby": "time",
        "limit": max_results,
    }

    try:
        resp = requests.get(USGS_FDSN_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  ⚠ USGS API 查询失败: {e}")
        return pd.DataFrame()

    features = data.get("features", [])
    if not features:
        return pd.DataFrame()

    records = []
    for feat in features:
        props = feat["properties"]
        geom = feat["geometry"]["coordinates"]
        records.append({
            "id": feat["id"],
            "time": pd.to_datetime(props["time"], unit="ms", utc=True),
            "latitude": float(geom[1]),
            "longitude": float(geom[0]),
            "depth": float(geom[2]),
            "mag": float(props["mag"]),
            "place": str(props.get("place", "")),
            "type": str(props.get("type", "")),
            "url": str(props.get("url", "")),
        })

    df = pd.DataFrame(records)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def query_aftershocks(
    mainshock_time: datetime,
    mainshock_lat: float,
    mainshock_lon: float,
    obs_days: float = DEFAULT_OBS_DAYS,
    radius_km: float = DEFAULT_SPATIAL_RADIUS_KM,
    min_magnitude: float = 4.0,
) -> pd.DataFrame:
    """查询主震后的早期余震（观测窗口内、空间半径内）。

    USGS API 支持按圆形区域查询:
      latitude + longitude + maxradiuskm
    """
    obs_end = mainshock_time + timedelta(days=obs_days)
    now = datetime.now(timezone.utc)
    query_end = min(obs_end, now)

    if query_end <= mainshock_time:
        return pd.DataFrame()

    params = {
        "format": "geojson",
        "starttime": mainshock_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": query_end.strftime("%Y-%m-%dT%H:%M:%S"),
        "latitude": mainshock_lat,
        "longitude": mainshock_lon,
        "maxradiuskm": radius_km,
        "minmagnitude": min_magnitude,
        "orderby": "time",
        "limit": 2000,
    }

    try:
        resp = requests.get(USGS_FDSN_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    ⚠ 余震查询失败: {e}")
        return pd.DataFrame()

    features = data.get("features", [])
    records = []
    for feat in features:
        props = feat["properties"]
        geom = feat["geometry"]["coordinates"]
        records.append({
            "id": feat["id"],
            "time": pd.to_datetime(props["time"], unit="ms", utc=True),
            "latitude": float(geom[1]),
            "longitude": float(geom[0]),
            "depth": float(geom[2]),
            "mag": float(props["mag"]),
            "place": str(props.get("place", "")),
        })

    return pd.DataFrame(records).sort_values("time").reset_index(drop=True)


# ============================================================
#  状态管理
# ============================================================

def load_state() -> dict:
    """加载处理状态。"""
    if STATE_FILE.exists():
        with STATE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed_events": {}, "last_poll_time": None}


def save_state(state: dict) -> None:
    """持久化处理状态。"""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, default=str)


def event_key(event_row: pd.Series) -> str:
    """为地震事件生成唯一键（基于 USGS event ID 的哈希）。"""
    raw = str(event_row.get("id", ""))
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def is_new_event(event_row: pd.Series, state: dict) -> bool:
    """判断事件是否未被处理过。"""
    key = event_key(event_row)
    return key not in state.get("processed_events", {})


def mark_event_processing(key: str, state: dict, status: str, **extra) -> None:
    """标记事件处理状态。"""
    state.setdefault("processed_events", {})[key] = {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **extra,
    }


# ============================================================
#  核心推理
# ============================================================

def build_event_table(mainshock: pd.Series, aftershocks: pd.DataFrame) -> pd.DataFrame:
    """将主震 + 余震合并为统一事件表（兼容 make_submission 的 normalize_event_table）。"""
    main_row = pd.DataFrame([{
        "time": mainshock["time"],
        "latitude": float(mainshock["latitude"]),
        "longitude": float(mainshock["longitude"]),
        "depth": float(mainshock["depth"]),
        "mag": float(mainshock["mag"]),
        "id": str(mainshock.get("id", "")),
        "place": str(mainshock.get("place", "")),
    }])

    if aftershocks.empty:
        return main_row

    aft_rows = aftershocks[["time", "latitude", "longitude", "depth", "mag", "id"]].copy()
    aft_rows["place"] = aftershocks.get("place", "")
    return pd.concat([main_row, aft_rows], ignore_index=True).sort_values("time").reset_index(drop=True)


def predict_single_mainshock(
    event_df: pd.DataFrame,
    model_dir: Path = MODEL_DIR,
    plate_path: Path = PLATE_PATH,
    gcmt_path: Path = GCMT_PATH,
    obs_days: float = DEFAULT_OBS_DAYS,
    spatial_radius_km: float = DEFAULT_SPATIAL_RADIUS_KM,
    use_gating: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """对单条主震序列运行完整集成推理。

    返回包含预测结果和元信息的字典。
    """
    # 导入推理模块
    from scripts.make_submission import (
        normalize_event_table,
        build_single_sequence_features,
        add_derived_features,
        load_feature_cols,
        make_model_matrix,
        load_ensemble_weights,
        predict_with_baseline,
        predict_with_dl,
        predict_with_gnn,
        postprocess_prediction,
        check_feature_consistency,
        get_positive_probability,
        rule_fallback_prediction,
    )

    result: dict[str, Any] = {
        "success": False,
        "mainshock_id": None,
        "mainshock_mag": None,
        "predicted_mag": None,
        "predicted_time_days": None,
        "early_aftershock_count": 0,
        "models_used": [],
        "prob_has_aftershock": None,
        "gated": False,
        "error": None,
    }

    try:
        # 1. 规范化事件表
        norm_df = normalize_event_table(event_df)
        if norm_df.empty:
            result["error"] = "事件表为空"
            return result

        # 2. 提取特征
        gcmt = gcmt_path if gcmt_path.exists() else None
        feature_df, early_events = build_single_sequence_features(
            norm_df,
            plate_boundaries_path=plate_path,
            gcmt_catalog_path=gcmt,
            obs_days=obs_days,
            spatial_radius_km=spatial_radius_km,
        )

        mainshock_id = str(feature_df.loc[0, "mainshock_id"])
        mainshock_mag = float(feature_df.loc[0, "mainshock_mag"])
        early_count = int(len(early_events))

        result["mainshock_id"] = mainshock_id
        result["mainshock_mag"] = mainshock_mag
        result["early_aftershock_count"] = early_count

        enriched_df = add_derived_features(feature_df.copy())

        # 3. 加载特征列
        feature_cols_path = model_dir / "feature_cols.json"
        if not feature_cols_path.exists():
            result["error"] = "缺少 feature_cols.json"
            return result
        feature_cols = load_feature_cols(feature_cols_path)

        check_feature_consistency(enriched_df, feature_cols, strict=False)
        X = make_model_matrix(enriched_df, feature_cols)

        # 4. 门控分类器
        gated_no_aftershock = False
        if use_gating:
            clf_path = model_dir / "aftershock_classifier.joblib"
            cls_meta_path = model_dir / "classifier_meta.json"
            if clf_path.exists():
                import joblib
                clf = joblib.load(clf_path)
                threshold = 0.5
                feature_cols_cls = feature_cols
                if cls_meta_path.exists():
                    with cls_meta_path.open() as f:
                        cls_meta = json.load(f)
                    threshold = float(cls_meta.get("threshold", 0.5))
                    if cls_meta.get("feature_cols"):
                        feature_cols_cls = cls_meta["feature_cols"]

                X_gate = make_model_matrix(enriched_df, feature_cols_cls)
                prob = get_positive_probability(clf, X_gate)
                result["prob_has_aftershock"] = prob
                if prob is not None and prob < threshold:
                    gated_no_aftershock = True
                    result["gated"] = True
                    if verbose:
                        print(f"  🚪 门控判定无余震 (prob={prob:.4f} < {threshold})")

        if gated_no_aftershock:
            result["predicted_mag"] = 0.0
            result["predicted_time_days"] = 0.0
            result["success"] = True
            result["models_used"] = ["classifier_gate"]
            return result

        # 5. 加载融合权重并运行各模型
        weights_path = model_dir / "ensemble_weights.json"
        weights = load_ensemble_weights(weights_path)
        mag_weights = weights.get("mag", weights)
        time_weights = weights.get("time", weights)

        model_preds: dict[str, np.ndarray] = {}

        # --- 树模型 ---
        baseline_path = model_dir / "baseline_model.joblib"
        if baseline_path.exists():
            bm = mag_weights.get("baseline", 1.0)
            bt = time_weights.get("baseline", 1.0)
            if bm > 0 or bt > 0:
                model_preds["baseline"] = predict_with_baseline(baseline_path, X)

        xgb_path = model_dir / "xgboost_model.joblib"
        if xgb_path.exists():
            xm = mag_weights.get("xgboost", 0.0)
            xt = time_weights.get("xgboost", 0.0)
            if xm > 0 or xt > 0:
                model_preds["xgboost"] = predict_with_baseline(xgb_path, X)

        # --- 深度学习模型 ---
        dl_path = model_dir / "dl_model.pt"
        dl_meta = model_dir / "dl_meta.json"
        dm = max(float(mag_weights.get("dl", 0.0)), 0.0)
        dt_val = max(float(time_weights.get("dl", 0.0)), 0.0)
        if (dm > 0 or dt_val > 0) and dl_path.exists():
            dl_pred = predict_with_dl(dl_path, dl_meta, norm_df, enriched_df, device="cuda")
            if dl_pred is not None:
                model_preds["dl"] = dl_pred
            elif verbose:
                print("  ⚠ DL 模型不可用，跳过")

        gnn_path = model_dir / "gnn_model.pt"
        gnn_meta = model_dir / "gnn_meta.json"
        gm = max(float(mag_weights.get("gnn", 0.0)), 0.0)
        gt = max(float(time_weights.get("gnn", 0.0)), 0.0)
        if (gm > 0 or gt > 0) and gnn_path.exists():
            gnn_pred = predict_with_gnn(gnn_path, gnn_meta, norm_df, enriched_df, device="cuda")
            if gnn_pred is not None:
                model_preds["gnn"] = gnn_pred
            elif verbose:
                print("  ⚠ GNN 模型不可用，跳过")

        if not model_preds:
            # 兜底：经验规则
            fallback = rule_fallback_prediction(mainshock_mag, early_count)
            model_preds["rule_fallback"] = fallback
            if verbose:
                print("  ⚠ 无可用模型，使用经验规则兜底")

        # 6. 加权融合
        fused_mag, fused_time = 0.0, 0.0
        total_mag_w, total_time_w = 0.0, 0.0
        for name, pred in model_preds.items():
            w_mag = max(float(mag_weights.get(name, 1.0 / len(model_preds))), 0.0)
            w_time = max(float(time_weights.get(name, 1.0 / len(model_preds))), 0.0)
            if w_mag > 0:
                fused_mag += pred[0, 0] * w_mag
                total_mag_w += w_mag
            if w_time > 0:
                fused_time += pred[0, 1] * w_time
                total_time_w += w_time

        pred_mag = fused_mag / total_mag_w if total_mag_w > 0 else model_preds[next(iter(model_preds))][0, 0]
        pred_time = fused_time / total_time_w if total_time_w > 0 else model_preds[next(iter(model_preds))][0, 1]

        fused = np.array([[pred_mag, pred_time]], dtype=float)
        predicted_mag, predicted_time = postprocess_prediction(
            fused, mainshock_mag=mainshock_mag, early_count=early_count,
        )

        result["predicted_mag"] = float(predicted_mag)
        result["predicted_time_days"] = float(predicted_time)
        result["models_used"] = list(model_preds.keys())
        result["success"] = True

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        if verbose:
            print(f"  ❌ 推理失败: {result['error']}")

    return result


# ============================================================
#  预警判断
# ============================================================

def check_alert(result: dict) -> dict | None:
    """根据预测结果生成预警。

    Returns:
        预警字典或 None（无需预警）
    """
    if not result.get("success"):
        return None

    mag = result.get("predicted_mag", 0)
    time_days = result.get("predicted_time_days", 999)

    alerts = []
    if mag >= ALERT_MAG_THRESHOLD:
        alerts.append(f"强余震预警: 预测最大余震 Mw{mag:.1f} ≥ {ALERT_MAG_THRESHOLD}")

    if 0 < time_days <= ALERT_TIME_THRESHOLD:
        alerts.append(f"时间紧迫: 预测最大余震发生在 {time_days * 24:.0f}h 内")

    if not alerts:
        return None

    return {
        "mainshock_id": result["mainshock_id"],
        "mainshock_mag": result["mainshock_mag"],
        "predicted_mag": mag,
        "predicted_time_days": time_days,
        "alerts": alerts,
        "severity": "HIGH" if mag >= ALERT_MAG_THRESHOLD and time_days <= ALERT_TIME_THRESHOLD else "MEDIUM",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "early_aftershock_count": result.get("early_aftershock_count", 0),
    }


# ============================================================
#  主循环
# ============================================================

def print_header():
    """打印工具头部信息。"""
    print("=" * 70)
    print("  🌍 余震预测实时监控系统")
    print("  数据源: USGS FDSN Earthquake API")
    print(f"  模型目录: {MODEL_DIR}")
    print("=" * 70)


def check_model_availability() -> dict[str, bool]:
    """检查模型产物是否齐全。"""
    checks = {
        "baseline_model.joblib": (MODEL_DIR / "baseline_model.joblib").exists(),
        "xgboost_model.joblib": (MODEL_DIR / "xgboost_model.joblib").exists(),
        "dl_model.pt": (MODEL_DIR / "dl_model.pt").exists(),
        "gnn_model.pt": (MODEL_DIR / "gnn_model.pt").exists(),
        "feature_cols.json": (MODEL_DIR / "feature_cols.json").exists(),
        "ensemble_weights.json": (MODEL_DIR / "ensemble_weights.json").exists(),
        "aftershock_classifier.joblib": (MODEL_DIR / "aftershock_classifier.joblib").exists(),
        "PB2002_boundaries.json": PLATE_PATH.exists(),
    }
    return checks


def run_once(
    min_mag: float = DEFAULT_MIN_MAG,
    backfill_days: int = 0,
    obs_days: float = DEFAULT_OBS_DAYS,
    device: str = "cuda",
) -> list[dict]:
    """执行一次完整轮询 + 推理。

    Args:
        min_mag: 最小主震震级
        backfill_days: 回填天数（0 = 仅处理当前新事件）
        obs_days: 观测窗口天数
        device: 推理设备

    Returns:
        本轮处理结果的列表
    """
    # 初始化 CUDA
    setup_cuda(device, allow_tf32=True, benchmark=True)

    # 加载状态
    state = load_state()

    # 确定查询时间范围
    now = datetime.now(timezone.utc)
    if backfill_days > 0:
        query_start = now - timedelta(days=backfill_days)
    else:
        last_poll = state.get("last_poll_time")
        if last_poll:
            query_start = datetime.fromisoformat(last_poll) - timedelta(hours=1)
        else:
            query_start = now - timedelta(days=1)

    # 查询候选主震
    print(f"\n📡 查询 USGS API: {query_start.strftime('%Y-%m-%d %H:%M')} → "
          f"{now.strftime('%Y-%m-%d %H:%M')} UTC, Mw≥{min_mag}")
    candidates = query_usgs_events(query_start, now, min_magnitude=min_mag)

    if candidates.empty:
        print("  ✓ 无新的强震事件")
        state["last_poll_time"] = now.isoformat()
        save_state(state)
        return []

    print(f"  发现 {len(candidates)} 个候选事件")

    # 筛选新事件
    new_events = []
    for _, row in candidates.iterrows():
        key = event_key(row)
        if is_new_event(row, state):
            new_events.append((key, row))
        else:
            existing = state["processed_events"].get(key, {})
            if existing.get("status") != "predicted":
                # 之前只标记了但未推理 → 重试
                new_events.append((key, row))

    if not new_events:
        print("  ✓ 所有事件均已处理")
        state["last_poll_time"] = now.isoformat()
        save_state(state)
        return []

    print(f"\n🔍 处理 {len(new_events)} 个新事件 …")

    # 推理循环
    all_results = []
    for key, ms_row in tqdm(new_events, desc="推理进度", unit="event"):
        ms_time = ms_row["time"]
        hours_ago = (now - ms_time).total_seconds() / 3600

        # 检查观测窗口是否已结束
        obs_end = ms_time + timedelta(days=obs_days)
        obs_ready = now >= obs_end
        obs_progress = min(100.0, max(0.0, (now - ms_time).total_seconds() / (obs_days * 86400) * 100))

        if not obs_ready and backfill_days == 0:
            tqdm.write(f"  ⏳ {ms_row.get('place','?'):<30s} M{ms_row['mag']:.1f} "
                       f"观测窗口 {obs_progress:.0f}% ({hours_ago:.0f}h / {obs_days*24:.0f}h)"
                       f" → 暂不推理")
            mark_event_processing(key, state, "waiting",
                                  mag=float(ms_row["mag"]),
                                  time=ms_time.isoformat(),
                                  place=str(ms_row.get("place", "")))
            continue

        tqdm.write(f"  🧠 {ms_row.get('place','?'):<30s} M{ms_row['mag']:.1f} "
                   f"({hours_ago:.0f}h 前) → 开始推理 …")

        # 标记处理中
        mark_event_processing(key, state, "processing",
                              mag=float(ms_row["mag"]),
                              time=ms_time.isoformat(),
                              place=str(ms_row.get("place", "")))
        save_state(state)

        # 查询余震
        aftershocks = query_aftershocks(
            ms_time,
            float(ms_row["latitude"]),
            float(ms_row["longitude"]),
            obs_days=obs_days,
            radius_km=DEFAULT_SPATIAL_RADIUS_KM,
        )
        tqdm.write(f"    早期余震: {len(aftershocks)} 条 (Mw≥4.0, {DEFAULT_SPATIAL_RADIUS_KM}km)")

        # 构建事件表并推理
        event_df = build_event_table(ms_row, aftershocks)

        # 触发推理前先预热 CUDA（避免首次调用时延迟被计入）
        pred_result = predict_single_mainshock(
            event_df,
            model_dir=MODEL_DIR,
            obs_days=obs_days,
            verbose=False,
        )

        # 增强结果
        pred_result["event_time"] = ms_time.isoformat()
        pred_result["event_place"] = str(ms_row.get("place", ""))
        pred_result["event_lat"] = float(ms_row["latitude"])
        pred_result["event_lon"] = float(ms_row["longitude"])
        pred_result["event_depth"] = float(ms_row["depth"])
        pred_result["event_url"] = str(ms_row.get("url", ""))
        pred_result["hours_since_event"] = round(hours_ago, 1)
        pred_result["key"] = key

        if pred_result["success"]:
            tqdm.write(f"    ✅ 预测: Mw={pred_result['predicted_mag']:.2f}, "
                       f"时间={pred_result['predicted_time_days']:.1f}d "
                       f"({pred_result['predicted_time_days'] * 24:.0f}h)")
            tqdm.write(f"    模型: {', '.join(pred_result['models_used'])}")
            mark_event_processing(key, state, "predicted",
                                  mag=float(ms_row["mag"]),
                                  time=ms_time.isoformat(),
                                  place=str(ms_row.get("place", "")),
                                  predicted_mag=pred_result["predicted_mag"],
                                  predicted_time=pred_result["predicted_time_days"])
        else:
            tqdm.write(f"    ❌ 失败: {pred_result.get('error', 'unknown')}")
            mark_event_processing(key, state, "failed",
                                  mag=float(ms_row["mag"]),
                                  time=ms_time.isoformat(),
                                  place=str(ms_row.get("place", "")),
                                  error=str(pred_result.get("error", "")))

        all_results.append(pred_result)
        save_state(state)

        # 检查预警
        alert = check_alert(pred_result)
        if alert:
            _save_alert(alert)
            tqdm.write(f"    🚨 预警! ({alert['severity']}) {'; '.join(alert['alerts'])}")

    # 更新轮询时间
    state["last_poll_time"] = now.isoformat()
    save_state(state)

    # 追加到预测 CSV
    if all_results:
        _append_predictions_csv(all_results)

    return all_results


# ============================================================
#  持久化
# ============================================================

def _append_predictions_csv(results: list[dict]) -> None:
    """追加预测结果到 CSV。"""
    rows = []
    for r in results:
        rows.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mainshock_id": r.get("mainshock_id", ""),
            "event_time": r.get("event_time", ""),
            "event_place": r.get("event_place", ""),
            "event_mag": r.get("mainshock_mag"),
            "event_lat": r.get("event_lat"),
            "event_lon": r.get("event_lon"),
            "event_depth": r.get("event_depth"),
            "hours_since_event": r.get("hours_since_event"),
            "early_aftershock_count": r.get("early_aftershock_count"),
            "predicted_mag": r.get("predicted_mag"),
            "predicted_time_days": r.get("predicted_time_days"),
            "models_used": ",".join(r.get("models_used", [])),
            "prob_has_aftershock": r.get("prob_has_aftershock"),
            "gated": r.get("gated", False),
            "success": r.get("success", False),
            "error": r.get("error", ""),
        })

    df = pd.DataFrame(rows)
    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)

    if PREDICTIONS_FILE.exists():
        existing = pd.read_csv(PREDICTIONS_FILE)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(PREDICTIONS_FILE, index=False, encoding="utf-8")


def _save_alert(alert: dict) -> None:
    """保存预警到 JSON 文件。"""
    ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    alerts = []
    if ALERTS_FILE.exists():
        with ALERTS_FILE.open("r", encoding="utf-8") as f:
            try:
                alerts = json.load(f)
            except json.JSONDecodeError:
                pass

    alerts.append({k: str(v) if isinstance(v, (pd.Timestamp,)) else v
                   for k, v in alert.items()})

    with ALERTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(alerts, f, ensure_ascii=False, indent=2, default=str)


# ============================================================
#  打印摘要
# ============================================================

def print_summary(results: list[dict]) -> None:
    """打印本轮推理摘要。"""
    if not results:
        return

    success = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    print(f"\n{'─' * 60}")
    print(f"📊 本轮摘要: {len(success)} 成功, {len(failed)} 失败")

    if success:
        for r in success:
            mag = r.get("predicted_mag", "?")
            t = r.get("predicted_time_days", "?")
            if isinstance(t, (int, float)):
                t_str = f"{t:.1f}d ({t * 24:.0f}h)"
            else:
                t_str = str(t)
            print(f"  ✅ {r.get('event_place', '?'):<30s} "
                  f"→ 预测余震 Mw={mag}, 时间={t_str}")

    if failed:
        for r in failed:
            print(f"  ❌ {r.get('event_place', '?'):<30s} → {r.get('error', '?')}")

    print(f"\n📁 预测记录: {PREDICTIONS_FILE}")
    print(f"📁 预警记录: {ALERTS_FILE}")
    print(f"📁 状态文件: {STATE_FILE}")


# ============================================================
#  CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="🌍 余震预测实时监控与自动推理系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/realtime_monitor.py                     # 持续监控 (每 5 分钟)
  python scripts/realtime_monitor.py --once              # 单次运行
  python scripts/realtime_monitor.py --backfill 30       # 回填最近 30 天
  python scripts/realtime_monitor.py --min-mag 5.5       # 降低阈值
  python scripts/realtime_monitor.py --no-gating         # 关闭门控
  python scripts/realtime_monitor.py --device cpu        # CPU 推理
        """,
    )
    p.add_argument("--once", action="store_true",
                   help="单次运行后退出（默认持续监控）")
    p.add_argument("--interval", type=int, default=DEFAULT_POLL_INTERVAL,
                   help=f"轮询间隔 (秒, 默认 {DEFAULT_POLL_INTERVAL})")
    p.add_argument("--min-mag", type=float, default=DEFAULT_MIN_MAG,
                   help=f"最小主震震级 (默认 {DEFAULT_MIN_MAG})")
    p.add_argument("--obs-days", type=float, default=DEFAULT_OBS_DAYS,
                   help=f"观测窗口天数 (默认 {DEFAULT_OBS_DAYS})")
    p.add_argument("--backfill", type=int, default=0, metavar="DAYS",
                   help="回填最近 N 天内的强震 (不等待观测窗口)")
    p.add_argument("--no-gating", action="store_true",
                   help="禁用两阶段门控")
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "cpu", "auto"],
                   help="推理设备 (默认 cuda)")
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR,
                   help="模型产物目录")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)

    global MODEL_DIR
    MODEL_DIR = args.model_dir.resolve()

    print_header()

    # 检查模型
    print("\n📦 模型状态:")
    checks = check_model_availability()
    for name, ok in checks.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")
    if not any(checks.values()):
        print("\n❌ 无可用模型产物，请先运行训练。")
        sys.exit(1)

    print(f"\n⚙️  配置: 震级≥M{args.min_mag}, 观测窗口={args.obs_days}d, "
          f"设备={args.device}, 门控={'关' if args.no_gating else '开'}")
    print(f"   轮询间隔={args.interval}s, 模式={'单次' if args.once else '持续'}")

    if args.backfill > 0:
        print(f"\n⏪ 回填模式: 处理最近 {args.backfill} 天的强震 …")
        results = run_once(
            min_mag=args.min_mag,
            backfill_days=args.backfill,
            obs_days=args.obs_days,
            device=args.device,
        )
        print_summary(results)
        return

    if args.once:
        results = run_once(
            min_mag=args.min_mag,
            device=args.device,
            obs_days=args.obs_days,
        )
        print_summary(results)
        return

    # ── 持续监控模式 ──
    print(f"\n🔄 进入持续监控模式 (Ctrl+C 退出)")
    print(f"   下次轮询: {args.interval}s 后 …")
    try:
        while True:
            results = run_once(
                min_mag=args.min_mag,
                device=args.device,
                obs_days=args.obs_days,
            )
            print_summary(results)
            print(f"\n⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')} "
                  f"— 等待 {args.interval}s …")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n\n👋 监控已停止。")


if __name__ == "__main__":
    main()
