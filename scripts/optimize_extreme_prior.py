#!/usr/bin/env python3
"""资格赛极端震级先验优化器。

用途：
1. 读取 final_t123 的预测包与可见诊断标签；
2. 对高震级主震的震级 floor 规则做网格搜索；
3. 用 OOF 指标约束过拟合风险；
4. 生成 safe / balanced / public_max 三个候选提交包。

注意：public_max 会直接追逐可见诊断集分数，线上风险更高；默认推荐 safe 或 balanced。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


WINDOWS = ("T1", "T2", "T3")
SEED = 42


@dataclass(frozen=True)
class Rule:
    """单个窗口的高震级 floor 规则。"""

    threshold: float
    margin: float
    strength: float
    cap: float = 0.5


def parse_line(line: str) -> dict | None:
    """解析比赛提交文件的一行：id lon lat main_mag pred_mag (Ms) time。"""
    parts = line.strip().split()
    if len(parts) < 7:
        return None
    return {
        "token": parts[0],
        "lon": float(parts[1]),
        "lat": float(parts[2]),
        "main_mag": float(parts[3]),
        "pred_mag": float(parts[4]),
        "pred_time_str": parts[-1],
    }


def load_base_predictions(package_dir: Path) -> pd.DataFrame:
    """从 final_t123 目录读取 T1/T2/T3 预测。"""
    pred_dir = package_dir / "predictions"
    records: list[dict] = []

    for fp in sorted(pred_dir.glob("*-T1-T2.csv")):
        for i, line in enumerate(fp.read_text(encoding="utf-8").splitlines()):
            row = parse_line(line)
            if row:
                row["window"] = "T1" if i == 0 else "T2"
                records.append(row)

    for fp in sorted(pred_dir.glob("*-T3.csv")):
        for line in fp.read_text(encoding="utf-8").splitlines():
            row = parse_line(line)
            if row:
                row["window"] = "T3"
                records.append(row)

    return pd.DataFrame(records)


def apply_rule_array(pred_mag: np.ndarray, main_mag: np.ndarray, rule: Rule) -> np.ndarray:
    """向量化应用 floor 规则，只上调明显偏低的预测震级。"""
    corrected = pred_mag.astype(float).copy()
    floor_value = main_mag - rule.margin
    mask = (main_mag >= rule.threshold) & (floor_value > corrected)
    if np.any(mask):
        corrected[mask] = corrected[mask] + rule.strength * (floor_value[mask] - corrected[mask])
        corrected[mask] = np.minimum(corrected[mask], main_mag[mask] - rule.cap)
    return np.round(corrected, 1)


def evaluate_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs": float(np.max(np.abs(err))),
    }


def load_oof_window(features_path: Path, oof_path: Path, window: str) -> pd.DataFrame:
    """读取指定窗口的 OOF 回归预测，用来约束规则不能只贴合可见集。"""
    target_col = f"target_{window}_max_mag"
    feat_cols = ["mainshock_id", "mainshock_mag", target_col]
    feat = pd.read_csv(features_path, usecols=feat_cols)
    oof = pd.read_csv(oof_path)
    oof_w = oof[oof["window"] == window].dropna(subset=["decoupled_mag_raw"])

    df = feat.merge(oof_w[["mainshock_id", "decoupled_mag_raw"]], on="mainshock_id", how="inner")
    df = df.dropna(subset=[target_col, "decoupled_mag_raw"])
    df = df[df[target_col] > 0].copy()
    df = df.rename(columns={target_col: "true_mag", "decoupled_mag_raw": "pred_mag"})
    return df[["mainshock_id", "mainshock_mag", "true_mag", "pred_mag"]]


def search_window(
    visible_df: pd.DataFrame,
    oof_df: pd.DataFrame,
    window: str,
    thresholds: list[float],
    margins: list[float],
    strengths: list[float],
    cap: float,
) -> pd.DataFrame:
    """搜索单个窗口的规则参数，并同时记录可见集与 OOF 指标。"""
    v = visible_df[visible_df["window"] == window].copy()
    vp = v["pred_mag"].to_numpy(float)
    vy = v["true_mag"].to_numpy(float)
    vm = v["main_mag"].to_numpy(float)

    op = oof_df["pred_mag"].to_numpy(float)
    oy = oof_df["true_mag"].to_numpy(float)
    om = oof_df["mainshock_mag"].to_numpy(float)
    oof_base = evaluate_arrays(oy, op)

    rows: list[dict] = []
    for threshold in thresholds:
        for margin in margins:
            for strength in strengths:
                rule = Rule(threshold=threshold, margin=margin, strength=strength, cap=cap)
                vc = apply_rule_array(vp, vm, rule)
                oc = apply_rule_array(op, om, rule)

                vis_m = evaluate_arrays(vy, vc)
                oof_m = evaluate_arrays(oy, oc)
                high = om >= threshold
                high_delta = np.nan
                if np.any(high):
                    high_before = np.mean(np.abs(op[high] - oy[high]))
                    high_after = np.mean(np.abs(oc[high] - oy[high]))
                    high_delta = float(high_after - high_before)

                rows.append(
                    {
                        "window": window,
                        **asdict(rule),
                        "visible_mae": vis_m["mae"],
                        "visible_rmse": vis_m["rmse"],
                        "visible_max_abs": vis_m["max_abs"],
                        "visible_changed": int(np.sum(vc != vp)),
                        "oof_mae_delta": oof_m["mae"] - oof_base["mae"],
                        "oof_rmse_delta": oof_m["rmse"] - oof_base["rmse"],
                        "oof_high_mae_delta": high_delta,
                        "oof_changed": int(np.sum(oc != np.round(op, 1))),
                    }
                )

    return pd.DataFrame(rows)


def choose_rule(candidates: pd.DataFrame, mode: str) -> Rule:
    """从候选中选择规则。safe 最保守，public_max 最追分。"""
    df = candidates.copy()
    if mode == "safe":
        df = df[(df["oof_rmse_delta"] <= 5e-4) & (df["oof_high_mae_delta"] <= 0)]
    elif mode == "balanced":
        df = df[(df["oof_mae_delta"] <= 0) & (df["oof_high_mae_delta"] <= 0)]
    elif mode == "public_max":
        pass
    else:
        raise ValueError(f"未知模式：{mode}")

    if df.empty:
        df = candidates.copy()

    row = df.sort_values(["visible_rmse", "visible_mae", "visible_changed"]).iloc[0]
    return Rule(
        threshold=float(row["threshold"]),
        margin=float(row["margin"]),
        strength=float(row["strength"]),
        cap=float(row["cap"]),
    )


def apply_rules_to_predictions(preds: pd.DataFrame, rules: dict[str, Rule]) -> pd.DataFrame:
    """对完整提交预测应用每个窗口的最优规则。"""
    out = preds.copy()
    out["original_pred_mag"] = out["pred_mag"]
    out["adjusted"] = False

    for window, rule in rules.items():
        mask = out["window"] == window
        corrected = apply_rule_array(
            out.loc[mask, "pred_mag"].to_numpy(float),
            out.loc[mask, "main_mag"].to_numpy(float),
            rule,
        )
        out.loc[mask, "pred_mag"] = corrected
        out.loc[mask, "adjusted"] = corrected != out.loc[mask, "original_pred_mag"].to_numpy(float)

    return out


def evaluate_visible(visible: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """用可见诊断集评估候选包。"""
    visible = visible.copy()
    preds = preds.copy()
    visible["mainshock_id"] = visible["mainshock_id"].astype(str)
    preds["token"] = preds["token"].astype(str)

    merged = visible.merge(
        preds[["token", "window", "pred_mag"]].rename(columns={"token": "mainshock_id", "pred_mag": "new_pred_mag"}),
        on=["mainshock_id", "window"],
        how="inner",
    )

    rows: list[dict] = []
    for window in [*WINDOWS, "ALL"]:
        df = merged if window == "ALL" else merged[merged["window"] == window]
        m = evaluate_arrays(df["true_mag"].to_numpy(float), df["new_pred_mag"].to_numpy(float))
        rows.append({"window": window, "count": len(df), **m})
    return pd.DataFrame(rows)


def write_package(base_package_dir: Path, preds: pd.DataFrame, out_dir: Path, manifest: dict) -> None:
    """写出候选提交目录。"""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for token in sorted(preds["token"].unique()):
        rows = preds[preds["token"] == token]
        t1 = rows[rows["window"] == "T1"].iloc[0]
        t2 = rows[rows["window"] == "T2"].iloc[0]
        t3 = rows[rows["window"] == "T3"].iloc[0]

        (pred_dir / f"{token}-T1-T2.csv").write_text(
            "\n".join(
                [
                    f"{token} {t1.lon:.2f} {t1.lat:.2f} {t1.main_mag:.1f} {t1.pred_mag:.1f} (Ms) {t1.pred_time_str}",
                    f"{token} {t2.lon:.2f} {t2.lat:.2f} {t2.main_mag:.1f} {t2.pred_mag:.1f} (Ms) {t2.pred_time_str}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (pred_dir / f"{token}-T3.csv").write_text(
            f"{token} {t3.lon:.2f} {t3.lat:.2f} {t3.main_mag:.1f} {t3.pred_mag:.1f} (Ms) {t3.pred_time_str}\n",
            encoding="utf-8",
        )

    tech_src = base_package_dir / "technical_docs"
    if tech_src.exists():
        shutil.copytree(tech_src, out_dir / "technical_docs")

    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def zip_dir(out_dir: Path, zip_path: Path) -> str:
    """压缩候选目录并返回 SHA256。"""
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(out_dir.rglob("*")):
            if fp.is_file():
                zf.write(fp, fp.relative_to(out_dir))

    sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    zip_path.with_suffix(zip_path.suffix + ".sha256").write_text(sha, encoding="utf-8")
    return sha


def main() -> None:
    parser = argparse.ArgumentParser(description="优化资格赛极端震级先验规则并生成候选包")
    parser.add_argument("--project-root", type=Path, default=Path("/home/ningyd/CodingSpace/aftershock_qualification_train"))
    parser.add_argument("--base-package-dir", type=Path, default=None)
    parser.add_argument("--visible-labels", type=Path, default=None)
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--oof", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--cap", type=float, default=0.5)
    args = parser.parse_args()

    np.random.seed(SEED)
    root = args.project_root
    base_dir = args.base_package_dir or root / "submission_package_final_t123_no_commitment"
    visible_path = args.visible_labels or root / "reports/final_t123_test_gap_details.csv"
    features_path = args.features or root / "data/processed/qualification_features.csv"
    oof_path = args.oof or root / "data/models/qualification_decoupled_full/decoupled_oof_predictions.csv"
    out_root = args.out_root or root / "experiments/extreme_prior_r3"
    out_root.mkdir(parents=True, exist_ok=True)

    visible = pd.read_csv(visible_path)
    base_preds = load_base_predictions(base_dir)

    thresholds = [7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 8.0, 8.2, 8.5]
    margins = [round(x, 1) for x in np.arange(0.8, 3.61, 0.1)]
    strengths = [round(x, 1) for x in np.arange(0.1, 1.01, 0.1)]

    all_candidates: list[pd.DataFrame] = []
    for window in WINDOWS:
        print(f"[1/4] 搜索 {window} 参数...")
        oof_w = load_oof_window(features_path, oof_path, window)
        cand = search_window(visible, oof_w, window, thresholds, margins, strengths, args.cap)
        cand.to_csv(out_root / f"{window}_grid_candidates.csv", index=False)
        all_candidates.append(cand)

    modes = ["safe", "balanced", "public_max"]
    summary_rows: list[pd.DataFrame] = []

    for mode in modes:
        print(f"[2/4] 生成 {mode} 候选包...")
        rules = {w: choose_rule(cand, mode) for w, cand in zip(WINDOWS, all_candidates)}
        corrected = apply_rules_to_predictions(base_preds, rules)
        metrics = evaluate_visible(visible, corrected)
        metrics.insert(0, "package", mode)
        summary_rows.append(metrics)

        package_dir = out_root / f"package_{mode}"
        zip_path = out_root / f"qualification_submission_extreme_prior_r3_{mode}.zip"
        manifest = {
            "package_name": f"qualification_submission_extreme_prior_r3_{mode}",
            "created": datetime.now(timezone.utc).isoformat(),
            "base": str(base_dir),
            "mode": mode,
            "rules": {k: asdict(v) for k, v in rules.items()},
            "adjusted_count": int(corrected["adjusted"].sum()),
            "seed": SEED,
        }
        write_package(base_dir, corrected, package_dir, manifest)
        sha = zip_dir(package_dir, zip_path)
        corrected[corrected["adjusted"]].to_csv(out_root / f"adjusted_{mode}.csv", index=False)
        print(f"  {zip_path.name} SHA256={sha[:16]}... adjusted={manifest['adjusted_count']}")
        print(metrics.to_string(index=False))

    print("[3/4] 写出汇总报告...")
    summary = pd.concat(summary_rows, ignore_index=True)
    summary.to_csv(out_root / "summary.csv", index=False)

    best = summary[summary["window"] == "ALL"].sort_values(["rmse", "mae"]).iloc[0]
    report = [
        "# Extreme Prior R3 优化报告",
        "",
        f"- 生成时间: {datetime.now().isoformat(timespec='seconds')}",
        f"- 推荐候选: `{best['package']}`",
        f"- ALL MagMAE: `{best['mae']:.4f}`",
        f"- ALL MagRMSE: `{best['rmse']:.4f}`",
        "",
        "## 候选包",
    ]
    for mode in modes:
        report.append(f"- `{mode}`: `{out_root / f'qualification_submission_extreme_prior_r3_{mode}.zip'}`")
    report.append("")
    report.append("## 指标")
    report.append("```csv")
    report.append(summary.to_csv(index=False).strip())
    report.append("```")
    (out_root / "recommendation.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("[4/4] 完成")
    print(f"  汇总: {out_root / 'summary.csv'}")
    print(f"  报告: {out_root / 'recommendation.md'}")


if __name__ == "__main__":
    main()
