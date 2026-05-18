from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "aftershock_research_report"
FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"


def ensure_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def configure_plot_style() -> None:
    """设置适合中文研究报告的绘图风格。"""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        "STHeiti",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 240


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict, list[str]]:
    features = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "advanced_features.csv")
    features["mainshock_time"] = pd.to_datetime(
        features["mainshock_time"], utc=True, errors="coerce", format="mixed"
    )
    cv_metrics = pd.read_csv(PROJECT_ROOT / "data" / "models" / "cv_metrics.csv")
    submission = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "submission.csv")
    with (PROJECT_ROOT / "data" / "models" / "model_meta.json").open(
        "r", encoding="utf-8"
    ) as file:
        model_meta = json.load(file)
    with (PROJECT_ROOT / "data" / "models" / "ensemble_weights.json").open(
        "r", encoding="utf-8"
    ) as file:
        raw_ensemble_weights = json.load(file)
    with (PROJECT_ROOT / "data" / "models" / "feature_cols.json").open(
        "r", encoding="utf-8"
    ) as file:
        feature_cols = json.load(file)
    ensemble_weights = normalize_ensemble_weights(raw_ensemble_weights, model_meta)
    return features, cv_metrics, submission, model_meta, ensemble_weights, feature_cols


def normalize_ensemble_weights(raw_weights: dict, model_meta: dict) -> dict[str, float]:
    """兼容扁平权重和按目标拆分的嵌套权重格式。"""
    try:
        return {key: float(value) for key, value in raw_weights.items()}
    except (TypeError, ValueError):
        meta_weights = model_meta.get("ensemble_weights", {})
        if meta_weights:
            return {key: float(value) for key, value in meta_weights.items()}

    flat = {"baseline": 0.0, "xgboost": 0.0, "dl": 0.0, "gnn": 0.0}
    nested_groups = [
        value for value in raw_weights.values()
        if isinstance(value, dict)
    ]
    if nested_groups:
        for group in nested_groups:
            for key, value in group.items():
                flat[key] = flat.get(key, 0.0) + float(value) / len(nested_groups)
    return flat


def save_current_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / name, bbox_inches="tight")
    plt.close()


def latex_escape(value) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def write_table(path: Path, headers: list[str], rows: list[list], colspec: str) -> None:
    lines = [
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
    ]
    lines.append(" & ".join(latex_escape(h) for h in headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(latex_escape(item) for item in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_longtable(path: Path, headers: list[str], rows: list[list], colspec: str) -> None:
    lines = [
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{longtable}}{{{colspec}}}",
        r"\toprule",
    ]
    lines.append(" & ".join(latex_escape(h) for h in headers) + r" \\")
    lines.extend([r"\midrule", r"\endfirsthead", r"\toprule"])
    lines.append(" & ".join(latex_escape(h) for h in headers) + r" \\")
    lines.extend([r"\midrule", r"\endhead"])
    for row in rows:
        lines.append(" & ".join(latex_escape(item) for item in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\endgroup", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figures(
    features: pd.DataFrame,
    cv_metrics: pd.DataFrame,
    submission: pd.DataFrame,
    model_meta: dict,
    ensemble_weights: dict,
) -> None:
    years = features["mainshock_time"].dt.year.dropna().astype(int)

    plt.figure(figsize=(8, 4.5))
    sns.histplot(years, binwidth=2, color="#4267B2")
    plt.xlabel("主震年份")
    plt.ylabel("样本数")
    plt.title("主震样本时间分布")
    save_current_figure("mainshock_year_distribution.png")

    plt.figure(figsize=(7, 4.3))
    sns.histplot(features["mainshock_mag"], bins=28, color="#2A9D8F")
    plt.xlabel("主震矩震级 Mw")
    plt.ylabel("样本数")
    plt.title("主震震级分布")
    save_current_figure("mainshock_magnitude_distribution.png")

    plt.figure(figsize=(7, 4.3))
    sns.histplot(features["target_max_mag"], bins=32, color="#E76F51")
    plt.xlabel("未来窗口最大余震 Mw")
    plt.ylabel("样本数")
    plt.title("目标最大余震震级分布")
    save_current_figure("target_max_mag_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    sns.histplot(features["target_time_to_max_days"], bins=35, color="#6A4C93", ax=axes[0])
    axes[0].set_xlabel("真实发生时间差（天）")
    axes[0].set_title("原始时间目标")
    sns.histplot(np.log1p(features["target_time_to_max_days"]), bins=35, color="#F4A261", ax=axes[1])
    axes[1].set_xlabel(r"$\log(1+t)$")
    axes[1].set_title("log1p 后的时间目标")
    for ax in axes:
        ax.set_ylabel("样本数")
    save_current_figure("target_time_log_transform.png")

    plt.figure(figsize=(7.5, 4.3))
    capped = features["advanced_early_event_count"].clip(upper=60)
    sns.histplot(capped, bins=30, color="#577590")
    plt.xlabel("早期余震数量（>60 已截断）")
    plt.ylabel("样本数")
    plt.title("观测窗口内早期余震数量分布")
    save_current_figure("early_aftershock_count_distribution.png")

    validity = pd.DataFrame(
        {
            "特征模块": ["GR b值", "Omori", "空间各向异性", "ETAS", "Båth", "GCMT"],
            "有效样本数": [
                int(features["gr_valid"].fillna(False).sum()),
                int(features["omori_valid"].fillna(False).sum()),
                int(features["anisotropy_valid"].fillna(False).sum()),
                int(features["etas_valid"].fillna(False).sum()),
                int(features["bath_valid"].fillna(False).sum()),
                int(features["focal_mechanism_valid"].fillna(False).sum()),
            ],
        }
    )
    validity["有效率"] = validity["有效样本数"] / len(features)
    plt.figure(figsize=(8, 4.4))
    ax = sns.barplot(data=validity, x="特征模块", y="有效率", hue="特征模块", palette="Set2", legend=False)
    ax.set_ylim(0, 1)
    ax.set_ylabel("有效率")
    ax.set_title("高级地震学特征有效率")
    for idx, row in validity.iterrows():
        ax.text(idx, row["有效率"] + 0.02, f"{row['有效样本数']}", ha="center", fontsize=9)
    save_current_figure("feature_validity_bar.png")

    plt.figure(figsize=(6.5, 4.2))
    plate_counts = features["nearest_plate_boundary_type"].fillna("UNK").value_counts()
    sns.barplot(x=plate_counts.index, y=plate_counts.values, hue=plate_counts.index, palette="crest", legend=False)
    plt.xlabel("最近板块边界类型")
    plt.ylabel("样本数")
    plt.title("板块构造类型分布")
    save_current_figure("plate_type_distribution.png")

    plt.figure(figsize=(7, 4.2))
    fault_counts = features["fault_type"].fillna("Unknown").value_counts()
    sns.barplot(x=fault_counts.index, y=fault_counts.values, hue=fault_counts.index, palette="mako", legend=False)
    plt.xlabel("震源机制类型")
    plt.ylabel("样本数")
    plt.title("Global CMT 断层滑动类型分布")
    save_current_figure("fault_type_distribution.png")

    plt.figure(figsize=(7.5, 4.4))
    sampled = features.sample(min(len(features), 2500), random_state=42)
    sns.scatterplot(
        data=sampled,
        x="bath_deficit",
        y="target_max_mag",
        hue="fault_type",
        alpha=0.55,
        s=18,
        linewidth=0,
    )
    plt.xlabel("Båth 缺口：主震 Mw - 早期最大余震 Mw")
    plt.ylabel("未来窗口最大余震 Mw")
    plt.title("Båth's Law 特征与最大余震强度")
    plt.legend(title="断层类型", fontsize=8, title_fontsize=9)
    save_current_figure("bath_deficit_vs_target.png")

    plt.figure(figsize=(7.5, 4.4))
    valid_prod = features.dropna(subset=["productivity_index", "target_max_mag"]).copy()
    valid_prod = valid_prod.sample(min(len(valid_prod), 1800), random_state=42)
    sns.scatterplot(
        data=valid_prod,
        x="productivity_index",
        y="target_max_mag",
        hue="nearest_plate_boundary_type",
        alpha=0.6,
        s=20,
        linewidth=0,
    )
    plt.xlabel(r"生产率指数 $a-bM_{main}$")
    plt.ylabel("未来窗口最大余震 Mw")
    plt.title("生产率指数与最大余震强度")
    plt.legend(title="板块类型", fontsize=8, title_fontsize=9)
    save_current_figure("productivity_index_vs_target.png")

    cv_long = cv_metrics.melt(
        id_vars=["fold", "model"],
        value_vars=["mag_rmse", "time_rmse"],
        var_name="指标",
        value_name="数值",
    )
    cv_long["指标"] = cv_long["指标"].map({"mag_rmse": "震级RMSE", "time_rmse": "时间RMSE"})
    plt.figure(figsize=(8, 4.5))
    sns.lineplot(data=cv_long, x="fold", y="数值", hue="model", style="指标", markers=True, dashes=False)
    plt.xlabel("TimeSeriesSplit 折数")
    plt.ylabel("RMSE")
    plt.title("LightGBM 与 XGBoost 各折 RMSE 对比")
    save_current_figure("cv_rmse_by_fold.png")

    asym = cv_metrics.groupby("model", as_index=False)[
        ["time_mae", "time_asymmetric_mae", "time_asymmetric_rmse"]
    ].mean()
    asym_long = asym.melt(id_vars="model", var_name="指标", value_name="数值")
    asym_long["指标"] = asym_long["指标"].map(
        {
            "time_mae": "Time MAE",
            "time_asymmetric_mae": "Asym MAE",
            "time_asymmetric_rmse": "Asym RMSE",
        }
    )
    plt.figure(figsize=(7.5, 4.4))
    sns.barplot(data=asym_long, x="指标", y="数值", hue="model", palette="Set1")
    plt.xlabel("时间预测指标")
    plt.ylabel("平均值")
    plt.title("非对称时间惩罚指标对比")
    save_current_figure("time_asymmetric_metrics.png")

    weights = pd.DataFrame(
        {"模型": list(ensemble_weights.keys()), "权重": list(map(float, ensemble_weights.values()))}
    )
    plt.figure(figsize=(6.8, 4.1))
    sns.barplot(data=weights, x="模型", y="权重", hue="模型", palette="flare", legend=False)
    plt.ylim(0, max(1.0, weights["权重"].max() + 0.1))
    plt.title("OOF 搜索得到的融合权重")
    save_current_figure("ensemble_weights.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    sns.barplot(data=submission, x="mainshock_id", y="predicted_max_mag", color="#2A9D8F", ax=axes[0])
    axes[0].set_title("20 条测试序列预测震级")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("预测最大余震 Mw")
    axes[0].tick_params(axis="x", rotation=90, labelsize=6)
    sns.barplot(data=submission, x="mainshock_id", y="predicted_time_to_max", color="#E76F51", ax=axes[1])
    axes[1].set_title("20 条测试序列预测时间")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("预测时间差（天）")
    axes[1].tick_params(axis="x", rotation=90, labelsize=6)
    save_current_figure("submission_prediction_distribution.png")

    valid_gcmt = features[features["focal_mechanism_valid"].fillna(False)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    sns.histplot(valid_gcmt["gcmt_time_diff_seconds"], bins=30, color="#457B9D", ax=axes[0])
    axes[0].set_xlabel("GCMT 匹配时间差（秒）")
    axes[0].set_title("GCMT 时间匹配质量")
    sns.histplot(valid_gcmt["gcmt_distance_km"], bins=30, color="#A8DADC", ax=axes[1])
    axes[1].set_xlabel("GCMT 匹配空间距离（km）")
    axes[1].set_title("GCMT 空间匹配质量")
    save_current_figure("gcmt_match_quality.png")

    make_pipeline_figure()
    make_stgnn_figure()


def make_pipeline_figure() -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    nodes = [
        ("原始数据\nUSGS / PB2002 / GCMT", 0.08, 0.58, "#A8DADC"),
        ("序列构建\n主震-早期余震", 0.28, 0.58, "#F1FAEE"),
        ("高级特征\nGR / Omori / Båth / GCMT", 0.50, 0.58, "#FFE8A3"),
        ("模型训练\nLGBM / XGB / DL / GNN", 0.72, 0.58, "#FFCAD4"),
        ("推理提交\nsubmission.csv", 0.90, 0.58, "#CDB4DB"),
    ]
    for text, x, y, color in nodes:
        box = FancyBboxPatch(
            (x - 0.08, y - 0.12),
            0.16,
            0.24,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            fc=color,
            ec="#333333",
            lw=1.1,
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=10)
    for (_, x1, y1, _), (_, x2, y2, _) in zip(nodes[:-1], nodes[1:]):
        arrow = FancyArrowPatch(
            (x1 + 0.085, y1),
            (x2 - 0.085, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            lw=1.4,
            color="#333333",
        )
        ax.add_patch(arrow)
    ax.text(0.5, 0.22, "全流程固定随机种子，并采用时间序列切分防止未来信息泄漏", ha="center", fontsize=11)
    plt.title("余震预测工程流水线", fontsize=14)
    save_current_figure("pipeline_flow.png")


def make_stgnn_figure() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.axis("off")
    coords = np.array([[0.12, 0.28], [0.30, 0.55], [0.50, 0.38], [0.72, 0.70], [0.86, 0.43]])
    times = ["t1", "t2", "t3", "t4", "t5"]
    for i, ((x, y), label) in enumerate(zip(coords, times), start=1):
        ax.scatter([x], [y], s=520, color="#4C78A8", edgecolor="white", zorder=3)
        ax.text(x, y, f"e{i}\n{label}", ha="center", va="center", color="white", fontsize=10, zorder=4)
    edges = [(0, 1), (1, 2), (2, 3), (2, 4), (0, 2)]
    for i, j in edges:
        arrow = FancyArrowPatch(
            coords[i],
            coords[j],
            arrowstyle="-|>",
            mutation_scale=16,
            lw=1.5,
            color="#E76F51",
            alpha=0.85,
        )
        ax.add_patch(arrow)
    ax.text(
        0.5,
        0.08,
        "构图约束：距离 < R 且 time[source] < time[target]，信息仅从过去流向未来",
        ha="center",
        fontsize=11,
    )
    plt.title("ST-GNN 有向时空因果图示意", fontsize=14)
    save_current_figure("stgnn_causal_graph.png")


def make_tables(
    features: pd.DataFrame,
    cv_metrics: pd.DataFrame,
    submission: pd.DataFrame,
    model_meta: dict,
    ensemble_weights: dict,
    feature_cols: list[str],
) -> None:
    data_sources = [
        ["USGS Mw>=4.0 浅源目录", "USGS_Mw4.0_Depth70_1970-2023.csv", "重建主震后 3 天早期余震序列"],
        ["USGS Mw>=6.0 主震目录", "USGS_Mw6.0_Depth70_1970-2023.csv", "确定强震主震样本"],
        ["PB2002 板块边界", "PB2002_boundaries.json", "最近板块边界距离与边界类型"],
        ["Global CMT", "GlobalCMT_1976-2024.csv", "震源机制 strike/dip/rake 与断层类型"],
        ["比赛验证序列", "data/test_sequences/*_eq.csv", "20 条单序列推理验证"],
    ]
    write_table(TABLE_DIR / "data_sources.tex", ["数据源", "路径", "用途"], data_sources, "p{0.24\\textwidth}p{0.36\\textwidth}p{0.30\\textwidth}")

    feature_groups = [
        ["地震活动性", "early/count/energy", "早期余震数量、能量、分箱释放过程"],
        ["G-R 定律", "gr_b_value/gr_a_value/gr_mc", "MAXC 估计 Mc，Aki-Utsu MLE 估计 b 值"],
        ["大森-宇津", "omori_p/omori_c/omori_k", "非齐次泊松过程 MLE 拟合余震衰减"],
        ["Båth/生产率", "bath_deficit/productivity_index", "刻画主震后最大余震缺口与生产率水平"],
        ["空间各向异性", "anisotropy_*", "用协方差主轴表征破裂方向性"],
        ["地质构造", "plate_type_*/distance", "最近板块边界类型和距离"],
        ["震源机制", "strike/dip/rake/fault_type_*", "Global CMT 断层滑动类型与 P/T 轴"],
    ]
    write_table(TABLE_DIR / "feature_groups.tex", ["特征组", "代表字段", "物理含义"], feature_groups, "p{0.18\\textwidth}p{0.28\\textwidth}p{0.44\\textwidth}")

    validity_rows = []
    for name, col in [
        ("G-R b 值", "gr_valid"),
        ("Omori 参数", "omori_valid"),
        ("空间各向异性", "anisotropy_valid"),
        ("ETAS 参数", "etas_valid"),
        ("Båth 特征", "bath_valid"),
        ("GCMT 震源机制", "focal_mechanism_valid"),
    ]:
        valid = int(features[col].fillna(False).sum())
        validity_rows.append([name, valid, f"{valid / len(features):.2%}"])
    write_table(TABLE_DIR / "feature_validity.tex", ["特征模块", "有效样本数", "有效率"], validity_rows, "lrr")

    cv_rows = []
    for _, row in cv_metrics.iterrows():
        cv_rows.append(
            [
                int(row["fold"]),
                row["model"],
                row["valid_start"],
                row["valid_end"],
                f"{row['mag_rmse']:.3f}",
                f"{row['time_rmse']:.3f}",
                f"{row['time_asymmetric_mae']:.3f}",
            ]
        )
    write_longtable(
        TABLE_DIR / "cv_fold_metrics.tex",
        ["Fold", "模型", "验证起点", "验证终点", "Mag RMSE", "Time RMSE", "Asym MAE"],
        cv_rows,
        "clllrrr",
    )

    mean_rows = []
    mean_metrics = cv_metrics.groupby("model")[
        ["mag_rmse", "mag_mae", "time_rmse", "time_mae", "time_asymmetric_mae", "time_asymmetric_rmse"]
    ].mean()
    for model, row in mean_metrics.iterrows():
        mean_rows.append(
            [
                model,
                f"{row['mag_rmse']:.3f}",
                f"{row['mag_mae']:.3f}",
                f"{row['time_rmse']:.3f}",
                f"{row['time_mae']:.3f}",
                f"{row['time_asymmetric_mae']:.3f}",
                f"{row['time_asymmetric_rmse']:.3f}",
            ]
        )
    write_table(
        TABLE_DIR / "model_mean_metrics.tex",
        ["模型", "Mag RMSE", "Mag MAE", "Time RMSE", "Time MAE", "Asym MAE", "Asym RMSE"],
        mean_rows,
        "lrrrrrr",
    )

    ensemble_metrics = model_meta.get("ensemble_metrics", {})
    ensemble_rows = [
        ["baseline", f"{float(ensemble_weights.get('baseline', 0)):.2f}"],
        ["xgboost", f"{float(ensemble_weights.get('xgboost', 0)):.2f}"],
        ["dl", f"{float(ensemble_weights.get('dl', 0)):.2f}"],
        ["gnn", f"{float(ensemble_weights.get('gnn', 0)):.2f}"],
        ["OOF Mag RMSE", f"{ensemble_metrics.get('mag_rmse', np.nan):.3f}"],
        ["OOF Time RMSE", f"{ensemble_metrics.get('time_rmse', np.nan):.3f}"],
        ["OOF Ensemble Objective", f"{ensemble_metrics.get('ensemble_objective', np.nan):.3f}"],
    ]
    write_table(TABLE_DIR / "ensemble_summary.tex", ["项目", "数值"], ensemble_rows, "lr")

    sub_rows = []
    for _, row in submission.iterrows():
        sub_rows.append(
            [
                row["mainshock_id"],
                f"{row['predicted_max_mag']:.3f}",
                f"{row['predicted_time_to_max']:.3f}",
            ]
        )
    write_longtable(
        TABLE_DIR / "submission_results.tex",
        ["主震 ID", "预测最大余震 Mw", "预测时间差（天）"],
        sub_rows,
        "lrr",
    )

    modules = [
        ["main.py", "统一命令入口，串联下载、序列、特征、训练和推理"],
        ["run.sh", "一键比赛流水线，支持跳过下载和可选 DL/GNN"],
        ["src/data_loader.py", "主震-余震序列构建与目标生成"],
        ["src/features.py", "地震学特征、板块边界、GCMT 震源机制融合"],
        ["src/models.py", "LightGBM/XGBoost Baseline 与非对称目标"],
        ["src/dataset.py", "深度学习序列 Dataset、防泄漏预处理器"],
        ["src/models_dl.py", "Transformer 双输入融合模型"],
        ["src/models_gnn.py", "有向时空因果 ST-GNN"],
        ["scripts/make_submission.py", "端到端单序列推理与提交格式化"],
    ]
    write_table(TABLE_DIR / "module_structure.tex", ["模块", "职责"], modules, "p{0.28\\textwidth}p{0.62\\textwidth}")

    summary_rows = [
        ["主震样本数", len(features)],
        ["高级特征列数", len(features.columns)],
        ["模型输入特征数", len(feature_cols)],
        ["目标最大余震非零样本", int((features["target_max_mag"] > 0).sum())],
        ["目标时间非零样本", int((features["target_time_to_max_days"] > 0).sum())],
        ["提交序列数", len(submission)],
    ]
    write_table(TABLE_DIR / "dataset_summary.tex", ["统计项", "数值"], summary_rows, "lr")


def copy_logo() -> None:
    source = (
        Path("/Users/ningyedong/Coding/机器学习/机器学习小组作业/第15组大作业")
        / "实验报告"
        / "figures"
        / "pku_logo_wordmark.png"
    )
    if source.exists():
        shutil.copy2(source, FIGURE_DIR / "pku_logo_wordmark.png")


def write_report_stats(
    features: pd.DataFrame,
    submission: pd.DataFrame,
    model_meta: dict,
    feature_cols: list[str],
) -> None:
    stats = {
        "n_samples": int(len(features)),
        "n_feature_columns": int(len(features.columns)),
        "n_model_features": int(len(feature_cols)),
        "n_gcmt_valid": int(features["focal_mechanism_valid"].fillna(False).sum()),
        "n_gr_valid": int(features["gr_valid"].fillna(False).sum()),
        "n_omori_valid": int(features["omori_valid"].fillna(False).sum()),
        "n_anisotropy_valid": int(features["anisotropy_valid"].fillna(False).sum()),
        "n_submission": int(len(submission)),
        "submission_min_time": float(submission["predicted_time_to_max"].min()),
        "submission_max_time": float(submission["predicted_time_to_max"].max()),
        "ensemble": model_meta.get("ensemble_weights", {}),
    }
    (REPORT_DIR / "report_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    ensure_dirs()
    configure_plot_style()
    features, cv_metrics, submission, model_meta, ensemble_weights, feature_cols = load_inputs()
    make_figures(features, cv_metrics, submission, model_meta, ensemble_weights)
    make_tables(features, cv_metrics, submission, model_meta, ensemble_weights, feature_cols)
    copy_logo()
    write_report_stats(features, submission, model_meta, feature_cols)
    print(f"报告图表资产已生成: {REPORT_DIR}")
    print(f"PNG 图数量: {len(list(FIGURE_DIR.glob('*.png')))}")
    print(f"表格片段数量: {len(list(TABLE_DIR.glob('*.tex')))}")


if __name__ == "__main__":
    main()
