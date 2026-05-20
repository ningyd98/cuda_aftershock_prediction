# 余震预测技术国际大赛

本项目面向阿里云天池"余震预测技术国际大赛"，用于构建全球浅源强震的主震-余震样本、提取高级地震学特征，并为后续机器学习与深度学习模型开发预留工程接口。

## 工程架构

```
┌─────────────────────────────────────────────────────────────┐
│                       数据层 (Data Layer)                     │
│  data/raw/                                                   │
│  ├── USGS_Mw6.0_Depth70_1970-2023.csv   ← 主震目录 (Mw≥6.0) │
│  ├── USGS_Mw4.0_Depth70_1970-2023.csv   ← 完整目录 (特征用)  │
│  ├── PB2002_boundaries.json             ← 板块边界 GeoJSON   │
│  └── GlobalCMT_1976-2024.csv            ← 震源机制解目录     │
├─────────────────────────────────────────────────────────────┤
│                      特征层 (Feature Layer)                   │
│  src/features.py                                              │
│  ├── Gutenberg-Richter b 值 (Aki-Utsu MLE)                   │
│  ├── 大森-宇津定律 p/c/k 参数 (MLE)                          │
│  ├── Båth's Law 早期最大余震差值                              │
│  ├── 空间各向异性 (协方差分解)                                │
│  ├── 地质构造特征 (板块边界距离/类型 One-Hot)                │
│  ├── ETAS 事件级诊断 (branching ratio / 触发占比 / 背景率)   │
│  └── 按板块类型分层填补 (SUB/OSR/OTF/UNK 专属先验)           │
├─────────────────────────────────────────────────────────────┤
│                       模型层 (Model Layer)                    │
│  src/models.py          ← LightGBM / XGBoost / QuantileLGBM  │
│  src/models_dl.py       ← Seq2Seq Transformer (N×12 事件)    │
│  src/models_gnn.py      ← ST-GNN (震级感知边权 + GRU)        │
│  src/trainer.py         ← 时间序列交叉验证                   │
│  src/evaluator.py       ← 包含 Log-Time RMSE, 命中率(Hit Rate) 和 物理定律检验 等非对称+容忍度评测指标 │
├─────────────────────────────────────────────────────────────┤
│                      流水线 (Pipeline)                        │
│  run.sh                 ← 一键运行脚本                       │
│  main.py                ← 统一入口                           │
└─────────────────────────────────────────────────────────────┘
```

## 一键运行

> **默认启用 CUDA 混合精度加速 (bfloat16)**：所有 DL/GNN 训练默认使用 `--device cuda` 并自动开启 `bfloat16` AMP (Automatic Mixed Precision)，避免 NaN 的同时加速训练。
> 若无 GPU，会自动回退到 CPU 并打印醒目的警告横幅。

```bash
chmod +x run.sh
./run.sh --skip-download              # 快速稳定版：LightGBM + XGBoost (CUDA)
./run.sh --skip-download --with-dl    # 额外训练 Transformer (CUDA)
./run.sh --skip-download --with-gnn   # 额外训练 ST-GNN (CUDA)
./run.sh --skip-download --with-deep  # 同时训练 Transformer + ST-GNN (CUDA)
./run.sh --tune-full                  # 全模型联合超参数调优 (含进度条可视化)
./run.sh --tune-fast                  # 快速调优模式 (减少 DL/GNN 训练轮数)
./run.sh --realtime                   # 实时监控模式 (持续轮询 USGS API)
./run.sh train-only                   # 只重训模型并重新生成提交
```

## 分步命令

### 1. 下载数据

```bash
# 板块边界数据
python main.py download-pb2002

# USGS Mw≥6.0 强震目录
python main.py download-usgs

# USGS Mw≥4.0 完整目录（用于余震特征提取）
python main.py download-full-catalog --min-mag 4.0
```

### 2. 构建主震-余震序列

```bash
# 使用完整目录构建序列（推荐）
python src/data_loader.py \
    --input data/raw/USGS_Mw4.0_Depth70_1970-2023.csv \
    --output data/processed/ML_Ready_Sequences.csv

# 或通过统一入口
python main.py build-sequences
```

### 3. 提取高级特征

```bash
python main.py build-features

# 快速冒烟测试（前 50 条）
python main.py build-features --limit 50 --output data/processed/advanced_features_smoke.csv
```

如果 `configs/default.yaml` 中 `phase1.gcmt.enabled=true` 且本地缺少 `data/raw/GlobalCMT_1976-2024.csv`，特征脚本会自动下载 Global CMT 官方 NDK 目录并生成本地 CSV。

### 4. 训练模型

> 所有训练脚本默认启用 CUDA (`--device cuda`)，无需手动指定。

```bash
python main.py train-baseline \
    --data data/processed/advanced_features.csv \
    --n-splits 5 \
    --model-type both \
    --use-asymmetric-time-objective \
    --save-dir data/models

# 手动指定设备 (可选)
# --device cuda   强制 CUDA（不可用时优雅回退 CPU）
# --device cpu    纯 CPU 训练
# --device auto   自动检测
```

### 5. 实时监控与自动推理 🆕

基于 USGS FDSN API 实时监控全球强震，自动运行集成推理：

```bash
# 持续监控模式 (每 5 分钟轮询 USGS API)
python scripts/realtime_monitor.py

# 单次运行 (检查最近 24h 的强震)
python scripts/realtime_monitor.py --once

# 回填最近 30 天的历史强震
python scripts/realtime_monitor.py --backfill 30

# 降低震级阈值 (Mw≥5.5)
python scripts/realtime_monitor.py --min-mag 5.5

# 在 run.sh 中启动
./run.sh --realtime
```

**工作流**：
1. 轮询 USGS API → 发现新的 Mw≥6.0 强震
2. 等待 3 天观测窗口 → 自动查询早期余震
3. 运行特征提取 + 集成推理 (4 模型融合)
4. 输出预测 → 检测预警条件

**产物**：
| 文件 | 内容 |
|:---|:---|
| `data/processed/realtime_state.json` | 事件处理状态跟踪 |
| `data/processed/realtime_predictions.csv` | 历史预测记录 (累积追加) |
| `data/processed/realtime_alerts.json` | 预警事件 (强余震 + 时间紧迫) |

**预警规则**：
- 预测最大余震 Mw ≥ 6.5 → 强余震预警
- 预测时间 ≤ 1 天 → 时间紧迫预警

### 6. 生成预测

```bash
# 单条测试序列
python main.py make-submission \
    --input data/test_sequences/20230206011734_eq.csv \
    --output data/processed/submission.csv \
    --model-dir data/models \
    --allow-rule-fallback
```

### 7. 全模型超参数调优 (Optuna)

一键启动 LightGBM + XGBoost + Transformer + ST-GNN 联合调优，
每个 trial 自动完成：树模型 OOF CV → DL OOF CV → GNN OOF CV → 融合权重搜索。

```bash
# 完整联合调优 (100 trials，数小时)
./run.sh --tune-full --tune-trials 100

# 仅调优树模型 (LightGBM + XGBoost，较快)
./run.sh --tune --tune-trials 100

# 快速模式 (减少 DL/GNN 训练轮数)
./run.sh --tune-fast --tune-trials 50

# 多进程并行 Trial 提速 (慎用，需考量显存及 CPU 资源，可能与 DataLoader 产生死锁冲突)
# 若出现死锁或卡死情况，请务必去掉 `--optuna-n-jobs 2` 参数直接单线程运行。
python scripts/tune_all_models.py \
    --n-trials 100 --device cuda \
    --output-dir data/tuning_results
```

#### 进度条可视化

调优全流程配有完整的 **tqdm 进度条**，直观展示各步骤耗时：

| 步骤 | 进度条 | 说明 |
|:---|:---|:---|
| Trial 级别 | 🔬 `Trial N: 采样超参数并开始评估 …` | 每个 Optuna trial 的入口日志 |
| 特征工程 | 📐 特征选择状态 | 显示是否启用特征筛选 |
| 树模型 OOF | 🌲 `Tree OOF CV` (Fold 级) | 每折 train/valid 样本数实时显示 |
| Transformer OOF | 🧠 `DL OOF CV` → `DL Fold N epochs` → `DL Fold N 预测` | 三层嵌套: Fold / Epoch (val_loss) / Batch 预测 |
| ST-GNN OOF | 🔗 `GNN OOF CV` → `GNN Fold N epochs` → `GNN Fold N 预测` | 三层嵌套: Fold / Epoch (val_loss) / Batch 预测 |
| 融合权重搜索 | ⚖️ `融合权重搜索 (mag/time)` | Simplex 网格组合数进度 (≥3 模型) |
| Holdout 评估 | 🎯 `Holdout 评估` | 逐序列预测进度 (每 `eval_holdout_every` 轮) |
| Optuna 顶层 | `study.optimize(show_progress_bar=True)` | Optuna 内置 trial 进度条 |

调优产物：
| 文件 | 格式 | 说明 |
|:---|:---|:---|
| `tuning_report.md` | **Markdown 可读报告** | 📄 **重点**：总体概述 + OOF/Holdout 指标 + 融合权重 + 最优参数 + 使用建议 |
| `best_params.json` | JSON | 最优超参数 |
| `ensemble_weights.json` | JSON | 最优 mag/time 融合权重 |
| `holdout_predictions.csv` | CSV | 20 条测试序列的 holdout 预测 vs 真实 |
| `trials_history.csv` | CSV | 所有 trial 的完整记录 |
| `tuning_stats.json` | JSON | 调优统计汇总 (含 best_holdout 指标) |

### 8. 多模型 OOF 融合 (推荐)

```bash
# 单独运行融合脚本（当所有 OOF CSV 已就绪时）
python scripts/train_ensemble.py --model-dir data/models --grid-step 0.02

# 或在 run.sh 中一键完成（含树模型 OOF + DL OOF + GNN OOF）
./run.sh --skip-download --train-oof-ensemble
```

### OOF 的含义

OOF (Out-Of-Fold) 预测是时间序列交叉验证过程中，
每条训练样本在"该样本未被用于训练"的 fold 中得到的预测值。
OOF 指标比训练集指标更能反映模型对未来数据的泛化能力，
因此融合权重必须在 **OOF 预测**上搜索。

### 为什么震级和时间使用不同权重？

地震学中，**震级预测**和**时间预测**是两种截然不同的物理任务：

- **震级 (Mag)**: 主震释放能量后地壳应力降控制最大余震震级；
  G-R 定律 b 值和 Båth 定律 Δm 是核心特征；
  LightGBM 通常在震级任务上最优。

- **时间 (Time)**: 大森-宇津定律 p 值控制余震衰减速率；
  ETAS 模型参数 μ/K0/α 捕捉触发级联；
  Transformer/ST-GNN 的自注意力机制擅长捕获
  "早期余震的时间模式"。

因此为 `mag` 和 `time` 目标**独立搜索**最优融合权重，比共用一组
权重效果更好。

算法采用完整 simplex 网格搜索：
- 2 模型：遍历 `(w, 1-w)` 一维网格
- 3 模型：遍历 `(w1, w2, 1-w1-w2)` 二维网格
- 4 模型：遍历 `(w1, w2, w3, 1-w1-w2-w3)` 三维网格

### 查看融合结果

```bash
# 查看最优权重
cat data/models/ensemble_weights.json

# 查看融合指标
cat data/models/ensemble_metrics.json

# 查看融合后的 OOF 预测（含各单模型预测列）
head data/models/ensemble_oof_predictions.csv
```

### 融合权重文件格式

```json
{
  "mag": {
    "baseline": 0.60,
    "xgboost": 0.25,
    "dl": 0.10,
    "gnn": 0.05
  },
  "time": {
    "baseline": 0.45,
    "xgboost": 0.20,
    "dl": 0.25,
    "gnn": 0.10
  }
}
```

### 比较单模型与融合模型

```bash
# 运行融合后会输出对比表：
python scripts/train_ensemble.py --model-dir data/models
```

重点关注 OOF 指标，而非训练集指标。OOF 指标直接反映
模型对"未见过的未来地震"的预测能力。

### OOF → Full-Fit 闭环

OOF 流程不仅搜索融合权重，还会自动训练最终模型：

```bash
./run.sh --skip-download --train-oof-ensemble
```

完整产物清单:
- `data/models/oof_predictions.csv` — 树模型 OOF 预测
- `data/models/dl_oof_predictions.csv` — Transformer OOF 预测
- `data/models/gnn_oof_predictions.csv` — ST-GNN OOF 预测
- `data/models/ensemble_weights.json` — 双目标融�的合权重
- `data/models/ensemble_metrics.json` — 融合 + 各单模型 OOF 指标
- `data/models/ensemble_oof_predictions.csv` — 融合后 OOF 预测
- `data/models/baseline_model.joblib` — 全量数据 LGBM 模型
- `data/models/dl_model.pt` — 全量数据 Transformer 模型 (OOF 后自动训练)
- `data/models/gnn_model.pt` — 全量数据 ST-GNN 模型 (OOF�的�自动训练)

> OOF 模型用于调权，Full-Fit 模型用于最终推理。

### 结果可视化

```bash
python scripts/visualize_results.py \
    --oof data/models/ensemble_oof_predictions.csv \
    --features data/processed/advanced_features.csv \
    --weights data/models/ensemble_weights.json \
    --output-dir reports/figures
```

生成 5 张图:
- `mag_scatter.png` — 实际 vs 预测震级散点
- `time_scatter_log.png` — log1p 时间散点
- `residual_dist.png` — 残差分布
- `error_by_plate.png` — 按板块类型的误差
- `ensemble_weights.png` — 融合权重柱状图

## 两阶段零膨胀门控预测

### 原理

大量主震没有后续强余震，直接回归会扭曲分布。两阶段方案:
1. **分类器** (`aftershock_classifier.joblib`): 预测"是否有余震"
2. **回归器** (现有 ensemble): 仅对"有余震"的样本预测 magnitude/time

### 训练

分类器随 `train_baseline.py` 自动训练：
```bash
python scripts/train_baseline.py --model-type both --save-dir data/models
# 输出: aftershock_classifier.joblib + classifier_meta.json
```

### 推理

```bash
# 启用门控
python scripts/make_submission.py \
    --input data/test_sequences/20230206011734_eq.csv \
    --output submission.csv \
    --model-dir data/models \
    --use-gating --gating-threshold 0.5 \
    --no-aftershock-mag 0.0 --no-aftershock-time 0.0
```

当 `prob_has_aftershock < threshold` 时直接输出
`no_aftershock_mag` / `no_aftershock_time`，不再调回归器。

## Transformer 深度学习方案

### 架构

```
┌─────────────────────────────────────────────────────┐
│                 Transformer 预测器                    │
│                                                      │
│  早期余震序列 (N×12) ──▶ EventProjection ──▶ PE       │
│  [dt, log_dt, x, y, dist, depth, mag,              │
│   azimuth, mag_diff, Δt_interval, local_b, norm_E]  │
│                                           ▼          │
│                              TransformerEncoder ×3   │
│                              (d_model=128, nhead=4)   │
│                                           │          │
│                                           ▼          │
│  全局手工特征 (D_global) ──▶ MLP Encoder  │          │
│  [G-R/Omori/ETAS/构造...]                │          │
│                                           ▼          │
│                              Concat ──▶ FusionMLP    │
│                                           │          │
│                                           ▼          │
│                              [pred_mag, pred_time]    │
└─────────────────────────────────────────────────────┘
```

- **事件特征维**: 12 (时间差、对数时间差、相对经纬度 (km)、距离、深度、震级、方位角、震级差、事件间隔、局部 b 值代理、归一化能量)
- **位置编码**: 正弦位置编码 (sinusoidal PE)，支持最大 256 时间步
- **空序列处理**: 对无早期余震的主震，自动产生零向量表示
- **时间目标**: 训练时使用 log1p 压缩长尾分布，推理时还原为真实天数
- **预处理器**: RobustScaler + 领域先验填充 (b=1.0, p=1.0, c=0.05)

### 训练

```bash
# 常规训练 (80% train / 20% val)
python scripts/train_dl.py \
    --features data/processed/advanced_features.csv \
    --event-catalog data/raw/USGS_Mw4.0_Depth70_1970-2023.csv \
    --epochs 50 --batch-size 32 --lr 1e-3 \
    --d-model 128 --nhead 4 --num-layers 3 \
    --save-dir data/models

# OOF 交叉验证 (用于融合)
python scripts/train_dl.py --oof --n-splits 5 --purge-days 30 \
    --save-dir data/models
```

### 分析与可解释性

```bash
# 单条序列分析 + 事件贡献度
python scripts/analyze_transformer.py \
    --input data/test_sequences/20230206011734_eq.csv \
    --model-dir data/models \
    --event-contributions

# 批量分析所有测试序列
python scripts/analyze_transformer.py \
    --input-dir data/test_sequences \
    --model-dir data/models \
    --output-dir reports/transformer_analysis
```

输出 JSON 包含:
- Transformer 预测 (pred_mag, pred_time)
- 各早期余震事件对预测的贡献度排名
- 注意力权重矩阵 (可选 `--extract-attention`)

## 关键设计

| 维度 | 方案 |
|------|------|
| **目标** | 预测 Mw≥6.0 强震后 30 天内最大余震的震级和时间 |
| **观测窗口** | 主震后 3 天 (72h) 内的早期余震序列 |
| **空间窗口** | 主震震中 100 km 半径 |
| **特征工程** | G-R b值、大森-宇津 p/c/k、Båth's Law、空间各向异性、板块构造、ETAS事件级诊断 (branching ratio/触发占比/背景率)、按板块分层填补 |
| **验证策略** | 时间序列交叉验证 (TimeSeriesSplit, n=5) |
| **评估指标** | Mag RMSE/MAE + Time RMSE/MAE + 非对称时间惩罚 (late_weight=2.0) |
| **Baseline** | LightGBM 非对称时间目标 + XGBoost + QuantileLGBM (时间分位数)，多输出回归，基于 OOF 搜索融合权重 |
| **深度模型输入** | Transformer: N×12 事件特征 + RobustScaler；ST-GNN: 震级感知有向边权重 + GRU 时序编码 |
| **GPU 加速** | 全程序默认 CUDA (LightGBM GPU + PyTorch CUDA + cuDNN benchmark)；无 GPU 时优雅回退 CPU |
| **超参数调优** | Optuna TPESampler 多变量联合采样，全模型 OOF CV + simplex 融合搜索，含完整 tqdm 进度条 |

## 目录结构

```text
.
├── main.py                         ← 统一入口
├── run.sh                          ← 一键运行脚本
├── configs/default.yaml            ← 全局配置
├── data/
│   ├── raw/                        ← 原始数据
│   ├── processed/                  ← 处理后的特征和序列
│   ├── test_sequences/             ← 测试序列 (每条主震一个 CSV)
│   └── models/                     ← 训练产物
├── src/
│   ├── data_loader.py              ← 序列构建
│   ├── features.py                 ← 地震学特征工程
│   ├── models.py                   ← 模型定义
│   ├── trainer.py                  ← 训练 & CV
│   ├── evaluator.py                ← 评估指标
│   └── utils.py                    ← 工具函数
└── scripts/
    ├── download_usgs.py            ← 下载 USGS Mw≥6.0 目录
    ├── download_full_catalog.py    ← 下载 USGS Mw≥4.0 目录
    ├── download_pb2002.py          ← 下载板块边界
    ├── build_features.py           ← 并行特征生成
    ├── train_baseline.py           ← Baseline 训练
    └── make_submission.py          ← 生成提交文件
```

## 当前状态

- ✅ Gutenberg-Richter b 值 (Aki-Utsu MLE + MAXC)
- ✅ 大森-宇津定律 MLE 参数拟合 (p/c/k)
- ✅ 空间各向异性 (协方差分解)
- ✅ 板块构造特征 (PB2002 边界距离 + One-Hot)
- ✅ Båth's Law 特征 (bath_deficit / bath_early_max_mag / bath_valid)
- ✅ 时空分箱特征 (1h/6h/12h/24h/72h 频次+能量分布)
- ✅ ETAS 模型参数 + 事件级诊断 (branching ratio / 触发占比 / 期望计数 / 背景率)
- ✅ 震源机制解特征 (Global CMT: strike/dip/rake, P/T轴, 断层类型)
- ✅ Gardner & Knopoff 去聚类算法 (src/utils.py)
- ✅ joblib 并行特征生成
- ✅ LightGBM 非对称时间目标 + XGBoost Baseline 时间序列 CV
- ✅ **QuantileLGBM 时间分位数回归** (5分位数独立训练，改善长尾时间预测)
- ✅ **按板块类型分层缺失值填补** (SUB/OSR/OTF/UNK 专属地震学先验)
- ✅ 基于 OOF 的树模型融合权重搜索
- ✅ Transformer 深度学习模型 (可选训练，**N×12 事件维**双输入融合，稳健归一化)
- ✅ ST-GNN 时空图神经网络 (可选训练，**震级感知有向边权重** + TemporalGRU)
- ✅ 多模型加权融合推理 (默认树模型；DL/GNN 产物存在且权重大于 0 时参与)
- ✅ 非对称时间惩罚评估指标
- ✅ 一键运行脚本 (run.sh)
- ✅ 模拟线上评测系统 (mock_evaluation.py)
- ✅ 全程序 CUDA 加速 (LightGBM GPU + PyTorch CUDA，默认启用，优雅回退)
- ✅ 全模型联合超参数调优 (Optuna: LGB + XGB + Transformer + ST-GNN)
- ✅ 调优全流程 tqdm 进度条可视化 (Fold/Epoch/Batch/Simplex/Holdout 全覆盖)
- ⬜ LLM 微调 (Chronos/TimeGPT)
