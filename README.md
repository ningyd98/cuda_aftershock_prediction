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
│  └── 地质构造特征 (板块边界距离/类型 One-Hot)                │
├─────────────────────────────────────────────────────────────┤
│                       模型层 (Model Layer)                    │
│  src/models.py          ← LightGBM / XGBoost / DL 接口       │
│  src/trainer.py         ← 时间序列交叉验证                   │
│  src/evaluator.py       ← 非对称时间惩罚指标                 │
├─────────────────────────────────────────────────────────────┤
│                      流水线 (Pipeline)                        │
│  run.sh                 ← 一键运行脚本                       │
│  main.py                ← 统一入口                           │
└─────────────────────────────────────────────────────────────┘
```

## 一键运行

```bash
chmod +x run.sh
./run.sh --skip-download              # 快速稳定版：LightGBM + XGBoost
./run.sh --skip-download --with-dl    # 额外训练 Transformer
./run.sh --skip-download --with-gnn   # 额外训练 ST-GNN
./run.sh --skip-download --with-deep  # 同时训练 Transformer + ST-GNN
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

```bash
python main.py train-baseline \
    --data data/processed/advanced_features.csv \
    --n-splits 5 \
    --model-type both \
    --use-asymmetric-time-objective \
    --save-dir data/models
```

### 5. 生成预测

```bash
# 单条测试序列
python main.py make-submission \
    --input data/test_sequences/20230206011734_eq.csv \
    --output data/processed/submission.csv \
    --model-dir data/models \
    --allow-rule-fallback
```

### 6. 多模型 OOF 融合 (推荐)

```bash
# 在已训练树模型的基础上，生成 DL/GNN 的 OOF 预测并搜索融合权重
python scripts/train_dl.py --oof --n-splits 5 --purge-days 30 --save-dir data/models
python scripts/train_gnn.py --oof --n-splits 5 --purge-days 30 --save-dir data/models

# 搜索震级和时间独立权重
python scripts/train_ensemble.py --model-dir data/models --grid-step 0.02

# 或在 run.sh 中一键完成
./run.sh --skip-download --train-oof-ensemble
```

## 模型融合策略

### OOF 融合原理

每个模型在时间序列交叉验证 (TimeSeriesSplit, n=5) 中生成
Out-Of-Fold (OOF) 预测，即每条训练样本都有一条"该样本未被用于训练时"
的预测。融合权重在 OOF 预测上搜索，而非训练集指标，以保证泛化性。

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

## Transformer 深度学习方案

### 架构

```
┌─────────────────────────────────────────────────────┐
│                 Transformer 预测器                    │
│                                                      │
│  早期余震序列 (N×7) ──▶ EventProjection ──▶ PE       │
│  [dt, log_dt, x, y, dist, depth, mag]    │          │
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

- **事件特征维**: 7 (时间差、对数时间差、相对经纬度 (km)、距离、深度、震级)
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
| **特征工程** | G-R b值、大森-宇津 p/c/k、Båth's Law、空间各向异性、板块构造 |
| **验证策略** | 时间序列交叉验证 (TimeSeriesSplit, n=5) |
| **评估指标** | Mag RMSE/MAE + Time RMSE/MAE + 非对称时间惩罚 (late_weight=2.0) |
| **Baseline** | LightGBM 非对称时间目标 + XGBoost，多输出回归，基于 OOF 搜索融合权重 |
| **深度模型输入** | 训练集拟合 RobustScaler，领域先验填充 + missing indicator，推理复用同一预处理器 |

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
- ✅ 简化 ETAS 模型参数 (μ/K0/α)
- ✅ 震源机制解特征 (Global CMT: strike/dip/rake, P/T轴, 断层类型)
- ✅ Gardner & Knopoff 去聚类算法 (src/utils.py)
- ✅ joblib 并行特征生成
- ✅ LightGBM 非对称时间目标 + XGBoost Baseline 时间序列 CV
- ✅ 基于 OOF 的树模型融合权重搜索
- ✅ Transformer 深度学习模型 (可选训练，双输入融合，稳健归一化)
- ✅ ST-GNN 时空图神经网络 (可选训练，SpatialGraphConv + TemporalGRU，稳健归一化)
- ✅ 多模型加权融合推理 (默认树模型；DL/GNN 产物存在且权重大于 0 时参与)
- ✅ 非对称时间惩罚评估指标
- ✅ 一键运行脚本 (run.sh)
- ✅ 模拟线上评测系统 (mock_evaluation.py)
- ⬜ LLM 微调 (Chronos/TimeGPT)
