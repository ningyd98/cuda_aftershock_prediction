# 余震预测技术国际大赛

本项目面向阿里云天池"余震预测技术国际大赛"，用于构建全球浅源强震的主震-余震样本、提取高级地震学特征，并为后续机器学习与深度学习模型开发预留工程接口。

## 工程架构

```
┌─────────────────────────────────────────────────────────────┐
│                       数据层 (Data Layer)                     │
│  data/raw/                                                   │
│  ├── USGS_Mw6.0_Depth70_1970-2023.csv   ← 主震目录 (Mw≥6.0) │
│  ├── USGS_Mw4.5_Depth70_1970-2023.csv   ← 完整目录 (特征用)  │
│  └── PB2002_boundaries.json             ← 板块边界 GeoJSON   │
├─────────────────────────────────────────────────────────────┤
│                      特征层 (Feature Layer)                   │
│  src/features.py                                              │
│  ├── Gutenberg-Richter b 值 (Aki-Utsu MLE)                   │
│  ├── 大森-宇津定律 p/c/k 参数 (MLE)                          │
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
./run.sh              # 全流程：下载 → 序列 → 特征 → 训练 → 预测
./run.sh --skip-dl    # 跳过下载（数据已就绪）
./run.sh train-only   # 仅重新训练
```

## 分步命令

### 1. 下载数据

```bash
# 板块边界数据
python main.py download-pb2002

# USGS Mw≥6.0 强震目录
python main.py download-usgs

# USGS Mw≥4.5 完整目录（用于余震特征提取）
python main.py download-full-catalog
```

### 2. 构建主震-余震序列

```bash
# 使用完整目录构建序列（推荐）
python src/data_loader.py \
    --input data/raw/USGS_Mw4.5_Depth70_1970-2023.csv \
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

### 4. 训练模型

```bash
python main.py train-baseline \
    --data data/processed/advanced_features.csv \
    --n-splits 5 \
    --save-dir data/models
```

### 5. 生成预测

```bash
# 单条测试序列
python main.py make-submission \
    --input data/test_sequences/20230206011734_eq.csv \
    --output data/processed/submission.csv \
    --baseline-model data/models/baseline_model.joblib \
    --feature-cols data/models/feature_cols.json \
    --allow-rule-fallback
```

## 关键设计

| 维度 | 方案 |
|------|------|
| **目标** | 预测 Mw≥6.0 强震后 30 天内最大余震的震级和时间 |
| **观测窗口** | 主震后 3 天 (72h) 内的早期余震序列 |
| **空间窗口** | 主震震中 100 km 半径 |
| **特征工程** | G-R b值、大森-宇津 p/c/k、空间各向异性、板块构造 |
| **验证策略** | 时间序列交叉验证 (TimeSeriesSplit, n=5) |
| **评估指标** | Mag RMSE/MAE + Time RMSE/MAE + 非对称时间惩罚 (late_weight=2.0) |
| **Baseline** | LightGBM 多输出回归 (300 trees, lr=0.03) |

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
    ├── download_full_catalog.py    ← 下载 USGS Mw≥4.5 目录
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
- ✅ 时空分箱特征 (1h/6h/12h/24h/72h 频次+能量分布)
- ✅ 简化 ETAS 模型参数 (μ/K0/α)
- ✅ 震源机制解特征 (Global CMT: strike/dip/rake, P/T轴, 断层类型)
- ✅ Gardner & Knopoff 去聚类算法 (src/utils.py)
- ✅ joblib 并行特征生成
- ✅ LightGBM Baseline 时间序列 CV
- ✅ XGBoost 第二 Baseline 时间序列 CV
- ✅ Transformer 深度学习模型 (双输入融合)
- ✅ ST-GNN 时空图神经网络 (SpatialGraphConv + TemporalGRU)
- ✅ 多模型加权融合推理 (LightGBM + XGBoost + DL + GNN)
- ✅ 非对称时间惩罚评估指标
- ✅ 一键运行脚本 (run.sh)
- ✅ 模拟线上评测系统 (mock_evaluation.py)
- ⬜ LLM 微调 (Chronos/TimeGPT)
