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
