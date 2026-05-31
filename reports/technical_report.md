# 余震预测技术报告

## 一、问题定义与竞赛背景

本项目面向全球浅源强震后的最大余震预测问题。具体而言：给定一次 Mw≥6.0、深度≤70 km 的主震及其震后 72 小时内（半径 100 km 内）的早期余震序列，预测该主震在三个时间窗口内的最大余震震级和发生时刻：

| 窗口 | 时间范围 | 输出文件 |
|------|---------|---------|
| T1 | 0–24 h（主震后第 1 天） | `{mainshock}-T1-T2.csv` (第 1 行) |
| T2 | 24–72 h（主震后第 2–3 天） | `{mainshock}-T1-T2.csv` (第 2 行) |
| T3 | 72–168 h（主震后第 4–7 天） | `{mainshock}-T3.csv` (1 行) |

此外，为满足比赛可能要求每个测试序列输出单一 0–168 h 预测的需求，系统同时支持 H168 单窗口路径，输出唯一文件 `predictions/qualification_predictions.csv`。

## 二、科学原理与特征工程

余震发生受控于主震引起的静态库仑应力变化、区域构造背景和地壳流变性质。本项目从以下地震学定律出发构建特征体系。

### 2.1 Gutenberg-Richter 定律与 b 值

G-R 定律描述了震级-频度关系：

$$\log_{10} N(M) = a - bM$$

其中 b 值反映区域应力水平和介质非均匀性。低 b 值通常对应高应力积累区，与较大余震概率正相关。本系统使用 Aki-Utsu 极大似然估计 (MLE) 计算 b 值，辅以 MAXC 方法自动检测完整性震级 $M_c$。特征包含 `gr_b_value`、`gr_a_value`、`gr_mc`、`gr_valid` 等。

### 2.2 大森-宇津定律 (Omori-Utsu Law)

余震频度随时间衰减遵循修正大森定律：

$$n(t) = \frac{K}{(t + c)^p}$$

其中 p 值通常为 0.9–1.4，c 为时间偏移常数。通过 MLE 拟合 p/c/k 参数，生成 `omori_p`、`omori_c`、`omori_k` 等特征。p 值越大表示余震衰减越快，越不容易在晚期发生强余震。

### 2.3 Båth 定律

Båth 定律指出最大余震震级通常比主震低约 1.1–1.2 级（Δm ≈ 1.2）。特征 `bath_deficit = M_main − M_early_max` 刻画早期已释放的最大余震强度缺口，缺口越大则未来可能出现更大余震的风险越高。

### 2.4 ETAS 模型诊断

ETAS (Epidemic-Type Aftershock Sequence) 模型将地震序列分解为背景地震和由已有事件触发的级联地震。对每条训练样本，使用早期余震序列快速拟合简化 ETAS 模型，提取分支比 (branching ratio)、触发占比、背景率等特征。高分支比意味着序列具有强触发级联性质，更可能持续产出余震。

### 2.5 空间各向异性

对早期余震震中进行协方差矩阵分解，提取余震分布的主轴方向、长短轴比和方位角。这些特征反映区域应力场的空间投影，有助于判断余震区展布形态。特征包含 `anisotropy_major_axis_km`、`anisotropy_axis_ratio`、`anisotropy_azimuth_deg` 等。

### 2.6 地质构造特征

基于 PB2002 全球板块边界模型，计算主震震中到最近板块边界的距离，并按边界类型（SUB 俯冲带、OSR 洋脊、OTF 转换断层等）做 One-Hot 编码。板块构造环境直接影响区域应力释放模式。

### 2.7 震源机制解 (Global CMT)

从 Global CMT 项目获取主震的矩张量解，提取两组节面参数 (strike1/dip1/rake1, strike2/dip2/rake2)、P 轴/T 轴方位与倾角、断层类型分类以及 CLVD 分量。这些参数编码了主震破裂的几何和力学信息。

### 2.8 时空分箱特征

在 1h/6h/12h/24h/72h 时间窗口内累积统计余震计数和释放能量，并按子窗口拆分得到非累积量。设计能量比率和计数比率特征以刻画余震活动的时序分布模式。

### 2.9 缺失值处理

按板块类型 (SUB/OSR/OTF/UNK) 分层填补缺失值，各类型使用专属的地震学先验（如俯冲带 b≈1.0、洋脊 b≈1.2）以保持领域一致性。

## 三、模型架构

### 3.1 T1/T2/T3 Hybrid Calibrated（三窗口混合校准，默认方案）

**Score Model（分数模型）**：对 T1/T2/T3 三个窗口独立训练 LightGBM + XGBoost 多输出回归器，使用非对称时间目标函数（过晚预测施加 2× 惩罚），预测 (mag, time)。

**Legal-Risk Model（合法窗口与极端风险模型）**：使用 `reconstruct_legal_window_features` 重建各窗口合法可用的特征子集——T1 仅使用主震元信息 + 静态特征（观测时间=0h），T2 允许 0–24h 累计计数/能量信号，T3 允许 0–72h 完整特征。每个窗口额外训练 LightGBM 二分类器（ExtremeClassifier），标记目标震级 ≥ mainshock_mag − margin 的高风险样本。

后校准逻辑：极端大余震概率超过阈值时，震级上修到 floor = max(mag_pred, mainshock_mag − margin)，时间向窗口早期 30% 处偏移；T1 高风险样本额外加 `t1_early_delta_bonus`。

### 3.2 Decoupled Pipeline（震级/时间分离管道）

**设计动机**：同时预测震级和时间的多输出回归器被迫在两个分布不同、物理规律相异的目标间折中。分离模型允许各自利用最相关的物理特征：G-R b 值和 Båth's Law Δm 主要驱动震级，Omori-Utsu p 值和 ETAS α 参数主要驱动时间。

每个窗口独立训练三个模型：

| 模型 | 类型 | 算法 |
|------|------|------|
| MagModel | 回归 | LightGBM + XGBoost 集成 |
| TimeBucketModel | 4 类分类 | LightGBM 多分类，桶中心加权期望 |
| ExtremeClassifier | 二分类 | LightGBM class_weight=balanced |

**时间桶方法**：将连续时间预测转为离散分类 + 期望计算。H168 窗口的 4 桶划分为 (0,12]、(12,48]、(48,96]、(96,168] 小时。期望时间按 $E[t] = \sum_{k} p_k \cdot \text{center}_k$ 计算。分类方法有以下优势：

- 在每个桶内学习局部分布，不受桶间方差干扰；
- 即便选错桶，相邻桶的预测仍然合理；
- 分类概率可直接用于不确定性量化。

**OOF 融合**：在时间序列交叉验证的 OOF (Out-Of-Fold) 预测上，通过 simplex 网格搜索学习震级和时间各自的融合权重。震级候选含 `oof_mag_lgbm` 和 `oof_mag_xgb`；时间候选含 `oof_time_bucket_raw`（桶分类期望）、`oof_time_direct_lgbm_raw`（独立对数时间回归 LGBM）和 `oof_time_direct_xgb_raw`（独立对数时间回归 XGBoost）。权重非负且和为 1。

### 3.3 Single-Horizon H168（单窗口 0–168h 路径，辅助实验）

H168 移除 T1/T2/T3 分窗口概念，将 0–168h 作为统一预测窗口。标签通过取 T1/T2/T3 三窗口中最强余震（并列取最早）自动推导；无余震时回退 mag=0.0, time=84.0h。特征默认仅使用主震静态特征（无余震观测泄漏）。同时支持 LGBM+XGBoost 震级双候选和时间三候选（桶分类 + 对数时间回归 LGBM + 对数时间回归 XGBoost）的 OOF 融合。

### 3.4 Transformer 深度学习模型（可选增强）

采用序列到序列风格的 Transformer 编码器架构：早期余震序列 (N×12 事件特征) 经 EventProjection 层映射到 d_model=128 维空间，叠加正弦位置编码，通过 3 层 TransformerEncoder (nhead=4) 编码序列上下文；全局手工特征（G-R、Omori、ETAS 等）经独立 MLP 编码后与序列表示拼接，经 FusionMLP 输出 (mag, time)。训练时 log1p 压缩时间目标的长尾分布。

### 3.5 ST-GNN 时空图神经网络（可选增强）

基于震级感知的有向边权重建图——事件间以震级差和空间距离加权连接，经图卷积层聚合邻域信息，再由 GRU 时序层编码事件序列演化。与 Transformer 一样，作为 OOF 融合的候选模型。

## 四、验证与评估

### 4.1 交叉验证策略

采用 `TimeSeriesSplit(n_splits=5)` 时间序列交叉验证，严格保证训练集时间早于验证集，避免未来信息泄漏。所有调参和融合权重搜索均在 OOF 预测上进行，不使用测试集标签。

### 4.2 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| Mag MAE | $\frac{1}{n}\sum |\hat{m}_i - m_i|$ | 震级平均绝对误差 |
| Mag RMSE | $\sqrt{\frac{1}{n}\sum (\hat{m}_i - m_i)^2}$ | 震级均方根误差 |
| Time MAE | $\frac{1}{n}\sum |\hat{t}_i - t_i|$ (小时) | 时间平均绝对误差 |
| Time Asymmetric RMSE | $\sqrt{\frac{1}{n}\sum w_i(\hat{t}_i - t_i)^2}$ | 过晚预测 2× 惩罚（$w_i=2$ 当 $\hat{t}_i > t_i$） |
| Extreme Mag MAE | $\frac{1}{|E|}\sum_{i\in E}|\hat{m}_i - m_i|$ | 仅计算极端大余震样本 |
| Hit Rate | $\frac{1}{n}\sum \mathbf{1}[|\hat{t}_i-t_i|\le \max(0.2 t_i,3.0)]$ | 时间预测命中率 |

### 4.3 当前性能（T1/T2/T3 Hybrid Calibrated，20 条测试序列）

| 窗口 | Mag MAE | Mag RMSE | Time MAE (h) | Time RMSE (h) | Within ±0.5 Mag | Within ±24h |
|------|---------|----------|-------------|--------------|-----------------|-------------|
| T1 | 0.42 | 0.57 | 2.91 | 4.03 | 80% | 100% |
| T2 | 0.28 | 0.48 | 8.01 | 10.45 | 85% | 100% |
| T3 | 0.40 | 0.48 | 22.43 | 26.20 | 75% | 50% |
| ALL | 0.36 | 0.51 | 11.12 | 16.45 | 80% | 83% |

**分析**：T1 和 T2 窗口的预测效果较好（震级误差 ≤0.5，时间命中率 100%）。T3 窗口（72–168h）是最大短板——时间 MAE 达 22.43h，24h 命中率仅 50%。主要原因是 72–168h 窗口跨度 96h，可用的早期余震观测信号在时间上的信息衰减显著。

## 五、开发计划与路线图

### 5.1 已完成

- ✅ Gutenberg-Richter b 值 MLE、大森-宇津 p/c/k 拟合、Båth 定律、空间各向异性、板块构造、ETAS 事件级诊断、震源机制解特征
- ✅ 按板块类型分层缺失值填补
- ✅ LightGBM + XGBoost baseline，非对称时间目标 + TimeSeriesSplit CV
- ✅ QuantileLGBM 时间分位数回归 (5 分位数)
- ✅ 基于 OOF 的树模型融合权重搜索 (simplex 网格)
- ✅ Hybrid Calibrated 三窗口打包（score model + legal-risk model + 后校准）
- ✅ Decoupled Pipeline：震级/时间分离训练 + 时间桶 4 分类 + Optuna 超参数调优 + OOF 融合
- ✅ Single-Horizon H168 完整调参/训练/打包流程
- ✅ Transformer 深度学习（N×12 事件序列编码 + 全局特征融合）
- ✅ ST-GNN 时空图神经网络（震级感知边权 + GRU 时序）
- ✅ Optuna 全模型联合超参数调优，含完整进度条可视化
- ✅ run.sh 一键运行入口，支持 qualification 子命令自动分发
- ✅ 模拟线上评测、批量 OOF 推理、实时监控

### 5.2 进行中

- 🔄 Decoupled v2 模型在完整训练集上的最终性能评估
- 🔄 T3 窗口时间预测精度专项优化

### 5.3 计划中

- ⬜ T3 分层 hazard/ranking 模型：先预测窗口内日级风险，再回归小时级峰值
- ⬜ 极端大余震时间分层 OOF 校准
- ⬜ 引入更多物理先验作为网络正则化项（Physics-Informed ML）
- ⬜ LLM 微调（Chronos/TimeGPT）用于时间序列预测
- ⬜ `configs/qualification.yaml` 集中配置管理

## 六、关键设计决策

| 维度 | 方案 | 理由 |
|------|------|------|
| 验证策略 | TimeSeriesSplit (n=5) | 严格保持时间顺序，贴近真实预测场景 |
| 融合权重建模 | OOF predictions + simplex 网格搜索 | 避免在训练集指标上过拟合权重 |
| 震级/时间分离 | Decoupled mag + time models | 两目标分布和物理规律不同，分离允许各取最优 |
| 时间离散化 | 4-桶分类 + 期望计算 | 降低长尾回归困难，分类概率天然提供置信度 |
| 特征合法性 | `reconstruct_legal_window_features` | T1/T2 不泄漏未来观测信息 |
| 后校准 | 基于 OOF 偏差分析 | 不直接拟合测试样本 |
| 高性能 vs 兼容性 | GPU 优先，CPU 自动回退 | 生产环境灵活部署 |

## 七、代码工程化

### 7.1 统一入口

```bash
# 一键获取帮助
./run.sh qualification --help

# 默认 hybrid calibrated 打包
./run.sh qualification --device cpu --no-install

# decoupled 完整流程
./run.sh qualification --tune-decoupled --retrain-decoupled --use-decoupled --device cpu --no-install

# H168 single-horizon 完整流程
./run.sh qualification --tune-single-horizon --retrain-single-horizon --use-single-horizon --device cpu --no-install
```

### 7.2 目录结构

```
├── main.py                          # 统一命令入口
├── run.sh                           # 一键运行（含 qualification 子命令分发）
├── src/
│   ├── qualification.py             # 窗口定义、标签推导、格式、裁剪
│   ├── features.py                  # 地震学特征工程
│   ├── time_buckets.py              # 时间桶分类 + 概率对齐 + 极端概率提取
│   ├── models.py / models_dl.py / models_gnn.py  # 模型定义
│   ├── trainer.py / evaluator.py    # 训练 & 评估
│   └── utils.py                     # 工具函数
├── scripts/
│   ├── tune_decoupled_models.py     # Decoupled 调参
│   ├── train_decoupled_window_models.py  # Decoupled 训练
│   ├── make_decoupled_qualification_package.py  # Decoupled 打包
│   ├── tune_single_horizon_models.py  # H168 调参
│   ├── train_single_horizon_models.py  # H168 训练
│   ├── make_single_horizon_package.py  # H168 打包
│   ├── oof_fusion.py                # OOF 融合权重搜索
│   ├── run_qualification_pipeline.sh   # 资格赛一键流程
│   └── ...
├── data/
│   ├── raw/                         # USGS、GCMT、PB2002 原始数据
│   ├── processed/                   # 特征和标签
│   ├── models/                      # 训练产物
│   └── test_sequences/              # 测试震例
└── reports/                         # 报告与文档
```

## 八、结论

本系统以 Gutenberg-Richter、Omori-Utsu、Båth、ETAS 等经典地震学定律为特征基础，采用混合架构（树模型 + 选配 Transformer/ST-GNN）实现多窗口余震预测。通过震级/时间目标分离、时间桶分类离散化、OOF 融合权重搜索和后校准等设计，在 T1/T2 窗口取得了较好的预测性能（震级 MAE < 0.42，时间命中率 100%）。T3 窗口的时间预测仍是主要挑战，后续将通过层次化 hazard/ranking 模型和物理信息正则化等手段持续优化。
