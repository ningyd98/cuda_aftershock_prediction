# Decoupled Pipeline 调参与训练方案

## 一、为什么拆分震级和时间？

在当前 hybrid calibrated 方案中，每个窗口使用同一个多输出回归器（MultiOutputRegressor）同时预测震级和时间。这有三个问题：

1. **目标分布不同**：震级分布近似正态，时间分布严重长尾（尤其是 T3 窗口 72-168h）。共享模型需要在两个不同物理规律的目标之间折中。
2. **特征相关性不同**：G-R 定律 b 值、Båth's Law Δm 主要驱动震级；大森-宇津 p 值、ETAS α 参数主要驱动时间。混在一起降低了各自可利用的物理先验。
3. **T3 时间误差特大**：当前 T3 时间 MAE ≈ 22.43h，RMSE ≈ 26.20h。这是因为回归器很难在 96h 宽的时间窗口内做高精度点预测。

## 二、为什么使用时间桶分类？

将时间预测从回归问题转为多分类 + 期望计算：

- **T1**: 4 个桶 (0-3h, 3-6h, 6-12h, 12-24h)
- **T2**: 4 个桶 (24-36h, 36-48h, 48-60h, 60-72h)
- **T3**: 4 个桶 (72-96h, 96-120h, 120-144h, 144-168h)

推理时，使用分类概率加权桶中心得到期望时间：E[t] = Σ p_k × center_k。

优势：
1. 分类任务在每个桶内学习分布模式，不受桶间方差干扰。
2. 对极值/异常值更鲁棒——即便模型选错桶，相邻桶的预测仍然是合理的。
3. 分类概率可直接用于不确定性量化（熵/置信度）。

## 三、模型结构

每个窗口（T1/T2/T3）独立训练：

| 模型 | 类型 | 目标 |
|------|------|------|
| MagModel | LightGBM/XGBoost 回归 | 最大余震震级 |
| TimeBucketModel | LightGBM 4 分类 | 最大余震所在时间桶 |
| ExtremeClassifier | LightGBM 二分类 | 是否 extreme 大余震（target_mag ≥ mainshock_mag - margin） |

后处理：
- `mag = MagModel 输出 + mag_bias`
- `extreme_prob > threshold` 时推向上尾分位数
- `time = 时间桶概率加权中心 + time_bias`
- `extreme_prob` 高时，时间向早期桶轻微移动
- 所有预测裁剪到窗口合法范围

## 四、调参搜索空间

### 4.1 震级模型参数
- n_estimators: [200, 300, 500, 800, 1000]
- learning_rate: [0.01, 0.02, 0.03, 0.05, 0.07]
- num_leaves: [15, 31, 63, 127, 255]
- min_child_samples: [5, 10, 20, 50, 100]
- subsample: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
- colsample_bytree: [0.4-1.0]
- reg_alpha: [0.0-1.0]
- reg_lambda: [0.0-5.0]

### 4.2 时间桶分类器参数
- 同上，额外关注 class_weight 策略

### 4.3 极端大余震分类器参数
- extreme_margin: [0.8, 1.0, 1.2, 1.4, 1.6, 2.0]
- n_estimators, learning_rate, num_leaves

### 4.4 后处理参数
- mag_bias_T1/T2/T3
- time_bias_T1/T2/T3
- extreme_prob_threshold
- high_risk_mag_quantile_weight
- early_time_shift_strength

## 五、评分函数定义

### balanced（默认推荐）
```
score = 1.0 × mean_mag_rmse
      + 0.03 × mean_time_rmse
      + 0.5 × extreme_mag_mae
      + 0.2 × T3_time_rmse / 24h
```

### time（侧重时间优化）
```
score = mean_time_mae + 0.5 × mean_time_rmse + 0.5 × T3_time_mae
```

### mag（侧重震级优化）
```
score = mean_mag_mae + mean_mag_rmse + extreme_mag_mae
```

### official_like（模拟官方评分）
```
score = mean_mag_mae + mean_mag_rmse
      + 0.02 × mean_time_mae + 0.02 × mean_time_rmse
      + 0.3 × late_penalty
      + 0.5 × extreme_underestimate_penalty
```

## 六、如何运行

### 快速 smoke test
```bash
# 极小搜索空间，3 trials，用于验证代码可运行
./run.sh qualification \
  --tune-decoupled \
  --tune-fast \
  --tune-trials 3 \
  --retrain-decoupled \
  --use-decoupled \
  --no-install
```

### 完整调参训练
```bash
# 80 trials，balanced 目标，完整搜索空间
./run.sh qualification \
  --tune-decoupled \
  --tune-trials 80 \
  --tune-objective balanced \
  --retrain-decoupled \
  --use-decoupled \
  --no-install
```

### 分步运行
```bash
# Step 1: 只做调参
python main.py tune-decoupled-models \
  --data data/processed/qualification_features.csv \
  --n-trials 80 \
  --objective balanced

# Step 2: 用最优参数训练最终模型
python main.py train-decoupled-window-models \
  --data data/processed/qualification_features.csv \
  --best-params data/tuning_results/decoupled_XXXX/best_params.json \
  --save-dir data/models/qualification_decoupled

# Step 3: 打包
python main.py make-decoupled-qualification-package \
  --model-path data/models/qualification_decoupled/qualification_decoupled_models.joblib \
  --skip-commitment
```

## 七、如何对比 hybrid calibrated 和 decoupled tuned

1. 生成 hybrid calibrated 包（默认流程）：
```bash
./run.sh qualification --no-install
```

2. 生成 decoupled tuned 包：
```bash
./run.sh qualification --tune-decoupled --tune-trials 80 \
  --retrain-decoupled --use-decoupled --no-install
```

3. 对比：
   - `data/models/qualification_window_metrics.json`（hybrid 指标）
   - `data/models/qualification_decoupled/decoupled_metrics.json`（decoupled OOF 指标）
   - `data/tuning_results/decoupled_XXXX/tuning_summary.json`（调参汇总）
   - 注意：OOF 指标反映训练数据的交叉验证性能，不代表测试集表现。

## 八、重要说明

- **候选方案**：decoupled pipeline 是候选实验方案，**不替代当前 hybrid calibrated 默认方案**。默认流程仍生成 hybrid calibrated ZIP。
- **未运行完整训练时，不得声称效果提升**。本文档仅描述方案设计和运行方法，不包含任何未经验证的性能声明。
- **调参优化 OOF 指标**：所有调参基于 TimeSeriesSplit OOF（Out-Of-Fold），不使用测试集标签。
- **后校准来自 OOF 验证**，不是从测试集拟合。
- **时间桶分类**：将时间回归转为 4 分类 + 期望计算，旨在降低 T3 时间预测误差。
- **上尾校正使用简单的 floor 校正**，并非真正的分位数回归（quantile regression）模型。extreme 后处理通过 floor（mainshock_mag - margin）上修震级，并通过早期偏移调整时间，但未训练 p75/p90 分位数模型。
- **默认保留 hybrid calibrated 流程**，decoupled 管道为可选增强方案。使用 `--use-decoupled` 切换。
- 调参支持 Optuna TPESampler（优先）和随机搜索回退（若 Optuna 未安装）。
