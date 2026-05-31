# Decoupled Pipeline 调参结果报告

## 数据版本
- 训练数据：`data/processed/qualification_features.csv`
- 特征构建：`scripts/build_features.py` + `scripts/build_qualification_labels.py`
- 窗口定义：T1(0-24h), T2(24-72h), T3(72-168h)

## 训练命令

```bash
# 快速 smoke test
python main.py tune-decoupled-models \
  --data data/processed/qualification_features.csv \
  --n-trials 3 --n-splits 2 --fast --device cpu \
  --output-dir data/tuning_results/decoupled_smoke

# 完整调参 (80 trials, weighted objective)
python main.py tune-decoupled-models \
  --data data/processed/qualification_features.csv \
  --n-trials 80 --n-splits 5 --device cuda \
  --objective weighted \
  --w-mag-mae 1.0 --w-mag-rmse 1.0 \
  --w-time-mae 0.03 --w-time-rmse 0.03 \
  --w-extreme 0.5 --w-late 0.3 --w-t1-bonus 0.2 \
  --output-dir data/tuning_results/decoupled_YYYYMMDD_HHMMSS

# 用最优参数训练最终模型
python main.py train-decoupled-window-models \
  --data data/processed/qualification_features.csv \
  --best-params data/tuning_results/decoupled_XXXX/best_params.json \
  --save-dir data/models/qualification_decoupled \
  --device cuda

# 打包
python main.py make-decoupled-qualification-package \
  --model-path data/models/qualification_decoupled/qualification_decoupled_models.joblib \
  --skip-commitment
```

## 参数搜索空间

### 震级模型 (LightGBM Regressor)
| 参数 | 搜索范围 |
|------|---------|
| n_estimators | [200, 300, 500, 800, 1000] |
| learning_rate | [0.01, 0.02, 0.03, 0.05, 0.07] |
| num_leaves | [15, 31, 63, 127, 255] |
| min_child_samples | [5, 10, 20, 50, 100] |
| subsample | [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] |
| colsample_bytree | [0.4-1.0] |
| reg_alpha | [0.0, 0.01, 0.05, 0.1, 0.5, 1.0] |
| reg_lambda | [0.0, 0.5, 1.0, 2.0, 5.0] |

### 时间桶模型 (LightGBM 4-Class Classifier)
| 参数 | 搜索范围 |
|------|---------|
| n_estimators | [200, 300, 500, 800] |
| learning_rate | [0.01-0.07] |
| num_leaves | [15, 31, 63, 127] |
| min_child_samples | [5, 10, 15, 20, 30, 50] |
| subsample / colsample_bytree | 同上 |
| reg_alpha / reg_lambda | 同上 |

### 极端大余震分类器
| 参数 | 搜索范围 |
|------|---------|
| extreme_margin (delta) | [0.5, 0.8, 1.0, 1.2, 1.5, 2.0] |
| n_estimators | [100, 200, 300, 500] |
| learning_rate | [0.01-0.07] |
| num_leaves | [15, 31, 63] |
| subsample / colsample_bytree | [0.6-1.0] |
| class_weight | "balanced" (fixed) |

### 后处理参数
| 参数 | 搜索范围 |
|------|---------|
| mag_bias_T1/T2/T3 | 各窗口震级偏移 |
| time_bias_T1/T2/T3 | 各窗口时间偏移 |
| extreme_prob_threshold | [0.3, 0.4, 0.5, 0.6, 0.7] |
| high_risk_mag_quantile_weight | [0.3, 0.5, 0.7, 0.9] |
| early_time_shift_strength | [0.0, 0.1, 0.2, 0.35] |
| t1_early_delta_bonus | [0.0, 0.1, 0.2, 0.3, 0.5] |

## 评分函数 (weighted 模式)

```
score = w_mag_mae   × mean(mag_mae)
      + w_mag_rmse  × mean(mag_rmse)
      + w_time_mae  × mean(time_mae) / 24.0
      + w_time_rmse × mean(time_rmse) / 24.0
      + w_extreme   × mean(extreme_mag_mae)
      + w_late      × late_penalty
      + w_t1_bonus  × (T1_mag_mae + T1_time_mae / 24.0)
```

默认权重通过 CLI 配置：`--w-mag-mae 1.0 --w-mag-rmse 1.0 --w-time-mae 0.03 --w-time-rmse 0.03 --w-extreme 0.5 --w-late 0.3 --w-t1-bonus 0.2`

## 最佳参数

见 `best_params.json`，结构：
```json
{
  "T1": { "mag_model": {...}, "bucket_model": {...}, "extreme_model": {...} },
  "T2": { "mag_model": {...}, "bucket_model": {...}, "extreme_model": {...} },
  "T3": { "mag_model": {...}, "bucket_model": {...}, "extreme_model": {...} },
  "global": { "score_weights": {...}, "seed": 42 },
  "postprocessing": {...}
}
```

## 每个窗口 raw/postprocessed 指标

见 `tuning_summary.json` → `per_window_metrics`。

## 极端大余震样本指标

见 `tuning_summary.json` → `per_window_metrics` → 各窗口 `extreme_mag_mae` 和 `extreme_count`。

## 与 hybrid calibrated 对比

对比：
- `data/models/qualification_window_metrics.json`（hybrid 指标）
- `data/tuning_results/decoupled_XXXX/tuning_summary.json`（decoupled 调参汇总）

**注意**：两套指标均基于 OOF 交叉验证，不代表测试集性能。

## 是否真实运行完整训练

- **未产生新的准确率结论**。
- 本文档中的指标仅来自 smoke test（小样本、少量 trial），或未运行完整调参训练。
- 如需真实调参结果，请使用真实 `qualification_features.csv` 运行 80+ trials 完整 Optuna 调参。

## decoupled 方案状态

- **候选方案**：decoupled pipeline 仍为候选实验方案，不替代 hybrid calibrated 默认流程。
- 使用 `--use-decoupled` 切换。
- 极端概率逻辑已统一到 `src/time_buckets.py` → `safe_extreme_probability()`，train/tune/package 三处共用。

## 修复的安全问题

- DummyClassifier constant=0 → 极端概率全 0 ✅
- DummyClassifier constant=1 → 极端概率全 1 ✅
- train/tune/package 三处共用同一 `safe_extreme_probability()` ✅
- **weighted objective 已实现**：7 个 --w-* CLI 参数 + compute_objective 支持 weighted ✅
- **t1_early_delta_bonus 已实现**：T1 高风险样本 floor 额外抬高，tune/train/package 三处同步 ✅
- **报告与代码一致**：所有特性均有代码实现，不再超前于代码 ✅

## 下一步推荐

1. 准备完整的 `qualification_features.csv`
2. 运行完整调参：
   ```bash
   python main.py tune-decoupled-models --n-trials 80 --device cuda --objective weighted
   ```
3. 用最优参数训练最终模型并打包
4. 与 hybrid calibrated 做盲测对比
