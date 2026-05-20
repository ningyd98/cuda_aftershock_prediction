# 余震预测全模型超参数调优报告

> 生成时间: 2026-05-20 09:35 UTC
> 调优算法: Optuna TPESampler (多变量联合采样)
> 试验次数: 100

## 1. 总体概述

| 指标 | 值 |
|:---|---:|
| 最优 Trial 编号 | **#91** |
| 最优 OOF 综合分 (越低越好) | **10.3567** |
| 参与融合的模型 | baseline, xgboost |
| 调优 DL 模型 | ✅ |
| 调优 GNN 模型 | ✅ |
| 时间非对称惩罚权重 | 2.0× |

## 2. OOF 交叉验证指标

OOF (Out-of-Fold) 指标反映模型对**未见过的未来数据**的预测能力。

| 指标 | 最优 Trial 值 | 说明 |
|:---|---:|:---|
| 震级 RMSE | 0.5170 | 预测余震震级的均方根误差 |
| 时间非对称 RMSE | 8.8058 | 预测偏晚惩罚 2× 的时间误差 |
| **综合分 (mag × 3 + time)** | **10.3567** | Optuna 优化目标 |

## 3. Holdout 测试集评估

在 **20 条独立测试序列**上的最终评估（未参与训练/验证）：

### 3.1 震级预测
| 指标 | 值 | 说明 |
|:---|---:|:---|
| RMSE | 2.5063 | 震级均方根误差 |
| MAE | 1.9712 | 平均绝对误差 |
| MedAE | 1.4146 | 中位数绝对误差（鲁棒） |
| Energy Ratio (median) | 132.69× | 典型能量偏差倍数 |

### 3.2 时间预测
| 指标 | 值 | 说明 |
|:---|---:|:---|
| RMSE | 8.6065 | 时间均方根误差（天） |
| MAE | 7.2998 | 平均绝对误差（天） |
| MedAE | 5.6895 | 中位数绝对误差（天） |
| 非对称 RMSE | 9.7904 | 预测偏晚 2× 惩罚 |
| 非对称 MAE | 11.1538 | 非对称平均绝对误差 |

### 3.3 物理一致性
| 指标 | 值 | 说明 |
|:---|---:|:---|
| Båth ΔM Deviation | 1.1690 | ΔM 预测偏差，越低越好 |
| **Holdout 综合分** | **12.2966** | mag_rmse + time_asymmetric_rmse |

## 4. 最优融合权重

震级 (mag) 和时间 (time) 目标独立搜索最优权重。

### 震级预测 (Mag) 融合权重

| 模型 | 权重 | 占比 |
|:---|---:|---:|
| LightGBM (基线) | 0.4000 | 40.0% |
| XGBoost | 0.6000 | 60.0% |
| Transformer | 0.0000 | 0.0% |
| ST-GNN | 0.0000 | 0.0% |

### 时间预测 (Time) 融合权重

| 模型 | 权重 | 占比 |
|:---|---:|---:|
| LightGBM (基线) | 0.4800 | 48.0% |
| XGBoost | 0.5200 | 52.0% |
| Transformer | 0.0000 | 0.0% |
| ST-GNN | 0.0000 | 0.0% |

> 权重为 0 的模型不参与最终推理。

## 5. 最优超参数

### LightGBM

| 参数 | 最优值 |
|:---|---:|
| `lgb_colsample_bytree` | 0.444857 |
| `lgb_lr` | 0.012231 |
| `lgb_max_depth` | 8 |
| `lgb_min_child_samples` | 92 |
| `lgb_num_leaves` | 233 |
| `lgb_reg_alpha` | 0.065368 |
| `lgb_reg_lambda` | 4.745790 |
| `lgb_subsample` | 0.895300 |

### XGBoost

| 参数 | 最优值 |
|:---|---:|
| `xgb_colsample_bytree` | 0.909363 |
| `xgb_lr` | 0.011611 |
| `xgb_max_depth` | 3 |
| `xgb_min_child_weight` | 31 |
| `xgb_reg_alpha` | 4.762595 |
| `xgb_reg_lambda` | 7.771949 |
| `xgb_subsample` | 0.635267 |

### Transformer (DL)

| 参数 | 最优值 |
|:---|---:|
| `dl_batch_size` | 32 |
| `dl_d_model` | 256 |
| `dl_dim_ff` | 512 |
| `dl_dropout` | 0.172844 |
| `dl_epochs` | 38 |
| `dl_fusion_hidden` | 128 |
| `dl_global_hidden` | 256 |
| `dl_lr` | 0.002559 |
| `dl_nhead` | 4 |
| `dl_num_layers` | 1 |

### ST-GNN

| 参数 | 最优值 |
|:---|---:|
| `gnn_batch_size` | 16 |
| `gnn_dropout` | 0.145867 |
| `gnn_epochs` | 25 |
| `gnn_fusion_hidden` | 64 |
| `gnn_global_hidden` | 64 |
| `gnn_gru_hidden` | 32 |
| `gnn_gru_layers` | 1 |
| `gnn_layers` | 4 |
| `gnn_lr` | 0.001302 |
| `gnn_node_hidden` | 32 |
| `gnn_radius_km` | 119.028878 |
| `gnn_sigma` | 28.510090 |

### 特征工程 & OOF

| 参数 | 最优值 |
|:---|---:|
| `ensemble_grid_step` | 0.020000 |
| `feature_selection` | False |
| `feature_selection_min` | 47 |
| `feature_selection_ratio` | 0.607335 |
| `mag_weight` | 3.000000 |
| `min_purge_days` | 15.698728 |
| `purge_days` | 84.001430 |

## 6. 使用建议

### 使用最优参数重新训练

```bash
# 查看最优参数
cat data/tuning_results/20260520_151227/tune_20260520_071229/best_params.json

# 查看融合权重
cat data/tuning_results/20260520_151227/tune_20260520_071229/ensemble_weights.json

# 使用最优融合权重运行 OOF 全流程
cp data/tuning_results/20260520_151227/tune_20260520_071229/ensemble_weights.json data/models/ensemble_weights.json
./run.sh --skip-download --train-oof-ensemble
```

### 产物清单

| 文件 | 说明 |
|:---|:---|
| `data/tuning_results/20260520_151227/tune_20260520_071229/best_params.json` | 最优超参数 (JSON) |
| `data/tuning_results/20260520_151227/tune_20260520_071229/ensemble_weights.json` | 双目标最优融合权重 (JSON) |
| `data/tuning_results/20260520_151227/tune_20260520_071229/tuning_stats.json` | 调优统计汇总 (JSON) |
| `data/tuning_results/20260520_151227/tune_20260520_071229/holdout_predictions.csv` | Holdout 预测 vs 真实值 |
| `data/tuning_results/20260520_151227/tune_20260520_071229/trials_history.csv` | 所有 trial 记录 |
| `data/tuning_results/20260520_151227/tune_20260520_071229/tuning_report.md` | **本报告** |
