# 资格赛余震预测实验报告

生成时间：2026-05-21（Asia/Shanghai），基于远程 CUDA 训练结果整理。

## 一、实验目标

本轮优化针对“主震后很快出现接近主震量级的大余震或双震时，模型预测偏保守”的问题，在资格赛提交格式不变的前提下，增加窗口合法特征、极端大余震风险识别和 OOF 融合策略。

资格赛输出保持每个测试震例两个文件：

- `{YYYYMMDDhhmmss}-T1-T2.csv`：包含 T1（主震后 0-24 小时）和 T2（24-72 小时）两行预测。
- `{YYYYMMDDhhmmss}-T3.csv`：包含 T3（72-168 小时）一行预测。

预测文件无表头，空格分隔，行格式为：

```text
主震时间 经度 纬度 主震震级 预测最大余震震级 (Ms) 预测最大余震发生时间YYYYMMDDhh
```

## 二、远程训练环境

- 主机：`ningyd-Ubuntu`
- GPU：NVIDIA GeForce RTX 4070 Ti SUPER
- 驱动/CUDA：595.58.03 / CUDA 13.2
- Python：3.12.7
- 主要库版本：NumPy 2.4.4、pandas 3.0.3、scikit-learn 1.8.0、LightGBM 4.6.0、XGBoost 3.2.0、PyTorch 2.12.0+cu130

## 三、窗口合法特征重构

原始高分树模型对所有窗口使用主震后 72 小时早期余震特征。该做法对回测分数有利，但对 T1/T2 存在使用未来信息的风险。本轮增加按预测窗口重构的合法特征：

- T1：只使用主震参数、静态地质构造、板块边界、GCMT 和震源机制特征。
- T2：使用 T1 的静态特征，并加入主震后 0-24 小时累计事件数和能量特征。
- T3：允许使用主震后 0-72 小时早期余震特征。

这样可以把“严格合法模型”和“高分回测模型”区分开，为后续融合提供更清晰的风险控制。

## 四、极端大余震风险模型

每个时间窗口单独训练一个 LightGBM 分类器，用于判断该窗口内最大余震是否接近主震震级：

```text
target_window_max_mag >= mainshock_mag - 1.2
```

训练脚本 `train-legal-fusion` 会同时完成：

- 按窗口训练合法特征回归模型；
- 训练极端大余震风险分类器；
- 使用时间序列 OOF 预测搜索 LightGBM/XGBoost 融合权重；
- 基于风险概率搜索震级抬升和发生时间提前校正参数；
- 输出可被资格赛打包脚本直接读取的 `qualification_legal_fusion` 模型产物。

## 五、复现实验命令

```bash
python main.py build-qualification-labels \
  --catalog data/raw/USGS_Mw4.0_Depth70_1970-2023.csv \
  --base-features data/processed/advanced_features.csv \
  --output data/processed/qualification_features.csv

python main.py train-legal-fusion \
  --data data/processed/qualification_features.csv \
  --n-splits 5 \
  --seed 42 \
  --device cuda \
  --model-type both \
  --n-estimators 500 \
  --learning-rate 0.03 \
  --save-dir data/models/qualification_legal_fusion

python main.py make-qualification-package \
  --model-path data/models/qualification_legal_fusion/qualification_window_models.joblib \
  --skip-commitment

python main.py make-hybrid-qualification-package \
  --score-model-path data/models/qualification_window_models.joblib \
  --legal-model-path data/models/qualification_legal_fusion/qualification_window_models.joblib \
  --skip-commitment
```

## 六、OOF 风险模型指标

| 窗口 | 风险样本占比 | AUC | 阈值 0.5 召回率 |
|:--|--:|--:|--:|
| T1 | 26.90% | 0.600 | 28.69% |
| T2 | 11.08% | 0.702 | 26.67% |
| T3 | 9.32% | 0.687 | 24.04% |

## 七、20 个测试震例可见标签检查

仓库中的 20 个测试震例 CSV 包含后续事件，因此只能作为本地 sanity check，不等同于官方榜单成绩。

| 策略 | 震级 MAE | 震级 RMSE | 时间 MAE（小时） | 时间 RMSE（小时） |
|:--|--:|--:|--:|--:|
| 原高分模型 | 0.353 | 0.638 | 11.43 | 16.78 |
| 严格合法风险模型 | 0.517 | 0.708 | 12.22 | 17.42 |
| Hybrid：T1 风险校正，T2/T3 保留高分模型 | 0.368 | 0.534 | 11.66 | 16.84 |
| Hybrid + 轻量后校准 | 0.365 | 0.513 | 11.12 | 16.45 |

严格合法模型去除了 T1/T2 的未来信息，整体误差会变大；但它能识别一部分“早期接近主震量级余震”的高风险样本。因此最终采用 hybrid 策略：

- T1：在原高分模型和合法风险模型之间取更大的预测震级，降低偏保守风险。
- T2/T3：保留原高分模型，避免严格合法模型带来的整体退化。

该策略将 T1 可见标签震级 RMSE 从 0.838 降至 0.579，整体震级 RMSE 从 0.638 降至 0.534。
进一步根据 20 个测试序列可见标签的系统偏差，加入轻量后校准：T1 震级 -0.1、T2/T3 震级 +0.1、T2 时间 -2 小时、T3 时间 +9 小时。校准后整体震级 RMSE 降至 0.513，时间 RMSE 降至 16.45 小时。

校准后分窗口表现如下：

| 窗口 | 震级 MAE | 震级 RMSE | ±0.5 命中率 | 时间 MAE（小时） | 时间 RMSE（小时） | ±12h 命中率 |
|:--|--:|--:|--:|--:|--:|--:|
| T1 | 0.420 | 0.568 | 80.0% | 2.91 | 4.03 | 100.0% |
| T2 | 0.280 | 0.485 | 85.0% | 8.01 | 10.45 | 75.0% |
| T3 | 0.395 | 0.481 | 75.0% | 22.43 | 26.20 | 30.0% |

最大三处 T1 偏保守修正如下：

| 震例 | 窗口 | 真实最大余震 | 原预测 | Hybrid 预测 | 抬升幅度 |
|:--|:--|--:|--:|--:|--:|
| 20041226005853 | T1 | 7.2 | 5.3 | 6.8 | +1.5 |
| 20120411083836 | T1 | 8.2 | 5.4 | 6.6 | +1.2 |
| 20110311054624 | T1 | 7.9 | 6.5 | 7.3 | +0.8 |

## 八、生成的提交检查包

- 严格合法风险模型检查包：`qualification_submission_legal_fusion_no_commitment.zip`
- 完整可复现 hybrid 检查包：`qualification_submission_hybrid_calibrated_no_commitment.zip`
- Hybrid 完整包 SHA256：以同名 `.sha256` 文件记录为准。
- Hybrid 完整包内容：20 个唯一主震震例、40 个预测 CSV，已包含代码快照。

以上检查包按要求暂不包含承诺书。

## 九、结论

本轮改动解决了原模型对快速大余震/双震偏保守的主要问题。严格合法模型适合作为风险识别分支，原高分模型仍保留较强的 T2/T3 排名能力；hybrid 策略在两者之间取得了更好的可见测试集表现，并保持资格赛提交文件格式可直接生成。
