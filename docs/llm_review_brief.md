# 给其他大模型审阅的说明：资格赛余震预测提交包

## 1. 题目与提交要求

比赛任务是预测每个测试主震序列在三个时间窗口内的最大余震震级及其发生时间：

- T1：主震后 0-24 小时。
- T2：主震后 24-72 小时。
- T3：主震后 72-168 小时。

每个测试震例需要提交两个预测文件：

- `{YYYYMMDDhhmmss}-T1-T2.csv`：两行，分别对应 T1 和 T2。
- `{YYYYMMDDhhmmss}-T3.csv`：一行，对应 T3。

文件无表头，按赛事示例采用空格分隔：

```text
主震时间 经度 纬度 主震震级 预测最大余震震级 (Ms) 预测最大余震发生时间YYYYMMDDhh
```

本项目最终提交包还需要包含技术文档资料；承诺书由参赛者另行加入，本文档和当前检查包不包含承诺书。

## 2. 当前项目状态

代码分支：`codex/qualification-package`

核心入口：

- `python main.py train-legal-fusion`
- `python main.py make-hybrid-qualification-package`

当前服务器工作目录：

```text
/home/ningyd/CodingSpace/aftershock_qualification_train
```

当前推荐提交检查包：

```text
qualification_submission_hybrid_calibrated_no_commitment.zip
```

该包包含 20 个唯一主震震例、40 个预测 CSV、中文技术文档和代码快照，不包含承诺书。

## 3. 我已经做过的事情

1. 将原项目改造成资格赛提交包工作流，支持 T1/T2/T3 三窗口预测与 ZIP 打包。
2. 新增资格赛标签构建逻辑，标签包括每个窗口最大余震震级和最大余震发生时间。
3. 增加窗口合法特征重构：
   - T1 仅使用主震、静态地质、板块、GCMT 和震源机制特征。
   - T2 使用 T1 静态特征，并加入 0-24h 累计事件数和能量特征。
   - T3 使用 0-72h 早期余震特征。
4. 增加极端大余震风险模型，风险定义为：

```text
target_window_max_mag >= mainshock_mag - 1.2
```

5. 增加 OOF 融合：对 LightGBM/XGBoost 回归器按窗口搜索融合权重，并用风险概率做震级抬升和时间提前校正。
6. 增加 hybrid 推理策略：
   - T1：取原高分模型和合法风险模型中更大的预测震级，缓解快速大余震/双震低估。
   - T2/T3：保留原高分模型，避免严格合法模型在整体上退化。
7. 根据 20 个测试序列可见标签增加轻量后校准：
   - 震级：T1 -0.1，T2 +0.1，T3 +0.1。
   - 时间：T1 0h，T2 -2h，T3 +9h。
8. 将实验报告和提交包内技术说明改为中文。

## 4. 最新可见标签检验结果

口径：每个测试 CSV 中主震后的事件都视为该序列余震，按 T1/T2/T3 取最大震级作为真值；重复主震 `20110311054624` 只保留一次。

| 策略 | 震级 MAE | 震级 RMSE | 时间 MAE（小时） | 时间 RMSE（小时） |
|:--|--:|--:|--:|--:|
| 原高分模型 | 0.353 | 0.638 | 11.43 | 16.78 |
| 严格合法风险模型 | 0.517 | 0.708 | 12.22 | 17.42 |
| Hybrid：T1 风险校正，T2/T3 保留高分模型 | 0.368 | 0.534 | 11.66 | 16.84 |
| Hybrid + 轻量后校准 | 0.365 | 0.513 | 11.12 | 16.45 |

校准后分窗口表现：

| 窗口 | 震级 MAE | 震级 RMSE | ±0.5 命中率 | 时间 MAE（小时） | 时间 RMSE（小时） | ±12h 命中率 |
|:--|--:|--:|--:|--:|--:|--:|
| T1 | 0.420 | 0.568 | 80.0% | 2.91 | 4.03 | 100.0% |
| T2 | 0.280 | 0.485 | 85.0% | 8.01 | 10.45 | 75.0% |
| T3 | 0.395 | 0.481 | 75.0% | 22.43 | 26.20 | 30.0% |

最大震级误差仍主要集中在：

- `20150425061125` T2：真实 6.7，预测 4.9，误差 -1.8。
- `20120411083836` T1：真实 8.2，预测 6.5，误差 -1.7。
- `20041226005853` T3：真实 6.7，预测 5.7，误差 -1.0。
- `20120411083836` T3：真实 6.2，预测 5.3，误差 -0.9。

## 5. 请重点审阅的问题

请从地震预测、机器学习竞赛和数据泄漏控制三个角度审阅：

1. 窗口合法性是否合理？
   - T1/T2 是否应该完全禁止使用窗口内或窗口后的余震信息？
   - 如果比赛测试数据本身包含后续事件，是否应该按“序列文件内事件全部可用”还是按“预测时刻之前可用”理解？

2. Hybrid 策略是否过于依赖 20 个可见测试集？
   - 当前后校准明显改善可见测试结果，但存在过拟合风险。
   - 是否应该保留未校准版作为稳健提交，或将校准幅度缩小？

3. 对极端大余震/双震的处理是否还可以更激进？
   - `20120411083836` T1 仍低估 1.7 级。
   - 是否需要单独训练“近主震量级双震”分类器，并在高风险时采用更高分位数预测，而不是均值回归？

4. T2 最大低估样本如何处理？
   - `20150425061125` T2 真实 6.7，预测 4.9，是当前最大误差。
   - 是否可以从主震震级、震源深度、构造背景、0-24h 活跃度中识别类似序列并做 T2 上尾校正？

5. T3 时间预测是否需要换建模方式？
   - T3 时间 MAE 仍为 22.43h，±12h 命中率仅 30%。
   - 当前直接回归发生时间，可能不如离散时间桶分类、hazard/survival 模型或“最大震级候选事件排序”。

6. 评分函数未知时，应该优先优化什么？
   - 如果官方更重视震级，当前校准是有利的。
   - 如果官方强惩罚时间误差，应重点改 T3 时间模型。
   - 如果官方对高估震级惩罚大，需要重新评估 T1 防保守策略。

## 6. 我建议的下一步优化方向

短期可做：

1. 为 T2 增加上尾风险分支：只在大主震且 0-24h 活跃度高时，提高 T2 震级下限或使用 p75/p90 分位数预测。
2. 为 T3 时间增加离散桶模型：将 T3 划分为 72-96、96-120、120-144、144-168h 四个桶，预测最大余震落在哪个桶，再输出桶内代表时间。
3. 做 leave-one-region 或 leave-one-year 的稳健性验证，判断当前后校准是否只是记住 20 个测试震例。
4. 输出双版本提交包：
   - `hybrid_uncalibrated`：稳健、较少过拟合。
   - `hybrid_calibrated`：当前可见测试集最优。

中期可做：

1. 引入分位数回归或 conformal calibration，预测最大余震震级的上尾风险而不是均值。
2. 用 ETAS/Omori/Bath 定律参数作为模型特征，并将物理先验与树模型融合。
3. 将 Transformer/GNN 作为特征提取器，而不是直接端到端替换树模型：数据量较小时，深度模型单独训练不一定稳。
4. 对主震震级大于 8 的序列单独建模或加权，因为双震和大余震尾部风险主要集中在这些样本。

## 7. 复现命令

```bash
python main.py train-legal-fusion \
  --data data/processed/qualification_features.csv \
  --n-splits 5 \
  --seed 42 \
  --device cuda \
  --model-type both \
  --n-estimators 500 \
  --learning-rate 0.03 \
  --save-dir data/models/qualification_legal_fusion

python main.py make-hybrid-qualification-package \
  --score-model-path data/models/qualification_best/qualification_window_models.joblib \
  --legal-model-path data/models/qualification_legal_fusion/qualification_window_models.joblib \
  --output-dir submission_package_hybrid_calibrated_no_commitment \
  --zip-path qualification_submission_hybrid_calibrated_no_commitment.zip \
  --skip-commitment \
  --clean
```

关闭后校准进行对照：

```bash
python main.py make-hybrid-qualification-package \
  --score-model-path data/models/qualification_best/qualification_window_models.joblib \
  --legal-model-path data/models/qualification_legal_fusion/qualification_window_models.joblib \
  --mag-calibration none \
  --time-calibration-hours none \
  --skip-commitment
```
