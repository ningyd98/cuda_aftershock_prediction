# 资格赛余震预测项目架构审阅

## 一、总体结论

当前代码库已经从原始“单一最大余震预测”扩展为面向资格赛的三窗口提交体系：`T1(0-24h)`、`T2(24-72h)`、`T3(72-168h)`。核心建模、合法特征重构、极端大余震风险模型、hybrid 打包和报告生成已经形成闭环，可以直接生成不含承诺书的检查包，加入官方承诺书后可生成正式提交包。

主要架构问题不在模型代码，而在入口层：旧版 `run.sh` 仍以 30 天单目标预测为主，资格赛最终流程隐藏在多个脚本里。此次更新后，推荐入口固定为：

```bash
./run.sh qualification --no-install
```

需要重训时使用：

```bash
./run.sh qualification --full-train --no-install
```

## 二、当前模块边界

| 层级 | 主要文件 | 职责 |
|---|---|---|
| 统一入口 | `main.py` | 将命令映射到训练、标签、打包脚本 |
| 一键流程 | `run.sh`、`scripts/run_qualification_pipeline.sh` | 资格赛一键构建、可选重训、ZIP 校验 |
| 资格赛领域逻辑 | `src/qualification.py` | T1/T2/T3 窗口定义、合法特征重构、预测文件格式、时间/震级裁剪 |
| 标签构建 | `scripts/build_qualification_labels.py` | 从完整地震目录追加三窗口目标列 |
| score model | `scripts/train_window_baseline.py` | 三窗口树模型，分别预测最大余震震级和发生时间 |
| legal-risk model | `scripts/train_legal_fusion.py` | 按窗口重构合法特征，并加入极端大余震风险建模与 OOF 融合 |
| 提交包 | `scripts/make_qualification_package.py`、`scripts/make_hybrid_qualification_package.py`、`scripts/make_decoupled_qualification_package.py` | 生成 `predictions/`、`technical_docs/`、`commitment/` 和 ZIP |
| 说明与审阅 | `reports/experiment_report.md`、`reports/llm_review_brief.md`、`reports/decoupled_tuning_plan.md` | 中文实验报告、设计说明和调参方案 |

## 三、数据流

```mermaid
flowchart LR
    A["完整地震目录"] --> B["build_qualification_labels"]
    C["advanced_features.csv"] --> B
    B --> D["qualification_features.csv"]
    D --> E["train_window_baseline score model"]
    D --> F["train_legal_fusion legal-risk model"]
    G["20 个测试震例"] --> H["make_hybrid_qualification_package"]
    E --> H
    F --> H
    H --> I["predictions/*.csv"]
    H --> J["technical_docs/*"]
    H --> K[\"qualification_submission_*.zip\"]
    D --> L[\"tune_decoupled_models\"] 
    L --> M[\"train_decoupled_window_models\"] 
    M --> N[\"make_decoupled_qualification_package\"]
    G --> N
    N --> I
    N --> J
```

## 四、优点

- 资格赛窗口已经显式建模，输出文件名、行数、列顺序、无表头、空格分隔、`(Ms)` 类型和小时级时间都集中在 `src/qualification.py`。
- `reconstruct_legal_window_features` 把 T1/T2/T3 可用观测信息分开，减少了 T1/T2 使用未来 72 小时特征的风险。
- hybrid 策略把原高分模型与合法风险模型解耦，便于比较 calibrated 和 uncalibrated 包。
- 打包脚本会写入 manifest、报告、模型指标、代码快照和 SHA256，复查性比旧版单 CSV 输出更强。
- 新增 `./run.sh qualification` 后，训练服务器和本地都可以用同一个入口复现最终包。

## 五、主要风险

- 当前仓库同时保留旧的 30 天单目标流程和资格赛三窗口流程，概念上容易混淆；README 需要持续把“资格赛推荐入口”放在最前。
- 后校准参数来自 20 个可见测试序列的偏差分析，能改善当前可见准确度，但存在榜外泛化风险；建议同时保留 `--no-calibration` 包用于对照。
- T3 时间误差仍是最大短板，说明 72-168h 余震发生时刻的信息量不足，后续应重点做 T3 hazard/ranking 模型。
- 承诺书被用户要求暂不处理；正式提交时必须通过 `--commitment-template` 放入官方模板。
- 模型权重不进入 GitHub，服务器和本地路径需要通过脚本参数显式声明，避免其他机器直接运行时找不到模型。

## 六、已更新的一键脚本

新增 `scripts/run_qualification_pipeline.sh`，并让根目录 `run.sh` 支持：

```bash
./run.sh qualification --no-install
```

默认行为：

- 自动选择 `data/models/qualification_best/qualification_window_models.joblib` 作为 score model；如果不存在，则回退到 `data/models/qualification_window_models.joblib`。
- 使用 `data/models/qualification_legal_fusion/qualification_window_models.joblib` 作为 legal-risk model。
- 默认生成 `qualification_submission_hybrid_calibrated_no_commitment.zip`。
- 默认跳过承诺书，符合当前“承诺书不用管”的执行要求。
- 生成后校验 ZIP 中是否包含 `predictions/`、`technical_docs/` 和 `MANIFEST.json`，并打印 SHA256。

常用命令：

```bash
# 只打包当前服务器已有最优模型
./run.sh qualification --no-install

# 重建标签，并重训 score model 与 legal-risk model
./run.sh qualification --full-train --no-install

# 只重训 legal-risk model，然后重新打包
./run.sh qualification --retrain-legal --no-install

# 生成未校准对照包
./run.sh qualification --no-calibration --zip-path qualification_submission_hybrid_uncalibrated_no_commitment.zip --no-install

# 加入官方承诺书，生成正式 ZIP
./run.sh qualification --commitment-template /path/to/commitment.docx --no-install
```

## 七、下一步优化建议

1. 建立 `configs/qualification.yaml`，把模型路径、校准参数、窗口定义、输出包名集中配置，减少脚本参数漂移。
2. 增加 `scripts/evaluate_qualification_predictions.py`，把 20 个测试集的可见准确度检查纳入一键流程的可选步骤。
3. 针对 T3 时间建立分布式 hazard/ranking 模型，先预测窗口内日级风险，再回归小时级峰值时间。
4. 对极端大余震样本做时间分层 OOF 校准，避免只按 20 个测试序列调整导致过拟合。
5. 将 `run_tuning.sh` 改造为资格赛专用调参入口，或在 README 中明确标记为旧流程。

## 八、Decoupled Pipeline（新增）

新增震级/时间分离训练管道（详见 `reports/decoupled_tuning_plan.md`）：

| 脚本 | 功能 |
|------|------|
| `src/time_buckets.py` | 时间桶定义，4 桶分类 + 期望时间计算 |
| `scripts/train_decoupled_window_models.py` | 分离训练 MagModel + TimeBucketModel + ExtremeClassifier |
| `scripts/tune_decoupled_models.py` | Optuna/随机搜索超参数调优（支持 balanced/mag/time/official_like 评分） |
| `scripts/make_decoupled_qualification_package.py` | Decoupled 模型推理打包 |

一键运行：
```bash
# Smake test
./run.sh qualification --tune-decoupled --tune-fast --tune-trials 3 \
  --retrain-decoupled --use-decoupled --no-install

# 完整调参
./run.sh qualification --tune-decoupled --tune-trials 80 \
  --tune-objective balanced --retrain-decoupled --use-decoupled --no-install
```
