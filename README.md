# 余震预测技术国际大赛

本项目面向阿里云天池“余震预测技术国际大赛”，用于构建全球浅源强震的主震-余震样本、提取高级地震学特征，并为后续机器学习与深度学习模型开发预留工程接口。

## 目录

```text
.
├── main.py
├── configs/default.yaml
├── data/raw/
├── data/processed/
├── data/test_sequences/
├── src/
├── scripts/
└── docs/
```

## 常用命令

生成基础主震-余震样本：

```bash
python main.py build-sequences
```

生成阶段一高级特征：

```bash
python main.py build-features
```

快速测试前 50 条样本：

```bash
python main.py build-features --limit 50 --output data/processed/advanced_features_smoke.csv
```

下载原始数据：

```bash
python main.py download-usgs
python main.py download-pb2002
```

## 当前状态

- 已实现 Gutenberg-Richter b 值、MAXC 完整性震级、大森-宇津 MLE 参数拟合。
- 已实现 `joblib` 并行特征生成入口。
- 当前 `data/raw/USGS_Mw6.0_Depth70_1970-2023.csv` 是强震目录，因此可稳定拟合 b 值和大森参数的样本较少；后续建议补充低震级局部余震目录。
