from __future__ import annotations

"""
模型定义模块。

阶段二将在这里加入 LightGBM/XGBoost 多输出回归基线，以及 PyTorch
时序 Transformer 或 ST-GNN 架构。当前先保留清晰入口，避免工程结构缺口。
"""


class ModelNotImplementedError(NotImplementedError):
    """模型尚未进入当前阶段实现时抛出的异常。"""


def build_model(model_name: str, **kwargs):
    """根据模型名称构建预测模型。"""
    raise ModelNotImplementedError(f"模型 {model_name} 将在阶段二实现。")
