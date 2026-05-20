from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

TIME_TARGET_INDEX = 1


def transform_targets_for_fit(y, transform_time_target: bool = True) -> np.ndarray:
    """训练前对时间目标做 log1p；震级目标保持原尺度。"""
    y_array = np.asarray(y, dtype=float).copy()
    if transform_time_target:
        y_array[:, TIME_TARGET_INDEX] = np.log1p(
            np.clip(y_array[:, TIME_TARGET_INDEX], a_min=0.0, a_max=None)
        )
    return y_array


def inverse_transform_model_predictions(
    preds,
    transform_time_target: bool = True,
) -> np.ndarray:
    """模型预测后将时间目标还原为真实天数尺度。"""
    pred_array = np.asarray(preds, dtype=float).copy()
    if transform_time_target:
        pred_array[:, TIME_TARGET_INDEX] = np.expm1(
            np.clip(pred_array[:, TIME_TARGET_INDEX], a_min=-50.0, a_max=50.0)
        )
    pred_array[:, TIME_TARGET_INDEX] = np.clip(
        pred_array[:, TIME_TARGET_INDEX],
        a_min=0.0,
        a_max=None,
    )
    return pred_array


def asymmetric_mse_objective(
    y_true,
    y_pred,
    late_weight: float = 2.0,
    log_space: bool = True,
):
    """
    LightGBM 自定义非对称 MSE 目标。

    默认 y_true/y_pred 为 log1p(time) 尺度；先还原到真实天数判断
    “预测偏晚”，再通过链式法则返回相对 log-pred 的梯度和 Hessian。
    """
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if log_space:
        true_log = np.clip(true, a_min=-50.0, a_max=50.0)
        pred_log = np.clip(pred, a_min=-50.0, a_max=50.0)
        true_time = np.expm1(true_log)
        pred_time = np.expm1(pred_log)
        error = pred_time - true_time
        jacobian = np.exp(pred_log)
        weights = np.where(error > 0.0, float(late_weight), 1.0)
        grad = 2.0 * weights * error * jacobian
        # 使用正定 Gauss-Newton Hessian 近似，避免极端负误差导致 Hessian 非正。
        hess = 2.0 * weights * np.square(jacobian)
        return grad, np.clip(hess, a_min=1e-6, a_max=None)

    error = pred - true
    weights = np.where(error > 0.0, float(late_weight), 1.0)
    grad = 2.0 * weights * error
    hess = 2.0 * weights
    return grad, hess


@dataclass
class AsymmetricTimeObjective:
    """可序列化的 LightGBM 自定义目标包装器。"""

    late_weight: float = 2.0
    log_space: bool = True

    def __call__(self, y_true, y_pred):
        return asymmetric_mse_objective(
            y_true=y_true,
            y_pred=y_pred,
            late_weight=self.late_weight,
            log_space=self.log_space,
        )


class BaselineLGBM:
    """
    多输出树模型基线。

    优先使用 LightGBM；若本地未安装 lightgbm，则回退到 sklearn 的
    HistGradientBoostingRegressor，保证训练流程能端到端运行。
    """

    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 500,
        learning_rate: float = 0.02,
        num_leaves: int = 63,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.7,
        reg_alpha: float = 0.05,
        reg_lambda: float = 1.0,
        n_jobs: int = -1,
        use_asymmetric_time_objective: bool = False,
        late_weight: float = 2.0,
        transform_time_target: bool = True,
        device: str = "cpu",
        gpu_use_dp: bool = False,
        **model_kwargs,
    ) -> None:
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs
        self.use_asymmetric_time_objective = use_asymmetric_time_objective
        self.late_weight = late_weight
        self.transform_time_target = transform_time_target
        self.device = device
        self.gpu_use_dp = gpu_use_dp
        self.model_kwargs = model_kwargs
        self.backend = "lightgbm"
        self.model = self._build_model()

    def _build_model(self):
        """构建多输出回归器，支持 GPU 加速。"""
        try:
            from lightgbm import LGBMRegressor

            common_lgbm_params = {
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves,
                "max_depth": self.max_depth,
                "min_child_samples": self.min_child_samples,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "reg_alpha": self.reg_alpha,
                "reg_lambda": self.reg_lambda,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                "verbosity": -1,
                **self.model_kwargs,
            }

            # GPU 加速：LightGBM 支持 device="cuda"
            if self.device == "cuda":
                common_lgbm_params["device"] = "cuda"
                # gpu_use_dp: 使用双精度（更精确但略慢）
                if self.gpu_use_dp:
                    common_lgbm_params["gpu_use_dp"] = True
                self.backend = "lightgbm_cuda"

            if self.use_asymmetric_time_objective:
                self.backend = (
                    "lightgbm_asymmetric_time_cuda"
                    if self.device == "cuda"
                    else "lightgbm_asymmetric_time"
                )
                self.mag_model = LGBMRegressor(
                    objective="regression",
                    **common_lgbm_params,
                )
                self.time_model = LGBMRegressor(
                    objective=AsymmetricTimeObjective(
                        late_weight=self.late_weight,
                        log_space=self.transform_time_target,
                    ),
                    **common_lgbm_params,
                )
                return None

            base_model = LGBMRegressor(
                objective="regression",
                **common_lgbm_params,
            )
        except ImportError:
            self.backend = "sklearn_hist_gradient_boosting"
            self.use_asymmetric_time_objective = False
            base_model = HistGradientBoostingRegressor(
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_leaf_nodes=self.num_leaves,
                max_depth=None if self.max_depth == -1 else self.max_depth,
                random_state=self.random_state,
                **self.model_kwargs,
            )

        return MultiOutputRegressor(base_model)

    def fit(self, X, y):
        """训练多目标回归模型。"""
        use_asymmetric = getattr(self, "use_asymmetric_time_objective", False)
        transform_time = getattr(self, "transform_time_target", False)
        y_array = transform_targets_for_fit(y, transform_time_target=transform_time)
        if use_asymmetric and (self.backend or "").startswith("lightgbm_asymmetric_time"):
            if y_array.ndim != 2 or y_array.shape[1] != 2:
                raise ValueError("BaselineLGBM 自定义时间目标要求 y 为两列：[震级, 时间]。")
            self.mag_model.fit(X, y_array[:, 0])
            self.time_model.fit(X, y_array[:, 1])
            return self

        if self.model is None:
            raise RuntimeError(
                "BaselineLGBM.model 未初始化。"
                "请检查 use_asymmetric_time_objective 与 backend 配置是否一致。"
            )
        self.model.fit(X, y_array)
        return self

    def predict(self, X):
        """预测最大余震震级和发生时间。"""
        use_asymmetric = getattr(self, "use_asymmetric_time_objective", False)
        transform_time = getattr(self, "transform_time_target", False)
        if use_asymmetric and (self.backend or "").startswith("lightgbm_asymmetric_time"):
            mag_pred = np.asarray(self.mag_model.predict(X), dtype=float)
            time_pred = np.asarray(self.time_model.predict(X), dtype=float)
            raw_preds = np.column_stack([mag_pred, time_pred])
            return inverse_transform_model_predictions(
                raw_preds,
                transform_time_target=transform_time,
            )
        if self.model is None:
            raise RuntimeError(
                "BaselineLGBM.model 未初始化。"
                "请检查 use_asymmetric_time_objective 与 backend 配置是否一致。"
            )
        raw_preds = self.model.predict(X)
        return inverse_transform_model_predictions(
            raw_preds,
            transform_time_target=transform_time,
        )


class BaselineXGBoost:
    """
    XGBoost 多输出回归模型（第二 Baseline）。

    对应 project_plan 第 4.1 节，用于与 LightGBM 做模型融合。
    """

    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 500,
        learning_rate: float = 0.02,
        max_depth: int = 7,
        subsample: float = 0.8,
        colsample_bytree: float = 0.7,
        reg_alpha: float = 0.05,
        reg_lambda: float = 1.5,
        min_child_weight: int = 5,
        gamma: float = 0.1,
        n_jobs: int = -1,
        transform_time_target: bool = True,
        **model_kwargs,
    ) -> None:
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.n_jobs = n_jobs
        self.transform_time_target = transform_time_target
        self.model_kwargs = model_kwargs
        self.backend = "xgboost"
        self.model = self._build_model()

    def _build_model(self):
        """构建 XGBoost 多输出回归器。"""
        try:
            from xgboost import XGBRegressor

            base_model = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                min_child_weight=self.min_child_weight,
                gamma=self.gamma,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=0,
                tree_method="hist",
                **self.model_kwargs,
            )
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor

            self.backend = "sklearn_hist_gradient_boosting"
            base_model = HistGradientBoostingRegressor(
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                **self.model_kwargs,
            )

        return MultiOutputRegressor(base_model)

    def fit(self, X, y):
        """训练多目标回归模型。"""
        y_array = transform_targets_for_fit(
            y,
            transform_time_target=getattr(self, "transform_time_target", False),
        )
        self.model.fit(X, y_array)
        return self

    def predict(self, X):
        """预测最大余震震级和发生时间。"""
        raw_preds = self.model.predict(X)
        return inverse_transform_model_predictions(
            raw_preds,
            transform_time_target=getattr(self, "transform_time_target", False),
        )


def build_model(model_name: str, **kwargs):
    """根据模型名称构建预测模型。"""
    normalized_name = model_name.lower()
    if normalized_name in {"baseline_lgbm", "lgbm", "lightgbm"}:
        return BaselineLGBM(**kwargs)
    if normalized_name in {"baseline_xgboost", "xgboost", "xgb"}:
        return BaselineXGBoost(**kwargs)
    if normalized_name in {"baseline_quantile_lgbm", "quantile_lgbm", "qlgbm"}:
        return BaselineQuantileLGBM(**kwargs)
    raise ValueError(f"未知模型名称: {model_name}")


# ============================================================
#  时间分位数回归模型（改善时间预测的长尾分布偏差）
# ============================================================

class BaselineQuantileLGBM:
    """
    多分位数 LightGBM 回归模型，专门改善时间预测的长尾分布问题。

    训练时对多个分位数（默认 [0.1, 0.25, 0.5, 0.75, 0.9]）各训练一个模型。
    推理时输出中位数 (q=0.5) 作为点预测，同时可提供不确定性区间。

    震级目标仍使用标准 MSE 回归。
    """

    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 300,
        learning_rate: float = 0.03,
        num_leaves: int = 63,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.7,
        reg_alpha: float = 0.05,
        reg_lambda: float = 1.0,
        n_jobs: int = -1,
        transform_time_target: bool = True,
        device: str = "cpu",
        quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
        **model_kwargs,
    ) -> None:
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs
        self.transform_time_target = transform_time_target
        self.device = device
        self.quantiles = tuple(quantiles)
        self.model_kwargs = model_kwargs
        self.backend = "lightgbm_quantile"
        self.mag_model = None
        self.time_models: dict[float, object] = {}

    def _build_single_lgbm(self, objective: str, alpha: float | None = None):
        """构建单个 LightGBM 回归器。"""
        try:
            from lightgbm import LGBMRegressor

            params = {
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves,
                "max_depth": self.max_depth,
                "min_child_samples": self.min_child_samples,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "reg_alpha": self.reg_alpha,
                "reg_lambda": self.reg_lambda,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                "verbosity": -1,
                **self.model_kwargs,
            }
            if self.device == "cuda":
                params["device"] = "cuda"
            if objective == "quantile" and alpha is not None:
                params["objective"] = "quantile"
                params["alpha"] = alpha
            else:
                params["objective"] = "regression"
            return LGBMRegressor(**params)
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor

            if objective == "quantile" and alpha is not None:
                return HistGradientBoostingRegressor(
                    loss="quantile", quantile=alpha,
                    max_iter=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_leaf_nodes=self.num_leaves,
                    max_depth=None if self.max_depth == -1 else self.max_depth,
                    random_state=self.random_state,
                )
            return HistGradientBoostingRegressor(
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_leaf_nodes=self.num_leaves,
                max_depth=None if self.max_depth == -1 else self.max_depth,
                random_state=self.random_state,
            )

    def fit(self, X, y):
        """训练多分位数模型。mag 用标准回归，time 用分位数回归。"""
        y_array = np.asarray(y, dtype=float)
        if y_array.ndim != 2 or y_array.shape[1] != 2:
            raise ValueError("BaselineQuantileLGBM 要求 y 为两列：[震级, 时间]。")

        # 震级模型（标准 MSE 回归）
        self.mag_model = self._build_single_lgbm("regression")
        self.mag_model.fit(X, y_array[:, 0])

        # 时间分位数模型
        time_target = y_array[:, 1]
        for q in self.quantiles:
            model = self._build_single_lgbm("quantile", alpha=float(q))
            model.fit(X, time_target)
            self.time_models[float(q)] = model
        return self

    def predict(self, X):
        """输出 [mag, time_median] 预测。"""
        mag_pred = np.asarray(self.mag_model.predict(X), dtype=float)
        if 0.5 in self.time_models:
            time_pred = np.asarray(self.time_models[0.5].predict(X), dtype=float)
        else:
            time_preds = [
                np.asarray(m.predict(X), dtype=float) for m in self.time_models.values()
            ]
            time_pred = np.mean(time_preds, axis=0)
        return np.column_stack([mag_pred, time_pred])

    def predict_quantiles(self, X) -> dict[float, np.ndarray]:
        """返回各分位数的时间预测（用于不确定性分析）。"""
        return {
            q: np.asarray(m.predict(X), dtype=float)
            for q, m in self.time_models.items()
        }
