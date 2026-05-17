from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


class BaselineLGBM:
    """
    多输出树模型基线。

    优先使用 LightGBM；若本地未安装 lightgbm，则回退到 sklearn 的
    HistGradientBoostingRegressor，保证训练流程能端到端运行。
    """

    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 300,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        max_depth: int = -1,
        n_jobs: int = -1,
        **model_kwargs,
    ) -> None:
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.model_kwargs = model_kwargs
        self.backend = "lightgbm"
        self.model = self._build_model()

    def _build_model(self):
        """构建多输出回归器。"""
        try:
            from lightgbm import LGBMRegressor

            base_model = LGBMRegressor(
                objective="regression",
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=-1,
                **self.model_kwargs,
            )
        except ImportError:
            self.backend = "sklearn_hist_gradient_boosting"
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
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """预测最大余震震级和发生时间。"""
        return self.model.predict(X)


class BaselineXGBoost:
    """
    XGBoost 多输出回归模型（第二 Baseline）。

    对应 project_plan 第 4.1 节，用于与 LightGBM 做模型融合。
    """

    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 300,
        learning_rate: float = 0.03,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        n_jobs: int = -1,
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
        self.n_jobs = n_jobs
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
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=0,
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
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """预测最大余震震级和发生时间。"""
        return self.model.predict(X)


def build_model(model_name: str, **kwargs):
    """根据模型名称构建预测模型。"""
    normalized_name = model_name.lower()
    if normalized_name in {"baseline_lgbm", "lgbm", "lightgbm"}:
        return BaselineLGBM(**kwargs)
    if normalized_name in {"baseline_xgboost", "xgboost", "xgb"}:
        return BaselineXGBoost(**kwargs)
    raise ValueError(f"未知模型名称: {model_name}")
