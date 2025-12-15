from __future__ import annotations
import pandas as pd
from typing import Optional, Dict, Any
import logging

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

logger = logging.getLogger(__name__)


class MLPredictor:

    def __init__(self, model_name: str = "random_forest", **kwargs):

        self.model_name = model_name.lower()
        self.kwargs = kwargs
        self.model: Optional[Any] = None

    def _create_model(self) -> Any:
        """创建指定类型的模型实例。
        
        支持的模型：
        - 'rf' or 'random_forest': RandomForestRegressor
        - 'gbr' or 'gradient_boosting': GradientBoostingRegressor
        - 'svr': Support Vector Regressor
        - 'lr' or 'linear': LinearRegression
        - 'xgb': XGBRegressor（需要已安装）
        
        Returns:
            sklearn或xgboost模型实例
            
        Raises:
            ValueError: 当model_name不被支持时
            ImportError: 当选择XGBoost但未安装时
        """
        if self.model_name in ["rf", "random_forest"]:
            return RandomForestRegressor(**self.kwargs)
        elif self.model_name in ["gbr", "gradient_boosting"]:
            return GradientBoostingRegressor(**self.kwargs)
        elif self.model_name == "svr":
            return SVR(**self.kwargs)
        elif self.model_name in ["lr", "linear"]:
            return LinearRegression(**self.kwargs)
        elif self.model_name == "xgb":
            if not XGB_AVAILABLE:
                raise ImportError("XGBoost is not installed")
            return XGBRegressor(**self.kwargs)
        else:
            raise ValueError(f"Unsupported model name {self.model_name}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MLPredictor:
        """拟合模型。
        
        在提供的训练数据上学习模型参数。自动创建并初始化指定的回归模型。
        
        Args:
            X: 特征数据框，形状为 (n_samples, n_features)
            y: 目标变量Series，形状为 (n_samples,)
            
        Returns:
            self，用于链式调用
            
        Raises:
            ValueError: 当model_name无效时（通过_create_model()传播）
            ImportError: 当选择XGBoost但未安装时
            
        Examples:
            >>> predictor = MLPredictor('gbr', n_estimators=100)
            >>> predictor.fit(X_train, y_train)
            >>> predictor  # 返回self用于链式调用
        """
        logger.info(f"Fitting model: {self.model_name}")
        self.model = self._create_model()
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """使用已拟合的模型进行预测。
        
        在提供的特征数据上生成预测值。
        
        Args:
            X: 特征数据框，形状为 (n_samples, n_features)。
               必须与训练数据具有相同的列数和语义。
            
        Returns:
            预测结果Series，形状为 (n_samples,)，名称为'prediction'
            
        Raises:
            RuntimeError: 当模型未先拟合（即fit()未被调用）时
            
        Examples:
            >>> predictor = MLPredictor('gbr')
            >>> predictor.fit(X_train, y_train)
            >>> predictions = predictor.predict(X_test)
            >>> print(predictions.shape)  # (n_test,)
        """
        if self.model is None:
            raise RuntimeError("Call fit() first")
        preds = self.model.predict(X)
        return pd.Series(preds, name="prediction")

    def fit_predict(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """同时拟合和预测。
        
        Args:
            X: 特征数据框
            y: 目标变量Series
            
        Returns:
            预测结果Series
        """
        self.fit(X, y)
        return self.predict(X)
