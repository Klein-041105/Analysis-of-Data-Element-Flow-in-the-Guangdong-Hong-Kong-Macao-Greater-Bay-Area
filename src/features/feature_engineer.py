from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    PolynomialFeatures,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:

    def __init__(
        self,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        scaling: str = "standard",
        add_polynomial: bool = False,
        poly_degree: int = 2,
    ):

        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.scaling = scaling
        self.add_polynomial = add_polynomial
        self.poly_degree = poly_degree

        self.pipeline: Optional[Pipeline] = None

    def _build_pipeline(self) -> Pipeline:
        """构建特征工程流水线。
        
        包括数值特征缩放、分类特征编码和可选的多项式特征生成。
        
        Returns:
            构建好的sklearn Pipeline对象
            
        Raises:
            ValueError: 当scaling参数不正确时
        """
        logger.info("Building feature engineering pipeline...")

        transformers = []

        # Scaling
        if self.numeric_features:
            if self.scaling == "standard":
                scaler = StandardScaler()
            elif self.scaling == "minmax":
                scaler = MinMaxScaler()
            else:
                raise ValueError("scaling must be 'standard' or 'minmax'.")

            transformers.append(
                ("num", scaler, self.numeric_features)
            )

        # One-hot encoding
        if self.categorical_features:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.categorical_features,
                )
            )

        column_tf = ColumnTransformer(transformers, remainder="drop")

        steps = [("column_transform", column_tf)]

        # Polynomial features
        if self.add_polynomial:
            steps.append(
                ("poly", PolynomialFeatures(self.poly_degree, include_bias=False))
            )

        return Pipeline(steps)

    def fit(self, df: pd.DataFrame) -> FeatureEngineer:
        """拟合特征工程流水线。
        
        在提供的数据上学习特征转换参数，包括：
        - 数值特征的缩放参数（均值、标准差）
        - 分类特征的编码类别
        - 多项式特征的展开规则
        
        Args:
            df: 输入数据框。必须包含初始化时指定的numeric_features和categorical_features
            
        Returns:
            self，用于链式调用
            
        Examples:
            >>> eng = FeatureEngineer(
            ...     numeric_features=['age', 'income'],
            ...     categorical_features=['gender']
            ... )
            >>> eng.fit(df_train)
            >>> eng  # 返回self
        """
        self.pipeline = self._build_pipeline()
        logger.info("Fitting feature engineering pipeline...")
        self.pipeline.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用已拟合的流水线变换数据。
        
        应用学习到的转换参数将输入特征转换为工程后的特征：
        - 数值特征标准化/归一化
        - 分类特征one-hot编码
        - 可选的多项式特征生成
        
        Args:
            df: 输入数据框，形状为 (n_samples, n_features)。
               必须与训练数据具有相同的特征列。
            
        Returns:
            变换后的数据框，包含所有编码和变换后的特征
            
        Raises:
            RuntimeError: 当流水线未先拟合时
            
        Examples:
            >>> eng = FeatureEngineer(numeric_features=['age', 'income'])
            >>> eng.fit(X_train)
            >>> X_transformed = eng.transform(X_test)
            >>> print(X_transformed.shape)  # 可能大于原始特征数（多项式）
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        logger.info("Transforming data via feature engineering pipeline...")
        arr = self.pipeline.transform(df)
        return pd.DataFrame(arr)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """同时拟合和变换数据。
        
        Args:
            df: 输入数据框
            
        Returns:
            变换后的数据框
        """
        return self.fit(df).transform(df)
