from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


class PCAAnalyzer:

    def __init__(self, n_components: Optional[int] = None):

        self.n_components = n_components
        self.pca: Optional[PCA] = None
        self.components_: Optional[pd.DataFrame] = None

    def fit(self, df: pd.DataFrame) -> PCAAnalyzer:
        """在数据上拟合PCA。
        
        学习主成分方向和方差信息。
        
        Args:
            df: 输入数据框，形状为 (n_samples, n_features)
            
        Returns:
            self，用于链式调用
            
        Examples:
            >>> analyzer = PCAAnalyzer(n_components=2)
            >>> analyzer.fit(data)
            >>> analyzer  # 返回self
        """
        logger.info("Fitting PCA...")
        self.pca = PCA(n_components=self.n_components, random_state=42)
        self.pca.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """将数据投影到主成分空间。
        
        使用已拟合的PCA模型将数据变换到低维主成分空间。
        
        Args:
            df: 输入数据框，形状为 (n_samples, n_features)。
               必须与训练数据具有相同的特征数。
            
        Returns:
            投影后的数据框，形状为 (n_samples, n_components)。
            列名为 'PC1', 'PC2', ... 'PCn'
            
        Raises:
            RuntimeError: 当PCA未先拟合时
            
        Examples:
            >>> analyzer = PCAAnalyzer(n_components=2)
            >>> analyzer.fit(X_train)
            >>> X_pca = analyzer.transform(X_test)
            >>> print(X_pca.shape)  # (n_test, 2)
        """
        if self.pca is None:
            raise RuntimeError("PCA not fitted")

        arr = self.pca.transform(df)
        columns = [f"PC{i+1}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=columns)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """同时拟合和变换数据。
        
        Args:
            df: 输入数据框
            
        Returns:
            PCA变换后的数据框
        """
        self.fit(df)
        return self.transform(df)

    def explained_variance(self) -> Dict[str, Any]:
        """获取PCA方差解释信息。
        
        Returns:
            包含以下键的字典：
            - 'variance_ratio': 各主成分的方差比例列表
            - 'components': PCA成分矩阵
            
        Raises:
            RuntimeError: 当PCA未拟合时
        """
        if self.pca is None:
            raise RuntimeError("Call fit() first")

        return {
            "variance_ratio": self.pca.explained_variance_ratio_.tolist(),
            "components": self.pca.components_.tolist(),
        }
