from __future__ import annotations
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ClusterAnalyzer:

    def __init__(self, method: str = "kmeans", **kwargs):

        self.method = method.lower()
        self.kwargs = kwargs
        self.model: Optional[Any] = None

    def _create_model(self) -> Any:
        """创建指定聚类算法的模型实例。
        
        支持的方法：
        - 'kmeans': K-Means聚类
        - 'gmm': 高斯混合模型
        - 'dbscan': DBSCAN密度聚类
        - 'hierarchical': 层次聚类
        
        Returns:
            sklearn聚类模型实例
            
        Raises:
            ValueError: 当method不被支持时
        """
        if self.method == "kmeans":
            return KMeans(**self.kwargs)
        elif self.method == "gmm":
            return GaussianMixture(**self.kwargs)
        elif self.method == "dbscan":
            return DBSCAN(**self.kwargs)
        elif self.method == "hierarchical":
            return AgglomerativeClustering(**self.kwargs)
        else:
            raise ValueError(f"Unsupported clustering method: {self.method}")

    def fit(self, df: pd.DataFrame) -> ClusterAnalyzer:
        """拟合聚类模型。
        
        在提供的数据上学习聚类结构。自动创建并初始化指定的聚类算法。
        
        Args:
            df: 输入数据框，形状为 (n_samples, n_features)
            
        Returns:
            self，用于链式调用
            
        Raises:
            ValueError: 当method无效时（通过_create_model()传播）
            
        Examples:
            >>> analyzer = ClusterAnalyzer('kmeans', n_clusters=4)
            >>> analyzer.fit(data)
            >>> analyzer  # 返回self用于链式调用
        """
        logger.info(f"Fitting clustering model: {self.method}")
        self.model = self._create_model()
        self.model.fit(df)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """预测样本的聚类标签。
        
        使用已拟合的聚类模型预测新数据的聚类归属。
        对于不支持predict()的算法（如DBSCAN、Hierarchical），
        重新应用fit_predict()方法。
        
        Args:
            df: 输入数据框，形状为 (n_samples, n_features)
            
        Returns:
            聚类标签Series，形状为 (n_samples,)，值为 [0, 1, ..., n_clusters-1]
            
        Raises:
            RuntimeError: 当模型未先拟合时
            
        Examples:
            >>> analyzer = ClusterAnalyzer('kmeans', n_clusters=4)
            >>> analyzer.fit(X_train)
            >>> labels = analyzer.predict(X_test)
            >>> print(labels.value_counts())  # 各聚类的样本数
        """
        if self.model is None:
            raise RuntimeError("Clustering model not fitted")

        if hasattr(self.model, "predict"):
            labels = self.model.predict(df)
        else:
            labels = self.model.fit_predict(df)

        return pd.Series(labels, name="cluster")

    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        """同时拟合和预测聚类标签。
        
        Args:
            df: 输入数据框
            
        Returns:
            聚类标签Series
        """
        self.fit(df)
        return self.predict(df)
