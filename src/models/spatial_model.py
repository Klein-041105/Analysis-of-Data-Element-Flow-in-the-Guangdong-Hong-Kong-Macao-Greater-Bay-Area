from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class SpatialModel:
    """空间分析模型。
    
    支持基于权重矩阵的空间自相关分析。
    """

    def __init__(self, weight_matrix: Optional[pd.DataFrame] = None):
        """初始化SpatialModel。
        
        Args:
            weight_matrix: 空间权重矩阵DataFrame，形状为 (n, n)。
                           行列对应空间单元，元素为单元间的空间权重。
                           通常由邻接关系、距离衰减或其他空间关系定义。
                           
        Attributes:
            weight_matrix: 存储的权重矩阵
            
        Examples:
            >>> # 简单的邻接权重矩阵
            >>> W = pd.DataFrame([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
            >>> spatial = SpatialModel(weight_matrix=W)
        """
        self.weight_matrix = weight_matrix

    def moran_i(self, values: pd.Series) -> float:
        """计算Moran's I空间自相关系数。
        
        Moran's I用于检测空间数据的自相关性：
        - I > 0: 正相关（相似的值聚集在一起）
        - I ≈ 0: 无相关性
        - I < 0: 负相关（不相似的值聚集在一起）
        
        Args:
            values: 空间变量值Series
            
        Returns:
            Moran's I系数（范围约 -1 到 1）
            
        Raises:
            ValueError: 当weight_matrix未提供时
        """
        if self.weight_matrix is None:
            raise ValueError("weight_matrix not provided")

        logger.info("Computing Moran's I...")
        x = values.to_numpy()
        w = self.weight_matrix.to_numpy()
        x_mean = x.mean()
        n = len(x)

        numerator = 0.0
        for i in range(n):
            for j in range(n):
                numerator += w[i, j] * (x[i] - x_mean) * (x[j] - x_mean)

        denominator = np.sum((x - x_mean) ** 2)
        W = np.sum(w)

        I = (n / W) * (numerator / denominator)
        return float(I)
